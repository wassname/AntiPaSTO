"""Dataset creation and loading for AntiPaSTO training."""

import hashlib
import json
import random
from pathlib import Path
from typing import List, Optional

from datasets import Dataset
from loguru import logger
from transformers import PreTrainedTokenizerBase

from antipasto import make_dataset
from antipasto.config import TrainingConfig, proj_root


def _stable_u64(s: str) -> int:
    # Stable across processes and machines (unlike Python's hash()).
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def load_train_suffixes(
    data_dir: Path = proj_root / "nbs/data", max_per_file: Optional[int] = None,
    data_seed: int = 42,
) -> List[str]:
    """Load dataset suffixes from JSON files.
    
    Args:
        data_seed: Fixed seed for data loading (deterministic subset selection).
            Separate from training seed so we compare runs on same data.
    """
    suffix_files = sorted(data_dir.glob("*.json"))
    if not suffix_files:
        raise FileNotFoundError(f"No .json suffix files found in {data_dir}")

    # Deterministic, prefix-stable sampling:
    # - sort files (filesystem order can vary)
    # - shuffle within each file with a local RNG seeded by (data_seed, filename)
    # - round-robin interleave across files (so increasing max_per_file only appends)
    per_file_suffixes: List[List[str]] = []
    for sf in suffix_files:
        with open(sf) as f:
            f_suffixes = json.load(f)
        rng = random.Random(data_seed ^ _stable_u64(sf.name))
        rng.shuffle(f_suffixes)
        if max_per_file is not None:
            f_suffixes = f_suffixes[:max_per_file]
        per_file_suffixes.append(f_suffixes)

    max_len = max((len(x) for x in per_file_suffixes), default=0)
    suffixes: List[str] = []
    for i in range(max_len):
        for f_suffixes in per_file_suffixes:
            if i < len(f_suffixes):
                suffixes.append(f_suffixes[i])

    logger.info(
        f"Loaded {len(suffixes)} suffixes from {data_dir} ({len(suffix_files)} files)"
    )
    return suffixes


def create_train_dataset(config: TrainingConfig, tokenizer: PreTrainedTokenizerBase, max_size: Optional[int] = None):
    """Create contrastive dataset with train/val split."""
    suffixes = load_train_suffixes(
        max_per_file=max_size // 4 if max_size is not None else None,
        data_seed=config.data_seed,
    )

    honest_dataset = make_dataset(
        config.PROMPT,
        config.PERSONAS[0],
        config.PERSONAS[1],
        suffixes,
        tokenizer,
    )

    data = []
    for ex in honest_dataset:
        data.append({"s": ex.positive})
        data.append({"s": ex.negative})

    dataset = Dataset.from_list(data)

    if (max_size is not None) and (max_size < len(dataset) // 2):
        # To get max_size training pairs after split, expand by 1/(1-val_split)
        max_size2 = int(max_size / (1 - config.val_split))
        max_size2 = min(max_size2, len(dataset) // 2)
        dataset = dataset.select(range(max_size2 * 2))
        honest_dataset = honest_dataset[:max_size2]
        logger.debug(
            f"Cropping to {max_size2} pairs (will split to ~{max_size} train)."
        )

    # Split into train/val
    val_size = int(config.val_split * len(honest_dataset))
    train_honest = honest_dataset[val_size:]
    val_honest = honest_dataset[:val_size]

    # Create separate datasets for train and val
    train_data = []
    for ex in train_honest:
        train_data.append({"s": ex.positive})
        train_data.append({"s": ex.negative})

    val_data = []
    for ex in val_honest:
        val_data.append({"s": ex.positive})
        val_data.append({"s": ex.negative})

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    logger.info(
        f"Dataset: {len(train_dataset)} train examples ({len(train_honest)} pairs), "
        f"{len(val_dataset)} val examples ({len(val_honest)} pairs)"
    )

    # Tokenize both
    train_dataset_pt = train_dataset.map(
        lambda examples: tokenizer(examples["s"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["s"],
    )
    train_dataset_pt.set_format(type="torch", columns=["input_ids", "attention_mask"])

    val_dataset_pt = val_dataset.map(
        lambda examples: tokenizer(examples["s"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["s"],
    )
    val_dataset_pt.set_format(type="torch", columns=["input_ids", "attention_mask"])

    s = tokenizer.batch_decode(train_dataset_pt[:2]['input_ids'])
    logger.debug(f"Train dataset preview: {s}")

    return train_honest, train_dataset_pt, val_honest, val_dataset_pt
