#!/usr/bin/env python3
"""Upload trained AntiPaSTO adapter to HuggingFace Hub.

Usage:
    uv run python scripts/upload_to_hf.py outputs/adapters/20260113_074001_g270m-antisym-r64
    uv run python scripts/upload_to_hf.py outputs/adapters/20260113_074001_g270m-antisym-r64 --repo-id wassname/antipasto-gemma-3-270m-honesty
"""
import json
import sys
from pathlib import Path
import tyro
from dataclasses import dataclass
from huggingface_hub import HfApi, create_repo
from loguru import logger


@dataclass
class UploadConfig:
    adapter_path: Path
    """Path to trained adapter folder"""
    
    repo_id: str = None
    """HuggingFace repo ID (e.g., 'wassname/antipasto-gemma-3-1b-honesty'). Auto-generated if not provided."""
    
    private: bool = False
    """Make repo private"""


# Model name mappings for readable repo names
MODEL_SHORTCUTS = {
    "g270m": "gemma-3-270m",
    "g1b": "gemma-3-1b", 
    "g4b": "gemma-3-4b",
    "q06b": "qwen2.5-0.6b",
    "q4b": "qwen2.5-4b",
    "q14b": "qwen2.5-14b",
}


def generate_repo_id(adapter_path: Path, username: str = "wassname") -> str:
    """Generate HuggingFace repo ID from adapter folder name."""
    name = adapter_path.name
    # Extract model shortcode from name like "20260113_074001_g270m-antisym-r64"
    parts = name.split("_")
    if len(parts) >= 3:
        model_code = parts[2].split("-")[0]  # e.g., "g270m"
        model_name = MODEL_SHORTCUTS.get(model_code, model_code)
        return f"{username}/antipasto-{model_name}-honesty"
    return f"{username}/antipasto-adapter"


def create_model_card(adapter_path: Path, repo_id: str) -> str:
    """Generate HuggingFace model card."""
    # Load training config if available
    config_path = adapter_path / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            training_config = json.load(f)
        base_model = training_config.get("model_name", "unknown")
    else:
        base_model = "unknown"
    
    # Load adapter config
    adapter_config_path = adapter_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path", base_model)
        rank = adapter_config.get("r", "unknown")
    else:
        rank = "unknown"
    
    return f'''---
tags:
  - antipasto
  - peft
  - moral-steering
  - honesty
  - alignment
base_model: {base_model}
library_name: peft
license: apache-2.0
---

# AntiPaSTO: Honesty Steering Adapter

[![arXiv](https://img.shields.io/badge/arXiv-2601.07473-b31b1b.svg)](https://arxiv.org/abs/2601.07473)

🍝 **Anti-Pa**rallel **S**ubspace **T**raining for **O**rdered steering.

This adapter steers language model responses toward honest or deceptive reasoning on moral dilemmas.

## Usage

```python
# Install
pip install git+https://github.com/wassname/AntiPaSTO.git

from antipasto.peft_utils.load import load_adapter
from antipasto.gen import gen, ScaleAdapter

# Load adapter
model, tokenizer, _ = load_adapter("{repo_id}", quantization_type="4bit")

# Steer: coeff > 0 = honest, coeff < 0 = deceptive
prompt = "Should I tell my boss I was late because I overslept?"
with ScaleAdapter(model, coeff=1.0):
    output = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), max_new_tokens=64)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Model Details

- **Base model**: `{base_model}`
- **Adapter rank**: {rank}
- **Training data**: 800 synthetic honest/dishonest contrast pairs
- **Evaluation**: 1,360 Daily Dilemmas across 9 value dimensions

## Citation

```bibtex
@misc{{clark2026antipasto,
  title = {{AntiPaSTO: Self-Supervised Steering of Moral Reasoning}},
  author = {{Clark, Michael J.}},
  year = {{2026}},
  eprint = {{2601.07473}},
  archivePrefix = {{arXiv}},
  primaryClass = {{cs.LG}},
  url = {{https://arxiv.org/abs/2601.07473}}
}}
```
'''


def main(config: UploadConfig):
    adapter_path = Path(config.adapter_path).resolve()
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    
    # Check required files exist
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "0_svd_bases.safetensors",
    ]
    for fname in required_files:
        if not (adapter_path / fname).exists():
            raise FileNotFoundError(f"Required file missing: {adapter_path / fname}")
    
    # Generate repo_id if not provided
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    
    repo_id = config.repo_id or generate_repo_id(adapter_path, username)
    logger.info(f"Uploading to: {repo_id}")
    
    # Create repo if needed
    create_repo(repo_id, exist_ok=True, private=config.private)
    
    # Create model card
    model_card = create_model_card(adapter_path, repo_id)
    readme_path = adapter_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    
    # Files to upload
    files_to_upload = [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "0_svd_bases.safetensors",
        "0_layer_selection.json",
        "training_config.json",
    ]
    
    for fname in files_to_upload:
        fpath = adapter_path / fname
        if fpath.exists():
            logger.info(f"Uploading: {fname}")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
            )
    
    logger.info(f"✅ Uploaded to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    config = tyro.cli(UploadConfig)
    main(config)
