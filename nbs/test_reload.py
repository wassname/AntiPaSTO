#!/usr/bin/env python3
"""Test script to verify SVD bases persistence works correctly.

Loads a trained adapter and compares example outputs to the saved TSV.
This validates that the trained rotation parameters remain aligned with SVD bases.

Usage:
    uv run python nbs/test_reload.py outputs/adapters/20260113_071046_g270m-antisym-r64
    # Or use latest:
    uv run python nbs/test_reload.py  # auto-finds most recent adapter
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from antipasto.eval import get_choice_ids
from antipasto.peft_utils.load import load_adapter
from antipasto.train.train_adapter import log_example_outputs


def find_latest_adapter() -> Path:
    """Find most recent adapter directory."""
    adapters_dir = Path("outputs/adapters")
    dirs = sorted([d for d in adapters_dir.iterdir() if d.is_dir()], reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No adapters found in {adapters_dir}")
    return dirs[0]


def main():
    # Determine adapter path
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = find_latest_adapter()
        print(f"Using latest adapter: {results_dir}")

    # Load adapter with same quantization as training
    model, tokenizer, layer_selection = load_adapter(results_dir, quantization_type=None)
    choice_ids = get_choice_ids(tokenizer)

    # Generate outputs at standard coefficients
    coeffs = [-1.0, 0.0, 1.0]
    log_example_outputs(
        model, tokenizer, choice_ids, coeffs,
        title="RELOAD TEST",
        save_folder=None,  # Don't save, just print
    )

    # Load and display expected values from training TSV if exists
    tsv_path = results_dir / "examples_after_training_example_outputs_at_different_steeri.tsv"
    if tsv_path.exists():
        print("\n=== EXPECTED (from training) ===")
        df = pd.read_csv(tsv_path, sep="\t")
        for _, row in df.iterrows():
            text_preview = row["text"].strip()[:30].replace("\n", " ")
            print(f"coeff={row['coeff']:+.1f} | score={row['score']:+.3f} | {text_preview}")
    else:
        print(f"\nNo comparison TSV at {tsv_path}")


if __name__ == "__main__":
    main()
