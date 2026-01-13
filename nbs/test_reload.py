#!/usr/bin/env python3
"""Test script to verify SVD bases persistence works correctly.

Loads a trained adapter and compares example outputs to the saved TSV.
This validates that the trained rotation parameters remain aligned with SVD bases.

Usage:
    uv run python nbs/test_reload.py outputs/adapters/20260113_071046_g270m-antisym-r64
    uv run python nbs/test_reload.py  # auto-finds most recent adapter
    uv run python nbs/test_reload.py --no-indices  # test without precomputed_indices
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


def test_reload(results_dir: Path, skip_indices: bool = False):
    """Test loading adapter and compare outputs to saved TSV."""
    
    # Optionally hide precomputed_indices to verify svd_bases is sufficient
    indices_path = results_dir / "0_precomputed_indices.pt"
    indices_backup = results_dir / "0_precomputed_indices.pt.bak"
    
    if skip_indices and indices_path.exists():
        print(">>> Temporarily hiding 0_precomputed_indices.pt to test svd_bases-only reload")
        indices_path.rename(indices_backup)
    
    try:
        # Load adapter
        model, tokenizer, layer_selection = load_adapter(results_dir, quantization_type=None)
        choice_ids = get_choice_ids(tokenizer)

        # Generate outputs
        coeffs = [-1.0, 0.0, 1.0]
        log_example_outputs(
            model, tokenizer, choice_ids, coeffs,
            title="RELOAD TEST" + (" (no precomputed_indices)" if skip_indices else ""),
            save_folder=None,
        )

        # Compare to expected
        tsv_path = results_dir / "examples_after_training_example_outputs_at_different_steeri.tsv"
        if tsv_path.exists():
            print("\n=== EXPECTED (from training) ===")
            df = pd.read_csv(tsv_path, sep="\t")
            for _, row in df.iterrows():
                text_preview = row["text"].strip()[:30].replace("\n", " ")
                print(f"coeff={row['coeff']:+.1f} | score={row['score']:+.3f} | {text_preview}")
        else:
            print(f"\nNo comparison TSV at {tsv_path}")
            
    finally:
        # Restore indices file
        if skip_indices and indices_backup.exists():
            indices_backup.rename(indices_path)
            print(">>> Restored 0_precomputed_indices.pt")


def main():
    # Parse args
    skip_indices = "--no-indices" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    
    if args:
        results_dir = Path(args[0])
    else:
        results_dir = find_latest_adapter()
        print(f"Using latest adapter: {results_dir}")
    
    test_reload(results_dir, skip_indices=skip_indices)


if __name__ == "__main__":
    main()
