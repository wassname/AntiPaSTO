"""Integration tests: train pipeline end-to-end."""
import re
import subprocess
import sys

import pytest


def _run_train(config: str, *extra_args: str, timeout: int = 300):
    """Run train.py with given config and assert exit code 0."""
    cmd = [sys.executable, "nbs/train.py", config, *extra_args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
    return result


def _parse_val_loss(stdout: str) -> float:
    """Extract validation total loss from the final losses table."""
    # Matches the val row: "val    -9.2    +0.043    +0    -9.1"
    match = re.search(r"^val\s+[\d.e+-]+\s+[\d.e+-]+\s+[\d.e+-]+\s+([\d.e+-]+)", stdout, re.MULTILINE)
    assert match, f"Could not find val loss in output"
    return float(match.group(1))


def test_train_rnd():
    """Smoke test: 5-layer random model, ~3min."""
    result = _run_train("rnd")
    assert "Saved adapter" in result.stdout
    val_loss = _parse_val_loss(result.stdout)
    print(f"rnd val loss: {val_loss}")


@pytest.mark.slow
def test_train_tiny():
    """Larger test: gemma-3-270m-it with --quick, ~5min."""
    result = _run_train("tiny", "--quick", timeout=600)
    assert "Saved adapter" in result.stdout
    val_loss = _parse_val_loss(result.stdout)
    print(f"tiny val loss: {val_loss}")
    assert val_loss < 0, f"Expected negative projection loss, got {val_loss}"
