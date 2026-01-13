

default:
    #!/bin/bash
    set -e
    uv run python nbs/train.py tiny --quick
    uv run python nbs/train.py tiny
    uv run python nbs/train.py q06b-24gb
    # uv run nbs/test_reload.py
    uv run python nbs/train.py gemma1b-24gb
    uv run python nbs/train.py q4b-24gb