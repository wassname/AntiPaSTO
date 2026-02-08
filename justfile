

default:
    #!/bin/bash
    set -e
    uv run pytest tests/test_train.py::test_train_rnd -v
    uv run pytest tests/test_train.py::test_train_tiny -v

    uv run python nbs/train.py tiny
    uv run python nbs/train.py q06b-24gb
    
    uv run python nbs/train.py gemma1b-24gb
    uv run python nbs/train.py q4b-24gb