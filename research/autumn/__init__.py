"""The Autumn adaptation of the PRO-LONG agent harness.

A sibling of `research/arc-agi-3/`, not a fork of it: `agent/`, `metrics/` and `utils/`
are imported unchanged, and only the parts that were ARC-shaped -- the environment, the
prompt, the action alphabet, the runner's level/score machinery -- are replaced. Keeping
it in its own directory is what keeps the diff against upstream readable.
"""
import sys as _sys
from pathlib import Path as _Path

# `agent/`, `metrics/` and `utils/` are imported unchanged from the ARC tree next door,
# which is a plain directory rather than an installed package (RGB-Agent has no root
# pyproject). Putting it on the path here is what makes `python -m research.autumn.*`
# work from the repo root, as the launcher's usage line promises.
_ARC = _Path(__file__).resolve().parents[1] / "arc-agi-3"
if _ARC.is_dir() and str(_ARC) not in _sys.path:
    _sys.path.insert(0, str(_ARC))
