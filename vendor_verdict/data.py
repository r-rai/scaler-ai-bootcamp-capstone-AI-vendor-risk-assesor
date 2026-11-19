"""Competition data loading: a thin wrapper over the runner's own loaders, so every module reads cases the way the runner does."""
from __future__ import annotations

import sys
from pathlib import Path

from .config import ROOT


def _run_kaggle():
    """eval/run_kaggle.py, imported lazily so that importing this module neither touches sys.path nor loads the runner."""
    eval_dir = str(ROOT / "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import run_kaggle
    return run_kaggle


def load(data_dir: Path, split: str, subset: str, limit: int | None = None,
         only: list[str] | None = None) -> tuple[dict[str, dict], dict[str, list[dict]], list[str]]:
    """(cases by id, pages by case id, selected case ids). `subset`, `limit` and `only` go straight to run_kaggle.select_cases.

    The pages mapping is the runner's defaultdict: a case without pages reads as an empty list."""
    rk = _run_kaggle()
    cases = rk.load_cases(Path(data_dir))
    pages = rk.load_pages(Path(data_dir))
    return cases, pages, rk.select_cases(cases, split, subset, limit, only)
