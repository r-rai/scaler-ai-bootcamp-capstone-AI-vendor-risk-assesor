"""Apply the retention policy: delete old run folders, expired cache entries and rotated logs.

    python scripts/retention.py                  # dry run: say what would go
    python scripts/retention.py --apply          # do it, and write one audit record
    python scripts/retention.py --run-days 90 --log-days 30 --apply

Never touched: the `assessments` table (reports and approvals), the audit log, vendor memory, and the run folder of any
assessment still waiting for approval. The run folder is only an export of what is already in the database, so
removing it loses the raw trace and fetched text, not the record of the decision.
"""
from __future__ import annotations

import argparse
import getpass
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assessor.cache import Cache  # noqa: E402
from assessor.config import RUNS_DIR  # noqa: E402
from assessor.db import DB_PATH, audit  # noqa: E402


def pending_runs(db: Path) -> set[str]:
    if not db.exists():
        return set()
    with sqlite3.connect(db) as c:
        return {r[0] for r in c.execute("SELECT run_id FROM assessments WHERE approval_status = 'pending'")}


def old_paths(root: Path, days: float, *, keep: set[str] = frozenset(), only_dirs: bool = False) -> list[Path]:
    cutoff = time.time() - days * 86400
    out = []
    for p in sorted(root.iterdir()) if root.exists() else []:
        if p.name in keep or (only_dirs and not p.is_dir()):
            continue
        if p.stat().st_mtime < cutoff:
            out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-days", type=float, default=90, help="delete run folders older than this")
    ap.add_argument("--log-days", type=float, default=30, help="delete rotated log files older than this")
    ap.add_argument("--apply", action="store_true", help="actually delete (default is a dry run)")
    a = ap.parse_args(argv)

    keep = pending_runs(DB_PATH)
    runs = [p for p in old_paths(RUNS_DIR, a.run_days, keep=keep, only_dirs=True)
            if p.name[:2].isdigit()]                        # run folders are named YYYYMMDD-...; skip cache/, issues/, logs/
    logs = [p for p in old_paths(RUNS_DIR / "logs", a.log_days) if p.is_file() and p.name != "app.jsonl"]
    print(f"{len(runs)} run folder(s) older than {a.run_days:g} days, {len(logs)} rotated log file(s) older than {a.log_days:g} days"
          f" ({len(keep)} pending-approval run(s) kept)")
    for p in runs + logs:
        print("  " + ("delete " if a.apply else "would delete ") + str(p.relative_to(RUNS_DIR)))
    if not a.apply:
        print("dry run: nothing deleted (use --apply)")
        return 0
    for p in runs:
        shutil.rmtree(p, ignore_errors=True)
    for p in logs:
        p.unlink(missing_ok=True)
    expired = Cache().purge(expired_only=True)
    audit("retention", getpass.getuser(), "runs+logs+cache",
          {"run_folders": len(runs), "log_files": len(logs), "cache_entries": expired,
           "run_days": a.run_days, "log_days": a.log_days})
    print(f"deleted {len(runs)} run folder(s), {len(logs)} log file(s), {expired} expired cache entrie(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
