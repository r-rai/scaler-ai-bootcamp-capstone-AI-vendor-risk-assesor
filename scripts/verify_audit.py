"""Verify the audit log's hash chain (assessor/db.py). Exit code 0 = intact, 1 = tampered.

    python scripts/verify_audit.py            # verify
    python scripts/verify_audit.py --tail 20  # verify and print the last 20 records

Each record stores the SHA-256 of the previous record, so editing or deleting any row breaks every hash
after it. Run this on a schedule and before relying on the log in a review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assessor.db import DB_PATH, get_audit_log, verify_audit_chain  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--tail", type=int, default=0)
    a = ap.parse_args(argv)
    res = verify_audit_chain(Path(a.db))
    if a.tail:
        for e in reversed(get_audit_log(a.tail, db_path=Path(a.db))):
            print(f"{e['id']:>6} {e['ts']} {e['actor']:<32} {e['action']:<22} {e['subject']:<34} {json.dumps(e['details'])[:100]}")
    if res["ok"]:
        print(f"OK: {res['rows']} audit record(s), hash chain intact ({a.db}).")
        return 0
    print(f"TAMPERED: record {res['first_bad_id']} — {res['problem']} ({res['rows']} records in {a.db}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
