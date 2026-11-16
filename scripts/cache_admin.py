"""Inspect and purge the caches (runs/cache/cache.db). Every purge is written to the audit log.

    python scripts/cache_admin.py stats
    python scripts/cache_admin.py purge --expired              # drop entries past their TTL
    python scripts/cache_admin.py purge --vendor "Zoho Desk"   # page / extraction entries tagged with that vendor
    python scripts/cache_admin.py purge --ns semantic          # one namespace: search, page, extract, llm, semantic, embeddings
    python scripts/cache_admin.py purge --all --yes            # everything (asks first unless --yes)

Purge a vendor after they publish new terms, or the whole semantic cache after editing a policy by hand.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assessor.cache import Cache  # noqa: E402
from assessor.db import audit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    p = sub.add_parser("purge")
    p.add_argument("--expired", action="store_true")
    p.add_argument("--vendor")
    p.add_argument("--ns")
    p.add_argument("--all", action="store_true")
    p.add_argument("--yes", action="store_true", help="do not ask before --all")
    a = ap.parse_args(argv)

    cache = Cache()
    if a.cmd == "stats":
        stats = cache.stats()
        if not stats:
            print("cache is empty")
        for ns, s in sorted(stats.items()):
            print(f"{ns:<12} entries={s['entries']:<6} hits={s['hits']:<6} size={s['bytes'] / 1024:.0f} KiB")
        return 0

    chosen = [bool(a.expired), bool(a.vendor), bool(a.ns), bool(a.all)]
    if sum(chosen) != 1:
        print("choose exactly one of --expired, --vendor, --ns, --all", file=sys.stderr)
        return 2
    if a.all and not a.yes and input("Purge EVERY cache entry? [y/N] ").strip().lower() != "y":
        print("cancelled")
        return 1
    if a.expired:
        n, what = cache.purge(expired_only=True), "expired entries"
    elif a.vendor:
        n, what = cache.purge(tag=a.vendor), f"entries for vendor '{a.vendor}'"
    elif a.ns:
        n, what = cache.purge(ns=a.ns), f"namespace '{a.ns}'"
    else:
        n, what = cache.purge(), "all entries"
    audit("cache_purge", getpass.getuser(), what, {"removed": n})
    print(f"removed {n} {what}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
