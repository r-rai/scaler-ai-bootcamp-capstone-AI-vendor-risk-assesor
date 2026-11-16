"""scripts/cache_admin.py and scripts/retention.py: they must delete only what they say, and leave an audit record."""
import importlib.util
import os
import time
from pathlib import Path

import pytest

from assessor.cache import Cache
from assessor.db import get_audit_log

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vendor_purge_removes_only_that_vendors_entries_and_is_audited(capsys):
    admin = load("cache_admin")
    cache = Cache()
    cache.put("page", "u1", {"t": 1}, tag="Acme AI")
    cache.put("page", "u2", {"t": 2}, tag="Other Corp")
    assert admin.main(["purge", "--vendor", "acme ai"]) == 0         # case-insensitive
    assert cache.get("page", "u1") is None and cache.get("page", "u2") == {"t": 2}
    assert any(e["action"] == "cache_purge" and e["details"].get("removed") == 1 for e in get_audit_log(10))


def test_purge_needs_exactly_one_selector():
    admin = load("cache_admin")
    assert admin.main(["purge"]) == 2
    assert admin.main(["purge", "--expired", "--all", "--yes"]) == 2


def test_retention_only_selects_old_run_folders_and_never_pending_ones(tmp_path):
    ret = load("retention")
    old, new, pending = tmp_path / "20260101-000000-assess", tmp_path / "20261101-000000-assess", tmp_path / "20260102-000000-assess"
    for d in (old, new, pending):
        d.mkdir()
    ancient = time.time() - 200 * 86400
    for d in (old, pending):
        os.utime(d, (ancient, ancient))
    picked = ret.old_paths(tmp_path, 90, keep={pending.name}, only_dirs=True)
    assert picked == [old]


def test_retention_dry_run_deletes_nothing(tmp_path, monkeypatch):
    ret = load("retention")
    run = tmp_path / "20260101-000000-assess"
    run.mkdir()
    ancient = time.time() - 200 * 86400
    os.utime(run, (ancient, ancient))
    monkeypatch.setattr(ret, "RUNS_DIR", tmp_path)
    assert ret.main(["--run-days", "90"]) == 0 and run.exists()
    assert ret.main(["--run-days", "90", "--apply"]) == 0 and not run.exists()
    assert any(e["action"] == "retention" for e in get_audit_log(10))
