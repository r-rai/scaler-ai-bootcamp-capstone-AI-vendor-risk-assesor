"""
scripts/run_all_tests.py — runs the entire test suite (all tests/test_*.py files).
Uses `pytest` if installed, or a lightweight built-in runner when `pytest` is absent.
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def _install_pytest_shim() -> None:
    try:
        import pytest  # noqa: F401
        return
    except ImportError:
        pass

    mod = types.ModuleType("pytest")

    class _Mark:
        @staticmethod
        def parametrize(argnames, argvalues, ids=None):
            names = [a.strip() for a in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

            def deco(fn):
                fn._parametrize = (names, list(argvalues))
                return fn

            return deco

    def fixture(scope="function"):
        def deco(fn):
            fn._is_fixture = True
            return fn

        return deco

    class _Approx:
        def __init__(self, val, abs=1e-6):
            self.val = val
            self.abs = abs

        def __eq__(self, other):
            return abs(float(other) - float(self.val)) <= self.abs

    class _Skip(Exception):
        pass

    def importorskip(name):
        try:
            return __import__(name)
        except ImportError:
            raise _Skip(f"skipped: {name} not installed")

    mod.mark = _Mark()  # type: ignore[attr-defined]
    mod.fixture = fixture  # type: ignore[attr-defined]
    mod.approx = lambda val, abs=1e-6: _Approx(val, abs=abs)  # type: ignore[attr-defined]
    mod.importorskip = importorskip  # type: ignore[attr-defined]
    mod._Skip = _Skip  # type: ignore[attr-defined]
    sys.modules["pytest"] = mod


def main() -> int:
    _install_pytest_shim()
    import conftest  # noqa: WPS433

    fixtures = {"library": conftest.library()}
    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    passed = 0
    skipped = 0
    failed = 0

    for tf in test_files:
        ns: dict = {}
        code = compile(tf.read_text(encoding="utf-8"), str(tf), "exec")
        exec(code, ns)  # noqa: S102
        local_fixtures = dict(fixtures)
        for name, obj in ns.items():
            if callable(obj) and getattr(obj, "_is_fixture", False):
                sig = inspect.signature(obj)
                kwargs = {p: local_fixtures[p] for p in sig.parameters if p in local_fixtures}
                local_fixtures[name] = obj(**kwargs)

        for name, obj in ns.items():
            if not (callable(obj) and name.startswith("test_")):
                continue
            sig = inspect.signature(obj)
            if hasattr(obj, "_parametrize"):
                pnames, pvals = obj._parametrize
                for row in pvals:
                    row_tuple = row if isinstance(row, (tuple, list)) else (row,)
                    kw = {p: local_fixtures[p] for p in sig.parameters if p in local_fixtures}
                    for pn, pv in zip(pnames, row_tuple):
                        kw[pn] = pv
                    try:
                        obj(**kw)
                        passed += 1
                    except Exception as exc:  # noqa: BLE001
                        if exc.__class__.__name__ == "_Skip":
                            skipped += 1
                        else:
                            failed += 1
                            print(f"FAIL {tf.name}::{name}{row_tuple}: {exc}")
            else:
                kw = {p: local_fixtures[p] for p in sig.parameters if p in local_fixtures}
                try:
                    obj(**kw)
                    passed += 1
                except Exception as exc:  # noqa: BLE001
                    if exc.__class__.__name__ == "_Skip":
                        skipped += 1
                    else:
                        failed += 1
                        print(f"FAIL {tf.name}::{name}: {exc}")

    print(f"\nTest summary: {passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
