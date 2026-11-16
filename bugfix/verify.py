"""verify.py — did the fix work, and did it break anything? Plus a scanner for 'fixes' that cheat."""

from __future__ import annotations

import re
from pathlib import Path

from .detectors import detect_eval_regressions, run_pytest
from .models import Issue, FixProposal

_ADDED_CHEAT = [
    (re.compile(r"pytest\.mark\.(skip|xfail)|pytest\.skip\(|@unittest\.skip"), "adds a test skip/xfail"),
    (re.compile(r"^\s*except\s*:\s*$|^\s*except\s+(Base)?Exception\s*:\s*(pass)?\s*$"), "adds a broad/bare except"),
    (re.compile(r"PYTEST_CURRENT_TEST|\"pytest\"\s+in\s+sys\.modules|'pytest'\s+in\s+sys\.modules"), "special-cases running under pytest"),
    (re.compile(r"^\s*pass\s*$"), "adds a bare `pass` (possible swallowed error)"),
]
_REMOVED_ASSERT = re.compile(r"^\s*assert\b")


def scan_diff(diff_text: str) -> list[str]:
    """Flag diff patterns that usually mean 'made the check pass' rather than 'fixed the bug'."""
    flags: list[str] = []
    current = "?"
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            for rx, msg in _ADDED_CHEAT:
                if rx.search(line[1:]):
                    flags.append(f"{current}: {msg}: {line[1:].strip()[:80]}")
        elif line.startswith("-") and not line.startswith("---"):
            if _REMOVED_ASSERT.match(line[1:]) and (current.startswith("tests/") or "test" in Path(current).name):
                flags.append(f"{current}: removes an assertion: {line[1:].strip()[:80]}")
    if re.search(r"^\+\+\+ b/(tests/|data/gold_set)", diff_text, re.M):
        flags.append("modifies tests or gold sets (explicitly approved, but review carefully)")
    return flags


def targeted_ids(issue: Issue, proposal: FixProposal) -> list[str]:
    ids = [t for t in proposal.verification_tests if t.startswith("tests/")]
    if not ids and issue.kind == "test":
        ids = [t for t in issue.related if t.startswith("tests/")]
    return ids[:100]


def verify(issue: Issue, proposal: FixProposal, baseline_failing: set[str], root: Path, *, live_env: bool = False) -> dict:
    """Targeted check + full-suite comparison BY TEST ID against the pre-fix baseline."""
    out: dict = {"targeted_ok": None, "new_failures": [], "still_failing": [], "timeout": False, "notes": []}
    ids = targeted_ids(issue, proposal)
    if ids:
        t = run_pytest(ids, live_env=live_env, root=root)
        if t["timeout"]:
            out["timeout"] = True
            out["notes"].append("targeted tests timed out")
        else:
            out["targeted_ok"] = not t["failures"] and t["rc"] in (0,)
            out["still_failing"] = sorted(t["failures"])
            if t["rc"] not in (0, 1):
                out["targeted_ok"] = False
                out["notes"].append(f"pytest exited {t['rc']} on targeted tests")
    elif issue.kind == "eval":
        out["targeted_ok"] = not any(i.signature == issue.signature for i in detect_eval_regressions(live_env=live_env))
    full = run_pytest(live_env=live_env, root=root)
    if full["timeout"]:
        out["timeout"] = True
        out["notes"].append("full suite timed out")
    else:
        out["new_failures"] = sorted(set(full["failures"]) - baseline_failing)
        if full["rc"] not in (0, 1):
            out["notes"].append(f"full-suite pytest exited {full['rc']}")
            out["new_failures"] = out["new_failures"] or ["<pytest could not complete>"]
    return out
