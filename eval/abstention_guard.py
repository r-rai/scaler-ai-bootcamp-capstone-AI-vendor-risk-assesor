"""
eval/abstention_guard.py — did a prompt change get "stable" only by answering less? Label-free; reads two finished runs' cases.jsonl.

A prompt that stops extracting is trivially stable and not better. Two checks, both written down before the run they judge:
  * per requirement: the share of UNKNOWN + NEEDS_REVIEW verdicts in B stays within --max-shift points of A
  * per enum field (trains_on_customer_data, dpa_available, sso_saml): B still extracts at least --min-ratio of the values A extracted

  python eval/abstention_guard.py --a reports/kaggle_test_v6 --b reports/kaggle_test_v7
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ABSTAIN = ("UNKNOWN", "NEEDS_REVIEW")
GUARDED_FIELDS = ("trains_on_customer_data", "dpa_available", "sso_saml")


def load(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def abstention_rates(cases: list[dict]) -> dict[str, tuple[int, int]]:
    """requirement -> (abstaining verdicts, verdicts)."""
    out: dict[str, list[int]] = {}
    for c in cases:
        for rid, v in c["verdicts"].items():
            n = out.setdefault(rid, [0, 0])
            n[0] += v in ABSTAIN
            n[1] += 1
    return {k: (a, n) for k, (a, n) in out.items()}


def extracted(cases: list[dict], field: str) -> int:
    return sum(1 for c in cases if (c["facts"].get(field) or {}).get("value") not in (None, [], {}))


def judge(a: list[dict], b: list[dict], max_shift: float, min_ratio: float) -> tuple[list[str], list[str]]:
    lines, problems = [], []
    ra, rb = abstention_rates(a), abstention_rates(b)
    lines += ["| requirement | abstain in A | abstain in B | shift (points) |", "|---|---|---|---|"]
    for rid in sorted(set(ra) | set(rb)):
        (xa, na), (xb, nb) = ra.get(rid, (0, 0)), rb.get(rid, (0, 0))
        pa, pb = 100 * xa / max(1, na), 100 * xb / max(1, nb)
        flag = " **OVER**" if abs(pb - pa) > max_shift else ""
        if flag:
            problems.append(f"{rid}: abstention moved {pb - pa:+.1f} points (limit {max_shift})")
        lines.append(f"| {rid} | {xa}/{na} = {pa:.1f}% | {xb}/{nb} = {pb:.1f}% | {pb - pa:+.1f}{flag} |")
    lines += ["", "| field | values extracted in A | in B | ratio |", "|---|---|---|---|"]
    for f in GUARDED_FIELDS:
        ea, eb = extracted(a, f), extracted(b, f)
        ratio = eb / max(1, ea)
        flag = " **UNDER**" if ratio < min_ratio else ""
        if flag:
            problems.append(f"{f}: B extracts {ratio:.1%} of A's values (limit {min_ratio:.0%})")
        lines.append(f"| {f} | {ea} | {eb} | {ratio:.1%}{flag} |")
    va, vb = Counter(c["overall"] for c in a), Counter(c["overall"] for c in b)
    lines += ["", f"OVERALL distribution A {dict(va)} / B {dict(vb)} (reported, not a gate)"]
    return lines, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="run dir of the reference (e.g. submission 1's test run)")
    ap.add_argument("--b", required=True, help="run dir of the new prompt")
    ap.add_argument("--max-shift", type=float, default=3.0)
    ap.add_argument("--min-ratio", type=float, default=0.95)
    a = ap.parse_args()
    ca, cb = load(Path(a.a)), load(Path(a.b))
    if {c["case_id"] for c in ca} != {c["case_id"] for c in cb}:
        sys.exit("the two runs do not cover the same cases")
    lines, problems = judge(ca, cb, a.max_shift, a.min_ratio)
    print("\n".join(lines))
    print("\nABSTENTION GUARD: " + ("PASSED" if not problems else "FAILED: " + "; ".join(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
