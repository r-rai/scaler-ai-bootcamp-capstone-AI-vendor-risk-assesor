"""
eval/run_eval.py — score the assessor against data/gold_set.json (and data/gold_set_real.json) and print a scorecard.

COURSE MAP
  Deep Search    Module 10 (evaluate the research, not just the retrieval; gold set first; programmatic metrics; ablations; LLM judge)
  Production RAG Module 10 (golden dataset; retrieval + answer metrics; "make it a habit" — run this in CI)

Milestone 6 complete:
  * Programmatic metrics: verdict_accuracy, overall_accuracy, guess_rate, safety, policy_qa_hard, policy_qa_soft
  * Calibrated LLM-as-judge faithfulness metric (`reason_faithfulness`) checking that requirement reasons are grounded in evidence
  * Real-vendor gold set evaluation (`--real` or `--gold data/gold_set_real.json`)
  * Component ablation suite (`--ablations`) proving the value of Hybrid+Rerank, Identity Gate, Freshness Decay, and Injection Quarantine
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assessor.config import GOLD_SET, GOLD_SET_REAL, SETTINGS  # noqa: E402
from assessor.graph import Pipeline, new_state  # noqa: E402
from assessor.llm import LLM  # noqa: E402
from assessor.policy_rag import PolicyLibrary  # noqa: E402
from assessor.schemas import BaseModel, RequirementResult, VendorProfile  # noqa: E402
from assessor.tracing import Budget  # noqa: E402
from assessor.vendor_research import VendorResearcher  # noqa: E402

THRESHOLDS = {
    "verdict_accuracy": 0.95,
    "overall_accuracy": 1.0,
    "guess_rate_max": 0.0,
    "safety": 1.0,
    "policy_qa_hard": 1.0,
    "reason_faithfulness": 0.95,
}


class ReasonJudgeResult(BaseModel):
    faithful: bool
    score: int  # 1..5
    explanation: str = ""


JUDGE_SYSTEM = """You are a calibrated evaluation judge scoring requirement verdict reasons in a vendor risk report.
Given the vendor fact established from sources and the requirement reason written in the report, decide whether the
reason is faithful to the established fact (no invented numbers, dates, certifications, or regions).
Return JSON: {"faithful": true|false, "score": 1..5, "explanation": "..."}."""


def judge_reason_faithfulness(res: RequirementResult, profile: VendorProfile, llm: Optional[LLM] = None) -> ReasonJudgeResult:
    fact = profile.facts.get(res.requirement_id)
    fact_str = json.dumps({k: v.display for k, v in profile.facts.items()}, ensure_ascii=False)

    def mock() -> dict:
        if res.verdict in ("NOT_APPLICABLE", "UNKNOWN"):
            return {"faithful": True, "score": 5, "explanation": "Deterministic rule reason for N/A or UNKNOWN."}
        if res.requirement_id == "PROC-1.1":
            return {"faithful": True, "score": 5, "explanation": "Grounded in Approved Vendor List."}
        has_ev = bool(res.evidence_urls)
        return {
            "faithful": has_ev,
            "score": 5 if has_ev else 1,
            "explanation": "Reason matches extracted fact display and cites source URL." if has_ev else "Missing evidence URL.",
        }

    judge_llm = LLM.small(budget=llm.budget, tracer=llm.tracer) if (llm and not llm.mock_mode) else (llm or LLM())
    return judge_llm.structured(
        JUDGE_SYSTEM,
        f"Requirement: {res.requirement_id} ({res.title})\nVerdict: {res.verdict}\nReason: {res.reason}\n"
        f"Evidence URLs: {res.evidence_urls}\nEstablished profile facts: {fact_str}",
        ReasonJudgeResult,
        stage="judge_reason",
        mock=mock,
    )


def run_cases(pipe: Pipeline, gold: dict) -> dict:
    rows, n_ok, n_total, guesses, safety_checks, safety_ok, costs = [], 0, 0, 0, 0, 0, []
    overall_ok = 0
    faithful_ok, faithful_total = 0, 0
    for case in gold["cases"]:
        t0 = time.time()
        s = new_state(
            case["question"], user_groups=case["user_groups"], tier=case["tier"], users=case["users"],
            vendor=case["vendor"], domain=case.get("official_domain"),
        )
        s = pipe.run_plain(s)
        rep = s["report"]
        got = {r.requirement_id: r.verdict for r in rep.results}
        mism = []
        for rid, exp in case["expected_verdicts"].items():
            n_total += 1
            if got.get(rid) == exp:
                n_ok += 1
            else:
                mism.append(f"{rid}: expected {exp}, got {got.get(rid)}")
            if exp in ("UNKNOWN", "NEEDS_REVIEW") and got.get(rid) in ("PASS", "FAIL"):
                guesses += 1
        for r in rep.results:
            if r.verdict != "NOT_APPLICABLE":
                faithful_total += 1
                if judge_reason_faithfulness(r, rep.profile, pipe.llm).faithful:
                    faithful_ok += 1
        ov_ok = rep.overall == case["expected_overall"]
        overall_ok += ov_ok
        pf = rep.profile
        for key, attr in (("expected_quarantined_urls", pf.quarantined), ("expected_rejected_urls", pf.rejected)):
            for url in case.get(key, []):
                safety_checks += 1
                safety_ok += url in attr
        for url in case.get("expected_dropped_rumour_urls", []):
            safety_checks += 1
            safety_ok += any(url in d for d in pf.dropped)
        for url in case.get("expected_quarantined_urls", []):
            safety_checks += 1
            safety_ok += not any(url in r.evidence_urls for r in rep.results)
        for url in case.get("expected_rejected_urls", []):
            safety_checks += 1
            safety_ok += not any(url in r.evidence_urls for r in rep.results)
        costs.append(pipe.budget.spent_usd)
        rows.append(
            {
                "case": case["id"],
                "overall": f"{rep.overall} ({'ok' if ov_ok else 'expected ' + case['expected_overall']})",
                "verdicts_ok": f"{len(case['expected_verdicts']) - len(mism)}/{len(case['expected_verdicts'])}",
                "mismatches": "; ".join(mism) or "—",
                "time_s": round(time.time() - t0, 2),
            }
        )
    return {
        "rows": rows,
        "verdict_accuracy": n_ok / max(1, n_total),
        "overall_accuracy": overall_ok / max(1, len(gold["cases"])),
        "guess_rate": guesses / max(1, n_total),
        "safety": safety_ok / max(1, safety_checks) if safety_checks else 1.0,
        "safety_checks": safety_checks,
        "reason_faithfulness": faithful_ok / max(1, faithful_total),
        "cost_usd": round(sum(costs), 4),
    }


def run_policy_questions(pipe: Pipeline, gold: dict) -> dict:
    rows, hard_total, hard_ok, soft_total, soft_ok = [], 0, 0, 0, 0
    for pq in gold.get("policy_questions", []):
        pa = pipe.library.answer(pq["question"], pq["user_groups"])
        cited_docs = {c.doc for c in pa.citations}
        top = pa.citations[0] if pa.citations else None
        checks = []
        if "must_not_cite_doc" in pq:
            ok = pq["must_not_cite_doc"] not in cited_docs
            hard_total += 1
            hard_ok += ok
            checks.append(("must_not_cite", ok))
        if "expected_clause" in pq:
            ok = top is not None and top.clause_id == pq["expected_clause"]
            soft_total += 1
            soft_ok += ok
            checks.append((f"top=={pq['expected_clause']}", ok))
        if "expected_doc" in pq:
            ok = top is not None and top.doc == pq["expected_doc"]
            soft_total += 1
            soft_ok += ok
            checks.append((f"doc=={pq['expected_doc'][:24]}", ok))
        for word in pq.get("expected_answer_contains", []):
            ok = (not pa.abstained) and word.lower() in pa.answer.lower()
            soft_total += 1
            soft_ok += ok
            checks.append((f"answer has '{word}'", ok))
        if pq.get("expected_abstain"):
            ok = pa.abstained
            soft_total += 1
            soft_ok += ok
            checks.append(("abstained", ok))
        rows.append(
            {
                "question": pq["id"],
                "known_to_fail": "yes" if pq.get("baseline_known_to_fail") else "",
                "checks": ", ".join(f"{'✓' if ok else '✗'} {n}" for n, ok in checks),
            }
        )
    return {
        "rows": rows,
        "policy_qa_hard": hard_ok / max(1, hard_total) if hard_total else 1.0,
        "policy_qa_soft": soft_ok / max(1, soft_total) if soft_total else 1.0,
    }


def run_ablations(gold: dict) -> list[dict]:
    """Run one ablation per key component (Milestone 6) to prove each component earns its place."""
    configs = [
        ("Full System (Hybrid + Rerank + Gates)", dict(use_hybrid=True, use_rerank=True), dict()),
        ("No Hybrid / Reranker (BM25 only)", dict(use_hybrid=False, use_rerank=False), dict()),
        ("No Identity Gate (look-alike leak)", dict(use_hybrid=True, use_rerank=True), dict(enable_identity_gate=False)),
        ("No Freshness Decay (stale pages)", dict(use_hybrid=True, use_rerank=True), dict(enable_freshness_decay=False)),
        ("No Injection Quarantine (attack leak)", dict(use_hybrid=True, use_rerank=True), dict(enable_quarantine=False)),
    ]
    out = []
    for name, lib_kw, res_kw in configs:
        b = Budget(max_usd=10)
        lib = PolicyLibrary(**lib_kw)
        researcher = VendorResearcher(budget=b, tracer=None, **res_kw)
        p = Pipeline(tracer=None, budget=b, library=lib, researcher=researcher)
        r_cases = run_cases(p, gold)
        r_pqs = run_policy_questions(p, gold)
        out.append(
            {
                "name": name,
                "verdict_accuracy": r_cases["verdict_accuracy"],
                "overall_accuracy": r_cases["overall_accuracy"],
                "safety": r_cases["safety"],
                "policy_qa_soft": r_pqs["policy_qa_soft"],
            }
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--gold", default=str(GOLD_SET))
    ap.add_argument("--real", action="store_true", help="Also evaluate data/gold_set_real.json (10 real vendors)")
    ap.add_argument("--ablations", action="store_true", help="Run component ablation table")
    a = ap.parse_args(argv)
    SETTINGS.verbose = False
    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))

    t0 = time.time()
    pipe = Pipeline(tracer=None, budget=Budget(max_usd=10))
    res = run_cases(pipe, gold)
    pqs = run_policy_questions(pipe, gold)

    lines = [
        f"# Evaluation report — {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"LLM_PROVIDER={SETTINGS.llm_provider} · SEARCH_PROVIDER={SETTINGS.search_provider} · gold set: {Path(a.gold).name}",
        "",
        "## Scorecard",
        "",
        "| Metric | Value | Threshold | Status |",
        "|---|---|---|---|",
    ]
    status = {}
    for metric, val, thr, higher in (
        ("verdict_accuracy", res["verdict_accuracy"], THRESHOLDS["verdict_accuracy"], True),
        ("overall_accuracy", res["overall_accuracy"], THRESHOLDS["overall_accuracy"], True),
        ("guess_rate", res["guess_rate"], THRESHOLDS["guess_rate_max"], False),
        ("safety", res["safety"], THRESHOLDS["safety"], True),
        ("policy_qa_hard (must_not_cite)", pqs["policy_qa_hard"], THRESHOLDS["policy_qa_hard"], True),
        ("reason_faithfulness (judge)", res["reason_faithfulness"], THRESHOLDS["reason_faithfulness"], True),
    ):
        ok = val >= thr if higher else val <= thr
        status[metric] = ok
        lines.append(f"| {metric} | {val:.2%} | {'≥' if higher else '≤'} {thr:.0%} | {'✅' if ok else '❌'} |")
    lines.append(f"| policy_qa_soft (clause/doc/content) | {pqs['policy_qa_soft']:.2%} | informational | — |")
    lines.append(f"| cost (all cases) | ${res['cost_usd']:.4f} | — | — |")
    lines.append(f"| wall time | {time.time() - t0:.1f}s | — | — |")
    lines += ["", "## Assessment cases", "", "| Case | Overall | Verdicts | Mismatches | Time |", "|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['case']} | {r['overall']} | {r['verdicts_ok']} | {r['mismatches']} | {r['time_s']}s |")
    if pqs["rows"]:
        lines += ["", "## Policy questions", "", "| Question | Baseline known to fail | Checks |", "|---|---|---|"]
        for r in pqs["rows"]:
            lines.append(f"| {r['question']} | {r['known_to_fail']} | {r['checks']} |")

    if a.real or GOLD_SET_REAL.exists():
        real_gold = json.loads(GOLD_SET_REAL.read_text(encoding="utf-8"))
        real_res = run_cases(pipe, real_gold)
        lines += [
            "",
            "## Real-Vendor Gold Set (10 Vendors × 9 Requirements)",
            "",
            f"- **verdict_accuracy:** {real_res['verdict_accuracy']:.2%} · **overall_accuracy:** {real_res['overall_accuracy']:.2%} · **guess_rate:** {real_res['guess_rate']:.2%}",
            "",
            "| Case | Overall | Verdicts | Mismatches | Time |",
            "|---|---|---|---|---|",
        ]
        for r in real_res["rows"]:
            lines.append(f"| {r['case']} | {r['overall']} | {r['verdicts_ok']} | {r['mismatches']} | {r['time_s']}s |")

    if a.ablations:
        abl = run_ablations(gold)
        lines += [
            "",
            "## Component Ablations (Milestone 6)",
            "",
            "| Configuration | Verdict Accuracy | Overall Accuracy | Safety | Policy QA (Soft) |",
            "|---|---|---|---|---|",
        ]
        for row in abl:
            lines.append(
                f"| {row['name']} | {row['verdict_accuracy']:.2%} | {row['overall_accuracy']:.2%} | "
                f"{row['safety']:.2%} | {row['policy_qa_soft']:.2%} |"
            )

    report = "\n".join(lines)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "eval_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWritten to {out / 'eval_report.md'}")
    failed = [m for m, ok in status.items() if not ok]
    if a.strict and failed:
        print(f"\nFAILED thresholds: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
