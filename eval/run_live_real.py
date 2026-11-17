"""
eval/run_live_real.py — measure LIVE accuracy of the assessor against data/gold_set_real.json.

Why a separate harness: eval/run_eval.py scores the gold set OFFLINE and calls a paid LLM judge per requirement.
This one runs the real pipeline (live search + fetch + extraction) one vendor at a time and only scores verdicts.

What "accuracy" means here: the real gold set stores EXPECTED VERDICTS per requirement (not expected fact values),
so field accuracy = per-requirement verdict accuracy. Verdicts are deterministic given the extracted facts, so a miss
is classified as: unknown_deadline / unknown_not_found / needs_review / wrong_value / guess / other. The gate number
covers only web-driven requirements; PROC-1.1 (internal Approved Vendor List) is reported separately.

Safety rails: per-run cap (VRA_MAX_USD), daily/monthly caps via check_spend_caps(), a batch-total stop
(--max-batch-usd), sequential runs (latency is meaningful), spend saved to llm_usage, the search provider asserted
from the trace, and each run's profile kept under reports/live_real_<stamp>/ so it can be rescored for free.
The gold set is opened read-only and never edited.

  LLM_PROVIDER=mock SEARCH_PROVIDER=offline python eval/run_live_real.py --dry     # free: must equal the offline --real score
  SEARCH_PROVIDER=tavily VRA_CACHE=0 python eval/run_live_real.py --cases real-zoho-zia-t3
  SEARCH_PROVIDER=tavily VRA_CACHE=0 python eval/run_live_real.py                  # all 10 vendors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from assessor.config import GOLD_SET_REAL, SETTINGS  # noqa: E402
from assessor.costs import SpendCapExceeded, check_spend_caps  # noqa: E402
from assessor.db import save_usage  # noqa: E402
from assessor.graph import Pipeline, new_state  # noqa: E402
from assessor.tracing import Budget, Tracer  # noqa: E402

GATE = 0.80                      # extraction may move to the local model only at >= this web-driven accuracy
NON_WEB_FIELDS = {"approved_vendor_list"}   # filled from the internal policy library, not extracted from the web
CATEGORIES = ("correct", "unknown_deadline", "unknown_not_found", "needs_review", "wrong_value", "guess", "other")


def requirement_fields() -> dict[str, str]:
    data = yaml.safe_load((ROOT / "data" / "requirements.yaml").read_text(encoding="utf-8"))
    reqs = data["requirements"] if isinstance(data, dict) else data
    return {r["id"]: r["field"] for r in reqs}


def classify(expected: str, got: str | None, *, deadline_hit: bool) -> str:
    if got == expected:
        return "correct"
    if got == "UNKNOWN":
        return "unknown_deadline" if deadline_hit else "unknown_not_found"
    if got == "NEEDS_REVIEW" and expected in ("PASS", "FAIL"):
        return "needs_review"            # conservative miss: the matcher refused to decide
    if {expected, got} == {"PASS", "FAIL"}:
        return "wrong_value"
    if expected in ("UNKNOWN", "NEEDS_REVIEW") and got in ("PASS", "FAIL"):
        return "guess"                   # the dangerous one: decided where the truth is "not established"
    return "other"


def fact_snapshot(profile, field: str) -> dict | None:
    f = profile.facts.get(field)
    if f is None:
        return None
    return {"display": f.display, "confidence": f.confidence, "evidence_level": f.evidence_level,
            "as_of": str(f.as_of) if f.as_of else None, "sources": f.sources[:3],
            "conflicts": f.conflicts[:3], "unresolved_conflict": f.unresolved_conflict}


def score_requirements(case: dict, results: list, profile, fields: dict[str, str], deadline_hit: bool) -> dict:
    got = {r.requirement_id: r for r in results}
    out = {}
    for rid, exp in case["expected_verdicts"].items():
        r = got.get(rid)
        verdict = r.verdict if r else None
        field = fields.get(rid, "?")
        out[rid] = {
            "field": field, "expected": exp, "got": verdict, "web_driven": field not in NON_WEB_FIELDS,
            "category": classify(exp, verdict, deadline_hit=deadline_hit),
            "reason": (r.reason[:200] if r else None), "fact": fact_snapshot(profile, field)}
    return out


def rescore(src: Path, out_dir: Path) -> int:
    """Free: re-run ONLY the deterministic matcher over the profiles a live run saved (no LLM, no search)."""
    from assessor.matcher import Matcher
    from assessor.policy_rag import PolicyLibrary
    from assessor.schemas import AssessmentRequest, VendorProfile

    saved = json.loads((src / "results.json").read_text(encoding="utf-8"))
    gold = {c["id"]: c for c in json.loads(GOLD_SET_REAL.read_text(encoding="utf-8"))["cases"]}
    matcher, fields, rows = Matcher(PolicyLibrary()), requirement_fields(), []
    for old in saved["rows"]:
        case = gold[old["id"]]
        pf_path = src / f"{case['id']}.profile.json"
        row = {k: v for k, v in old.items() if k not in ("requirements", "got_overall")}
        if not pf_path.exists():
            row.update(requirements={}, got_overall=None)
            rows.append(row)
            continue
        profile = VendorProfile.model_validate_json(pf_path.read_text(encoding="utf-8"))
        req = AssessmentRequest(question=case["question"], vendor=case["vendor"], official_domain=profile.official_domain,
                                tier=case["tier"], users=case["users"], user_groups=case["user_groups"])
        results = matcher.evaluate(profile, req)
        row["got_overall"] = matcher.overall(results)[0]
        row["requirements"] = score_requirements(case, results, profile, fields, bool(old["deadline_hit"]))
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    summ = summarise(rows)
    meta = dict(saved["meta"], stamp=f"rescore of {src.name}", dry=False, provider_check=saved["meta"]["provider_check"])
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "summary": summ, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_dir / "report.md", meta, rows, summ)
    w = summ["web_driven"]
    print(f"rescored {src.name}: web-driven {w['correct']}/{w['n']} guesses={summ['guesses']} -> {out_dir}/report.md")
    return 0


def run_case(case: dict, fields: dict[str, str], out_dir: Path) -> dict:
    budget = Budget()
    tracer = Tracer("livereal")
    pipe = Pipeline(tracer=tracer, budget=budget)
    t0 = time.time()
    error = None
    rep = None
    try:
        s = new_state(case["question"], user_groups=case["user_groups"], tier=case["tier"], users=case["users"],
                      vendor=case["vendor"], domain=case.get("official_domain"))
        s = pipe.run_plain(s)
        rep = s.get("report")
    except Exception as exc:  # noqa: BLE001 — a crashed run is a result, not a reason to stop the batch
        error = f"{type(exc).__name__}: {exc}"
    finally:
        wall = time.time() - t0
        tracer.close()
        save_usage(tracer.run_id, budget.calls, route="assessment", tools=budget.tools)

    events = list(tracer.events)
    providers = Counter(e.get("provider") for e in events if e.get("stage") == "search")
    deadline_hit = any(e.get("stage") == "deadline" for e in events)
    row = {"id": case["id"], "vendor": case["vendor"], "run_id": tracer.run_id, "run_dir": str(tracer.dir),
           "wall_s": round(wall, 1), "usd": round(budget.spent_usd, 5), "deadline_hit": deadline_hit,
           "search_calls": dict(providers), "search_failed": sum(e.get("stage") == "search_failed" for e in events),
           "pages_read": getattr(rep.profile, "pages_read", 0) if rep else 0, "error": error,
           "expected_overall": case["expected_overall"], "got_overall": rep.overall if rep else None,
           "requirements": {}}
    if rep is None:
        return row
    (out_dir / f"{case['id']}.profile.json").write_text(
        json.dumps(rep.profile.model_dump(mode="json"), indent=1, ensure_ascii=False), encoding="utf-8")
    row["requirements"] = score_requirements(case, rep.results, rep.profile, fields, deadline_hit)
    return row


def summarise(rows: list[dict]) -> dict:
    def acc(items: list[dict]) -> dict:
        n = len(items)
        c = Counter(i["category"] for i in items)
        return {"n": n, "correct": c["correct"], "accuracy": round(c["correct"] / n, 4) if n else None,
                "categories": {k: c[k] for k in CATEGORIES if c[k]}}

    reqs = [(rid, r) for row in rows for rid, r in row["requirements"].items()]
    web = [r for _, r in reqs if r["web_driven"]]
    web_no_deadline = [r for r in web if r["category"] != "unknown_deadline"]
    per_req: dict[str, list[dict]] = defaultdict(list)
    for rid, r in reqs:
        per_req[rid].append(r)
    runs_ok = [r for r in rows if not r["error"]]
    walls = sorted(r["wall_s"] for r in runs_ok)
    return {
        "gate": GATE,
        "web_driven": acc(web),
        "web_driven_excluding_deadline_unknowns": acc(web_no_deadline),
        "proc_1_1_internal": acc([r for _, r in reqs if not r["web_driven"]]),
        "per_requirement": {rid: acc(v) for rid, v in sorted(per_req.items())},
        "overall_verdict_matches": sum(r["got_overall"] == r["expected_overall"] for r in rows),
        "cases": len(rows), "crashed": [r["id"] for r in rows if r["error"]],
        "deadline_runs": [r["id"] for r in rows if r["deadline_hit"]],
        "guesses": sum(r["category"] == "guess" for r in web),
        "usd_total": round(sum(r["usd"] for r in rows), 4),
        "wall_s": {"median": walls[len(walls) // 2] if walls else None, "max": walls[-1] if walls else None},
        "search_calls": dict(sum((Counter(r["search_calls"]) for r in rows), Counter())),
    }


def write_markdown(path: Path, meta: dict, rows: list[dict], summ: dict) -> None:
    w = summ["web_driven"]
    verdict = "n/a" if w["accuracy"] is None else ("MEETS" if w["accuracy"] >= GATE else "BELOW")
    L = [f"# Live accuracy vs data/gold_set_real.json — {meta['stamp']}", "",
         f"- **Config:** llm={meta['llm_provider']} (main `{meta['llm_model']}`, small `{meta['llm_small_model']}`), "
         f"search={meta['search_provider']}, VRA_CACHE={meta['cache']}, as_of={meta['as_of']}, deadline={meta['max_seconds']}s",
         f"- **Mode:** {'DRY (mock/offline, scoring validation only)' if meta['dry'] else 'LIVE'}",
         f"- **Web-driven requirement accuracy:** {w['correct']}/{w['n']} = "
         f"{'n/a' if w['accuracy'] is None else format(w['accuracy'], '.1%')} — {verdict} the {GATE:.0%} gate",
         f"- **Excluding deadline-UNKNOWNs:** {summ['web_driven_excluding_deadline_unknowns']['correct']}/"
         f"{summ['web_driven_excluding_deadline_unknowns']['n']}",
         f"- **PROC-1.1 (internal AVL, not extracted):** {summ['proc_1_1_internal']['correct']}/{summ['proc_1_1_internal']['n']}",
         f"- **Guesses (decided where truth is UNKNOWN/NEEDS_REVIEW):** {summ['guesses']}",
         f"- **Overall verdict matches:** {summ['overall_verdict_matches']}/{summ['cases']}; crashed: {summ['crashed'] or 'none'}",
         f"- **Cost:** ${summ['usd_total']} total; wall median {summ['wall_s']['median']} s, max {summ['wall_s']['max']} s; "
         f"deadline hit in {len(summ['deadline_runs'])} run(s)",
         f"- **Search calls by provider:** {summ['search_calls']} (provider check: {meta['provider_check']})", "",
         "Caveats: gold set checked_on 2026-11-01, the live web has moved since; small sample (one run per vendor); "
         "live runs are non-deterministic. Category counts below classify every non-correct verdict.", "",
         "## Per requirement", "", "| requirement | correct | accuracy | misses |", "|---|---|---|---|"]
    for rid, a in summ["per_requirement"].items():
        miss = {k: v for k, v in a["categories"].items() if k != "correct"}
        L.append(f"| {rid} | {a['correct']}/{a['n']} | {a['accuracy']:.0%} | {miss or ''} |")
    L += ["", "## Per vendor", "", "| vendor | overall (exp/got) | wall s | $ | pages | deadline | non-correct requirements |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        bad = [f"{rid}:{x['expected']}→{x['got']} ({x['category']})" for rid, x in r["requirements"].items()
               if x["category"] != "correct"]
        L.append(f"| {r['vendor']} | {r['expected_overall']}/{r['got_overall']} | {r['wall_s']} | {r['usd']} | "
                 f"{r['pages_read']} | {'yes' if r['deadline_hit'] else ''} | {'; '.join(bad) or r['error'] or 'all correct'} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", help="Only these case ids (default: all)")
    ap.add_argument("--max-batch-usd", type=float, default=3.0, help="Stop the batch when cumulative spend passes this")
    ap.add_argument("--dry", action="store_true", help="Mock/offline scoring validation: allowed only with LLM_PROVIDER=mock")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rescore", metavar="DIR", help="Free: re-run the matcher over a previous run's saved profiles")
    a = ap.parse_args(argv)

    if a.rescore:
        src = Path(a.rescore)
        return rescore(src, Path(a.out_dir) if a.out_dir else src.with_name(src.name + "_rescored"))

    live = SETTINGS.llm_provider != "mock" or SETTINGS.search_provider != "offline"
    if a.dry and live:
        print("--dry needs LLM_PROVIDER=mock SEARCH_PROVIDER=offline (refusing to spend on a dry run).")
        return 2
    if live and SETTINGS.cache_enabled:
        print("Refusing to measure live accuracy with caches on: set VRA_CACHE=0.")
        return 2

    gold = json.loads(GOLD_SET_REAL.read_text(encoding="utf-8"))          # read-only
    cases = [c for c in gold["cases"] if not a.cases or c["id"] in a.cases]
    if not cases:
        print("No matching cases.")
        return 2
    fields = requirement_fields()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "reports" / f"live_real_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.verbose = False

    rows: list[dict] = []
    stopped = None
    for i, case in enumerate(cases, 1):
        try:
            check_spend_caps()
        except SpendCapExceeded as exc:
            stopped = f"spend cap: {exc}"
            break
        spent = sum(r["usd"] for r in rows)
        if spent >= a.max_batch_usd:
            stopped = f"batch cap ${a.max_batch_usd} reached (spent ${spent:.3f})"
            break
        print(f"[{i}/{len(cases)}] {case['vendor']} ...", flush=True)
        row = run_case(case, fields, out_dir)
        rows.append(row)
        ok = sum(x["category"] == "correct" for x in row["requirements"].values())
        print(f"    {row['wall_s']} s  ${row['usd']}  {ok}/{len(row['requirements'])} correct  "
              f"overall {row['got_overall']} (exp {row['expected_overall']})  search={row['search_calls']}"
              f"{'  DEADLINE' if row['deadline_hit'] else ''}{'  ERROR ' + row['error'] if row['error'] else ''}", flush=True)

    summ = summarise(rows)
    used = set(summ["search_calls"])
    expected_provider = SETTINGS.search_provider
    provider_check = ("n/a (offline)" if expected_provider == "offline" else
                      "OK" if used <= {expected_provider} else f"CONTAMINATED: expected only {expected_provider}, saw {sorted(map(str, used))}")
    meta = {"stamp": stamp, "dry": a.dry, "llm_provider": SETTINGS.llm_provider, "llm_model": SETTINGS.llm_model,
            "llm_small_model": SETTINGS.llm_small_model, "search_provider": SETTINGS.search_provider,
            "cache": int(SETTINGS.cache_enabled), "as_of": str(SETTINGS.as_of), "max_seconds": SETTINGS.max_seconds,
            "provider_check": provider_check, "stopped": stopped}
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "summary": summ, "rows": rows}, indent=1,
                                                     ensure_ascii=False), encoding="utf-8")
    write_markdown(out_dir / "report.md", meta, rows, summ)
    w = summ["web_driven"]
    print(f"\nweb-driven accuracy {w['correct']}/{w['n']}  guesses={summ['guesses']}  ${summ['usd_total']}  "
          f"provider_check={provider_check}{'  STOPPED: ' + stopped if stopped else ''}")
    print(f"Report: {out_dir}/report.md")
    return 0 if not stopped else 3


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
