"""
eval/run_kaggle.py — an OFFLINE runner for the Kaggle competition "Vendor Verdict: Can We Trust This AI Tool?".

    pages.jsonl ─► page triage (URL host decides who may speak; look-alikes, news, reviews, forums, blogs ignored)
                ─► injection quarantine ─► LLM extraction (one page at a time, cached by content) ─► claims.normalise_claim
                ─► registry holder check ─► newest-`last_updated`-wins merge ─► facts
                ─► rules_lite.evaluate / overall (the competition's own deterministic rules) ─► CSV

No web and no search: everything the runner knows is in the case's pages. The LLM only READS pages; the verdicts come from
deterministic code. Our own assessor/matcher.py runs alongside as a cross-check and every row where it disagrees with
rules_lite is written to disagreements.csv.

The competition files stay in a gitignored folder (default data_kaggle/) and are treated as untrusted data: rules_lite.py and
metric.py are loaded by path with importlib and are never put on sys.path.

  python eval/run_kaggle.py --split train --subset dev --limit 20          # small slice, real LLM, cost-capped
  python eval/run_kaggle.py --split train --subset dev --cache-only        # free re-run from the extraction cache
  python eval/run_kaggle.py --split test                                   # writes submission.csv (never uploaded by this script)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("VRA_CACHE", "0")          # our own content-addressed cache below; no competition text in runs/cache

import kaggle_policy as kp  # noqa: E402
from assessor.llm import LLM, LLMError  # noqa: E402
from assessor.matcher import Matcher  # noqa: E402
from assessor.schemas import AssessmentRequest, Claim, ClaimList, Requirement, VendorFact, VendorProfile  # noqa: E402
from assessor.tracing import Budget, BudgetExceeded, estimate_tokens, price_for  # noqa: E402

REQ_LABELS = ("PASS", "FAIL", "UNKNOWN", "NEEDS_REVIEW")
OVERALL_LABELS = ("BLOCKED", "CONDITIONAL", "READY_FOR_APPROVAL")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ data
def load_cases(data: Path) -> dict[str, dict]:
    with (data / "cases.csv").open(encoding="utf-8") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def load_pages(data: Path) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    with (data / "pages.jsonl").open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            by[p["case_id"]].append(p)
    return by


def select_cases(cases: dict[str, dict], split: str, subset: str, limit: int | None, only: list[str] | None) -> list[str]:
    ids = sorted(c for c, r in cases.items() if r["split"] == split)
    if split == "train" and subset in ("dev", "holdout"):
        # deterministic, label-blind halves: even positions iterate, odd positions are the held-out estimate
        ids = [c for i, c in enumerate(ids) if (i % 2 == 0) == (subset == "dev")]
    if only:
        ids = [c for c in ids if c in set(only)]
    return ids[:limit] if limit else ids


# ------------------------------------------------------------------------------------------------ extraction (cached)
class Extractor:
    """One page in, the model's raw claims out. The cache key covers page text + URL + prompt + schema + model."""

    def __init__(self, llm: LLM, cache_dir: Path, cache_only: bool):
        self.llm, self.dir, self.cache_only = llm, cache_dir, cache_only
        self.model = llm.model                      # part of the cache key, so --cache-only must know it too
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    def run(self, page: dict) -> dict:
        key = kp.cache_key(self.model, page)
        path = self.dir / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            with self.lock:
                self.hits += 1
            return {"status": "ok", "claims": rec["claims"], "in": rec["in"], "out": rec["out"], "cached": True}
        if self.cache_only:
            return {"status": "not_cached", "claims": [], "in": 0, "out": 0, "cached": False}
        res, err = None, ""
        for attempt in range(5):
            try:
                res = self.llm.structured(kp.EXTRACT_SYSTEM, kp.user_prompt(page), ClaimList, stage="extract")
                break
            except BudgetExceeded:
                raise
            except LLMError as exc:
                err = str(exc)[:200]
                if "429" not in err and "rate limit" not in err.lower():
                    break                                   # only a rate limit is worth waiting for
                time.sleep(min(60, 4 * 3 ** attempt))       # 4, 12, 36, 60 s
        if res is None:
            return {"status": "failed", "error": err, "claims": [], "in": 0, "out": 0, "cached": False}
        claims = [c.model_dump(mode="json") for c in res.claims]
        # ~4 chars/token estimates (per-page usage cannot be read back reliably under concurrency); real spend comes from the Budget
        tin = estimate_tokens(kp.EXTRACT_SYSTEM + kp.user_prompt(page))
        tout = estimate_tokens(json.dumps(claims))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model": self.model, "claims": claims, "in": tin, "out": tout}), encoding="utf-8")
        tmp.replace(path)
        with self.lock:
            self.misses += 1
        return {"status": "ok", "claims": claims, "in": tin, "out": tout, "cached": False}


# ------------------------------------------------------------------------------------------------ verdicts
def our_matcher_verdicts(reqs: list[dict], case: dict, facts: dict, as_of: date) -> tuple[dict[str, str], str]:
    matcher = Matcher(None, [Requirement.model_validate(r) for r in reqs])    # the library is only used by PROC-1.1
    vf = {k: VendorFact(field=k, value=v["value"], display=json.dumps(v["value"], default=str)[:300], confidence=1.0,
                        evidence_level=v.get("evidence_level", "documented")) for k, v in facts.items()}
    profile = VendorProfile(vendor=case["vendor"], as_of=as_of, facts=vf)
    res = matcher.evaluate(profile, AssessmentRequest(question=case["question"], vendor=case["vendor"], tier=case["tier"], users=int(case["users"])))
    results = [r for r in res if r.verdict != "NOT_APPLICABLE"]
    return {r.requirement_id: r.verdict for r in results}, Matcher.overall(results)[0]


def strip_internal(facts: dict) -> dict:
    """What rules_lite gets: no holder key, nothing the page did not state."""
    out = {}
    for k, v in facts.items():
        v = json.loads(json.dumps(v))
        if k == "security_certifications":
            for c in v["value"]:
                c.pop("holder", None)
        out[k] = v
    return out


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data_kaggle"))
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--subset", choices=["dev", "holdout", "all"], default="dev", help="train only: label-blind halves by case order")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", default="", help="comma-separated case ids")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-usd", type=float, default=1.0, help="hard cap for LLM spend in THIS run")
    ap.add_argument("--cache-only", action="store_true", help="never call the LLM; pages not in the cache are reported as not extracted")
    ap.add_argument("--subprocessors", choices=["newest_page", "union"], default="newest_page")
    ap.add_argument("--certs", choices=["newest_page", "union_by_name"], default="newest_page")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    data = Path(a.data)
    cases, pages = load_cases(data), load_pages(data)
    reqs = json.loads((data / "requirements.json").read_text(encoding="utf-8"))
    rl = load_module("kaggle_rules_lite", data / "rules_lite.py")
    metric = load_module("kaggle_metric", data / "metric.py")
    ids = select_cases(cases, a.split, a.subset, a.limit or None, [c for c in a.cases.split(",") if c])
    out = Path(a.out) if a.out else ROOT / "reports" / f"kaggle_{a.split}_{a.subset}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    budget = Budget(max_usd=a.max_usd, max_seconds=0, max_tokens=0, max_tool_calls=0)
    llm = LLM.extractor(budget=budget)
    if llm.mock_mode:
        sys.exit("The extractor is on the mock provider: the cache is keyed by model, so set the real provider (via .env) even for --cache-only.")
    extractor = Extractor(llm, data / "cache" / "extract", a.cache_only)
    print(f"extractor: provider={llm.provider} model={llm.model}  prompt={kp.PROMPT_VERSION}  cap=${a.max_usd:.2f}"
          + ("  (cache-only: no calls)" if a.cache_only else ""))

    # ---- 1. triage (free) ----
    plan: dict[str, list[dict]] = {}
    for cid in ids:
        recs = []
        for p in pages[cid]:
            role, why = kp.classify_page(p, cases[cid])
            rec = {"page": p, "role": role, "why": why, "quarantine": [], "extract": None}
            if role != "ignored":
                if len((p.get("text") or "").strip()) < kp.MIN_PAGE_CHARS:
                    rec["role"], rec["why"] = "ignored", "no readable text"
                else:
                    rec["quarantine"] = kp.quarantine_hits(p)
            recs.append(rec)
        plan[cid] = recs
    jobs = [(cid, r) for cid in ids for r in plan[cid] if r["role"] != "ignored" and not r["quarantine"]]
    print(f"{len(ids)} cases, {sum(len(plan[c]) for c in ids)} pages, {len(jobs)} to read "
          f"({Counter(r['role'] for c in ids for r in plan[c])})")

    # ---- 2. extraction ----
    t0, stopped, done = time.time(), "", 0
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futs = {pool.submit(extractor.run, r["page"]): r for _, r in jobs}
        try:
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    r["extract"] = fut.result()
                except BudgetExceeded as exc:
                    stopped = str(exc)
                    for f in futs:
                        f.cancel()
                    break
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(jobs)} pages  ${budget.spent_usd:.4f}  {time.time() - t0:.0f}s", flush=True)
        finally:
            pass
    wall = time.time() - t0

    # ---- 3. per case: normalise -> holder check -> merge -> verdicts ----
    preds: dict[str, str] = {}
    complete: list[str] = []
    case_log: list[dict] = []
    disagreements: list[dict] = []
    for cid in ids:
        case, as_of = cases[cid], date.fromisoformat(cases[cid]["as_of"])
        items, notes, bad = [], [], 0
        for r in plan[cid]:
            if r["role"] == "ignored" or r["quarantine"]:
                continue
            ex = r["extract"]
            if not ex or ex["status"] != "ok":
                bad += 1
                continue
            claims, why = kp.normalise_page_claims([Claim.model_validate(c) for c in ex["claims"]])
            notes += [f"{r['page']['page_id']}: dropped {w}" for w in why]
            pc = kp.PageClaims(r["page"]["page_id"], r["role"], kp.page_date(r["page"]), claims)
            notes += kp.check_holders(pc, case["legal_entity"])
            items.append(pc)
        merged = kp.build_facts(items, as_of, subprocessors=a.subprocessors, certs=a.certs)
        facts = strip_internal(merged.facts)
        v = rl.evaluate(facts, reqs, case["tier"], int(case["users"]), as_of)
        ov = rl.overall(v, reqs)
        ours, ours_ov = our_matcher_verdicts(reqs, case, facts, as_of)
        for rid, vv in v.items():
            if ours.get(rid) != vv:
                disagreements.append({"case_id": cid, "requirement_id": rid, "rules_lite": vv, "assessor_matcher": ours.get(rid)})
        if ours_ov != ov:
            disagreements.append({"case_id": cid, "requirement_id": "OVERALL", "rules_lite": ov, "assessor_matcher": ours_ov})
        is_complete = bad == 0
        if is_complete:
            complete.append(cid)
        for rid, vv in v.items():
            preds[f"{cid}__{rid}"] = vv
        preds[f"{cid}__OVERALL"] = ov
        case_log.append({"case_id": cid, "complete": is_complete, "unextracted_pages": bad, "facts": facts, "notes": notes + merged.notes,
                         "conflicts": merged.conflicts, "verdicts": v, "overall": ov,
                         "pages": [{"id": r["page"]["page_id"], "role": r["role"], "why": r["why"], "date": kp.page_date(r["page"]),
                                    "quarantine": r["quarantine"], "extract": (r["extract"] or {}).get("status"), "error": (r["extract"] or {}).get("error"),
                                    "claims": [c["field"] for c in (r["extract"] or {}).get("claims", [])]}
                                   for r in plan[cid]]})
    (out / "cases.jsonl").write_text("\n".join(json.dumps(c, default=str) for c in case_log) + "\n", encoding="utf-8")
    with (out / "disagreements.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "requirement_id", "rules_lite", "assessor_matcher"])
        w.writeheader()
        w.writerows(disagreements)

    fails = Counter((r["extract"] or {}).get("error", "no result")[:110] for c in ids for r in plan[c]
                  if r["role"] != "ignored" and not r["quarantine"] and (r["extract"] is None or r["extract"]["status"] == "failed"))
    # ---- 4. cost ----
    p_in, p_out = price_for(llm.model, "extract")
    tin = sum(r["extract"]["in"] for c in ids for r in plan[c] if r["extract"])
    tout = sum(r["extract"]["out"] for c in ids for r in plan[c] if r["extract"])
    est_all = (tin * p_in + tout * p_out) / 1e6
    lines = [f"# Kaggle run: {a.split}/{a.subset}", "",
             f"- cases: {len(ids)} ({len(complete)} complete, {len(ids) - len(complete)} with pages not extracted)",
             f"- pages read by the LLM: {len(jobs)}  (cache hits {extractor.hits}, new calls {extractor.misses})",
             f"- spend this run: ${budget.spent_usd:.4f}; tokens recorded for these pages in/out {tin:,}/{tout:,} = ${est_all:.4f} at configured prices"
             f" (${est_all / max(1, len(ids)):.5f} per case, {len(jobs) / max(1, len(ids)):.1f} pages per case)",
             f"- wall time {wall:.0f}s with {a.workers} workers" + (f"; STOPPED: {stopped}" if stopped else ""),
             *( [f"- extraction FAILURES ({sum(fails.values())}): " + "; ".join(f"{n}x {m!r}" for m, n in fails.most_common(4))] if fails else [] ),
             f"- prompt version {kp.PROMPT_VERSION}; our-matcher vs rules_lite disagreements: {len(disagreements)}"]

    # ---- 5. outputs / score ----
    if a.split == "test":
        test_rows = list(csv.DictReader((data / "test.csv").open(encoding="utf-8")))
        if len(complete) == len(ids) and len(ids) == len({r["case_id"] for r in test_rows}):
            with (out / "submission.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "verdict"])
                for r in test_rows:
                    w.writerow([r["id"], preds.get(r["id"], "UNKNOWN" if not r["id"].endswith("__OVERALL") else "CONDITIONAL")])
            lines.append(f"- wrote {out / 'submission.csv'} ({len(test_rows)} rows). NOT uploaded.")
        else:
            lines.append("- submission.csv NOT written: not every test case is complete.")
    elif not complete:
        lines.append("\nNo complete cases, so nothing was scored.")
    else:
        import pandas as pd
        gold = {r["id"]: r["verdict"] for r in csv.DictReader((data / "train_labels.csv").open(encoding="utf-8"))}
        rows = [i for i in gold if i.split("__")[0] in set(complete)]
        sol = pd.DataFrame({"id": rows, "verdict": [gold[i] for i in rows]})
        sub = pd.DataFrame({"id": rows, "verdict": [preds[i] for i in rows]})
        comp = metric.components(sol, sub)
        lines += ["", f"## Score on {len(complete)} complete cases ({len(rows)} rows)", "",
                  f"- **score {comp['score']:.4f}**  = 0.60 x macro-F1 {comp['macro_f1']:.4f} + 0.20 x overall-acc {comp['overall_accuracy']:.4f}"
                  f" + 0.20 x (1 - false-confidence {comp['false_confidence_rate']:.4f})"]
        base = sub.copy()
        base["verdict"] = ["CONDITIONAL" if i.endswith("__OVERALL") else "NEEDS_REVIEW" for i in base["id"]]
        lines.append(f"- constant baseline on the same rows: {metric.components(sol, base)['score']:.4f}")
        merged_df = sol.merge(sub, on="id", suffixes=("_gold", "_pred"))
        merged_df["req"] = merged_df["id"].str.split("__").str[1]
        lines += ["", "### Accuracy per requirement", "", "| requirement | rows | correct | accuracy |", "|---|---|---|---|"]
        for req, g in merged_df.groupby("req"):
            lines.append(f"| {req} | {len(g)} | {(g.verdict_gold == g.verdict_pred).sum()} | {(g.verdict_gold == g.verdict_pred).mean():.3f} |")
        rq = merged_df[merged_df.req != "OVERALL"]
        lines += ["", "### Confusion (rows = gold, columns = predicted), requirement rows", "",
                  "| gold \\ pred | " + " | ".join(REQ_LABELS) + " |", "|---|" + "---|" * len(REQ_LABELS)]
        for g_ in REQ_LABELS:
            lines.append(f"| {g_} | " + " | ".join(str(int(((rq.verdict_gold == g_) & (rq.verdict_pred == p_)).sum())) for p_ in REQ_LABELS) + " |")
        ov = merged_df[merged_df.req == "OVERALL"]
        lines += ["", "Overall (rows = gold, columns = predicted)", "", "| gold \\ pred | " + " | ".join(OVERALL_LABELS) + " |", "|---|---|---|---|"]
        for g_ in OVERALL_LABELS:
            lines.append(f"| {g_} | " + " | ".join(str(int(((ov.verdict_gold == g_) & (ov.verdict_pred == p_)).sum())) for p_ in OVERALL_LABELS) + " |")
        miss = merged_df[merged_df.verdict_gold != merged_df.verdict_pred]
        with (out / "misses.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "gold", "pred"])
            for _, r in miss.iterrows():
                w.writerow([r["id"], r["verdict_gold"], r["verdict_pred"]])
        lines.append(f"\n{len(miss)} of {len(merged_df)} rows differ from gold: see misses.csv, cases.jsonl")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\noutput: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
