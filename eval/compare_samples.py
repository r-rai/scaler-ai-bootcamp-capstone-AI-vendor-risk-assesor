"""
eval/compare_samples.py — how noisy is extraction run to run? Compares two cached samples of the SAME prompt. No LLM calls.

Sample A is the normal cache entry, sample B the one drawn with `run_kaggle.py --cache-salt <salt>`. For every page both samples
read, the claims are compared with the paraphrase probe's own signature (so the two numbers are comparable); for every case both
samples are carried through the same deterministic pipeline and the verdicts are compared. A "flip" is one (case, requirement)
verdict, OVERALL included, that differs between A and B.

Decision rule, written down BEFORE the salted run (no number below was seen when it was written):
  * N = number of flipped verdicts on the compared cases. N is the noise floor.
  * a later config change (new prompt, new strategy) that moves at most N verdicts on the same cases is NOT shown to be an improvement
    or a regression; only differences clearly above N count.
  * labels are used only to report each sample's own score; nothing is fitted to them, and nothing here picks a "better" sample.

  python eval/compare_samples.py --split train --subset dev --salt noise1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("VRA_CACHE", "0")

import kaggle_policy as kp  # noqa: E402
import run_kaggle as rk  # noqa: E402
from assessor.llm import LLM  # noqa: E402
from assessor.schemas import Claim  # noqa: E402
from assessor.tracing import Budget  # noqa: E402


def load_sample(cache: Path, model: str, page: dict, salt: str) -> dict | None:
    path = cache / f"{kp.cache_key(model, page, salt)}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    return {"status": "ok", "claims": rec["claims"], "in": rec["in"], "out": rec["out"], "cached": True}


def page_agreement(claims_a: list[dict], claims_b: list[dict]) -> dict[str, bool]:
    """Per field the page says anything about in either sample: did both samples extract the same thing?"""
    sa = kp.claim_signature(kp.normalise_page_claims([Claim.model_validate(c) for c in claims_a])[0])
    sb = kp.claim_signature(kp.normalise_page_claims([Claim.model_validate(c) for c in claims_b])[0])
    return {f: sa.get(f) == sb.get(f) for f in set(sa) | set(sb)}


def flips(res_a: dict, res_b: dict) -> list[tuple[str, str, str]]:
    """(requirement, verdict in A, verdict in B) for every verdict that differs; OVERALL included."""
    out = [(rid, va, res_b["verdicts"].get(rid)) for rid, va in res_a["verdicts"].items() if res_b["verdicts"].get(rid) != va]
    if res_a["overall"] != res_b["overall"]:
        out.append(("OVERALL", res_a["overall"], res_b["overall"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data_kaggle"))
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--subset", choices=["dev", "holdout", "all"], default="dev")
    ap.add_argument("--salt", required=True, help="the --cache-salt of sample B")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if not a.salt:
        sys.exit("--salt must be non-empty: sample A is the unsalted cache")

    data = Path(a.data)
    cases, pages = rk.load_cases(data), rk.load_pages(data)
    reqs = json.loads((data / "requirements.json").read_text(encoding="utf-8"))
    rl = rk.load_module("kaggle_rules_lite", data / "rules_lite.py")
    metric = rk.load_module("kaggle_metric", data / "metric.py")
    ids = rk.select_cases(cases, a.split, a.subset, None, None)
    out = Path(a.out) if a.out else ROOT / "reports" / f"kaggle_noise_{a.split}_{a.subset}_{a.salt}"
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM.extractor(budget=Budget(max_usd=0.0, max_seconds=0, max_tokens=0, max_tool_calls=0))
    if llm.mock_mode:
        sys.exit("needs the real provider from .env: the cache key contains the model name")
    cache = data / "cache" / "extract"

    pages_total = pages_both = identical = 0
    pool_n = pool_identical = 0         # the paraphrase probe's pool: official pages that yielded claims in sample A
    per_field, per_field_diff = Counter(), Counter()
    page_diffs: list[dict] = []
    case_rows, all_flips = [], []
    preds_a, preds_b = {}, {}
    skipped: list[str] = []
    for cid in ids:
        recs_a, recs_b = rk.triage(cases[cid], pages[cid]), rk.triage(cases[cid], pages[cid])
        ok = True
        for ra, rb in zip(recs_a, recs_b):
            if not rk.readable(ra):
                continue
            pages_total += 1
            ra["extract"], rb["extract"] = load_sample(cache, llm.model, ra["page"], ""), load_sample(cache, llm.model, rb["page"], a.salt)
            if ra["extract"] is None or rb["extract"] is None:
                ok = False
                continue
            pages_both += 1
            agree = page_agreement(ra["extract"]["claims"], rb["extract"]["claims"])
            identical += all(agree.values())
            if ra["role"] == "official" and ra["extract"]["claims"]:
                pool_n += 1
                pool_identical += all(agree.values())
            for f, same in agree.items():
                per_field[f] += 1
                if not same:
                    per_field_diff[f] += 1
                    page_diffs.append({"case_id": cid, "page": ra["page"]["page_id"], "field": f,
                                       "A": [c for c in ra["extract"]["claims"] if c["field"] == f], "B": [c for c in rb["extract"]["claims"] if c["field"] == f]})
        if not ok:
            skipped.append(cid)         # a case with a page missing from either sample is left out of the verdict comparison
            continue
        res_a, res_b = rk.assemble_case(cases[cid], recs_a, reqs, rl), rk.assemble_case(cases[cid], recs_b, reqs, rl)
        fl = flips(res_a, res_b)
        case_rows.append(cid)
        all_flips += [{"case_id": cid, "requirement_id": rid, "A": va, "B": vb} for rid, va, vb in fl]
        for rid, vv in res_a["verdicts"].items():
            preds_a[f"{cid}__{rid}"], preds_b[f"{cid}__{rid}"] = vv, res_b["verdicts"].get(rid)
        preds_a[f"{cid}__OVERALL"], preds_b[f"{cid}__OVERALL"] = res_a["overall"], res_b["overall"]

    n_verdicts = len(preds_a)
    lines = [f"# Noise between two samples of prompt {kp.PROMPT_VERSION}: {a.split}/{a.subset}, salt {a.salt!r}", "",
             f"- cases compared: {len(case_rows)} of {len(ids)} ({len(skipped)} left out: a page missing from one of the samples)",
             f"- pages: {pages_both} read in both samples of {pages_total} readable; **identical claims on every field: {identical} ({identical / max(1, pages_both):.1%})**"
             f"",
             f"- like for like with the paraphrase probe (official pages that yielded claims in A): identical on every field {pool_identical} of {pool_n} "
             f"(**{pool_identical / max(1, pool_n):.1%}**; the probe measured 93.3% on reworded pages, so this is the baseline for that number)",
             f"- **verdicts that flip between A and B: N = {len(all_flips)} of {n_verdicts}** "
             f"(in {len({f['case_id'] for f in all_flips})} cases). Config changes that move at most N verdicts are not shown to matter.", ""]
    if a.split == "train" and preds_a:
        import pandas as pd
        gold = {r["id"]: r["verdict"] for r in csv.DictReader((data / "train_labels.csv").open(encoding="utf-8"))}
        rows = [i for i in preds_a if i in gold]
        sol = pd.DataFrame({"id": rows, "verdict": [gold[i] for i in rows]})
        sa = metric.components(sol, pd.DataFrame({"id": rows, "verdict": [preds_a[i] for i in rows]}))
        sb = metric.components(sol, pd.DataFrame({"id": rows, "verdict": [preds_b[i] for i in rows]}))
        lines += [f"- score of A {sa['score']:.4f}, score of B {sb['score']:.4f} on the same {len(rows)} rows (reported, never used to pick one)",
                  f"- rows wrong vs gold: A {sum(preds_a[i] != gold[i] for i in rows)}, B {sum(preds_b[i] != gold[i] for i in rows)}", ""]
    lines += ["## Field-level disagreement between the samples (differing / compared)", ""]
    lines += [f"- {f:26s} {per_field_diff[f]:4d} / {per_field[f]:4d}" for f in sorted(per_field)]
    if all_flips:
        lines += ["", "## Flipped verdicts by requirement", ""]
        lines += [f"- {r}: {n}" for r, n in Counter(f["requirement_id"] for f in all_flips).most_common()]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "flips.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "requirement_id", "A", "B"])
        w.writeheader()
        w.writerows(all_flips)
    (out / "page_diffs.jsonl").write_text("\n".join(json.dumps(d) for d in page_diffs) + ("\n" if page_diffs else ""), encoding="utf-8")
    print("\n".join(lines))
    print(f"\noutput: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
