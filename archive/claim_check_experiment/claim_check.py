"""
eval/claim_check.py — does the sentence a claim quotes actually support the claim? No LLM calls, no web; label-free on test.

The extractor returns a value AND the quote it rests on. Two cheap, deterministic checks can test that pair on the claims that decide
verdicts (trains_on_customer_data, sso_saml, dpa_available, data_regions). They can only REMOVE claims (a field then becomes UNKNOWN),
so they can fix an invented or unstable claim but never recover a missed one.

  G  grounding (no model): every "..."-separated segment of the quote occurs on the page the model was shown (title + first 12000
     chars), after lower-casing and collapsing punctuation/whitespace. An empty quote fails.
  N  entailment (BERT-style classifier): premise = the quote, hypothesis = a FIXED sentence per (field, value) from HYPOTHESES below.
     Model: cross-encoder/nli-deberta-v3-base (fp32 ONNX). The claim is supported iff entailment is the argmax of
     {contradiction, entailment, neutral}. No threshold exists to tune. (Laya, a typed yes/no decision model, is run as a side
     experiment under the identical rule: supported iff P(yes) > 0.5.)

PRE-REGISTRATION: written BEFORE any claim was scored. Templates, model and rule are frozen; one pass per model; nothing here is
tuned after a result is seen, and a changed template or model is a NEW experiment that must say so.
  Filters judged separately: F_G (drop claims failing G), F_N (drop claims failing N), F_GN (drop on either).
  R1  dev (prompt 9566ad4b6b95 samples A and noise1, prompt f45b4b77c99d sample): unfiltered dev is gold on every row, so ANY verdict
      the filter changes is a regression, and a filter with one is rejected as a hard filter.
  R2  positive control (descriptive, n = 1): holdout c0327's invented `trains_on_customer_data: no` (old prompt) should be removed.
  R3  test, label-free: N_f = verdicts that flip between the two filtered samples of the same prompt. Baselines (unfiltered,
      measured before): 14 for 9566ad4b6b95 (A vs noise1), 10 for f45b4b77c99d (v7 vs v7n1). The unfiltered recount here must give
      exactly those numbers, or the reconstruction is wrong and nothing below counts. Improvement is shown only if N_f <= 5 on BOTH
      prompts; 6-9 is inconclusive; >= 10 is no improvement.
  R4  abstention guard (eval/abstention_guard.py, 3 points / 95 %): filtered vs unfiltered verdicts of the same test sample.
  A filter is a candidate only if R1, R3 and R4 all hold. R2 is reported either way. On test nothing but counts and case ids is
  printed: no quote, claim or page text (the competition forbids hand-labelling test cases).

  python eval/claim_check.py export --out reports/claim_check_X          # unique (quote, hypothesis) pairs + coverage; no model
  python eval/nli_scorer.py --pairs reports/claim_check_X/pairs.jsonl --out reports/claim_check_X/scores_nli.jsonl
  python eval/claim_check.py judge  --out reports/claim_check_X --scores scores_nli.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import types
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # archived in archive/claim_check_experiment/; the repo root is two levels up
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))               # abstention_guard, compare_samples, kaggle_policy, run_kaggle stay in eval/
os.environ.setdefault("VRA_CACHE", "0")

import abstention_guard as ag  # noqa: E402
import compare_samples as cs  # noqa: E402
import kaggle_policy as kp  # noqa: E402
import run_kaggle as rk  # noqa: E402

MODEL = "MiniMax-M3.1-Flash"            # the cache key contains the extractor's model name
OLD_REV, OLD_PROMPT, NEW_PROMPT = "e42203d", "9566ad4b6b95", "f45b4b77c99d"
DECISIVE = ("trains_on_customer_data", "sso_saml", "dpa_available", "data_regions")

# Fixed hypotheses, written from the field definitions in requirements.json / the extraction prompt, not from any page.
HYPOTHESES = {
    "trains_on_customer_data": {
        "no": "Customer data is not used to train AI models.",
        "no_on_paid_plans": "Customer data on paid plans is not used to train AI models.",
        "opt_out_available": "Customer data may be used to train AI models, but customers can opt out.",
        "yes": "Customer data is used to train AI models and customers cannot opt out.",
    },
    "sso_saml": {
        "all_plans": "Single sign-on is available on every plan.",
        "paid_plans": "Single sign-on is available on paid plans.",
        "enterprise_only": "Single sign-on is available only on the enterprise plan.",
        "none": "The vendor does not offer single sign-on.",
    },
    "dpa_available": {
        "yes": "A data processing agreement is available to every customer.",
        "on_request": "A data processing agreement is available on request.",
        "no": "The vendor does not offer a data processing agreement.",
    },
}
REGION_HYPOTHESES = {"available": "Customer data is stored in {country}.", "planned": "Customer data will be stored in {country} in the future."}

SAMPLES = {            # name -> which cached extraction sample (split/subset, prompt, cache salt)
    "dev_old": dict(split="train", subset="dev", prompt="old", salt=""),
    "dev_old_n1": dict(split="train", subset="dev", prompt="old", salt="noise1"),
    "dev_new": dict(split="train", subset="dev", prompt="new", salt=""),
    "hold_old": dict(split="train", subset="holdout", prompt="old", salt=""),
    "test_old": dict(split="test", subset="all", prompt="old", salt=""),               # submission 1
    "test_old_n1": dict(split="test", subset="all", prompt="old", salt="noise1"),
    "test_new": dict(split="test", subset="all", prompt="new", salt=""),
    "test_new_n1": dict(split="test", subset="all", prompt="new", salt="v7n1"),
}
TEST_PAIRS = {"9566ad4b6b95": ("test_old", "test_old_n1", 14), "f45b4b77c99d": ("test_new", "test_new_n1", 10)}
MODES = ("none", "G", "N", "GN")


# --------------------------------------------------------------------------------------------- loading cached samples
def load_old_policy() -> types.ModuleType:
    """The policy file as it was when prompt 9566ad4b6b95 was current (its cache keys hash that prompt). Never touches the working tree."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_REV}:eval/kaggle_policy.py"], capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("kaggle_policy_old")
    sys.modules["kaggle_policy_old"] = mod
    exec(compile(src, f"kaggle_policy@{OLD_REV}", "exec"), mod.__dict__)  # noqa: S102 - our own committed file
    if mod.PROMPT_VERSION != OLD_PROMPT or kp.PROMPT_VERSION != NEW_PROMPT:
        sys.exit(f"prompt versions are not what this experiment was written for: old={mod.PROMPT_VERSION} new={kp.PROMPT_VERSION}")
    return mod


def load_sample(name: str, cases: dict, pages: dict, cache: Path, old) -> tuple[dict, int, int]:
    s = SAMPLES[name]
    pol = old if s["prompt"] == "old" else kp
    plan, readable, miss = {}, 0, 0
    for cid in rk.select_cases(cases, s["split"], s["subset"], None, None):
        recs = rk.triage(cases[cid], pages[cid])
        for r in recs:
            if not rk.readable(r):
                continue
            readable += 1
            path = cache / f"{pol.cache_key(MODEL, r['page'], s['salt'])}.json"
            if path.exists():
                rec = json.loads(path.read_text(encoding="utf-8"))
                r["extract"] = {"status": "ok", "claims": rec["claims"], "in": rec["in"], "out": rec["out"], "cached": True}
            else:
                miss += 1
                r["extract"] = None
        plan[cid] = recs
    return plan, readable, miss


# --------------------------------------------------------------------------------------------- the two checks
def norm(s) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", str(s or "")).lower()).split())


_SEG = re.compile(r"\s*(?:\[\.\.\.\]|\.\.\.|…)\s*")


def shown_text(page: dict) -> str:
    """Normalised text of what the extractor saw: the title and the first 12000 characters of the body (kp.user_prompt)."""
    return " " + norm(f"{page.get('title') or ''} {str(page.get('text') or '')[:12000]}") + " "


def grounding(quote, shown: str) -> str:
    segs = [norm(x) for x in _SEG.split(str(quote or "").strip())]
    segs = [x for x in segs if x]
    if not segs:
        return "no_quote"
    return "grounded" if all(f" {x} " in shown for x in segs) else "not_on_page"


def units(c: dict) -> list[tuple[str, int, str]]:
    """(kind, index, hypothesis) for everything the quote has to support. [] = no template for this value (uncheckable, kept)."""
    f, v = c.get("field"), c.get("value")
    if f in HYPOTHESES and isinstance(v, str):
        h = HYPOTHESES[f].get(re.sub(r"[\s-]+", "_", v.strip().lower()))
        return [("value", 0, h)] if h else []
    if f == "data_regions" and isinstance(v, dict):
        return [(kind, i, REGION_HYPOTHESES[kind].format(country=x.strip()))
                for kind in ("available", "planned") for i, x in enumerate(v.get(kind) or []) if isinstance(x, str) and x.strip()]
    return []


def pair_key(premise: str, hypothesis: str) -> str:
    return hashlib.sha1(f"{premise}\x00{hypothesis}".encode("utf-8")).hexdigest()


def flags(c: dict, page: dict, shown: str, scores: dict | None) -> dict:
    """Everything the filters need about one decisive claim. N is None when no scores were given or the claim is uncheckable."""
    g = grounding(c.get("quote"), shown)
    us = units(c)
    n = None
    if scores is not None and us:
        q = str(c.get("quote") or "").strip()
        n = {}
        for kind, i, h in us:
            if not q:
                n[(kind, i)] = False                                  # nothing to test the claim against
            else:
                k = pair_key(q, h)
                if k not in scores:
                    raise KeyError(f"no score for a pair ({c['field']}); run the scorer on the pairs file made by `export` first")
                n[(kind, i)] = bool(scores[k]["supported"])
    return {"g": g, "units": us, "n": n}


def filter_claims(claims: list[dict], page: dict, scores: dict | None, mode: str) -> list[dict]:
    if mode == "none":
        return claims
    shown, out = shown_text(page), []
    for c in claims:
        if c.get("field") not in DECISIVE:
            out.append(c)
            continue
        fl = flags(c, page, shown, scores if "N" in mode else None)
        if "G" in mode and fl["g"] != "grounded":
            continue
        if "N" in mode and fl["n"] is not None:
            if c["field"] == "data_regions":
                v = {kind: [x for i, x in enumerate(c["value"].get(kind) or []) if fl["n"].get((kind, i), True)] for kind in ("available", "planned")}
                if not v["available"] and not v["planned"]:
                    continue
                out.append({**c, "value": v})
                continue
            if not all(fl["n"].values()):
                continue
        out.append(c)
    return out


# --------------------------------------------------------------------------------------------- running a sample through the pipeline
def run_sample(plan: dict, mode: str, scores: dict | None, cases: dict, reqs: list, rl) -> dict:
    out = {}
    for cid, recs in plan.items():
        recs2 = [r if r["extract"] is None else {**r, "extract": {**r["extract"], "claims": filter_claims(r["extract"]["claims"], r["page"], scores, mode)}}
                 for r in recs]
        out[cid] = rk.assemble_case(cases[cid], recs2, reqs, rl)
    return out


def preds_of(results: dict) -> dict[str, str]:
    p = {}
    for cid, res in results.items():
        if res["bad"]:
            continue                     # a case with a page missing from the cache is left out
        for rid, v in res["verdicts"].items():
            p[f"{cid}__{rid}"] = v
        p[f"{cid}__OVERALL"] = res["overall"]
    return p


def flip_rows(pa: dict, pb: dict) -> list[str]:
    return sorted(i for i in pa if i in pb and pa[i] != pb[i])


# --------------------------------------------------------------------------------------------- commands
def setup(a):
    data = Path(a.data)
    cases, pages = rk.load_cases(data), rk.load_pages(data)
    reqs = json.loads((data / "requirements.json").read_text(encoding="utf-8"))
    rl = rk.load_module("kaggle_rules_lite", data / "rules_lite.py")
    rk.our_matcher_verdicts = lambda *args, **kw: ({}, "")      # the assessor cross-check is irrelevant here and slow; verdicts come from rules_lite
    return data, cases, pages, reqs, rl


def cmd_export(a) -> int:
    data, cases, pages, reqs, rl = setup(a)
    old, cache = load_old_policy(), Path(a.data) / "cache" / "extract"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs: dict[str, dict] = {}
    lines = ["# claim_check export", "", "| sample | cases | readable pages | not in cache | decisive claims | uncheckable (no template) | no quote | quote not on page |",
             "|---|---|---|---|---|---|---|---|"]
    bad = []
    for name in SAMPLES:
        plan, readable, miss = load_sample(name, cases, pages, cache, old)
        if readable and miss / readable > 0.01:
            bad.append(f"{name}: {miss}/{readable} pages missing from the cache: the key derivation is probably wrong")
        n = unch = noq = nop = 0
        for recs in plan.values():
            for r in recs:
                if r["extract"] is None:
                    continue
                shown = shown_text(r["page"])
                for c in r["extract"]["claims"]:
                    if c.get("field") not in DECISIVE:
                        continue
                    n += 1
                    fl = flags(c, r["page"], shown, None)
                    noq += fl["g"] == "no_quote"
                    nop += fl["g"] == "not_on_page"
                    if not fl["units"]:
                        unch += 1
                    q = str(c.get("quote") or "").strip()
                    for _, _, h in fl["units"]:
                        if q:
                            pairs[pair_key(q, h)] = {"key": pair_key(q, h), "premise": q[:1500], "hypothesis": h}
        lines.append(f"| {name} | {len(plan)} | {readable} | {miss} | {n} | {unch} | {noq} | {nop} |")
    (out / "pairs.jsonl").write_text("\n".join(json.dumps(p) for p in pairs.values()) + "\n", encoding="utf-8")
    lines += ["", f"unique (quote, hypothesis) pairs to score: {len(pairs)}"]
    (out / "export.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if bad:
        print("\nPROBLEMS:\n" + "\n".join(bad))
        return 1
    return 0


def cmd_judge(a) -> int:
    data, cases, pages, reqs, rl = setup(a)
    metric = rk.load_module("kaggle_metric", data / "metric.py")
    old, cache, out = load_old_policy(), data / "cache" / "extract", Path(a.out)
    scores = {}
    for line in (out / a.scores).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            scores[r["key"]] = r
    tag = a.scores.replace("scores_", "").replace(".jsonl", "")
    plans = {}
    for name in SAMPLES:
        plans[name], readable, miss = load_sample(name, cases, pages, cache, old)
        if readable and miss / readable > 0.01:
            sys.exit(f"{name}: {miss}/{readable} pages missing from the cache")
    gold = {r["id"]: r["verdict"] for r in csv.DictReader((data / "train_labels.csv").open(encoding="utf-8"))}
    import pandas as pd

    res = {(n, m): run_sample(plans[n], m, scores, cases, reqs, rl) for n in SAMPLES for m in MODES}
    pr = {k: preds_of(v) for k, v in res.items()}
    L = [f"# claim_check judge: {tag}", ""]

    # ---- descriptive: what the checks reject (counts only; test never shows text) ----
    L += ["## What each check rejects (decisive claims; counts)", "",
          "| sample | field | claims | failing G | failing N (any unit) | uncheckable |", "|---|---|---|---|---|---|"]
    for name in SAMPLES:
        tot = Counter()
        for recs in plans[name].values():
            for r in recs:
                if r["extract"] is None:
                    continue
                shown = shown_text(r["page"])
                for c in r["extract"]["claims"]:
                    if c.get("field") not in DECISIVE:
                        continue
                    fl = flags(c, r["page"], shown, scores)
                    f = c["field"]
                    tot[(f, "n")] += 1
                    tot[(f, "G")] += fl["g"] != "grounded"
                    tot[(f, "N")] += fl["n"] is not None and not all(fl["n"].values())
                    tot[(f, "U")] += not fl["units"]
        for f in DECISIVE:
            L.append(f"| {name} | {f} | {tot[(f, 'n')]} | {tot[(f, 'G')]} | {tot[(f, 'N')]} | {tot[(f, 'U')]} |")

    # ---- R1: dev regressions ----
    L += ["", "## R1  dev: verdict rows changed by the filter (unfiltered dev = gold on every row, so every change is a regression)", "",
          "| sample | filter | rows | changed | wrong vs gold | score |", "|---|---|---|---|---|---|"]
    r1_fail = {m: 0 for m in MODES}
    for name in ("dev_old", "dev_old_n1", "dev_new"):
        for m in MODES:
            p0, p1 = pr[(name, "none")], pr[(name, m)]
            changed = flip_rows(p0, p1)
            rows = [i for i in p1 if i in gold]
            wrong = sum(p1[i] != gold[i] for i in rows)
            sc = metric.components(pd.DataFrame({"id": rows, "verdict": [gold[i] for i in rows]}), pd.DataFrame({"id": rows, "verdict": [p1[i] for i in rows]}))["score"]
            r1_fail[m] += len(changed)
            L.append(f"| {name} | {m} | {len(rows)} | {len(changed)} | {wrong} | {sc:.4f} |")
    shown_n = 0
    L += ["", "Dev claims the filters rejected (train data, so text is shown; first 14):", ""]
    for name in ("dev_old", "dev_new"):
        for cid, recs in plans[name].items():
            for r in recs:
                if r["extract"] is None:
                    continue
                shown = shown_text(r["page"])
                for c in r["extract"]["claims"]:
                    if c.get("field") not in DECISIVE or shown_n >= 14:
                        continue
                    fl = flags(c, r["page"], shown, scores)
                    if fl["g"] != "grounded" or (fl["n"] is not None and not all(fl["n"].values())):
                        shown_n += 1
                        L.append(f"- {name} {cid} {r['page']['page_id']} {c['field']}={json.dumps(c['value'])[:70]} G={fl['g']} N={None if fl['n'] is None else list(fl['n'].values())} quote={str(c.get('quote'))[:160]!r}")

    # ---- R2: positive control ----
    L += ["", "## R2  positive control (holdout, prompt 9566ad4b6b95, case c0327; train data)", ""]
    for r in plans["hold_old"].get("c0327", []):
        for c in (r["extract"] or {}).get("claims", []):
            if c.get("field") == "trains_on_customer_data":
                fl = flags(c, r["page"], shown_text(r["page"]), scores)
                L.append(f"- {r['page']['page_id']}: value={c['value']!r} G={fl['g']} N={fl['n']} quote={str(c.get('quote'))[:200]!r}")
    for m in MODES:
        a0, a1 = pr[("hold_old", "none")].get("c0327__AI-2.1"), pr[("hold_old", m)].get("c0327__AI-2.1")
        L.append(f"- filter {m}: c0327 AI-2.1 verdict {a0} -> {a1} (gold {gold.get('c0327__AI-2.1')})")

    # ---- R3 / R4: test, counts only ----
    L += ["", "## R3  test: verdicts that flip between the two samples of the same prompt (label-free)", "",
          "| prompt | filter | flips N_f | cases with a flip | baseline (unfiltered, measured earlier) |", "|---|---|---|---|---|"]
    r3 = {}
    sanity_ok = True
    for prompt, (sa, sb, base) in TEST_PAIRS.items():
        for m in MODES:
            fl = flip_rows(pr[(sa, m)], pr[(sb, m)])
            r3[(prompt, m)] = len(fl)
            L.append(f"| {prompt} | {m} | {len(fl)} | {len({i.split('__')[0] for i in fl})} | {base} |")
        if r3[(prompt, "none")] != base:
            sanity_ok = False
    L += ["", f"Sanity: unfiltered recount reproduces the earlier flip counts (14 and 10): **{'YES' if sanity_ok else 'NO - the reconstruction differs from the original runs; treat everything on test as unreliable'}**", ""]
    L += ["## R4  abstention guard, filtered vs unfiltered (same test sample; 3 points / 95 %)", ""]
    guard = {}
    for name in ("test_old", "test_new"):
        for m in ("G", "N", "GN"):
            ca = [{"case_id": c, "verdicts": r["verdicts"], "facts": r["facts"], "overall": r["overall"]} for c, r in res[(name, "none")].items()]
            cb = [{"case_id": c, "verdicts": r["verdicts"], "facts": r["facts"], "overall": r["overall"]} for c, r in res[(name, m)].items()]
            lines, problems = ag.judge(ca, cb, 3.0, 0.95)
            guard[(name, m)] = problems
            L.append(f"- {name} / {m}: " + ("PASSED" if not problems else "FAILED: " + "; ".join(problems)))
    L += ["", "## Verdict per filter (rules above)", ""]
    for m in ("G", "N", "GN"):
        r1 = r1_fail[m] == 0
        r3ok = all(r3[(p, m)] <= 5 for p in TEST_PAIRS)
        r3inc = all(r3[(p, m)] <= 9 for p in TEST_PAIRS) and not r3ok
        r4 = not guard[("test_old", m)] and not guard[("test_new", m)]
        L.append(f"- F_{m}: R1 {'held' if r1 else f'FAILED ({r1_fail[m]} dev rows changed)'}; R3 N_f = " +
                 ", ".join(f"{p}: {r3[(p, m)]}" for p in TEST_PAIRS) + f" ({'improvement' if r3ok else 'inconclusive' if r3inc else 'no improvement'}); "
                 f"R4 {'passed' if r4 else 'FAILED'} => **{'CANDIDATE' if (r1 and r3ok and r4 and sanity_ok) else 'not a candidate'}**")
    (out / f"report_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "judge"])
    ap.add_argument("--data", default=str(ROOT / "data_kaggle"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", default="scores_nli.jsonl", help="judge: file inside --out written by the scorer")
    a = ap.parse_args()
    return cmd_export(a) if a.cmd == "export" else cmd_judge(a)


if __name__ == "__main__":
    raise SystemExit(main())
