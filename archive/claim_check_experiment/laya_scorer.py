"""
eval/laya_scorer.py — side experiment for eval/claim_check.py: scores the same (premise, hypothesis) pairs with Laya
(convaiinnovations/laya, ModernBERT-large typed-decision model), run zero-shot.

Frozen BEFORE running (one pass, nothing tuned afterwards):
  state     = the quote (premise)
  question  = one `noul` (yes/no) question: "Does the text state the following? <hypothesis>"
  supported = P(yes) > 0.5            (the card says the shipped checkpoint is over-confident and that its base is near chance zero-shot
                                        on its own benchmark; this run measures it on OUR claims, nothing more)
Writes the same JSONL as nli_scorer.py ({key, p_yes, supported}), so `claim_check.py judge --scores scores_laya.jsonl` judges it by the
same pre-registered rules.

Runs in its OWN venv (torch + laya + vegaml + transformers 5.x are not project dependencies and must never enter requirements*.txt):
  online once, to fetch the weights:   env -i HOME=... HF_HOME=... python eval/laya_scorer.py --smoke
  then offline, no network, no secrets: unshare -rn env -i HOME=... HF_HOME=... HF_HUB_OFFLINE=1 python eval/laya_scorer.py --pairs ... --out ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SMOKE = [("A man is eating pizza.", "A man eats something."),
         ("The sky is blue.", "The sky is not blue."),
         ("Your data is not used for model training.", "Customer data is not used to train AI models."),
         ("Data residency: workspaces can be hosted in Japan.", "Customer data is stored in Japan.")]


def question(hypothesis: str) -> dict:
    return {"q": {"type": "noul", "instructions": f"Does the text state the following? {hypothesis}"}}


def p_yes(router, premise: str, hypothesis: str) -> tuple[float, str]:
    res = router.predict(premise, question(hypothesis), model="english")
    return float(res["answers"]["q"]["noul"]), str(res.get("routing", {}).get("model"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs")
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true", help="only load the model (downloads the weights when online) and print a few answers")
    a = ap.parse_args()
    from laya import Router
    router = Router()
    for prem, hyp in SMOKE:
        py, route = p_yes(router, prem, hyp)
        print(f"smoke: P(yes)={py:.3f} routed={route} | {prem[:55]!r} => {hyp[:50]!r}", flush=True)
    if a.smoke:
        return 0
    pairs = [json.loads(line) for line in Path(a.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    t0, routes = time.time(), set()
    with Path(a.out).open("w", encoding="utf-8") as f:
        for n, p in enumerate(pairs, 1):
            py, route = p_yes(router, p["premise"], p["hypothesis"])
            routes.add(route)
            f.write(json.dumps({"key": p["key"], "p_yes": round(py, 5), "supported": py > 0.5}) + "\n")
            if n % 200 == 0:
                print(f"  {n}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"scored {len(pairs)} pairs in {time.time() - t0:.0f}s; routed to {sorted(routes)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
