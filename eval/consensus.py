r"""
eval/consensus.py — consensus extraction (stage A) and escalation of the unsure page-fields (stage B) for the Kaggle "Vendor Verdict" files.

PRE-REGISTRATION. Written on 2026-11-18 BEFORE any new sample is drawn and before any run that reads labels (spec:
docs/superpowers/specs/2026-11-18-kaggle-consensus-escalation-design.md, plan: docs/superpowers/plans/2026-11-18-kaggle-consensus-escalation.md).
Nothing below is changed after a number has been seen. A rule that fails is reported as failed; a threshold or a sample list is never
moved to make a rule pass. Labels are used only to report and to test these rules; nothing is fitted to them. Output (stdout, reports)
holds counts, field names and case ids only: never claims, quotes or page text.

What the code does. Vote at page-field level (one field as read from one readable page) over several samples of ONE unchanged prompt
(9566ad4b6b95, the prompt behind submission #1, selected from commit e42203d and never edited) and ONE model (MiniMax-M3.1-Flash). Silence is
a vote for "absent"; a page that failed to extract in a sample gives no vote. The consensus claims go through the unchanged
run_kaggle.assemble_case, so every requirement row and the OVERALL row stay consistent. Stage B re-reads only the pages behind unsure
page-fields with a stronger model (MiniMax-M3), with the same unchanged prompt.

---------------------------------------------------------------------------------------------------------------------------------
Spec section 6, verbatim
---------------------------------------------------------------------------------------------------------------------------------
These are written before any new sample is drawn, and the module docstring carries the same text. Labels are used only to report and to
test these rules; nothing is fitted to them. The numeric thresholds are my picks (flagged in section 9).

- **R0 plumbing.** With k = 1 the consensus path reproduces the single-sample pipeline verdict for verdict on dev, holdout and test
  (unit test on synthetic claims plus a cache-only regression run on existing data).
- **RA1 no harm on the clean holdout.** On the three holdout samples of prompt 9566ad4b6b95: wrong rows of the k = 3 consensus must be
  <= the mean wrong rows of the three single samples (rounded up), and false-confident rows of the consensus must be <= the mean of the
  three singles (rounded up). If this fails, A is not adopted: keep #1 and report the failure.
- **RA2 sanity on test.** The k = 5 consensus changes at most 30 verdict rows relative to the unsalted 9566 sample (submission #1). More means something
  other than noise is going on: investigate, do not submit.
- **RB0 adjudicator agrees on easy items.** A fixed-seed random draw of 100 settled test page-fields (label-blind): the adjudicator's
  signature equals the unanimous Flash signature on at least 95 of them. If not, the adjudicator reads the task differently and is not used.
- **RB1 calibration on the holdout.** Run the adjudicator on the holdout page-fields that trigger B (both triggers; the holdout gets one sample of the second-opinion prompt for trigger 2). Among rows B changes versus the A-only
  consensus, "wrong to right" must exceed "right to wrong" against gold. If B changes no holdout row, B is reported as **untested** and is
  not adopted unless the user decides otherwise with that stated. Expect little power: the holdout had 3 wrong rows of 1580.
- **RB2 decisiveness.** Report how many page-fields end unresolved; if more than 10 on test, B is reported as not decisive.

Candidate rule, fixed in advance (so no candidate is picked by its score): if R0, RA1, RA2 pass, the candidate is A-only; if RB0, RB1 and
RB2 also pass, the candidate is A+B; otherwise the candidate is none and #1 stays. At most two new submissions come out of this design,
each decided before any new score is seen, each needing the user's explicit yes. Final selection stays the user's call in the web UI.

(Spec 5.2 and 5.3, for the words the rules use: a page-field is SETTLED if at least 4 of 5 votes agree (test) or 3 of 3 (holdout);
trigger 1 of stage B = unsettled; trigger 2 = all samples of the second-opinion prompt f45b4b77c99d agree with each other and differ from the
consensus. Resolution: the adjudicator's strict majority of its 3 samples must equal the Flash consensus, otherwise the item is unresolved
and its claim is dropped, so "not stated means unknown" applies.)

---------------------------------------------------------------------------------------------------------------------------------
Decisions the spec left open (plan D1-D8, approved by the user as written; D2 is the tie rule)
---------------------------------------------------------------------------------------------------------------------------------
- D1 The adjudicator (MiniMax-M3) uses the base prompt 9566ad4b6b95, so stage B changes the model only. 5.3 says "the same unchanged
  extraction prompt" without naming which.
- D2 Ties: if Flash has no strict majority, the adjudicator's strict majority counts only when it equals one of the tied leaders (kind
  tie_broken); otherwise the item is unresolved and its claim is dropped. A confirmed majority changes nothing. B can therefore only break
  ties or drop claims; it never introduces a claim Flash did not produce.
- D3 "Settled" counts agreeing votes in absolute terms (4 with 4 valid votes is settled). A final stage-A run needs every sample to cover
  100% of readable pages (failed pages get re-run first); --allow-partial exists for dry runs only.
- D4 RB0 pool = test page-fields on which every valid Flash vote agrees and the signature is not "absent" (otherwise the check is vacuous).
  Seed 20261118. The adjudicator sample adj1 only; the threshold is agree*100 >= 95*n.
- D5 RB1 "untested" (B changes no holdout row) is not a pass: the candidate falls back to A-only.
- D6 New holdout draws use --no-score, so no per-sample holdout score is printed and the holdout never judges f45b4b77c99d. Consensus
  numbers come from consensus.py only.
- D7 If the candidate rule gives A+B, `candidate` also writes the A-only file (submission_A_only.csv, same self-check), so the user can
  decide yes/no on both files before any new score is read (6: "at most two new submissions ... each decided before any new score is
  seen"). Both decisions go into the results note.
- D8 The vote signature is the spec's claim_signature, which cannot see a certificate's `holder` (and check_holders acts on it for
  registry pages), and the consensus is built from raw claims carried per field, so normalise_page_claims can behave differently on the
  combined claims than on any single sample. Both gaps are measured, not fixed: stage A reports holder_split (a winning signature whose
  carriers disagree on holder) and post_vote_mismatch (the consensus record's signature differs from the voted winner). If either is
  non-empty, plan Task 11 stops and asks the user. Checked at plan time: evidence_level is not a hidden source of noise: rules_lite.py
  reads it only in the certificate rule, and build_facts derives it from issuer and date, both in the signature.

---------------------------------------------------------------------------------------------------------------------------------
Implementation notes (added 2026-11-18 after Tasks 1-9, still before any new sample was drawn and before any label was read for a rule;
none of them changes what a rule above measures)
---------------------------------------------------------------------------------------------------------------------------------
- N1 A page-field no valid Flash sample mentioned has the consensus "absent". Spec 5.3 trigger 2 reaches it: if all second-opinion samples
  agree on a claim there, it goes to the adjudicator. Under D2 the adjudicator can only confirm "absent" or end unresolved there (counted
  toward RB2), so no verdict row can change. Such stand-in items are never in the RB0 pool (its winners must not be "absent", D4).
- N2 rb0 with an empty draw is "untested", never a pass. ra1 with no single samples is "not_run".
- N3 `resolve` refuses to finish (exit 2) unless every planned page has all three adjudicator samples (RB0 pages: adj1) and every Flash and
  second-opinion sample covers 100% of the readable pages: a page the adjudicator never read would otherwise count as unresolved and drop a
  claim. `--allow-partial` is the dry-run override; dry runs of stage-a and resolve save no rule and read no labels.
- N4 RA1 and RB1 are scored on the labels of the holdout cases that are complete (every readable page extracted); RA2 compares the k = 5
  consensus with sample s0, which is submission #1 itself (R0 shows s0 alone reproduces its stored verdicts).
- N5 R0 is re-run (free, seconds) after any later code change, so it stays valid for the code that produces the candidate.

---------------------------------------------------------------------------------------------------------------------------------
Samples (the sample lists are fixed here; salts come from the run headers in reports/*.log and are confirmed by coverage, plan Task 9)
---------------------------------------------------------------------------------------------------------------------------------
Stage-A samples are the base prompt 9566ad4b6b95, model MiniMax-M3.1-Flash. Second-opinion samples are the current prompt f45b4b77c99d,
same model. The list order is the only tie-break anywhere (sample 0 = the unsalted sample).
- test    (split test, subset all):     a = s0 "" (this is submission #1 itself), noise1 "noise1", p9s3 "p9s3", p9s4 "p9s4", p9s5 "p9s5"
                                        second = v7 "", v7n1 "v7n1"
- holdout (split train, subset holdout): a = s0 "", p9h1 "p9h1", p9h2 "p9h2";   second = f45 ""
- dev     (split train, subset dev, R0 only): a = s0 "";   second = none

Not allowed in a vote: mixing prompts or models; hand-labelling or hand-adjudicating any row; reading the leaderboard for labels.

---------------------------------------------------------------------------------------------------------------------------------
Sensitivity analyses (added 2026-11-19, after RA1 and RA2 were recorded and after the label-free carrier-swap check was read, BEFORE the
analysis below reads any label; it changes no rule, threshold, sample list or decision above)
---------------------------------------------------------------------------------------------------------------------------------
Why. D8 found holder_split on both targets (test 102, holdout 74 page-fields, all security_certifications). The carrier-swap check
(label-free, reports/kaggle_consensus/{test,holdout}/carrier_swap.json) showed the test candidate is carrier-invariant (0 rows move) and
3 holdout rows move: c0060__SEC-4.2, c0312__SEC-4.2-T2, c0678__SEC-4.2-T2. The user decided (2026-11-19): first run the RA1 sensitivity
below, labelled, on the holdout; it never changes the carrier rule; if RA1 fails under some carrier, report it as a caveat and stop for
the user; then, if nothing else blocks, Task 12.

RA1 carrier sensitivity (holdout only; counts only; no sample is drawn).
- Space: for every holdout case with a holder_split page-field, every carrier combination over that case's split fields, exactly the
  carrier-swap definition (a carrier = a sample whose signature is the winner, deduplicated by the raw claims of that field; at most 5000
  combinations per case). A verdict row depends only on its own case, so the combinations of different cases are crossed. A case in which
  no combination moves any row contributes its default rows; this is VERIFIED in the run (not assumed) by the label-free gates below.
- Computed per combination: the consensus wrong rows and false-confident rows over the same gold rows RA1 used
  (gold = train_labels.csv rows of the holdout cases, minus incomplete cases), via score_rows.
- The RA1 limits stay as recorded (wrong <= 3, false-confident <= 1): they come from the three single samples, which have one carrier
  each, so no carrier choice can move them. RA1 itself is not re-decided; rules.json keeps what stage-a saved.
- Power. At most 3 rows can move (carrier_swap.json) and the consensus has 1 wrong row of the limit 3, so the check can fail: it is
  not vacuous. It cannot show that a carrier is right for a row and does not try.
- Reported (aggregate only): combinations in total, combinations on which RA1 would fail, min and max of wrong and of false-confident
  over all combinations, the cases whose rows can move (ids). Never printed or stored: which combination or which carrier is right for
  a row, or any per-row gold label, claim, quote or page text.
- Label-free gates, all passed BEFORE any label is read; any failure makes the analysis void (reported, nothing else done): (g1) the
  default combination reproduces preds_A.csv in 0 rows; (g2) the set of rows that move across all combinations, with OVERALL recomputed
  per combination, is exactly the 3 rows of carrier_swap.json, and the per-case and total combination counts equal carrier_swap.json
  (219). One gate after the labels are loaded, aggregate only: (g3) the default combination reproduces the recorded RA1 line in
  rules.json (consensus and single-sample wrong and false-confident counts) exactly; otherwise void.
- Decision. Every combination passes both limits: the sensitivity holds; Task 11 step 2 is ticked as measured, with the 3 moving rows
  and the combination counts stated; go to Task 12. Any combination fails: report it as a caveat with the counts and STOP for the user.
  In neither case does it change the carrier rule (the first carrier in sample-list order), rules.json, preds_A.csv, stage_a.json or the
  candidate rule; the result goes to reports/kaggle_consensus/holdout/ra1_carrier_sensitivity.json, not through save_rule.
- The script is a scratch script (precedent: the carrier-swap check): it is kept next to its output and has no unit tests; its validity
  is the three gates above, and no code under eval/ changes except this docstring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("VRA_CACHE", "0")

import kaggle_policy as kp  # noqa: E402
import run_kaggle as rk  # noqa: E402
from assessor.llm import LLM  # noqa: E402
from assessor.schemas import Claim  # noqa: E402
from assessor.tracing import Budget, BudgetExceeded  # noqa: E402

# ------------------------------------------------------------------------------------------------ pre-registered constants
MIN_VALID_VOTES = 3                  # a winner needs at least this many valid votes
SETTLED_MIN = {5: 4, 3: 3}           # k samples -> agreeing votes needed to call a page-field settled (any other k: all of them)
RA2_MAX_CHANGED = 30                 # RA2: most verdict rows the k = 5 consensus may change against submission #1
RB0_MIN_AGREE_PCT = 95               # RB0: adjudicator must match the unanimous Flash signature on this percent of the draw
RB0_SAMPLE = 100                     # RB0: size of the fixed-seed draw
RB0_SEED = 20261118                  # RB0: the seed
RB2_MAX_UNRESOLVED = 10              # RB2: more unresolved page-fields than this on test = B is not decisive
BASE_COMMIT = "e42203d"              # the commit holding the base prompt text (selected, never edited)
BASE_VERSION = "9566ad4b6b95"        # the base prompt, behind submission #1
SECOND_VERSION = "f45b4b77c99d"      # the current prompt, used only as a second opinion
ADJUDICATOR_MODEL = "MiniMax-M3"     # stage B model (the main role in .env.example)

Vote = Optional[str]                 # a field signature; None = the sample says nothing about the field ("absent")


def settled_min_for(k: int) -> int:
    return SETTLED_MIN.get(k, k)


# ------------------------------------------------------------------------------------------------ voting
@dataclass(frozen=True)
class Tally:
    valid: int            # votes counted (a failed sample gives no vote)
    has_winner: bool      # >= MIN_VALID_VOTES valid votes and a strict majority of them
    winner: Vote          # the winning signature (None is a real winner: "absent"); only meaningful when has_winner
    agree: int            # votes equal to the winner, or the top count when there is no winner
    settled: bool         # a winner AND agree >= settled_min
    leaders: frozenset    # the votes with the highest count


def tally(votes: Sequence[Vote], settled_min: int) -> Tally:
    counts = Counter(votes)
    top = max(counts.values()) if counts else 0
    leaders = frozenset(v for v, n in counts.items() if n == top)
    valid = len(votes)
    has_winner = valid >= MIN_VALID_VOTES and top * 2 > valid          # strict majority of the valid votes
    winner = next(iter(leaders)) if has_winner else None
    return Tally(valid, has_winner, winner, top, has_winner and top >= settled_min, leaders)


def sample_signatures(extract: Optional[dict]) -> Optional[dict[str, str]]:
    """Per field, what one sample says about one page, as the order-free claim signature after normalisation. None = no valid sample."""
    if not extract or extract.get("status") != "ok":
        return None
    return kp.claim_signature(kp.normalise_page_claims([Claim.model_validate(c) for c in extract["claims"]])[0])


def page_tallies(extracts: Sequence[Optional[dict]], settled_min: int) -> dict[str, Tally]:
    """One Tally per field any valid sample mentions. A valid sample that does not mention the field votes "absent" (None);
    a failed or missing sample (None) votes for nothing."""
    valid = [s for s in (sample_signatures(e) for e in extracts) if s is not None]
    fields = sorted(set().union(*(s.keys() for s in valid))) if valid else []
    return {f: tally([s.get(f) for s in valid], settled_min) for f in fields}


# ------------------------------------------------------------------------------------------------ samples
@dataclass(frozen=True)
class Sample:
    label: str
    system: str       # the extraction prompt text this sample was drawn with
    model: str
    salt: str

    @property
    def prompt_version(self) -> str:
        return hashlib.sha256(self.system.encode("utf-8")).hexdigest()[:12]


def require_one_prompt(samples: Sequence[Sample]) -> None:
    """One vote = one prompt and one model (salts may differ: they are the independent draws). Mixed votes are refused."""
    versions, models = {s.prompt_version for s in samples}, {s.model for s in samples}
    if len(versions) > 1 or len(models) > 1:
        raise ValueError(f"a vote needs one prompt and one model, got prompts {sorted(versions)} and models {sorted(models)}")


# (label, salt) lists as pre-registered in the module docstring; the "a" samples use the base prompt, the "second" ones the current prompt.
TARGETS = {
    "test": {"split": "test", "subset": "all",
             "a": [("s0", ""), ("noise1", "noise1"), ("p9s3", "p9s3"), ("p9s4", "p9s4"), ("p9s5", "p9s5")],
             "second": [("v7", ""), ("v7n1", "v7n1")]},
    "holdout": {"split": "train", "subset": "holdout",
                "a": [("s0", ""), ("p9h1", "p9h1"), ("p9h2", "p9h2")],
                "second": [("f45", "")]},
    "dev": {"split": "train", "subset": "dev", "a": [("s0", "")], "second": []},
}


def load_extract(cache: Path, sample: Sample, page: dict) -> Optional[dict]:
    """The cached extraction of one page for one sample, or None (not drawn yet, or the call failed: failures are never cached)."""
    path = cache / f"{kp.cache_key(sample.model, page, sample.salt, system=sample.system)}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    return {"status": "ok", "claims": rec["claims"], "in": rec["in"], "out": rec["out"], "cached": True}


# ------------------------------------------------------------------------------------------------ consensus claims and case records
def page_claims(extracts: Sequence[Optional[dict]], tallies: Mapping[str, Tally], overrides: Optional[Mapping[str, "Resolution"]] = None
                ) -> Optional[dict]:
    """The consensus extraction of one page, in the shape run_kaggle.assemble_case takes. None when no sample is valid.

    Per field: a winner of None ("absent") carries nothing; a winning signature carries the raw claims of the FIRST sample (list order)
    whose signature for the field equals it; no winner carries the first valid sample's raw claims for the field (a placeholder).
    `overrides` (stage B, per field): unresolved drops the field; tie_broken replaces the winner; confirmed changes nothing."""
    sigs = [sample_signatures(e) for e in extracts]
    first_valid = next((i for i, s in enumerate(sigs) if s is not None), None)
    if first_valid is None:
        return None
    claims: list[dict] = []
    for f in sorted(tallies):
        t = tallies[f]
        res = (overrides or {}).get(f)
        if res is not None and res.kind == "unresolved":
            continue
        has, winner = (True, res.winner) if res is not None and res.kind == "tie_broken" else (t.has_winner, t.winner)
        if has:
            if winner is None:
                continue
            i = next(i for i, s in enumerate(sigs) if s is not None and s.get(f) == winner)
        else:
            i = first_valid
        claims += [c for c in extracts[i]["claims"] if c.get("field") == f]
    return {"status": "ok", "claims": claims}


def _holders(extract: dict, field: str) -> tuple:
    claims = kp.normalise_page_claims([Claim.model_validate(c) for c in extract["claims"]])[0]
    return tuple(sorted(kp.norm_entity(x.get("holder")) for c in claims if c.field == field for x in c.value))


def case_records(recs: list[dict], per_sample: Sequence[Mapping[str, Optional[dict]]], settled_min: int,
                 overrides: Optional[Mapping[tuple[str, str], "Resolution"]] = None
                 ) -> tuple[list[dict], dict[tuple[str, str], Tally], dict[str, list[tuple[str, ...]]]]:
    """One case. per_sample[i][page_id] is sample i's extract of that page. Returns copies of `recs` whose readable entries carry the
    consensus extract (or a failed record when no sample extracted the page), the tallies keyed (page_id, field), and the D8 gaps.
    `overrides[(page_id, field)]` carries stage B's resolutions; an overridden field is not reported as a gap (it differs from the vote on purpose)."""
    out: list[dict] = []
    tallies: dict[tuple[str, str], Tally] = {}
    gaps: dict[str, list[tuple[str, ...]]] = {"holder_split": [], "post_vote_mismatch": []}
    for rec in recs:
        rec = dict(rec)
        out.append(rec)
        if not rk.readable(rec):
            continue
        pid = rec["page"]["page_id"]
        extracts = [ps.get(pid) for ps in per_sample]
        page_t = page_tallies(extracts, settled_min)
        tallies.update({(pid, f): t for f, t in page_t.items()})
        page_over = {f: r for (p, f), r in (overrides or {}).items() if p == pid}
        got = page_claims(extracts, page_t, page_over)
        if got is None:
            rec["extract"] = {"status": "failed", "error": "no sample extracted this page", "claims": [], "in": 0, "out": 0}
            continue
        rec["extract"] = got
        after, sigs = sample_signatures(got), [sample_signatures(e) for e in extracts]
        for f, t in page_t.items():
            if not t.has_winner or (f in page_over and page_over[f].kind != "confirmed"):
                continue
            if after.get(f) != t.winner:
                gaps["post_vote_mismatch"].append((pid, f))
            if f == "security_certifications" and t.winner is not None:
                carriers = {_holders(e, f) for e, s in zip(extracts, sigs) if s is not None and s.get(f) == t.winner}
                if len(carriers) > 1:
                    gaps["holder_split"].append((pid, f))
    return out, tallies, gaps


# ------------------------------------------------------------------------------------------------ scoring helpers and stage-A numbers
def score_rows(pred: Mapping[str, str], gold: Mapping[str, str]) -> dict:
    """Counts over the gold rows. false_confident = requirement rows (not OVERALL) whose gold is UNKNOWN/NEEDS_REVIEW and pred PASS/FAIL,
    the numerator of metric.py's false-confidence rate."""
    wrong = sum(1 for i, g in gold.items() if pred.get(i) != g)
    fc = sum(1 for i, g in gold.items() if not i.endswith("__OVERALL") and g in ("UNKNOWN", "NEEDS_REVIEW") and pred.get(i) in ("PASS", "FAIL"))
    return {"rows": len(gold), "wrong": wrong, "false_confident": fc}


def changed_rows(a: Mapping[str, str], b: Mapping[str, str]) -> list[str]:
    return sorted(i for i in set(a) | set(b) if a.get(i) != b.get(i))


def _consensus_over(case_ids: Sequence[str], recs_by_case: Mapping[str, list[dict]], per_sample: Sequence[Mapping[str, Mapping]],
                    settled_min: int, assemble: Callable[[str, list[dict]], dict],
                    overrides: Optional[Mapping[str, Mapping[tuple[str, str], "Resolution"]]] = None):
    """Every case through case_records + assemble for the given samples: (preds, tallies by case, gaps by case, incomplete case ids).
    `overrides[case_id][(page_id, field)]` carries stage B's resolutions."""
    preds: dict[str, str] = {}
    tallies_by_case: dict[str, dict] = {}
    gaps: dict[str, list[tuple[str, ...]]] = {"holder_split": [], "post_vote_mismatch": []}
    incomplete: list[str] = []
    for cid in case_ids:
        recs, tallies, g = case_records(recs_by_case[cid], [ps[cid] for ps in per_sample], settled_min, (overrides or {}).get(cid))
        res = assemble(cid, recs)
        preds.update({f"{cid}__{rid}": v for rid, v in res["verdicts"].items()})
        preds[f"{cid}__OVERALL"] = res["overall"]
        tallies_by_case[cid] = tallies
        for kind in gaps:
            gaps[kind] += [(cid, *x) for x in g[kind]]
        if res["bad"]:
            incomplete.append(cid)
    return preds, tallies_by_case, gaps, incomplete


def stage_a_numbers(case_ids: Sequence[str], recs_by_case: Mapping[str, list[dict]], per_sample: Sequence[Mapping[str, Mapping]],
                    settled_min: int, assemble: Callable[[str, list[dict]], dict]) -> dict:
    """The stage-A result. per_sample[i][case_id][page_id] is sample i's extract. Counts and ids only: no claim ever enters the result."""
    preds, tallies_by_case, gaps, incomplete = _consensus_over(case_ids, recs_by_case, per_sample, settled_min, assemble)
    singles = [_consensus_over(case_ids, recs_by_case, [ps], settled_min, assemble)[0] for ps in per_sample]
    fragile: set[str] = set()
    if len(per_sample) > 1:
        for j in range(len(per_sample)):
            rest = [ps for i, ps in enumerate(per_sample) if i != j]
            fragile.update(changed_rows(preds, _consensus_over(case_ids, recs_by_case, rest, settled_min, assemble)[0]))
    unsettled, per_field, settled = [], Counter(), 0
    for cid in case_ids:
        for (pid, f), t in sorted(tallies_by_case[cid].items()):
            if t.settled:
                settled += 1
            else:
                unsettled.append({"case_id": cid, "page_id": pid, "field": f, "valid": t.valid, "agree": t.agree})
                per_field[f] += 1
    voted = settled + len(unsettled)
    return {"k": len(per_sample), "cases": len(case_ids), "preds": preds, "singles": singles,
            "tallies_summary": {"voted": voted, "settled": settled, "unsettled": len(unsettled), "per_field_unsettled": dict(per_field)},
            "unsettled": unsettled, "changed_vs_first": changed_rows(preds, singles[0]) if singles else [],
            "fragile": sorted(fragile), "incomplete_cases": incomplete, "signature_gaps": gaps}


def _ids(xs: Sequence, cap: int = 25) -> str:
    shown = ", ".join(str(x) if not isinstance(x, tuple) else "/".join(map(str, x)) for x in list(xs)[:cap])
    return shown + (f", ... (+{len(xs) - cap} more)" if len(xs) > cap else "") if xs else "none"


def format_stage_a(r: dict) -> str:
    """The stage-A report text: counts, field names and ids only."""
    t, g = r["tallies_summary"], r["signature_gaps"]
    return "\n".join([
        f"# Stage A: consensus over {r['k']} samples, {r['cases']} cases", "",
        f"- page-fields voted: {t['voted']}; settled {t['settled']}; unsettled {t['unsettled']}",
        "- unsettled by field: " + (", ".join(f"{f} {n}" for f, n in sorted(t["per_field_unsettled"].items())) or "none"),
        f"- verdict rows changed versus the first sample: {len(r['changed_vs_first'])} ({_ids(r['changed_vs_first'])})",
        f"- fragile rows (change when any one sample is left out): {len(r['fragile'])} ({_ids(r['fragile'])})",
        f"- incomplete cases (a readable page no sample extracted): {len(r['incomplete_cases'])} ({_ids(r['incomplete_cases'])})",
        f"- D8 holder_split (case/page/field): {len(g['holder_split'])} ({_ids(g['holder_split'])})",
        f"- D8 post_vote_mismatch (case/page/field): {len(g['post_vote_mismatch'])} ({_ids(g['post_vote_mismatch'])})", ""])


# ------------------------------------------------------------------------------------------------ the pre-registered rules
Rule = dict                                  # {"status": "pass" | "fail" | "untested" | "not_run", "detail": str}
RULE_KEYS = ("R0", "RA1", "RA2", "RB0", "RB1", "RB2")
RULE_STATUSES = ("pass", "fail", "untested", "not_run")


def _rule(status: str, detail: str) -> Rule:
    return {"status": status, "detail": detail}


def _ceil_mean(xs: Sequence[int]) -> int:
    return -(-sum(xs) // len(xs))            # integer ceiling of the mean: no float rounding at the boundary


def ra1(consensus: Mapping[str, int], singles: Sequence[Mapping[str, int]]) -> Rule:
    """RA1: consensus wrong rows and false-confident rows must each be <= the (rounded-up) mean of the single samples."""
    if not singles:
        return _rule("not_run", "no single samples to compare with")
    lim_w, lim_f = _ceil_mean([s["wrong"] for s in singles]), _ceil_mean([s["false_confident"] for s in singles])
    good = consensus["wrong"] <= lim_w and consensus["false_confident"] <= lim_f
    return _rule("pass" if good else "fail", f"consensus wrong {consensus['wrong']} (limit {lim_w}), false-confident "
                 f"{consensus['false_confident']} (limit {lim_f}); singles wrong {[s['wrong'] for s in singles]}, "
                 f"false-confident {[s['false_confident'] for s in singles]}")


def ra2(changed: int) -> Rule:
    return _rule("pass" if changed <= RA2_MAX_CHANGED else "fail", f"{changed} verdict rows changed versus submission #1 (limit {RA2_MAX_CHANGED})")


def rb0(agree: int, n: int) -> Rule:
    if n < 1:
        return _rule("untested", "no page-fields were drawn")
    return _rule("pass" if agree * 100 >= RB0_MIN_AGREE_PCT * n else "fail", f"adjudicator agreed on {agree} of {n} (needs {RB0_MIN_AGREE_PCT}%)")


def rb1(changed: int, wrong_to_right: int, right_to_wrong: int) -> Rule:
    if changed == 0:
        return _rule("untested", "B changed no holdout row, so it is untested (not a pass)")
    return _rule("pass" if wrong_to_right > right_to_wrong else "fail",
                 f"B changed {changed} holdout rows: wrong to right {wrong_to_right}, right to wrong {right_to_wrong}")


def rb2(unresolved_on_test: int) -> Rule:
    return _rule("pass" if unresolved_on_test <= RB2_MAX_UNRESOLVED else "fail",
                 f"{unresolved_on_test} unresolved page-fields on test (limit {RB2_MAX_UNRESOLVED})")


def candidate_rule(rules: Mapping[str, Rule]) -> str:
    """"A" when R0, RA1, RA2 pass; "A+B" when RB0, RB1, RB2 also pass; else "none". untested / not_run / fail never count as a pass."""
    def passed(keys):
        return all(rules.get(k, {}).get("status") == "pass" for k in keys)
    if not passed(("R0", "RA1", "RA2")):
        return "none"
    return "A+B" if passed(("RB0", "RB1", "RB2")) else "A"


def load_rules(path: Path) -> dict[str, Rule]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def save_rule(path: Path, key: str, rule: Rule) -> None:
    """Read-modify-write one rule into rules.json. An unknown key or status is refused, so a typo cannot silently become a result."""
    if key not in RULE_KEYS or rule.get("status") not in RULE_STATUSES:
        raise ValueError(f"unknown rule key {key!r} or status {rule.get('status')!r}")
    path = Path(path)
    rules = load_rules(path)
    rules[key] = {"status": rule["status"], "detail": rule.get("detail", "")}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rules, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------ stage B: escalate the unsettled
@dataclass(frozen=True)
class Resolution:
    kind: Literal["confirmed", "tie_broken", "unresolved"]
    winner: Vote            # the signature to carry (tie_broken) or the confirmed consensus; None = "absent"


def triggers(flash: Tally, second_votes: Sequence[Vote]) -> tuple[bool, bool]:
    """(1: unsettled, 2: second opinion). Trigger 2 needs at least one second-opinion vote, all equal, a Flash winner, and a vote that
    differs from that winner. Silence ("absent", None) is a vote on both sides."""
    second = bool(second_votes) and len(set(second_votes)) == 1 and flash.has_winner and second_votes[0] != flash.winner
    return (not flash.settled, second)


def resolve(flash: Tally, adj_votes: Sequence[Vote]) -> Resolution:
    """D2. The adjudicator's strict majority (of at least 3 votes) counts only when it equals a Flash leader. B can therefore confirm,
    break a tie toward a tied leader, or drop a claim (unresolved); it never introduces a claim Flash did not produce."""
    adj = tally(adj_votes, 3)
    if not adj.has_winner or adj.winner not in flash.leaders:
        return Resolution("unresolved", None)
    if flash.has_winner:
        return Resolution("confirmed", flash.winner)
    return Resolution("tie_broken", adj.winner)


def b_plan(tallies_by_case: Mapping[str, Mapping[tuple[str, str], Tally]], second_votes: Mapping[tuple[str, str, str], Sequence[Vote]]) -> list[dict]:
    """One entry per page that has at least one triggered page-field (pages de-duplicated, sorted). Only fields with a tally are looked at;
    plan_inputs gives a field no Flash sample mentioned a stand-in "absent" tally, so the spec's trigger 2 reaches it too."""
    pages: dict[tuple[str, str], dict[str, list[int]]] = {}
    for cid in sorted(tallies_by_case):
        for (pid, f), t in sorted(tallies_by_case[cid].items()):
            fired = [n for n, hit in zip((1, 2), triggers(t, second_votes.get((cid, pid, f), []))) if hit]
            if fired:
                pages.setdefault((cid, pid), {})[f] = fired
    return [{"case_id": cid, "page_id": pid, "fields": sorted(fs), "triggers": dict(sorted(fs.items()))} for (cid, pid), fs in sorted(pages.items())]


def rb0_draw(tallies_by_case: Mapping[str, Mapping[tuple[str, str], Tally]], n: int = RB0_SAMPLE, seed: int = RB0_SEED) -> list[tuple[str, str, str]]:
    """RB0's label-blind draw (D4): page-fields on which every valid Flash vote agrees and the signature is not "absent"."""
    pool = sorted((cid, pid, f) for cid, ts in tallies_by_case.items() for (pid, f), t in ts.items()
                  if t.has_winner and t.winner is not None and t.agree == t.valid)
    return sorted(random.Random(seed).sample(pool, n)) if len(pool) > n else pool


def b_row_changes(a: Mapping[str, str], ab: Mapping[str, str], gold: Mapping[str, str]) -> dict:
    """What stage B did to the rows, judged against gold: rows changed, wrong-to-right (A wrong, A+B right), right-to-wrong."""
    changed = changed_rows(a, ab)
    return {"changed": len(changed),
            "wrong_to_right": sum(1 for i in changed if a.get(i) != gold.get(i) and ab.get(i) == gold.get(i)),
            "right_to_wrong": sum(1 for i in changed if a.get(i) == gold.get(i) and ab.get(i) != gold.get(i))}


def adjudicator_estimate(pages: list[dict], system: str, samples: int) -> dict:
    """Calls, tokens and minutes for `samples` adjudicator reads of `pages` (the measured per-page pace of the Flash runs: re-based on
    the first MiniMax-M3 run, see the plan)."""
    tin, _ = rk.estimate_call_tokens(pages, system)
    calls = len(pages) * samples
    return {"pages": len(pages), "samples": samples, "calls": calls, "tokens_in": tin * samples,
            "tokens_out": round(tin * samples * rk.OUT_PER_IN), "minutes": tuple(round(calls * s / 60) for s in rk.SEC_PER_PAGE)}


def _valid_sigs(per_sample: Sequence[Mapping[str, Mapping]], cid: str, pid: str) -> list[dict]:
    """The signatures of the samples that read this page successfully (a failed or missing sample is not in the list: it gives no vote)."""
    return [s for s in (sample_signatures(ps.get(cid, {}).get(pid)) for ps in per_sample) if s is not None]


def plan_inputs(case_ids: Sequence[str], recs_by_case: Mapping[str, list[dict]], flash: Sequence[Mapping[str, Mapping]],
                second: Sequence[Mapping[str, Mapping]], settled_min: int, assemble: Callable[[str, list[dict]], dict]):
    """(A-only preds, Flash tallies by case, second-opinion votes, B plan): the Flash consensus, the second-opinion votes on every tallied
    page-field, the plan. Spec 5.3 trigger 2 covers a field that NO valid Flash sample mentioned too: its consensus is "absent", so it gets
    a stand-in tally of as many "absent" votes as the page has valid Flash samples. (Under D2 the adjudicator can only confirm "absent" or end
    unresolved there, so no verdict row can change, but the item counts toward RB2 exactly as the spec says. Stand-ins are never in the
    RB0 pool, whose winners must not be "absent".)"""
    preds_a, tallies_by_case, _, _ = _consensus_over(case_ids, recs_by_case, flash, settled_min, assemble)
    for cid in case_ids:
        for rec in recs_by_case[cid]:
            if not rk.readable(rec):
                continue
            pid = rec["page"]["page_id"]
            n_flash = len(_valid_sigs(flash, cid, pid))
            for f in sorted({f for sg in _valid_sigs(second, cid, pid) for f in sg}) if n_flash else []:
                tallies_by_case[cid].setdefault((pid, f), tally([None] * n_flash, settled_min))
    second_votes: dict[tuple[str, str, str], list[Vote]] = {}
    for cid in case_ids:
        by_page: dict[str, list[dict]] = {}
        for pid, f in tallies_by_case[cid]:
            sigs = by_page.setdefault(pid, _valid_sigs(second, cid, pid))
            second_votes[(cid, pid, f)] = [sg.get(f) for sg in sigs]
    return preds_a, tallies_by_case, second_votes, b_plan(tallies_by_case, second_votes)


def stage_b_numbers(case_ids: Sequence[str], recs_by_case: Mapping[str, list[dict]], flash: Sequence[Mapping[str, Mapping]],
                    second: Sequence[Mapping[str, Mapping]], adj: Sequence[Mapping[str, Mapping]], settled_min: int,
                    assemble: Callable[[str, list[dict]], dict]) -> dict:
    """Stage A tallies -> second-opinion votes -> plan -> adjudicator resolutions -> A+B verdicts. Every list is per_sample[i][case_id][page_id].
    Counts and ids only. A page the adjudicator did not read has no votes, so its planned fields come out unresolved: callers must check
    adjudicator coverage of the plan BEFORE trusting the result (the CLI does)."""
    preds_a, tallies_by_case, _second_votes, plan = plan_inputs(case_ids, recs_by_case, flash, second, settled_min, assemble)
    overrides: dict[str, dict[tuple[str, str], Resolution]] = {}
    resolved: list[tuple[str, str, str, str]] = []
    for e in plan:
        cid, pid = e["case_id"], e["page_id"]
        sigs = _valid_sigs(adj, cid, pid)
        for f in e["fields"]:
            res = resolve(tallies_by_case[cid][(pid, f)], [s.get(f) for s in sigs])
            overrides.setdefault(cid, {})[(pid, f)] = res
            resolved.append((cid, pid, f, res.kind))
    preds_ab, _, _, incomplete = _consensus_over(case_ids, recs_by_case, flash, settled_min, assemble, overrides)
    kinds = Counter(k for *_, k in resolved)
    return {"plan": plan, "counts": {k: kinds.get(k, 0) for k in ("confirmed", "tie_broken", "unresolved")},
            "preds_A": preds_a, "preds_AB": preds_ab, "changed": changed_rows(preds_a, preds_ab),
            "unresolved": sorted((c, p, f) for c, p, f, k in resolved if k == "unresolved"), "incomplete_cases": incomplete}


def rb0_numbers(draw: Sequence[tuple[str, str, str]], tallies_by_case: Mapping[str, Mapping[tuple[str, str], Tally]],
                adj1: Mapping[str, Mapping]) -> tuple[int, int]:
    """RB0: of the drawn page-fields, on how many does the adjudicator's first sample have the unanimous Flash signature?
    A page the adjudicator did not read counts as a disagreement, never as a skipped row."""
    agree = 0
    for cid, pid, f in draw:
        sig = sample_signatures(adj1.get(cid, {}).get(pid))
        agree += sig is not None and sig.get(f) == tallies_by_case[cid][(pid, f)].winner
    return int(agree), len(draw)


def coverage(extracts_by_case: Mapping[str, Mapping[str, Optional[dict]]], recs_by_case: Mapping[str, list[dict]]) -> tuple[int, int]:
    """(readable pages with a valid extract in this sample, readable pages)."""
    have = need = 0
    for cid, recs in recs_by_case.items():
        for r in recs:
            if rk.readable(r):
                need += 1
                ex = extracts_by_case.get(cid, {}).get(r["page"]["page_id"])
                have += bool(ex and ex.get("status") == "ok")
    return have, need


def stored_verdicts(path: Path) -> dict[str, str]:
    """id -> verdict from a run_kaggle cases.jsonl. Reads the case id, the verdicts and the overall only; nothing else is looked at."""
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out.update({f"{rec['case_id']}__{rid}": v for rid, v in rec["verdicts"].items()})
        out[f"{rec['case_id']}__OVERALL"] = rec["overall"]
    return out


def r0_rule(results: Mapping[str, Mapping[str, int]]) -> Rule:
    """R0 over the targets checked so far: any difference fails at once; pass needs dev, holdout and test, all at zero."""
    bad = {t: r for t, r in results.items() if r["k1_diff"] or r["winner_path_diff"]}
    if bad:
        return _rule("fail", "; ".join(f"{t}: k=1 differs in {r['k1_diff']} rows, winner path in {r['winner_path_diff']}" for t, r in sorted(bad.items())))
    missing = [t for t in TARGETS if t not in results]
    if missing:
        return _rule("not_run", f"0 differences so far; still to check: {', '.join(missing)}")
    return _rule("pass", "k=1 and the winner path reproduce the stored single-sample verdicts on dev, holdout and test (0 differing rows)")


def format_stage_b(r: dict) -> str:
    """The stage-B report text: counts, field names and ids only."""
    only1 = sum(1 for e in r["plan"] for t in e["triggers"].values() if t == [1])
    only2 = sum(1 for e in r["plan"] for t in e["triggers"].values() if t == [2])
    both = sum(1 for e in r["plan"] for t in e["triggers"].values() if t == [1, 2])
    c = r["counts"]
    return "\n".join([
        "# Stage B: adjudication of the unsure page-fields", "",
        f"- planned: {only1 + only2 + both} page-fields on {len(r['plan'])} pages (trigger 1 only {only1}, trigger 2 only {only2}, both {both})",
        f"- resolutions: confirmed {c['confirmed']}, tie_broken {c['tie_broken']}, unresolved {c['unresolved']}",
        f"- verdict rows changed by B versus A-only: {len(r['changed'])} ({_ids(r['changed'])})",
        f"- unresolved page-fields (case/page/field): {_ids(r['unresolved'])}",
        f"- incomplete cases: {len(r['incomplete_cases'])} ({_ids(r['incomplete_cases'])})", ""])


# ------------------------------------------------------------------------------------------------ command line
OUT_ROOT = ROOT / "reports" / "kaggle_consensus"
V6 = {"dev": "kaggle_dev_v6", "holdout": "kaggle_holdout_v6", "test": "kaggle_test_v6"}      # the stored single-sample runs R0 compares with
ADJ_SALTS = ("adj1", "adj2", "adj3")


class Ctx:
    """Everything a command needs about one target, loaded once: cases, triage, rules, the two prompts, models and the extraction cache.
    Nothing here prints; commands print counts only."""

    def __init__(self, target: str, data: Path, out_root: Path = OUT_ROOT):
        self.target, self.spec, self.data = target, TARGETS[target], Path(data)
        self.out = Path(out_root) / target
        self.cache = self.data / "cache" / "extract"
        self.cases, pages = rk.load_cases(self.data), rk.load_pages(self.data)
        self.reqs = json.loads((self.data / "requirements.json").read_text(encoding="utf-8"))
        self.rl = rk.load_module("kaggle_rules_lite", self.data / "rules_lite.py")
        self.ids = rk.select_cases(self.cases, self.spec["split"], self.spec["subset"], None, None)
        self.recs = {cid: rk.triage(self.cases[cid], pages[cid]) for cid in self.ids}
        self.page_of = {(cid, r["page"]["page_id"]): r["page"] for cid, rs in self.recs.items() for r in rs}
        llm = LLM.extractor(budget=Budget(max_usd=1.0, max_seconds=0, max_tokens=0, max_tool_calls=0))
        if llm.mock_mode:
            sys.exit("The extractor is on the mock provider: the cache is keyed by model, so set the real provider (via .env).")
        self.flash_model = llm.model
        self.base_system = kp.prompt_text_from_git(BASE_COMMIT, BASE_VERSION)
        if kp.PROMPT_VERSION != SECOND_VERSION:
            sys.exit(f"the current prompt is {kp.PROMPT_VERSION}, not the registered second opinion {SECOND_VERSION}")
        self.a = [Sample(lab, self.base_system, self.flash_model, salt) for lab, salt in self.spec["a"]]
        self.second = [Sample(lab, kp.EXTRACT_SYSTEM, self.flash_model, salt) for lab, salt in self.spec["second"]]
        require_one_prompt(self.a)
        require_one_prompt(self.second)

    def adjudicator(self, budget: Optional[Budget] = None) -> LLM:
        llm = LLM(budget=budget or Budget(max_usd=1.0, max_seconds=0, max_tokens=0, max_tool_calls=0), role="extract", model=ADJUDICATOR_MODEL)
        if llm.mock_mode:
            sys.exit("The extractor is on the mock provider: set the real provider (via .env).")
        return llm

    def adj_samples(self, model: str) -> list[Sample]:
        return [Sample(salt, self.base_system, model, salt) for salt in ADJ_SALTS]

    def extracts(self, sample: Sample) -> dict[str, dict[str, Optional[dict]]]:
        """sample's cached extraction of every readable page: [case_id][page_id] (None = not cached)."""
        return {cid: {r["page"]["page_id"]: load_extract(self.cache, sample, r["page"]) for r in recs if rk.readable(r)}
                for cid, recs in self.recs.items()}

    def assemble(self, cid: str, recs: list[dict]) -> dict:
        res = rk.assemble_case(self.cases[cid], recs, self.reqs, self.rl)
        dis = sum(1 for rid, v in res["verdicts"].items() if res["ours"].get(rid) != v) + (res["ours_overall"] != res["overall"])
        return {"verdicts": res["verdicts"], "overall": res["overall"], "bad": res["bad"], "ours_disagreements": dis}

    def gold(self, exclude_cases: Sequence[str] = ()) -> dict[str, str]:
        with (self.data / "train_labels.csv").open(encoding="utf-8") as f:
            rows = {r["id"]: r["verdict"] for r in csv.DictReader(f)}
        keep, skip = set(self.ids), set(exclude_cases)
        return {i: v for i, v in rows.items() if i.split("__")[0] in keep and i.split("__")[0] not in skip}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")


def _write_preds(path: Path, preds: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "verdict"])
        w.writerows(sorted(preds.items()))


def _cov_line(label: str, have: int, need: int) -> str:
    return f"coverage {label}: {have}/{need} readable pages ({100 * have / need:.1f}%)" if need else f"coverage {label}: no readable pages"


def _coverage_lines(ctx: Ctx) -> tuple[list[str], bool]:
    lines, full = [], True
    for smp in [*ctx.a, *ctx.second]:
        have, need = coverage(ctx.extracts(smp), ctx.recs)
        lines.append(_cov_line(f"{smp.label} ({smp.prompt_version}, salt {smp.salt!r})", have, need))
        full &= have == need
    return lines, full


def cmd_check_r0(a) -> int:
    """R0: the consensus path must reproduce the stored single-sample verdicts (k=1, and the winner path with k identical samples)."""
    ctx = Ctx(a.target, a.data, a.out_root)
    stored = stored_verdicts(ROOT / "reports" / V6[a.target] / "cases.jsonl")
    ext = ctx.extracts(ctx.a[0])
    k = 5 if a.target == "test" else 3

    def diffs(per_sample):
        preds = _consensus_over(ctx.ids, ctx.recs, per_sample, settled_min_for(len(per_sample)), ctx.assemble)[0]
        return sorted(i for i, v in stored.items() if preds.get(i) != v)
    d1, dk = diffs([ext]), diffs([ext] * k)
    print(f"R0 {a.target}: {len(stored)} stored rows; k=1 differs in {len(d1)}; winner path (k={k} identical samples) differs in {len(dk)}")
    for name, d in (("k=1", d1), (f"k={k}", dk)):
        if d:
            print(f"  differing ids ({name}): {_ids(d, 12)}")
    _write_json(Path(a.out_root) / f"r0_{a.target}.json", {"rows": len(stored), "k1_diff": len(d1), "winner_path_diff": len(dk), "ids": sorted(set(d1 + dk))[:50]})
    results = {t: json.loads((Path(a.out_root) / f"r0_{t}.json").read_text()) for t in TARGETS if (Path(a.out_root) / f"r0_{t}.json").exists()}
    rule = r0_rule(results)
    save_rule(Path(a.out_root) / "rules.json", "R0", rule)
    print(f"R0 so far: {rule['status']} ({rule['detail']})")
    return 0 if rule["status"] != "fail" else 1


def cmd_stage_a(a) -> int:
    ctx = Ctx(a.target, a.data, a.out_root)
    cov, full = _coverage_lines(ctx)
    print("\n".join(cov))
    if not full and not a.allow_partial:
        print("refused: every sample must cover 100% of the readable pages (re-run the missing draws, or use --allow-partial for a dry run)")
        return 2
    per_sample = [ctx.extracts(smp) for smp in ctx.a]
    r = stage_a_numbers(ctx.ids, ctx.recs, per_sample, settled_min_for(len(per_sample)), ctx.assemble)
    text = format_stage_a(r)
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "stage_a.md").write_text("\n".join(cov) + ("\n\nPARTIAL (dry run): rules were not saved.\n" if not full else "\n") + "\n" + text, encoding="utf-8")
    _write_json(ctx.out / "stage_a.json", {k: v for k, v in r.items() if k not in ("preds", "singles", "unsettled")} | {"partial": not full})
    _write_json(ctx.out / "unsettled.json", r["unsettled"])
    _write_preds(ctx.out / "preds_A.csv", r["preds"])
    print(text)
    if not full:
        print("dry run: no rule saved")
        return 0
    rules = Path(a.out_root) / "rules.json"
    if a.target == "holdout":
        gold = ctx.gold(exclude_cases=r["incomplete_cases"])
        rule = ra1(score_rows(r["preds"], gold), [score_rows(sg, gold) for sg in r["singles"]])
        save_rule(rules, "RA1", rule)
        print(f"RA1 {rule['status']}: {rule['detail']}")
    elif a.target == "test":
        rule = ra2(len(r["changed_vs_first"]))
        save_rule(rules, "RA2", rule)
        print(f"RA2 {rule['status']}: {rule['detail']}")
    return 0


def _pages_of(ctx: Ctx, plan: Sequence[dict]) -> list[dict]:
    return [ctx.page_of[(e["case_id"], e["page_id"])] for e in plan]


def rb0_pages_from_draw(draw: Sequence[tuple[str, str, str]]) -> list[dict]:
    """The RB0 draw as the pages run-b reads: one entry per (case, page), sorted, the drawn fields in draw order."""
    return [{"case_id": c, "page_id": p, "fields": [f for cc, pp, f in draw if (cc, pp) == (c, p)]} for c, p in sorted({(c, p) for c, p, _ in draw})]


def plan_file_problems(plan_now: Sequence[dict], plan_disk: Optional[Sequence[dict]], rb0_now: Optional[Sequence[dict]] = None,
                       rb0_disk: Optional[Sequence[dict]] = None) -> list[str]:
    """Why the plan files run-b read cannot be trusted by `resolve` (counts only, never ids or contents). `plan_now` is the plan recomputed
    from the cached samples; `plan_disk` is b_plan.json (None = missing). The RB0 plan is compared only when both sides exist (no rb0_plan.json
    means RB0 is simply not computed; the holdout has no RB0). A plan written from other or partial samples would send the adjudicator to the
    wrong pages, and every page it never read would silently count as unresolved (its claim dropped)."""
    def same(x, y) -> bool:
        return json.loads(json.dumps(list(x))) == json.loads(json.dumps(list(y)))
    problems = []
    if plan_disk is None:
        problems.append("b_plan.json is missing (run plan-b)")
    elif not same(plan_now, plan_disk):
        problems.append(f"b_plan.json lists {len(plan_disk)} pages but the plan recomputed from the cached samples has {len(plan_now)}: it was "
                        "written from other or partial samples (re-run plan-b once every sample is complete, then draw again)")
    if rb0_now is not None and rb0_disk is not None and not same(rb0_now, rb0_disk):
        problems.append(f"rb0_plan.json lists {len(rb0_disk)} pages but the RB0 draw recomputed from the cached samples has {len(rb0_now)}: it is a "
                        "draw from another pool (re-run plan-b --rb0 once every sample is complete)")
    return problems


def save_plans(out: Path, plan: Sequence[dict], rb0_pages: Optional[Sequence[dict]], full: bool) -> list[str]:
    """plan-b's files. A dry run (some sample covers under 100% of the readable pages) writes nothing: run-b and resolve would read the
    file as if it were final. Returns the names written."""
    if not full:
        return []
    written = ["b_plan.json"]
    _write_json(Path(out) / "b_plan.json", list(plan))
    if rb0_pages is not None:
        _write_json(Path(out) / "rb0_plan.json", list(rb0_pages))
        written.append("rb0_plan.json")
    return written


def cmd_plan_b(a) -> int:
    if a.rb0 and a.target != "test":
        print("--rb0 applies to the test target only")
        return 2
    ctx = Ctx(a.target, a.data, a.out_root)
    cov, full = _coverage_lines(ctx)
    print("\n".join(cov))
    if not full:
        print("note: not every sample covers 100% of the readable pages; the plan below is a dry run and NO plan file is written")
    flash, second = [ctx.extracts(smp) for smp in ctx.a], [ctx.extracts(smp) for smp in ctx.second]
    _, tallies, _, plan = plan_inputs(ctx.ids, ctx.recs, flash, second, settled_min_for(len(flash)), ctx.assemble)
    draw = rb0_draw(tallies) if a.rb0 else None
    pages_rb0 = rb0_pages_from_draw(draw) if draw is not None else None
    written = save_plans(ctx.out, plan, pages_rb0, full)
    fired = Counter(tuple(t) for e in plan for t in e["triggers"].values())
    print(f"B plan {a.target}: {len(plan)} pages, {sum(len(e['fields']) for e in plan)} page-fields "
          f"(trigger 1 only {fired[(1,)]}, trigger 2 only {fired[(2,)]}, both {fired[(1, 2)]})")
    for n in (1, 3):
        print(f"  adjudicator, {n} sample(s): {adjudicator_estimate(_pages_of(ctx, plan), ctx.base_system, n)}")
    if draw is not None:
        print(f"RB0 draw: {len(draw)} page-fields on {len(pages_rb0)} pages; adjudicator, 1 sample: {adjudicator_estimate(_pages_of(ctx, pages_rb0), ctx.base_system, 1)}")
    print("written: " + (", ".join(written) if written else "nothing (dry run)"))
    return 0


def cmd_run_b(a) -> int:
    """The only command that spends: the adjudicator reads the plan's pages with the base prompt, one salt at a time."""
    ctx = Ctx(a.target, a.data, a.out_root)
    plan = json.loads((ctx.out / a.plan).read_text(encoding="utf-8"))
    pages = _pages_of(ctx, plan)
    salts = [x for x in a.salts.split(",") if x]
    budget = Budget(max_usd=a.max_usd, max_seconds=0, max_tokens=0, max_tool_calls=0)
    llm = ctx.adjudicator(budget)
    print(f"adjudicator: provider={llm.provider} model={llm.model} prompt={BASE_VERSION} cap=${a.max_usd:.2f} (not a real bound: every model gets the same fallback price)")
    todo_total = 0
    for salt in salts:
        cached, todo = rk.uncached_pages(pages, ctx.cache, llm.model, salt, ctx.base_system)
        todo_total += len(todo)
        print(f"estimate {salt}: " + rk.format_estimate(len(todo), len(cached), *rk.estimate_call_tokens(todo, ctx.base_system)))
    if a.estimate:
        return 0
    lines = [f"# run-b {a.target} {a.plan} salts {a.salts}", f"- pages {len(pages)}, calls still to make {todo_total}, model {llm.model}"]
    stopped, t0 = "", time.time()
    for salt in salts:
        ex = rk.Extractor(llm, ctx.cache, False, salt, system=ctx.base_system)
        done = fails = tin = tout = 0
        rate = 0
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            futs = {pool.submit(ex.run, pg): pg for pg in pages}
            try:
                for fut in as_completed(futs):
                    res = fut.result()
                    done += 1
                    tin, tout = tin + res["in"], tout + res["out"]
                    if res["status"] != "ok":
                        fails += 1
                        rate += ("429" in res.get("error", "") or "rate limit" in res.get("error", "").lower())
                    if done % 25 == 0:
                        print(f"  {salt}: {done}/{len(pages)} pages  {time.time() - t0:.0f}s", flush=True)
            except BudgetExceeded as exc:
                stopped = str(exc)[:80]
                for f in futs:
                    f.cancel()
        lines.append(f"- {salt}: pages {len(pages)}, cache hits {ex.hits}, new calls {ex.misses}, failures {fails} (rate-limit {rate}), "
                     f"tokens recorded in/out {tin:,}/{tout:,}, spend per budget ${budget.spent_usd:.4f}")
        print(lines[-1])
        if stopped:
            break
    wall = time.time() - t0
    lines.append(f"- wall time {wall:.0f}s with {a.workers} workers" + (f"; STOPPED: {stopped}" if stopped else ""))
    print(lines[-1])
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / f"run_b_{Path(a.plan).stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if stopped else 0


def cmd_resolve(a) -> int:
    ctx = Ctx(a.target, a.data, a.out_root)
    cov, full = _coverage_lines(ctx)
    print("\n".join(cov))
    adj_model = ctx.adjudicator().model
    adj_samples = ctx.adj_samples(adj_model)
    flash, second = [ctx.extracts(smp) for smp in ctx.a], [ctx.extracts(smp) for smp in ctx.second]
    # Coverage is judged against the plan RECOMPUTED from the cached samples, and the plan files run-b read must equal it: otherwise the
    # adjudicator may have read other pages and every page it never read would silently count as unresolved.
    _, tallies_now, _, plan = plan_inputs(ctx.ids, ctx.recs, flash, second, settled_min_for(len(flash)), ctx.assemble)
    plan_path, rb0_path = ctx.out / "b_plan.json", ctx.out / "rb0_plan.json"
    plan_disk = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    rb0_disk = json.loads(rb0_path.read_text(encoding="utf-8")) if a.target == "test" and rb0_path.exists() else None
    rb0_now = rb0_pages_from_draw(rb0_draw(tallies_now)) if a.target == "test" else None
    problems = plan_file_problems(plan, plan_disk, rb0_now, rb0_disk)
    rb0_pages = rb0_now if rb0_disk is not None else []
    adj = [ctx.extracts(smp) for smp in adj_samples]
    missing = sum(1 for e in plan for ex in adj if ex[e["case_id"]].get(e["page_id"]) is None)
    missing_rb0 = sum(1 for e in rb0_pages if adj[0][e["case_id"]].get(e["page_id"]) is None)
    print(f"adjudicator coverage: {len(plan) * len(adj) - missing}/{len(plan) * len(adj)} plan page-reads, "
          f"{len(rb0_pages) - missing_rb0}/{len(rb0_pages)} RB0 pages (adj1)")
    for p in problems:
        print("plan file problem: " + p)
    complete = full and not missing and not missing_rb0 and not problems
    if not complete and not a.allow_partial:
        print("refused: a page the adjudicator did not read would count as unresolved (claim dropped), or a plan file does not match the samples; "
              "finish the draws, or use --allow-partial for a dry run")
        return 2
    r = stage_b_numbers(ctx.ids, ctx.recs, flash, second, adj, settled_min_for(len(flash)), ctx.assemble)
    text = format_stage_b(r)
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "stage_b.md").write_text("\n".join(cov) + ("\n\nPARTIAL (dry run): rules were not saved.\n" if not complete else "\n") + "\n" + text, encoding="utf-8")
    _write_preds(ctx.out / "preds_AB.csv", r["preds_AB"])
    _write_json(ctx.out / "stage_b.json", {"partial": not complete, "counts": r["counts"], "rows_changed": len(r["changed"]),
                                           "unresolved": len(r["unresolved"]), "incomplete_cases": r["incomplete_cases"]})
    print(text)
    if not complete:
        print("dry run: no rule saved")
        return 0
    rules = Path(a.out_root) / "rules.json"
    if a.target == "test":
        rule = rb2(len(r["unresolved"]))
        save_rule(rules, "RB2", rule)
        print(f"RB2 {rule['status']}: {rule['detail']}")
        if rb0_pages:
            draw = [(e["case_id"], e["page_id"], f) for e in rb0_pages for f in e["fields"]]
            rule = rb0(*rb0_numbers(draw, tallies_now, adj[0]))
            save_rule(rules, "RB0", rule)
            print(f"RB0 {rule['status']}: {rule['detail']}")
        else:
            print("RB0 not computed: no rb0_plan.json")
    elif a.target == "holdout":
        gold = ctx.gold(exclude_cases=r["incomplete_cases"])
        rule = rb1(**b_row_changes(r["preds_A"], r["preds_AB"], gold))
        save_rule(rules, "RB1", rule)
        print(f"RB1 {rule['status']}: {rule['detail']}")
    return 0


def format_rules(rules: Mapping[str, Rule]) -> str:
    """The rule table: every pre-registered rule with its recorded status (not_run when nothing was recorded)."""
    rows = [(k, rules.get(k, {}).get("status", "not_run"), rules.get(k, {}).get("detail", "not recorded")) for k in RULE_KEYS]
    return "\n".join(f"{k:<4} {st:<9} {detail}" for k, st, detail in rows)


def make_candidate(rules: Mapping[str, Rule], incomplete_cases: Sequence[str], preds_a: Mapping[str, str], preds_ab: Mapping[str, str],
                   test_rows: list[dict], metric, out_dir: Path, baseline: Mapping[str, str], stage_b_partial: Optional[bool] = None) -> dict:
    """Apply the pre-registered candidate rule. none: nothing is written. A: submission.csv from A only. A+B: submission.csv from A+B and
    submission_A_only.csv (D7). Every file goes through run_kaggle.write_submission (the runner's own self-check). A test case with a page
    no sample extracted blocks the candidate. A+B also needs `stage_b_partial` = False (from `resolve`'s stage_b.json): a dry-run
    `resolve --allow-partial` overwrites preds_AB.csv with the claims of unread pages dropped, and None (no stage_b.json) is refused too.
    A-only never looks at it. Nothing is ever uploaded."""
    cand = candidate_rule(rules)
    res: dict = {"candidate": cand, "files": {}}
    if cand == "none":
        return res
    if incomplete_cases:
        res["refused"] = f"{len(incomplete_cases)} incomplete test cases (a readable page no sample extracted): {_ids(list(incomplete_cases), 10)}"
        return res
    if cand == "A+B" and stage_b_partial is not False:
        res["refused"] = ("the stage-B files come from a partial dry run" if stage_b_partial
                          else "the stage B status is unknown (no stage_b.json from a complete `resolve --target test`)")
        return res
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"submission.csv": preds_ab if cand == "A+B" else preds_a}
    if cand == "A+B":
        files["submission_A_only.csv"] = preds_a
    for name, preds in files.items():
        problems = rk.write_submission(out_dir / name, test_rows, preds, metric)
        with (out_dir / name).open(encoding="utf-8") as f:
            written = {r["id"]: r["verdict"] for r in csv.DictReader(f)}
        res["files"][name] = {"rows": len(written), "problems": problems,
                              "differs_from_baseline": sum(1 for i, v in written.items() if baseline.get(i) != v)}
    return res


def _read_preds(path: Path) -> dict[str, str]:
    with Path(path).open(encoding="utf-8") as f:
        return {r["id"]: r["verdict"] for r in csv.DictReader(f)}


def cmd_candidate(a) -> int:
    root, data = Path(a.out_root), Path(a.data)
    rules = load_rules(root / "rules.json")
    print(format_rules(rules))
    cand = candidate_rule(rules)
    print(f"candidate rule: {cand}")
    if cand == "none":
        print("no candidate: submission #1 stays; nothing was written")
        return 0
    stage_a = json.loads((root / "test" / "stage_a.json").read_text(encoding="utf-8"))
    if stage_a.get("partial"):
        print("refused: the test stage-A files come from a partial dry run")
        return 2
    with (data / "test.csv").open(encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    metric = rk.load_module("kaggle_metric", data / "metric.py")
    stage_b_partial = None                       # only A+B reads stage B; a missing stage_b.json stays None and is refused
    if cand == "A+B" and (root / "test" / "stage_b.json").exists():
        stage_b_partial = bool(json.loads((root / "test" / "stage_b.json").read_text(encoding="utf-8")).get("partial", True))
    res = make_candidate(rules, stage_a["incomplete_cases"], _read_preds(root / "test" / "preds_A.csv"),
                         _read_preds(root / "test" / "preds_AB.csv") if cand == "A+B" else {}, test_rows, metric, root / "candidate",
                         _read_preds(ROOT / "reports" / V6["test"] / "submission.csv"), stage_b_partial)
    if "refused" in res:
        print("refused: " + res["refused"])
        return 2
    bad = False
    for name, info in res["files"].items():
        bad |= bool(info["problems"])
        print(f"{name}: {info['rows']} rows, {info['differs_from_baseline']} differ from submission #1 (reports/{V6['test']}/submission.csv). Self-check: "
              + ("OK (ids match test.csv, no duplicates, valid labels, metric.py accepts it)" if not info["problems"] else "FAILED: " + "; ".join(info["problems"])))
    print(f"written under {root / 'candidate'}; NOT uploaded: a submission needs the user's explicit yes")
    return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Consensus extraction (stage A) and escalation (stage B) for the Vendor Verdict files. "
                                 "Counts and ids only; the pre-registration is in the module docstring.")
    ap.add_argument("--data", default=str(ROOT / "data_kaggle"))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, targets=("test", "holdout")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--target", choices=list(targets), required=True)
        sp.set_defaults(fn=fn)
        return sp
    add("check-r0", cmd_check_r0, "R0: the consensus path reproduces the stored single-sample verdicts", ("dev", "holdout", "test"))
    add("stage-a", cmd_stage_a, "vote over the cached samples; writes stage_a.md/json, preds_A.csv, unsettled.json; saves RA1 (holdout) or RA2 (test)") \
        .add_argument("--allow-partial", action="store_true", help="dry run on samples not fully drawn yet; no rule is saved")
    add("plan-b", cmd_plan_b, "which page-fields go to the adjudicator; writes b_plan.json (and rb0_plan.json with --rb0, test only)") \
        .add_argument("--rb0", action="store_true")
    sp = add("run-b", cmd_run_b, "the adjudicator reads the pages of a plan (the only command that spends)")
    sp.add_argument("--plan", default="b_plan.json", help="b_plan.json or rb0_plan.json, in reports/kaggle_consensus/<target>/")
    sp.add_argument("--salts", default="adj1,adj2,adj3")
    sp.add_argument("--estimate", action="store_true", help="print the estimate per salt and exit before any call")
    sp.add_argument("--workers", type=int, default=6)
    sp.add_argument("--max-usd", type=float, default=1.5)
    sub.add_parser("candidate", help="apply the candidate rule to rules.json and write the candidate file(s); never uploads").set_defaults(fn=cmd_candidate)
    add("resolve", cmd_resolve, "resolve the plan with the adjudicator samples; writes stage_b.md, preds_AB.csv; saves RB2/RB0 (test) or RB1 (holdout)") \
        .add_argument("--allow-partial", action="store_true", help="dry run before every draw is in; no rule is saved")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (SystemExit, KeyboardInterrupt):
        raise
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001  the message of a parsing error can quote page text: print where, not what
        where = "; ".join(f"{Path(f.filename).name}:{f.lineno} in {f.name}" for f in traceback.extract_tb(exc.__traceback__)[-4:])
        print(f"error: {type(exc).__name__} at {where} (message withheld: it may quote page text)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
