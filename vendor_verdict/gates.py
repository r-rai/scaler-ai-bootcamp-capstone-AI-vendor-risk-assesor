"""Pre-registered gates of the vendor_verdict multi-agent system (system B). This docstring is the pre-registration: it was committed
before any holdout or test run of system B, and the commit is the timestamp that proves the rules came first.

    python -m vendor_verdict gates --target dev|holdout|test [--replay DIR]

What the gates decide. System B becomes a submission candidate only if it passes the rules fixed here in advance. Each rule reads
the run directories the runner wrote (`run.json`, `preds.csv`, `submission.csv`), and the results are merged into
`reports/vendor_verdict/gates.json`. No threshold below may be changed after a holdout or test run has been seen.

The order of the steps (rulings R46 and R53). Each step's choice is fixed before the next step looks at anything, and no step's
choice depends on a label except R-DM and R-hold (where labels are only counted, with the 7 exposed rows excluded). The router
configuration depends on no label at all.
1. R-smoke, on dev (20 cases).
2. R-route on the holdout, which fixes the router configuration.
3. R-DM on the holdout, in that router configuration, which fixes the Decision Maker configuration.
4. R-hold on the holdout, on the run in the final configuration (router configuration plus Decision Maker configuration).
5. The test run in the final configuration: R-route if the router is on, then R-test.
6. The candidate.

The rules, in that order.
- R-smoke (dev, 20 cases). The run is not partial, it has 0 failures (no crash and no unparsed output, of any role), and 100% of the
  readable pages were read (`pages_read == pages_readable`). The dev run is exactly `--limit 20` (its folder is `train_dev_limit-20`).
  Dev is a smoke test only: it reads run.json and nothing else, never gold and never a label.
- R-route (label-free). Every baseline claim that survives the evidence policy is routed to the specialist of its field:
  covered == total, 100% recall against the baseline. This refines the spec's wording "every page-field on which the baseline Flash
  cache has a claim", before any run and with no label involved: the denominator counts only the 8 family fields (not
  `legal_entity`), only readable pages, and on a `registry` page only `security_certifications` (`baseline.route_baseline_fields`).
  The runner computes it as run.json `route_recall`; the gate applies covered == total. A run without a baseline cache has no
  `route_recall`, and then R-route is not computable.
  Router configuration (R53). R-route is first computed on the full holdout run (nothing ablated). If it passes, the router stays
  ON. If it fails or is not computable, the router is switched OFF: "router" is added to the ablation set, so every readable page goes
  to every specialist family, and R-route then holds by construction (it is still computed and reported). Router OFF applies
  everywhere in the candidate configuration: the holdout runs used for R-DM and R-hold, and the test run (which is router-ablated from
  the start). With the router ON, R-route on the test run remains a gate: if it fails there, there is no candidate. This fallback was
  added after the controller saw the dev-smoke R-route count (158 of 163 baseline page-fields covered: a label-free coverage count,
  no label and no page text), and before any holdout or test data was run or read. No routing prompt was edited and no
  field-specific routing rule was added for it.
- R-DM (holdout, the 7 exposed rows excluded), in the router configuration: router ON compares the runs ablating nothing and
  `decision_maker`; router OFF compares the runs ablating `router` and `router` + `decision_maker`. Over the rows where the run with
  the Decision Maker differs from the run where it is ablated (always abstains), the rows it turns from wrong to right must be at
  least the rows it turns from right to wrong. A tie passes. If R-DM fails or is not computable, `decision_maker` is added to the
  ablation set. The switch is part of the rule, not a later choice.
- R-hold (holdout, the 7 exposed rows excluded), on the run in the final configuration. B's wrong rows are at most the wrong rows of
  the baseline sample s0, and B's false-confident rows are at most s0's, both counted over the same rows. s0 is the submitted
  baseline: extractor model MiniMax-M3.1-Flash (pinned as `BASELINE_MODEL`; any other configured extractor model is refused, and the
  model name is recorded in the R-hold entry), prompt 9566ad4b6b95, unsalted, rebuilt by `run_kaggle.assemble_case` over the cached baseline extraction (the path rule
  R0 proved reproduces s0). If any holdout case has a readable page missing from that cache, s0 is refused (a missing page would
  make s0 look worse than it is). The rows are the gold rows of the holdout cases minus the exposed rows. A row with no prediction
  counts as wrong. A false-confident row is a requirement row (not OVERALL) whose gold label is UNKNOWN or NEEDS_REVIEW and whose
  prediction is PASS or FAIL.
- R-test (test, label-free), on the test run in the final configuration. The candidate's submission.csv changes at most 30 rows
  against submission #2 (`reports/kaggle_consensus/candidate/submission.csv`). A row present in only one of the two files counts as
  changed.

Candidate rule. When R-smoke, R-hold and R-test pass, and R-route on the test run passes or the router is off, the candidate is B
with the ablation set of the final configuration: nothing; `decision_maker`; `router`; or `router` + `decision_maker`. Otherwise no
candidate: B is not submitted, and the system is still presented with its honest numbers. A candidate always needs the user's yes to
submit, and nothing here uploads anything. Final selection (submission #2 plus B, if B was submitted) is the user's call in the web
UI. The runner writes submission.csv only for the ablation sets in its `SUBMISSION_ABLATIONS` (today: nothing, or
`decision_maker`); the candidate file for a router-OFF configuration depends on that runner rule being extended to `router` and
`router` + `decision_maker` (ruling R54, a separate change). Until then a router-OFF test run has no submission.csv, and R-test is
not computable.

How the steps meet in gates.json. `--target holdout` records the router and Decision Maker configuration it chose. `--target test`
reads that record and gates the test run in the same configuration (for example `test_all`, `test_all_ablate-router` or
`test_all_ablate-decision_maker+router`); with no holdout record the router and the Decision Maker count as off (R45). A test entry
made under another configuration than the current holdout record is stale and counts as failed. The candidate is recomputed from
the merged gates.json on every call.

Fresh inputs (ruling R55a). Each section of gates.json (dev, holdout, test) records the prompt versions of the runs it read and, for
every run directory it consulted (accepted or refused), the sha256 of that directory's run.json, preds.csv and submission.csv (or
"absent"). A section counts toward the candidate only while it is fresh: the frozen prompt versions are recorded, the section's prompt
versions equal them, and every consulted directory still hashes the same. Otherwise the section is treated as missing (not
computable, so FAIL under R45): a dev smoke on other prompts cannot stand in for B's prompts, and a run re-run after it was gated
forces a re-gate. The test step also reads the holdout configuration only from a fresh holdout section.

The record itself. gates.json has an append-only `history` list: every gate call appends one counts-only record (UTC time, target,
configuration, each rule's pass and computable flags, the input directory names, the candidate name) and never rewrites an earlier
one, so a re-gate stays visible. An unreadable gates.json, or one whose history is not a list, is refused (class name only) and left
untouched, never silently replaced; an unknown target is refused before anything is read or written.

Not computable (ruling R45). A rule whose input run is missing or refused is not computable, and a rule that is not computable counts
as FAIL. For the holdout R-route and for R-DM that fail is the pre-registered switch above (router off, Decision Maker off). For
R-smoke, R-hold, R-test, and R-route on the test run with the router on, it means no candidate. The operator order is fixed in
advance and does not depend on any gate result: if a replay a rule needs comes back partial, the replay is re-run once without
`--cache-only` (local calls are free and fill the local gaps; hosted calls are cached; a hosted call that would be paid needs the
user's go) before any gate is computed, and the gates then use the last run that is not partial. A candidate may be submitted only if
its cache-only replay is not partial and its preds.csv is byte-identical to the candidate's (`--replay DIR`, a directory other than
the candidate's, since the runner clears its output folder when it starts).

Which runs are accepted. A run directory is a gate input only if its run.json says all of this; anything else is refused with a
short reason, which makes the rule not computable:
- `partial` is false and `stopped_reason` is "complete" (an exit code is never read);
- `split` and `subset` match the target (dev = train/dev, holdout = train/holdout, test = test/all) and `salt` is empty;
- holdout and test: `limit` is null, `only` is null and `cases_total` equals the full size of the subset (from the runner's loaders);
- dev: `limit` is 20, `only` is null and `cases_total` is 20;
- a cache-only run has `not_cached` equal to 0;
- the ablation set equals what the rule needs in the configuration order above (a subset of `router` and `decision_maker`);
- holdout and test: `prompt_versions` equals the frozen prompt versions below. While they are not recorded, holdout and test are
  refused; dev may run without them.
The fields read from run.json (ruling R5): pages_readable, pages_read, failures, prompt_versions, route_recall (covered, total,
missed), partial, stopped_reason, limit, only, cases_total, cache_only, not_cached, ablate, salt, split, subset. Anything that needs
the candidate's submission.csv checks that the file exists; run.json `submission.written` alone is never trusted.

Reporting. Counts only, also when gold is read. The console and gates.json carry rule names, pass or fail, counts, `computable`
flags, run directory names and prompt versions, never a row id with a label, a verdict of a named row, page text, quotes or claims.
For R-DM the output is pass or fail and the two counts (wrong to right, right to wrong), whatever the number of changed rows; these
are counts, not a per-row analysis, so the few-rows rule (pass or fail and min/max only, under 10 rows) does not apply to them.
Labels are read only to count.

Exposed rows. The 7 holdout rows whose labels were exposed in an earlier session (`config.EXPOSED_HOLDOUT_ROWS`) are excluded from
R-hold and R-DM. The test target uses no label at all.

Frozen prompt versions: NOT YET RECORDED (Task 10 step 4)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import EXPOSED_HOLDOUT_ROWS, ROOT
from .runner import default_out

# Filled at Task 10 step 4 ({role: prompt version}, as run.json `prompt_versions`); holdout and test refuse while it is None.
FROZEN_PROMPT_VERSIONS: dict[str, str] | None = None

R_TEST_MAX_CHANGED = 30
BASELINE_MODEL = "MiniMax-M3.1-Flash"        # the extractor model of submission #1 (prompt 9566ad4b6b95): s0 is pinned to it (R55b)
HASHED_FILES = ("run.json", "preds.csv", "submission.csv")
DEV_LIMIT = 20
TARGETS: dict[str, tuple[str, str]] = {"dev": ("train", "dev"), "holdout": ("train", "holdout"), "test": ("test", "all")}
DM = frozenset({"decision_maker"})
ROUTER = frozenset({"router"})
REQUIRED = ("R-smoke", "R-route", "R-hold", "R-test")
ABSTAIN_LABELS = frozenset({"UNKNOWN", "NEEDS_REVIEW"})
CONFIDENT_LABELS = frozenset({"PASS", "FAIL"})
EXIT_USAGE = 64
EXIT_REFUSED = 2                             # the gates record is unreadable or the target unknown: nothing was written
RUNS_ROOT = ROOT / "reports" / "vendor_verdict"
SUBMISSION_2 = ROOT / "reports" / "kaggle_consensus" / "candidate" / "submission.csv"


class GateRefused(Exception):
    """A run (or another gate input) is not acceptable; the message is a short reason with field names and counts only."""


# ------------------------------------------------------------------------------------------------ input acceptance
def _is_int(v) -> bool:
    return type(v) is int


def check_run(run, target: str = "dev", *, ablate=frozenset(), cases_expected: int | None = None) -> None:
    """Raise GateRefused unless `run` (a run.json dict) is an acceptable input for `target` with the ablation set `ablate`."""
    if not isinstance(run, dict):
        raise GateRefused("run.json is not an object")
    if run.get("partial") is not False:
        raise GateRefused("partial run (or no partial flag)")
    if run.get("stopped_reason") != "complete":
        raise GateRefused("stopped_reason is not complete")
    if target not in TARGETS:
        raise GateRefused(f"unknown target {target}")
    split, subset = TARGETS[target]
    if run.get("split") != split or run.get("subset") != subset:
        raise GateRefused(f"not a {split}/{subset} run")
    if run.get("salt") != "":
        raise GateRefused("salted run (or no salt field)")
    if run.get("only") is not None:
        raise GateRefused("an --only run")
    if run.get("cache_only") not in (True, False):
        raise GateRefused("no cache_only flag")
    if run.get("cache_only") and run.get("not_cached") != 0:
        raise GateRefused("cache-only run with calls not in the cache")
    got = run.get("ablate")
    if not isinstance(got, list) or frozenset(got) != frozenset(ablate):
        raise GateRefused(f"ablation set {sorted(got) if isinstance(got, list) else 'missing'}, the rule needs {sorted(ablate)}")
    if target == "dev":
        if not (_is_int(run.get("limit")) and run["limit"] == DEV_LIMIT and run.get("cases_total") == DEV_LIMIT):
            raise GateRefused(f"the dev smoke run must be --limit {DEV_LIMIT} with {DEV_LIMIT} cases")
        return
    if run.get("limit") is not None:
        raise GateRefused("a --limit run")
    if not _is_int(cases_expected):
        raise GateRefused("the full subset size is unknown")
    if run.get("cases_total") != cases_expected:
        raise GateRefused(f"cases_total is not the full {split}/{subset} size {cases_expected}")
    if FROZEN_PROMPT_VERSIONS is None:
        raise GateRefused("frozen prompt versions NOT YET RECORDED (Task 10 step 4)")
    pv = run.get("prompt_versions")
    if not isinstance(pv, dict) or pv != FROZEN_PROMPT_VERSIONS:
        roles = sorted(set(FROZEN_PROMPT_VERSIONS) | set(pv if isinstance(pv, dict) else {}))
        diff = [r for r in roles if not isinstance(pv, dict) or pv.get(r) != FROZEN_PROMPT_VERSIONS.get(r)]
        raise GateRefused("prompt versions differ from the frozen ones: " + ", ".join(diff))


# ------------------------------------------------------------------------------------------------ the rules (pure)
def r_smoke(run: dict) -> bool:
    """`partial` is false, failures are 0, readable pages read = 100%."""
    readable, read = run.get("pages_readable"), run.get("pages_read")
    return (run.get("partial") is False and run.get("failures") == 0 and _is_int(readable) and _is_int(read)
            and read == readable)


def r_route(covered: int, total: int) -> bool:
    return covered == total


def wrong_counts(preds: dict[str, str], gold: dict[str, str], exclude: frozenset[str]) -> tuple[int, int]:
    """(wrong rows, false-confident rows) over the ids of `gold` minus `exclude`. A row with no prediction is wrong. False-confident:
    a requirement row (not OVERALL) whose gold is UNKNOWN or NEEDS_REVIEW and whose prediction is PASS or FAIL."""
    rows = [i for i in gold if i not in exclude]
    wrong = sum(1 for i in rows if preds.get(i) != gold[i])
    fc = sum(1 for i in rows if not i.endswith("__OVERALL") and gold[i] in ABSTAIN_LABELS and preds.get(i) in CONFIDENT_LABELS)
    return wrong, fc


def r_hold(b: tuple[int, int], s0: tuple[int, int]) -> bool:
    return b[0] <= s0[0] and b[1] <= s0[1]


def r_dm(preds_dm: dict[str, str], preds_abstain: dict[str, str], gold: dict[str, str], exclude: frozenset[str]) -> tuple[bool, int, int]:
    """(pass, wrong->right, right->wrong) over the gold rows (minus `exclude`) where the two runs differ; pass when w2r >= r2w."""
    w2r = r2w = 0
    for i in gold:
        if i in exclude or preds_dm.get(i) == preds_abstain.get(i):
            continue
        dm_ok, ab_ok = preds_dm.get(i) == gold[i], preds_abstain.get(i) == gold[i]
        w2r += int(dm_ok and not ab_ok)
        r2w += int(ab_ok and not dm_ok)
    return w2r >= r2w, w2r, r2w


def r_test(preds: dict[str, str], sub2: dict[str, str]) -> tuple[bool, int]:
    """(pass, rows changed against submission #2); a row in only one file is changed; pass when changed <= 30."""
    changed = sum(1 for i in set(preds) | set(sub2) if preds.get(i) != sub2.get(i))
    return changed <= R_TEST_MAX_CHANGED, changed


def router_config(holdout_route_pass: bool) -> frozenset[str]:
    """Step 1 of the configuration order (R53): the router stays on only when the holdout R-route passed (label-free)."""
    return frozenset() if holdout_route_pass is True else ROUTER


def configuration(holdout_route_pass: bool, dm_pass: bool) -> frozenset[str]:
    """The candidate configuration (R53, R46): the router ablations, then the Decision Maker's (off unless R-DM passed)."""
    return router_config(holdout_route_pass) | (frozenset() if dm_pass is True else DM)


def candidate(rules: dict[str, bool], *, router_off: bool = False) -> tuple[str, frozenset[str]]:
    """("B", ablations) or ("none", frozenset()). A missing rule counts as failed (R45). With the router on, R-route (test) is a
    gate; with the router off (the holdout R-route failed or was not computable) it holds by construction and "router" is ablated."""
    required = [k for k in REQUIRED if not (router_off and k == "R-route")]
    if all(rules.get(k) is True for k in required):
        return "B", configuration(not router_off, rules.get("R-DM") is True)
    return "none", frozenset()


def replay_identical(preds_a: bytes, preds_b: bytes, run_b: dict) -> bool:
    """The cache-only replay `run_b` is complete and its preds.csv bytes equal the candidate's."""
    return (isinstance(run_b, dict) and run_b.get("partial") is False and run_b.get("stopped_reason") == "complete"
            and run_b.get("cache_only") is True and run_b.get("not_cached") == 0 and preds_a == preds_b)


# ------------------------------------------------------------------------------------------------ reading runs
def run_dir(root: Path, target: str, ablate=frozenset()) -> Path:
    split, subset = TARGETS[target]
    return Path(root) / default_out(split, subset, frozenset(ablate), "", limit=DEV_LIMIT if target == "dev" else None).name


def read_rows(path: Path) -> dict[str, str]:
    """id -> verdict from an (id, verdict) csv; GateRefused when the file is missing or unreadable."""
    path = Path(path)
    if not path.is_file():
        raise GateRefused(f"{path.parent.name}/{path.name} missing")
    try:
        with path.open(encoding="utf-8") as f:
            return {r["id"]: r["verdict"] for r in csv.DictReader(f)}
    except (OSError, ValueError, KeyError) as exc:
        raise GateRefused(f"{path.parent.name}/{path.name} unreadable ({type(exc).__name__})") from None


def load_run(d: Path) -> dict:
    p = Path(d) / "run.json"
    if not p.is_file():
        raise GateRefused(f"{Path(d).name}: no run.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateRefused(f"{Path(d).name}: run.json unreadable ({type(exc).__name__})") from None


def accept(root: Path, target: str, ablate=frozenset(), cases_expected: int | None = None) -> tuple[Path, dict]:
    d = run_dir(root, target, ablate)
    run = load_run(d)
    try:
        check_run(run, target, ablate=ablate, cases_expected=cases_expected)
    except GateRefused as exc:
        raise GateRefused(f"{d.name}: {exc}") from None
    return d, run


def _nc(reason: str) -> dict:
    return {"pass": False, "computable": False, "reason": reason}


def _route_rule(d: Path, run: dict) -> dict:
    rr = run.get("route_recall")
    if not (isinstance(rr, dict) and _is_int(rr.get("covered")) and _is_int(rr.get("total"))):
        return _nc(f"{d.name}: no route_recall (no baseline cache)")
    return {"pass": r_route(rr["covered"], rr["total"]), "computable": True, "run": d.name, "covered": rr["covered"],
            "total": rr["total"], "missed": len(rr.get("missed") or [])}


# ------------------------------------------------------------------------------------------------ the targets
def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def fingerprint(d: Path) -> dict:
    """What a gate read from a run directory: its name, its path and the sha256 of each input file (None = absent) (R55a)."""
    d = Path(d)
    return {"dir": d.name, "path": str(d), **{name: _sha(d / name) for name in HASHED_FILES}}


def section_fresh(sec: dict | None) -> bool:
    """A recorded section counts only while the prompts are frozen, its prompt versions are the frozen ones, and every directory it
    consulted still hashes the same (a run re-run after gating needs a re-gate) (R55a)."""
    if not isinstance(sec, dict) or FROZEN_PROMPT_VERSIONS is None or sec.get("prompt_versions") != FROZEN_PROMPT_VERSIONS:
        return False
    inputs = sec.get("inputs")
    return isinstance(inputs, list) and all(isinstance(i, dict) and fingerprint(Path(str(i.get("path")))) == i for i in inputs)


def gate_dev(root: Path) -> dict:
    inputs = [fingerprint(run_dir(root, "dev"))]
    try:
        d, run = accept(root, "dev", frozenset(), DEV_LIMIT)
    except GateRefused as exc:
        return {"rules": {"R-smoke": _nc(str(exc))}, "inputs": inputs}
    return {"rules": {"R-smoke": {"pass": r_smoke(run), "computable": True, "run": d.name, "failures": run.get("failures"),
                                  "pages_read": run.get("pages_read"), "pages_readable": run.get("pages_readable")}},
            "prompt_versions": run.get("prompt_versions"), "inputs": inputs}


def gate_holdout(root: Path, cases_expected: int | None, gold: Callable[[], dict], s0: Callable[[], dict]) -> dict:
    """The configuration order (R53, R46): R-route on the full run fixes the router, R-DM in that router configuration fixes the
    Decision Maker, and R-hold reads the run in the final configuration."""
    runs: dict[frozenset, tuple[Path, dict, dict]] = {}
    why: dict[frozenset, str] = {}

    def get(ab: frozenset):
        if ab not in runs and ab not in why:
            try:
                d, run = accept(root, "holdout", ab, cases_expected)
                runs[ab] = (d, run, read_rows(d / "preds.csv"))
            except GateRefused as exc:
                why[ab] = str(exc)
        return runs.get(ab)

    rules: dict[str, dict] = {}
    full = get(frozenset())                                   # 1. the router: R-route on the full run, label-free
    rules["R-route"] = _route_rule(full[0], full[1]) if full else _nc(why[frozenset()])
    router = router_config(rules["R-route"]["pass"])
    on, off = get(router), get(router | DM)                   # 2. R-DM in the router configuration
    g, gold_why = None, None
    if on or off:
        try:
            g = gold()
        except GateRefused as exc:
            gold_why = str(exc)
    if on and off and g is not None:
        ok, w2r, r2w = r_dm(on[2], off[2], g, EXPOSED_HOLDOUT_ROWS)
        rules["R-DM"] = {"pass": ok, "computable": True, "runs": [on[0].name, off[0].name], "wrong_to_right": w2r,
                         "right_to_wrong": r2w}
    else:
        rules["R-DM"] = _nc("; ".join(r for r in (why.get(router), why.get(router | DM), gold_why) if r))
    config = configuration(rules["R-route"]["pass"], rules["R-DM"]["pass"])
    chosen = runs.get(config)                                 # 3. R-hold on the run in the final configuration
    if chosen is None:
        rules["R-hold"] = _nc(why[config])
    else:
        d, run, preds = chosen
        if router:                                            # router off: R-route holds by construction; computed and reported
            rules["R-route (final configuration)"] = _route_rule(d, run)
        if g is None:
            rules["R-hold"] = _nc(gold_why or "no gold")
        else:
            try:
                s0_preds = s0()
            except GateRefused as exc:
                rules["R-hold"] = _nc(str(exc))
            else:
                b, s = wrong_counts(preds, g, EXPOSED_HOLDOUT_ROWS), wrong_counts(s0_preds, g, EXPOSED_HOLDOUT_ROWS)
                rules["R-hold"] = {"pass": r_hold(b, s), "computable": True, "run": d.name, "b": list(b), "s0": list(s),
                                   "s0_model": BASELINE_MODEL,
                                   "rows": sum(1 for i in g if i not in EXPOSED_HOLDOUT_ROWS)}
    pv = next((r[1].get("prompt_versions") for r in runs.values()), None)
    inputs = [fingerprint(run_dir(root, "holdout", ab)) for ab in sorted({*runs, *why}, key=sorted)]
    return {"rules": rules, "router": "off" if router else "on", "config": sorted(config), "prompt_versions": pv, "inputs": inputs}


def _passed(section: dict | None, rule: str) -> bool:
    return bool(section) and ((section.get("rules") or {}).get(rule) or {}).get("pass") is True


def holdout_configuration(holdout: dict | None) -> frozenset[str]:
    """The final configuration recorded by the holdout step (no entry: router off and Decision Maker off, R45)."""
    return configuration(_passed(holdout, "R-route"), _passed(holdout, "R-DM"))


def gate_test(root: Path, cases_expected: int | None, sub2: Path, replay: Path | None, holdout: dict | None) -> dict:
    config = holdout_configuration(holdout)
    sec: dict = {"config": sorted(config), "router": "off" if "router" in config else "on", "rules": {},
                 "replay": {"checked": replay is not None, "identical": None},
                 "inputs": [fingerprint(run_dir(root, "test", config)), *([fingerprint(Path(replay))] if replay is not None else [])]}
    try:
        d, run = accept(root, "test", config, cases_expected)
    except GateRefused as exc:
        sec["rules"] = {"R-route": _nc(str(exc)), "R-test": _nc(str(exc))}
        if replay is not None:
            sec["replay"]["identical"] = False
        return sec
    sec["prompt_versions"] = run.get("prompt_versions")
    sec["rules"]["R-route"] = _route_rule(d, run)           # a gate only with the router on (summarize)
    try:                                                    # the FILE must exist: run.json `written` is never trusted
        ok, changed = r_test(read_rows(d / "submission.csv"), read_rows(sub2))
        sec["rules"]["R-test"] = {"pass": ok, "computable": True, "run": d.name, "changed": changed, "max": R_TEST_MAX_CHANGED}
    except GateRefused as exc:
        sec["rules"]["R-test"] = _nc(str(exc))
    if replay is not None:
        replay = Path(replay)
        sec["replay"]["dir"] = replay.name
        if replay.resolve() == d.resolve():
            sec["replay"].update(identical=False, reason="the replay directory is the candidate's own")
        else:
            try:
                rrun = load_run(replay)
                check_run(rrun, "test", ablate=config, cases_expected=cases_expected)
                a, b = d / "preds.csv", replay / "preds.csv"
                if not (a.is_file() and b.is_file()):
                    raise GateRefused("preds.csv missing")
                sec["replay"]["identical"] = replay_identical(a.read_bytes(), b.read_bytes(), rrun)
            except GateRefused as exc:
                sec["replay"].update(identical=False, reason=str(exc))
    return sec


def summarize(data: dict) -> tuple[dict[str, bool], dict, dict[str, bool]]:
    """The rule table, the candidate and the freshness of each section, from the merged gates.json (every call recomputes them; a
    missing or stale section counts as failed, R45/R55a). "R-route (holdout)" decides the router; "R-route" is the candidate's gate:
    the test half with the router on, true by construction with it off."""
    dev, hold, test = (data.get(k) if section_fresh(data.get(k)) else None for k in ("dev", "holdout", "test"))
    router_off = not _passed(hold, "R-route")
    want = sorted(holdout_configuration(hold))
    fresh = bool(test) and test.get("config") == want      # a test entry made under another configuration is stale
    rules = {"R-smoke": _passed(dev, "R-smoke"), "R-route (holdout)": not router_off,
             "R-route": True if router_off else fresh and _passed(test, "R-route"),
             "R-hold": _passed(hold, "R-hold"), "R-DM": _passed(hold, "R-DM"),
             "R-test": fresh and _passed(test, "R-test")}
    name, ab = candidate(rules, router_off=router_off)
    replay = (test.get("replay") or {}).get("identical") if fresh else None
    return rules, {"name": name, "ablate": sorted(ab), "router": "off" if router_off else "on", "test_entry_fresh": fresh,
                   "replay_identical": replay, "submittable": name == "B" and replay is True}, \
        {"dev": dev is not None, "holdout": hold is not None, "test": fresh}


def _line(name: str, r: dict) -> str:
    verdict = "pass" if r.get("pass") else "fail"
    if not r.get("computable"):
        return f"{name}: fail (not computable: {r.get('reason')})"
    extra = {k: v for k, v in r.items() if k not in ("pass", "computable")}
    return f"{name}: {verdict} (" + ", ".join(f"{k} {v}" for k, v in extra.items()) + ")"


def gate(target: str, *, runs_root: Path, gates_json: Path, cases_expected: int | None = None, gold: Callable[[], dict] | None = None,
         s0: Callable[[], dict] | None = None, sub2: Path = SUBMISSION_2, replay: Path | None = None, say=print) -> dict:
    """Compute `target`'s rules, merge them into `gates_json`, print counts only and return the merged dict."""
    if target not in TARGETS:
        raise GateRefused(f"unknown target {target}")
    gates_json = Path(gates_json)
    data: dict = {}
    if gates_json.exists():                                 # never replaced silently: an unreadable record is refused
        try:
            data = json.loads(gates_json.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GateRefused(f"{gates_json.name} unreadable ({type(exc).__name__}); fix or move it by hand") from None
        if not isinstance(data, dict) or not isinstance(data.get("history", []), list):
            raise GateRefused(f"{gates_json.name} is not a gates record (an object with a history list)")
    if target == "dev":
        sec = gate_dev(runs_root)
    elif target == "holdout":
        def no_input():
            raise GateRefused("no gold or s0 source")
        sec = gate_holdout(runs_root, cases_expected, gold or no_input, s0 or no_input)
    else:
        hold = data.get("holdout")
        sec = gate_test(runs_root, cases_expected, Path(sub2), replay, hold if section_fresh(hold) else None)
    data[target] = sec
    rules, cand, fresh = summarize(data)
    data["rules"], data["candidate"], data["fresh"] = rules, cand, fresh
    data["history"] = [*data.get("history", []), {          # append-only, counts only: a re-gate stays visible
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "target": target, "config": sec.get("config"),
        "rules": {k: {"pass": bool(r.get("pass")), "computable": bool(r.get("computable"))} for k, r in sec["rules"].items()},
        "inputs": [i["dir"] for i in sec.get("inputs", [])], "candidate": cand["name"]}]
    gates_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = gates_json.with_name(gates_json.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(gates_json)

    say(f"gates --target {target}" + (f" (configuration: ablate {'+'.join(sec['config']) or 'nothing'})"
                                      if "config" in sec else ""))
    for name, r in sec["rules"].items():
        say("  " + _line(name, r))
    if target == "test" and sec["replay"]["checked"]:
        say(f"  replay identical: {'pass' if sec['replay']['identical'] else 'fail'}")
    say("rules so far: " + ", ".join(f"{k} {'pass' if v else 'fail'}" for k, v in rules.items()))
    say(f"candidate: {cand['name']}" + (f" (ablate {'+'.join(cand['ablate'])})" if cand["ablate"] else "")
        + f"; replay identical: {'not checked' if cand['replay_identical'] is None else 'pass' if cand['replay_identical'] else 'fail'}"
        + f"; submittable: {'yes, with the user' + chr(39) + 's yes' if cand['submittable'] else 'no'}; nothing is uploaded")
    say(f"written: {gates_json}")
    return data


# ------------------------------------------------------------------------------------------------ real inputs (CLI only)
def subset_size(data_dir: Path, split: str, subset: str) -> int | None:
    from eval import run_kaggle as rk
    try:
        return len(rk.select_cases(rk.load_cases(Path(data_dir)), split, subset, None, None))
    except (OSError, ValueError, KeyError):
        return None


def load_gold(data_dir: Path) -> dict[str, str]:
    """The train labels of the holdout cases (read only to count)."""
    from eval import run_kaggle as rk
    try:
        ids = set(rk.select_cases(rk.load_cases(Path(data_dir)), "train", "holdout", None, None))
        with (Path(data_dir) / "train_labels.csv").open(encoding="utf-8") as f:
            return {r["id"]: r["verdict"] for r in csv.DictReader(f) if r["id"].split("__")[0] in ids}
    except (OSError, ValueError, KeyError) as exc:
        raise GateRefused(f"train labels unreadable ({type(exc).__name__})") from None


def s0_holdout_preds(data_dir: Path, *, llm=None, system: str | None = None) -> dict[str, str]:
    """s0 on the holdout: the cached baseline extraction (model `BASELINE_MODEL`, prompt 9566ad4b6b95, unsalted, never a model call)
    through `run_kaggle.assemble_case`. The model name is part of the cache key and run.json does not carry it: the configured
    extractor (or `llm`, tests) must be `BASELINE_MODEL`, and the mock provider is refused. `system` replaces the baseline prompt
    text (tests only)."""
    from assessor.llm import LLM
    from assessor.tracing import Budget
    from eval import run_kaggle as rk

    from . import baseline as baseline_mod
    if llm is None:
        llm = LLM.extractor(budget=Budget(max_usd=1.0, max_seconds=0, max_tokens=0, max_tool_calls=0))
    if llm.mock_mode:
        raise GateRefused("s0 baseline: the extractor is on the mock provider (the cache is keyed by model)")
    if llm.model != BASELINE_MODEL:
        raise GateRefused(f"s0 baseline: the configured extractor model is not the pinned {BASELINE_MODEL}")
    data_dir = Path(data_dir)
    cache = data_dir / "cache" / "extract"
    if not cache.is_dir():
        raise GateRefused("s0 baseline: no extraction cache")
    try:
        cases, pages = rk.load_cases(data_dir), rk.load_pages(data_dir)
        reqs = json.loads((data_dir / "requirements.json").read_text(encoding="utf-8"))
        rl = rk.load_module("kaggle_rules_lite", data_dir / "rules_lite.py")
        ex = rk.Extractor(baseline_mod._NoCallLLM(llm.model), cache, True, system=system or baseline_mod.baseline_prompt())
        preds: dict[str, str] = {}
        bad = 0
        for cid in rk.select_cases(cases, "train", "holdout", None, None):
            recs = rk.triage(cases[cid], pages[cid])
            for r in recs:
                if rk.readable(r):
                    r["extract"] = ex.run(r["page"])
            res = rk.assemble_case(cases[cid], recs, reqs, rl)
            bad += int(res["bad"] > 0)
            preds.update({f"{cid}__{rid}": v for rid, v in res["verdicts"].items()})
            preds[f"{cid}__OVERALL"] = res["overall"]
    except GateRefused:
        raise
    except Exception as exc:                                   # noqa: BLE001  class name only: never a message
        raise GateRefused(f"s0 baseline: {type(exc).__name__}") from None
    if bad:
        raise GateRefused(f"s0 baseline: {bad} holdout cases with a readable page not in the extraction cache")
    return preds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vendor_verdict gates", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--data", default=str(ROOT / "data_kaggle"))
    ap.add_argument("--runs-root", default=str(RUNS_ROOT), help="the folder holding the run directories")
    ap.add_argument("--gates-json", default="", help="default: <runs-root>/gates.json")
    ap.add_argument("--sub2", default=str(SUBMISSION_2), help="submission #2 (R-test)")
    ap.add_argument("--replay", default=None, help="test only: the candidate's cache-only replay directory (byte-identity check)")
    a = ap.parse_args(argv)
    if a.replay is not None and a.target != "test":
        print(f"refused (exit {EXIT_USAGE}): --replay is for --target test only")
        return EXIT_USAGE
    root = Path(a.runs_root)
    kw: dict = {"runs_root": root, "gates_json": Path(a.gates_json) if a.gates_json else root / "gates.json"}
    data = Path(a.data)
    if a.target == "holdout":
        kw.update(cases_expected=subset_size(data, "train", "holdout"), gold=lambda: load_gold(data), s0=lambda: s0_holdout_preds(data))
    elif a.target == "test":
        kw.update(cases_expected=subset_size(data, "test", "all"), sub2=Path(a.sub2),
                  replay=Path(a.replay) if a.replay else None)
    try:
        gate(a.target, **kw)
    except GateRefused as exc:
        print(f"refused: {exc}")
        return EXIT_REFUSED
    return 0
