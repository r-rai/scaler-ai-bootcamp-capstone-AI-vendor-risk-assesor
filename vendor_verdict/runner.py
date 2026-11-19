"""The vendor_verdict runner: one graph per case (cases in parallel, nodes sequential inside a case), then the run's outputs.

    python -m vendor_verdict run --split test|train --subset all|dev|holdout [--limit N] [--only ids] [--cache-only] [--estimate]
        [--salt S] [--engine plain|langgraph] [--ablate router,grader,decision_maker,iteration2] [--workers 6] [--out DIR]
        [--data DIR] [--max-usd X]

Outputs in `--out` (default `reports/vendor_verdict/<split>_<subset>[_ablate-a+b][_salt-S]/`):
- `preds.csv` (id, verdict) for every finished case, in case-id order (never completion order), so a cache-only rerun is byte-identical;
- `run.json`: counts only (see `run`); Task 9's gates read it, so its field names are a contract;
- `traces/<case_id>.json` and `.md`; `graph.mmd`;
- `submission.csv` only under `submission_rule` (ruling R4). It is never uploaded.
Every earlier output in `--out` is removed when a run starts, so a stale submission can never sit next to a later partial run.

Per-role accounting (cases run in parallel, so the global CallCache counters cannot be split by role):
- `RoleCache` remembers the role of every key it mints (`key(role, ...)` is always called before `get`/`put` on that key), so cache
  hits, misses, put failures and the tokens of each new record (the agents' ~4 chars/token estimates) are counted per role;
- `GuardedLLM` wraps each model and counts the calls actually sent to it per `stage` (the stage names the role); in a cache-only run it
  refuses the call (`NotCached`, an `LLMError`, so every agent treats it as its ordinary failed call) and counts `not_cached`;
- failures come from the node flags in each case's state, seconds from the case traces (node -> role).

Console output is counts, model names and case ids only: never page text, a reading, a note or an exception message (class names only).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import numbers
import re
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from assessor.llm import LLMError
from assessor.tracing import BudgetExceeded, estimate_tokens
from eval import run_kaggle as rk

from . import baseline as baseline_mod
from . import data as data_mod
from .agents import planner as planner_agent
from .agents import reviewer
from .agents import specialists as specialist_agents
from .cache import CallCache
from .config import FAMILIES, GRADER_FIELDS, LOCAL_SCOPE, ROOT
from .graph import build_langgraph, mermaid, new_state, run_plain
from .nodes import ABLATIONS, FAILED_PAGE_NOTE, FLAG_KEYS, Agents, _readable_pages
from .prompts import DECISION_SYSTEM, GRADER_SYSTEM, PLANNER_SYSTEM, ROUTER_SYSTEM, SPECIALIST_SYSTEM, page_user_prompt, version
from .safe import safe_id
from .trace import Trace

ROLES = ("planner", "specialist", "router", "grader", "decision_maker")
SUBMISSION_ABLATIONS = (frozenset(), frozenset({"decision_maker"}))     # R4: the only ablation gates.candidate() can select
_NODE_ROLE = {"plan": "planner", "specialists": "specialist", "route_pages": "router", "grade_evidence": "grader",
              "decide_conflicts": "decision_maker"}
_FLAG_ROLE = {"planner_failures": "planner", "specialist_failures": "specialist", "case_read_failures": "specialist",
              "router_fallbacks": "router", "grader_failures": "grader", "decision_failures": "decision_maker",
              "decision_invalid": "decision_maker"}
_OUTPUTS = ("submission.csv", "run.json", "preds.csv", "graph.mmd")
# Exceptions that mean "abandon this case, start no new case, finish cleanly" (the case writes no preds or trace). Task 8C adds its
# quota exception here; `_stop_name` gives the run.json `stopped` / `stopped_reason` value of each.
STOP_EXCEPTIONS: tuple[type[BaseException], ...] = (BudgetExceeded,)


def _stop_name(exc: BaseException) -> str:
    return "budget" if isinstance(exc, BudgetExceeded) else type(exc).__name__.lower()


def role_of(name: str) -> str:
    """The run.json role of a cache role or an LLM stage: "specialist:<family>" stays per family, "decision" is the decision maker."""
    return "decision_maker" if name == "decision" else name


def prompt_versions() -> dict[str, str]:
    return {"planner": version(PLANNER_SYSTEM), **{f"specialist:{f}": version(SPECIALIST_SYSTEM[f]) for f in FAMILIES},
            "router": version(ROUTER_SYSTEM), "grader": version(GRADER_SYSTEM), "decision_maker": version(DECISION_SYSTEM)}


# ------------------------------------------------------------------------------------------------ JSON hygiene
def jsonable(value):
    """A strict-JSON copy: plain int/float (numpy scalars too), NaN/inf -> None, tuples and sets -> lists, keys -> str."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return f"<{type(value).__name__}>"


def _dump(obj) -> str:
    return json.dumps(jsonable(obj), indent=2, allow_nan=False)          # default ensure_ascii: a lone surrogate cannot break utf-8


def _int(v) -> int:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    return int(f) if math.isfinite(f) else 0


def _float(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def clean_trace(trace: Trace) -> Trace:
    """A copy of `trace` whose tokens and seconds are plain finite numbers and whose summaries are dicts (Trace stores the numbers raw
    and `to_markdown` needs a dict summary), so `to_json` dumps strictly and `to_markdown` never fails."""
    out = Trace(trace.case_id)
    with trace._lock:               # the raw records: `to_json` itself sums the raw numbers and fails on a None
        raw = [dict(h) for h in trace._handovers]
    for h in raw:
        summary = h["summary"] if isinstance(h["summary"], dict) else {"summary": "<unrecordable>"}
        out.handover(h["agent"], h["to"], jsonable(summary), tokens_in=_int(h["tokens_in"]), tokens_out=_int(h["tokens_out"]),
                     seconds=_float(h["seconds"]))
    return out


# ------------------------------------------------------------------------------------------------ accounting
def _zero() -> dict:
    return {"calls": 0, "not_cached": 0, "cache_hits": 0, "cache_misses": 0, "put_failures": 0, "failures": 0, "tokens_in": 0,
            "tokens_out": 0, "seconds": 0.0}


class RoleCache(CallCache):
    """A CallCache that counts hits, misses, put failures and new-record tokens per role (the first argument of `key`)."""

    def __init__(self, root: Path):
        super().__init__(root)
        self._role_of: dict[str, str] = {}
        self.by_role: dict[str, Counter] = {}

    def _bump(self, key: str, **add: int) -> None:
        with self._lock:
            c = self.by_role.setdefault(self._role_of.get(key, "unknown"), Counter())
            c.update(add)

    def key(self, role: str, prompt_version: str, model: str, payload: str, salt: str = "") -> str:
        k = super().key(role, prompt_version, model, payload, salt)
        with self._lock:
            self._role_of[k] = role_of(role)
        return k

    def get(self, key: str) -> dict | None:
        rec = super().get(key)
        self._bump(key, cache_hits=int(rec is not None), cache_misses=int(rec is None))
        return rec

    def put(self, key: str, record: dict) -> None:
        super().put(key, record)                        # never raises; a failed write leaves no file behind
        self._bump(key, put_failures=int(not (self.root / f"{key}.json").exists()), tokens_in=_int(record.get("tokens_in")),
                   tokens_out=_int(record.get("tokens_out")))


class NotCached(LLMError):
    """A cache-only run refused a model call. An LLMError, so every agent handles it as its ordinary failed call. The message holds
    neither "429" nor "rate limit": the planner and the specialists would otherwise wait and retry."""


class RunStopped(BaseException):
    """The run is stopping (budget or interrupt): the case in flight ends at its next model call. A BaseException, so the agents'
    `except Exception` fallbacks cannot turn it into a counted failure."""


class GuardedLLM:
    """Forwards to a model, counting the calls sent per stage; refuses every call in a cache-only run or once the run is stopping."""

    def __init__(self, llm, cache_only: bool, stop: threading.Event):
        self._llm, self._cache_only, self._stop = llm, cache_only, stop
        self._lock = threading.Lock()
        self.calls: Counter = Counter()
        self.not_cached: Counter = Counter()

    @property
    def model(self):
        return self._llm.model

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def structured(self, system, user, schema, *args, stage: str = "", **kw):
        if self._stop.is_set():
            raise RunStopped()
        with self._lock:
            (self.not_cached if self._cache_only else self.calls)[role_of(stage)] += 1
        if self._cache_only:
            raise NotCached("not in the cache (cache-only run)")
        return self._llm.structured(system, user, schema, *args, stage=stage, **kw)


def _roles(cache: RoleCache, llms: list[GuardedLLM], flags: Counter, seconds: Counter) -> dict[str, dict]:
    out = {r: _zero() for r in ROLES} | {f"specialist:{f}": _zero() for f in FAMILIES}
    for name, c in cache.by_role.items():
        if name in out:
            for k, v in c.items():
                out[name][k] += v
    for g in llms:
        for name, n in g.calls.items():
            if name in out:
                out[name]["calls"] += n
        for name, n in g.not_cached.items():
            if name in out:
                out[name]["not_cached"] += n
    for f in FAMILIES:                                  # the specialist row is the sum of its families
        for k in ("calls", "not_cached", "cache_hits", "cache_misses", "put_failures", "tokens_in", "tokens_out"):
            out["specialist"][k] += out[f"specialist:{f}"][k]
    for k, v in flags.items():
        out[_FLAG_ROLE[k]]["failures"] += v
    for r, s in seconds.items():
        out[r]["seconds"] = round(s, 3)
    return out


# ------------------------------------------------------------------------------------------------ rules and paths
def submission_rule(split: str, salt: str, ablate, partial: bool, ids, test_case_ids) -> tuple[bool, str]:
    """(write submission.csv?, why). Ruling R4: the test split, no salt, not partial, ablate empty or exactly {decision_maker}; and, as
    run_kaggle requires, the run covers every test case and nothing else (write_submission would fill a missing case with UNKNOWN)."""
    if split != "test":
        return False, "not the test split"
    if salt:
        return False, "salted run: a noise sample, never a candidate"
    if frozenset(ablate) not in SUBMISSION_ABLATIONS:
        return False, "ablation other than decision_maker"
    if partial:
        return False, "partial run"
    if set(ids) != set(test_case_ids):
        return False, "the run does not cover exactly the test cases"
    return True, "ok"


def default_out(split: str, subset: str, ablate, salt: str) -> Path:
    name = f"{split}_{subset}"
    if ablate:
        name += "_ablate-" + "+".join(sorted(ablate))
    if salt:
        name += "_salt-" + re.sub(r"[^A-Za-z0-9_.-]", "_", salt)
    return ROOT / "reports" / "vendor_verdict" / name


def _file_id(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", case_id)[:80] or "_"


def load_split(data_dir: Path, split: str, subset: str, limit: int | None = None, only: list[str] | None = None):
    """(cases, pages, selected ids): `vendor_verdict.data.load`."""
    return data_mod.load(Path(data_dir), split, subset, limit, only)


# ------------------------------------------------------------------------------------------------ estimate
def _exists(root: Path, key: str) -> bool:
    return (root / f"{key}.json").exists()           # looks only: never cache.get (it would count) and never a write


def _tokens(tin: int) -> dict:
    return {"tokens_in": tin, "tokens_out": round(tin * rk.OUT_PER_IN)}


def estimate(cases: dict, pages: dict, ids: list[str], ag: Agents) -> dict:
    """Calls and tokens per role before the run, with no model call and no file write. Tokens are for the UNCACHED calls only
    (`run_kaggle.estimate_call_tokens` / `OUT_PER_IN`). Planner, router and per-page specialist calls are measured against the cache
    files on disk; the specialist "expected" figure assumes the baseline claims' families per page (needs a baseline cache); "worst"
    sends every readable page to all 4 families and the iteration-2 worst case re-reads every family of every case (upper bounds).
    Grader and decision maker calls depend on what is read, so they cannot be cache-checked before the run."""
    root = Path(ag.cache.root)
    main_m, reader_m, local_m = ag.main_llm.model, ag.reader_llm.model, ag.local_llm.model
    have_base = ag.baseline_dir is not None and Path(ag.baseline_dir).is_dir()
    plan = {"calls": 0, "uncached": 0, "tokens_in": 0}
    route = {"pages": 0, "calls": 0, "uncached": 0, "tokens_in": 0}
    worst = {"per_page": len(FAMILIES), "calls": 0, "uncached": 0, "tokens_in": 0}
    expected = {"calls": 0, "uncached": 0, "tokens_in": 0}
    it2 = {"calls": 0, "tokens_in": 0}
    readable_n = grader_n = 0
    for cid in ids:
        recs = rk.triage(cases[cid], pages[cid])
        readable = [r["page"] for r in recs if rk.readable(r)]
        uniq = _readable_pages({"recs": recs})
        readable_n += len(uniq)
        if readable:
            user = planner_agent.metadata_prompt(readable)
            plan["calls"] += 1
            if not _exists(root, ag.cache.key("planner", version(PLANNER_SYSTEM), main_m, planner_agent.payload(user, cid), ag.salt)):
                plan["uncached"] += 1
                plan["tokens_in"] += estimate_tokens(PLANNER_SYSTEM + user)
        if "router" not in ag.ablate and "router" in LOCAL_SCOPE:
            route["pages"] += len(readable)
            for p in readable:
                route["calls"] += 1
                if not _exists(root, ag.cache.key("router", version(ROUTER_SYSTEM), local_m, page_user_prompt(p))):
                    route["uncached"] += 1
                    route["tokens_in"] += rk.estimate_call_tokens([p], ROUTER_SYSTEM)[0]
        base_fams: dict[str, set[str]] = {}
        if have_base:
            base = baseline_mod.baseline_readings(ag.baseline_model, recs, Path(ag.baseline_dir))
            for pid, fields in baseline_mod.route_baseline_fields(base, recs).items():
                base_fams[pid] = {f for f in FAMILIES if set(FAMILIES[f]) & fields}
            grader_n += sum(1 for pid in uniq for r in base.get(pid, []) if r.get("field") in GRADER_FIELDS)
        for f in FAMILIES:
            system = SPECIALIST_SYSTEM[f]
            for pid, page in uniq.items():
                key = ag.cache.key(f"specialist:{f}", version(system), reader_m,
                                   specialist_agents.payload(page_user_prompt(page), f), ag.salt)
                cached = _exists(root, key)
                tin = 0 if cached else rk.estimate_call_tokens([page], system)[0]
                worst["calls"] += 1
                worst["uncached"] += int(not cached)
                worst["tokens_in"] += tin
                if f in base_fams.get(pid, ()):
                    expected["calls"] += 1
                    expected["uncached"] += int(not cached)
                    expected["tokens_in"] += tin
            if uniq and "iteration2" not in ag.ablate:
                it2["calls"] += 1
                it2["tokens_in"] += estimate_tokens(system + specialist_agents.case_user_prompt(list(uniq.values()), list(FAMILIES[f])))
    per_page = lambda n: n / readable_n if readable_n else 0.0      # noqa: E731
    exp = None
    if have_base:
        exp = {"per_page": per_page(expected["calls"]), "calls": expected["calls"], "uncached": expected["uncached"],
               **_tokens(expected["tokens_in"]), "basis": "the baseline claims' families per page; uncached measured"}
    return {
        "cases": len(ids), "pages_readable": readable_n,
        "planner": {"calls": plan["calls"], "uncached": plan["uncached"], **_tokens(plan["tokens_in"]), "basis": "measured"},
        "router": {"pages": route["pages"], "calls": route["calls"], "uncached": route["uncached"], **_tokens(route["tokens_in"]),
                   "basis": "ablated" if "router" in ag.ablate else "measured"},
        "specialist": {"expected": exp, "expected_note": None if have_base else "no baseline cache: expected per page unknown",
                       "worst": {"per_page": worst["per_page"], "calls": worst["calls"], "uncached": worst["uncached"],
                                 **_tokens(worst["tokens_in"]), "basis": "upper bound: every readable page to all 4; uncached measured"}},
        "iteration2_worst": {"calls": it2["calls"], **_tokens(it2["tokens_in"]),
                             "basis": "ablated" if "iteration2" in ag.ablate else "upper bound: every family of every case"},
        "grader": {"readings_expected": grader_n if have_base else None,
                   "basis": "the baseline claims' count; not cache-checkable before the run" if have_base else "no baseline cache"},
        "decision_maker": {"basis": "not estimable before the run (conflicts are known only after reading)"},
    }


def format_estimate(est: dict) -> list[str]:
    def tok(d):
        return f"tokens in {d['tokens_in']:,} / out {d['tokens_out']:,}"
    s = est["specialist"]
    lines = ["estimate (no model call made; tokens are for uncached calls; 'measured' = counted from the cache files on disk, "
             "'upper bound' = a worst case, 'expected' = from the baseline claims):",
             f"  cases {est['cases']}, readable pages {est['pages_readable']}",
             f"  planner [paid, main]: calls {est['planner']['calls']}, uncached {est['planner']['uncached']} (measured); {tok(est['planner'])}",
             f"  router [local]: pages {est['router']['pages']}, uncached {est['router']['uncached']} ({est['router']['basis']}); "
             f"{tok(est['router'])}"]
    if s["expected"] is not None:
        e = s["expected"]
        lines.append(f"  specialists [paid, reader] expected: {e['per_page']:.2f} per page = {e['calls']} calls, uncached {e['uncached']} "
                     f"(from the baseline claims' families; uncached measured); {tok(e)}")
    else:
        lines.append(f"  specialists [paid, reader] expected: unknown ({s['expected_note']})")
    w = s["worst"]
    lines.append(f"  specialists [paid, reader] worst case: {w['per_page']} per page = {w['calls']} calls, uncached {w['uncached']} "
                 f"(upper bound; uncached measured); {tok(w)}")
    i2 = est["iteration2_worst"]
    lines.append(f"  iteration 2 [paid, reader] worst case: {i2['calls']} case reads ({i2['basis']}); {tok(i2)}")
    g = est["grader"]
    lines.append(f"  grader [local]: readings expected {g['readings_expected'] if g['readings_expected'] is not None else 'unknown'} "
                 f"({g['basis']})")
    lines.append(f"  decision maker [local]: {est['decision_maker']['basis']}")
    return lines


# ------------------------------------------------------------------------------------------------ one case
def _run_case(case: dict, pages: list[dict], ag: Agents, app) -> dict:
    state = new_state(case, pages)
    final = run_plain(state, ag) if app is None else app.invoke(state)
    readable = _readable_pages(final)
    routed = {pid for ids in (final.get("routes") or {}).values() for pid in ids}
    notes = final.get("notes") or []
    failed = [n[len(FAILED_PAGE_NOTE):] for n in notes if n.startswith(FAILED_PAGE_NOTE)]
    recall = None
    if ag.baseline_dir is not None and Path(ag.baseline_dir).is_dir():
        recall = reviewer.route_recall(final.get("routes") or {},
                                       baseline_mod.route_baseline_fields(final.get("baseline") or {}, final.get("recs") or []))
    return {"verdicts": dict(final.get("verdicts") or {}), "overall": final.get("overall"), "flags": dict(final.get("flags") or {}),
            "readable": len(readable), "unrouted": sum(1 for p in readable if p not in routed), "failed_pages": failed,
            "iteration": int(final.get("iteration") or 1), "recall": recall, "trace": final["trace"]}


# ------------------------------------------------------------------------------------------------ the run
def run(data_dir, split: str, subset: str, *, agents: Agents, out, limit: int | None = None, only: list[str] | None = None,
        cache_only: bool = False, engine: str = "plain", workers: int = 6, metric=None, budget=None, say=print) -> dict:
    """Run the selected cases and write the outputs into `out`; returns the run.json dict. `agents` holds the models (the seam tests
    stub), the cache, rules_lite, the requirements, the baseline, the salt and the ablations. `metric` is the host's metric module
    (loaded from the data dir when a submission is written and none is given); `budget` (optional) is reported in `spend`."""
    if engine not in ("plain", "langgraph"):
        raise ValueError(f"unknown engine {engine!r}")
    t_start = time.perf_counter()
    data_dir, out = Path(data_dir), Path(out)
    cases, pages, ids = load_split(data_dir, split, subset, limit, only)
    ids = sorted(ids)
    for line in format_estimate(estimate(cases, pages, ids, agents)):
        say(line)

    out.mkdir(parents=True, exist_ok=True)
    for name in _OUTPUTS:                               # no stale output may survive into this run's folder
        (out / name).unlink(missing_ok=True)
    shutil.rmtree(out / "traces", ignore_errors=True)

    stop = threading.Event()
    cache = RoleCache(Path(agents.cache.root))
    llms = {k: GuardedLLM(getattr(agents, k), cache_only, stop) for k in ("main_llm", "reader_llm", "local_llm")}
    ag = replace(agents, cache=cache, **llms)
    app = build_langgraph(ag) if engine == "langgraph" else None
    mode = (["cache-only"] if cache_only else []) + (["salted"] if ag.salt else []) + \
        ([f"ablate {'+'.join(sorted(ag.ablate))}"] if ag.ablate else [])
    say(f"{len(ids)} cases; engine {engine}; workers {max(1, workers)}" + "".join(f"; {m}" for m in mode))

    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    stopped: str | None = None
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    futs = {pool.submit(_run_case, cases[cid], pages[cid], ag, app): cid for cid in ids}

    def halt(reason: str) -> None:
        nonlocal stopped
        stopped = stopped or reason
        stop.set()
        for f in futs:
            f.cancel()

    done = 0
    try:
        for fut in as_completed(futs):                  # progress and stop detection; outcomes are collected after the shutdown
            done += 1
            if fut.cancelled():
                continue
            exc = fut.exception()
            if isinstance(exc, STOP_EXCEPTIONS):
                halt(_stop_name(exc))
            elif isinstance(exc, KeyboardInterrupt):
                halt("interrupted")
            if done % 50 == 0:
                say(f"  {done}/{len(ids)} cases")
    except KeyboardInterrupt:
        halt("interrupted")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)   # in-flight cases end at their next model call once `stop` is set
    for fut, cid in futs.items():
        if fut.cancelled():
            continue                                    # never started: unfinished
        exc = fut.exception()
        if exc is None:
            results[cid] = fut.result()
        elif not isinstance(exc, (*STOP_EXCEPTIONS, KeyboardInterrupt, RunStopped)):
            errors[cid] = type(exc).__name__            # a bug in one case: counted by class name only, the other cases go on

    # ---- aggregate (counts only) ----
    finished = sorted(results)
    flags: Counter = Counter({k: 0 for k in FLAG_KEYS})
    seconds: Counter = Counter()
    failed_pages: list[str] = []
    recall_c = recall_t = 0
    missed: list[str] = []
    for cid in finished:
        r = results[cid]
        flags.update({k: _int(v) for k, v in r["flags"].items() if k in _FLAG_ROLE})
        failed_pages += [safe_id(p) for p in r["failed_pages"]]
        for h in r["trace"].to_json()["handovers"]:
            if h["agent"] in _NODE_ROLE:
                seconds[_NODE_ROLE[h["agent"]]] += _float(h["seconds"])
        if r["recall"] is not None:
            recall_c, recall_t = recall_c + r["recall"][0], recall_t + r["recall"][1]
            missed += [safe_id(m.rsplit(":", 1)[0]) + ":" + m.rsplit(":", 1)[1] for m in r["recall"][2]]
    readable = sum(results[c]["readable"] for c in finished)
    unrouted = sum(results[c]["unrouted"] for c in finished)
    unfinished = [c for c in ids if c not in results and c not in errors]
    reasons = [name for name, hit in (("failed_pages", bool(failed_pages)), (stopped or "", stopped is not None),
                                      ("case_errors", bool(errors)), ("unfinished_cases", bool(unfinished))) if hit]
    partial = bool(reasons)
    # one word for the run: a stop wins over failed pages, failed pages over a case error ("complete" iff not partial)
    stopped_reason = stopped or ("failed_pages" if failed_pages else "case_errors" if errors else
                                 "unfinished_cases" if unfinished else "complete")
    have_base = agents.baseline_dir is not None and Path(agents.baseline_dir).is_dir()

    # ---- preds.csv (case-id order, verdicts in rules_lite order, then OVERALL) ----
    preds: dict[str, str] = {}
    for cid in finished:
        for rid, v in results[cid]["verdicts"].items():
            preds[f"{cid}__{rid}"] = v
        preds[f"{cid}__OVERALL"] = results[cid]["overall"]
    with (out / "preds.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "verdict"])
        w.writerows(preds.items())

    # ---- traces, graph ----
    (out / "traces").mkdir(exist_ok=True)
    for cid in finished:
        t = clean_trace(results[cid]["trace"])
        (out / "traces" / f"{_file_id(cid)}.json").write_text(_dump(t.to_json()), encoding="utf-8")
        (out / "traces" / f"{_file_id(cid)}.md").write_text(t.to_markdown(), encoding="utf-8")
    (out / "graph.mmd").write_text(mermaid(ag), encoding="utf-8")

    # ---- submission (R4) ----
    test_ids: set[str] = set()
    test_rows: list[dict] = []
    if split == "test" and (data_dir / "test.csv").exists():
        with (data_dir / "test.csv").open(encoding="utf-8") as f:
            test_rows = list(csv.DictReader(f))
        test_ids = {r["case_id"] for r in test_rows}
    allowed, why = submission_rule(split, ag.salt, ag.ablate, partial, ids, test_ids)
    problems: list[str] = []
    if allowed:
        if metric is None:
            metric = rk.load_module("kaggle_metric", data_dir / "metric.py")
        problems = rk.write_submission(out / "submission.csv", test_rows, preds, metric)

    rj = {
        "split": split, "subset": subset, "engine": engine, "workers": max(1, workers), "limit": limit,
        "only": sorted(only) if only else None, "cache_only": cache_only, "salt": ag.salt, "ablate": sorted(ag.ablate),
        "partial": partial, "stopped_reason": stopped_reason, "partial_reasons": reasons, "stopped": stopped,
        "cases": {"selected": len(ids), "finished": len(finished), "errors": len(errors), "unfinished": len(unfinished)},
        "cases_total": len(ids), "cases_done": len(finished), "cases_pending": len(unfinished),
        "case_errors": dict(sorted(Counter(errors.values()).items())), "error_case_ids": [safe_id(c) for c in sorted(errors)],
        "unfinished_case_ids": [safe_id(c) for c in unfinished],
        "pages_readable": readable, "pages_read": readable - len(failed_pages) - unrouted, "pages_failed": len(failed_pages),
        "pages_unrouted": unrouted, "failed_page_ids": failed_pages,
        "failures": sum(flags.values()) + len(errors), "flags": dict(flags),
        "local_fallbacks": {"router_all_families": flags["router_fallbacks"], "grader_unsure": flags["grader_failures"],
                            "decision_abstain": flags["decision_failures"] + flags["decision_invalid"]},
        "iteration2_cases": sum(1 for c in finished if results[c]["iteration"] >= 2),
        "roles": _roles(cache, list(llms.values()), flags, seconds),
        "cache": {"hits": cache.hits, "misses": cache.misses, "put_failures": cache.put_failures},
        "prompt_versions": prompt_versions(), "LOCAL_SCOPE": sorted(LOCAL_SCOPE),
        "route_recall": {"covered": recall_c, "total": recall_t, "missed": sorted(missed)} if have_base else None,
        "route_recall_note": None if have_base else "no baseline cache: route recall not measured",
        "submission": {"written": allowed, "reason": why, "problems": problems},
        "preds_rows": len(preds),
        "spend": None if budget is None else {"usd": _float(budget.spent_usd), "tokens_in": _int(budget.input_tokens),
                                              "tokens_out": _int(budget.output_tokens), "llm_calls": _int(budget.llm_calls)},
        "wall_seconds": round(time.perf_counter() - t_start, 3),
    }
    (out / "run.json").write_text(_dump(rj), encoding="utf-8")

    say(f"finished {len(finished)}/{len(ids)} cases; readable pages {readable}, read {rj['pages_read']}, failed {len(failed_pages)}, "
        f"unrouted {unrouted}; failures {rj['failures']}; iteration-2 cases {rj['iteration2_cases']}")
    say("local fallbacks: " + ", ".join(f"{k} {v}" for k, v in rj["local_fallbacks"].items()))
    if partial:
        say("PARTIAL run (" + ", ".join(reasons) + "): no submission.csv; the gates refuse it")
        if errors:
            say("case errors: " + ", ".join(f"{safe_id(c)} {errors[c]}" for c in sorted(errors)))
    if allowed:
        say(f"submission.csv written ({len(test_rows)} rows), NOT uploaded; self-check "
            + ("OK" if not problems else f"FAILED ({len(problems)} problems, see run.json)"))
    else:
        say(f"no submission.csv: {why}")
    say(f"output: {out}")
    return rj


# ------------------------------------------------------------------------------------------------ CLI
def parse_ablate(text: str) -> frozenset[str]:
    got = frozenset(a.strip() for a in (text or "").split(",") if a.strip())
    unknown = got - ABLATIONS
    if unknown:
        raise SystemExit(f"unknown ablation(s) {sorted(unknown)}; known: {sorted(ABLATIONS)}")
    return got


def build_agents(a, data_dir: Path):
    """The real agents (CLI only; tests replace this function): MiniMax main and extractor models, the local llama.cpp model with no
    cloud fallback, the agent cache under the data dir, the competition's rules_lite and requirements, the baseline extraction cache.
    Building them makes no model call. Returns (agents, budget, metric or None)."""
    from assessor.llm import LLM
    from assessor.tracing import Budget

    budget = Budget(max_usd=a.max_usd, max_seconds=0, max_tokens=0, max_tool_calls=0)     # wall time and tokens are not capped
    main_llm, reader = LLM(budget=budget), LLM.extractor(budget=budget)
    if main_llm.mock_mode or reader.mock_mode:
        raise SystemExit("main or extractor model is on the mock provider: the cache is keyed by model, so set the real provider "
                         "(via .env) even for --cache-only or --estimate")
    local = LLM(budget=budget, role="small", provider="llamacpp")
    local.targets = [t for t in local.targets if t.get("local")]
    if local.mock_mode or not local.targets:
        raise SystemExit("no local llama.cpp endpoint configured for the local roles")
    reqs = json.loads((data_dir / "requirements.json").read_text(encoding="utf-8"))
    rl = rk.load_module("kaggle_rules_lite", data_dir / "rules_lite.py")
    metric = rk.load_module("kaggle_metric", data_dir / "metric.py") if a.split == "test" else None
    ag = Agents(main_llm=main_llm, reader_llm=reader, local_llm=local, cache=CallCache(data_dir / "cache" / "agents"), rl=rl, reqs=reqs,
                baseline_dir=data_dir / "cache" / "extract", baseline_model=reader.model, salt=a.salt, ablate=parse_ablate(a.ablate))
    return ag, budget, metric


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vendor_verdict", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the graph over a split")
    r.add_argument("--data", default=str(ROOT / "data_kaggle"))
    r.add_argument("--split", choices=["train", "test"], required=True)
    r.add_argument("--subset", choices=["all", "dev", "holdout"], required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--only", default="", help="comma-separated case ids")
    r.add_argument("--cache-only", action="store_true", help="never call a model: a cache miss is a failed call")
    r.add_argument("--estimate", action="store_true", help="print calls and tokens per role, then exit before any model call")
    r.add_argument("--salt", default="", help="an independent sample of the same inputs; never a submission")
    r.add_argument("--engine", choices=["plain", "langgraph"], default="plain")
    r.add_argument("--ablate", default="", help="comma-separated: " + ",".join(sorted(ABLATIONS)))
    r.add_argument("--workers", type=int, default=6, help="cases in parallel (nodes stay sequential inside a case)")
    r.add_argument("--max-usd", type=float, default=5.0, help="hard cap on spend at the configured prices (a run that hits it is partial)")
    r.add_argument("--out", default="")
    a = ap.parse_args(argv)

    data_dir = Path(a.data)
    ablate = parse_ablate(a.ablate)
    only = [c for c in a.only.split(",") if c] or None
    ag, budget, metric = build_agents(a, data_dir)
    print(f"models: main {ag.main_llm.model}, reader {ag.reader_llm.model}, local {ag.local_llm.model}; baseline {ag.baseline_model}")
    if a.estimate:
        cases, pages, ids = load_split(data_dir, a.split, a.subset, a.limit or None, only)
        for line in format_estimate(estimate(cases, pages, sorted(ids), ag)):
            print(line)
        return 0
    out = Path(a.out) if a.out else default_out(a.split, a.subset, ablate, a.salt)
    rj = run(data_dir, a.split, a.subset, agents=ag, out=out, limit=a.limit or None, only=only, cache_only=a.cache_only,
             engine=a.engine, workers=a.workers, metric=metric, budget=budget)
    if rj["stopped"] == "interrupted":
        return 130
    return 1 if rj["submission"]["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
