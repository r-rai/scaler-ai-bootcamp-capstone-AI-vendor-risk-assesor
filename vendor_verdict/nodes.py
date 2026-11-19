"""The agent nodes of the per-case graph. Each node is `(state: CaseState, ag: Agents) -> dict`, a partial state update that both
engines (vendor_verdict.graph) apply by REPLACING keys: an accumulating key (readings, notes, flags, grades) is always returned whole,
never as a delta, so the plain and the LangGraph engine give the same state.

Execution order (fixed and sequential within a case, so a cache-only rerun is byte-identical; families in `config.FAMILIES` order,
pages in `recs` order, fields in config order):
    validate_sources -> plan -> route_pages -> review_plan -> specialists -> grade_evidence -> evaluate
      -> (continue: specialists | stop: decide_conflicts) -> merge -> verdict -> audit

Models: `plan` uses `ag.main_llm`; the specialists use `ag.reader_llm`; the router, grader and decision maker use `ag.local_llm`, which
the caller must make local-only (no cloud fallback: the runner's job).

Failure handling: a failed local or MiniMax call never stops a case. It is counted in `flags`:
    planner_failures, router_fallbacks (page tagged with every family), specialist_failures (failed calls; the failed page ids are in
    `notes` as "specialists: failed page <id>" and in the trace), grader_failures (graded "unsure"), decision_failures (abstained).
`BudgetExceeded` always propagates.

Trace: every node calls `state["trace"].handover` exactly once per execution, with counts, ids, field and family names and verdict labels
only: never a quote, a value, display text, a title, a URL, page text, model output or an exception message. Notes follow the same rule
(they hold the id-only notes of the agent modules, prefixed with the node name). No node writes a file.
"""
from __future__ import annotations

import dataclasses
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from eval import run_kaggle as rk

from . import adapter
from . import baseline as baseline_mod
from .agents import decision_maker, grader, planner, reviewer, router
from .agents import specialists as specialist_agents   # the node below is named `specialists`
from .cache import CallCache
from .config import FAMILIES, MAX_ITERATIONS
from .safe import safe_id
from .state import CaseState

ABLATIONS: frozenset[str] = frozenset({"router", "grader", "decision_maker", "iteration2"})
FLAG_KEYS = ("planner_failures", "router_fallbacks", "specialist_failures", "grader_failures", "decision_failures")
FAILED_PAGE_NOTE = "specialists: failed page "        # + a page id: the runner reads the failed pages from these notes


@dataclasses.dataclass(frozen=True)
class Agents:
    main_llm: Any
    reader_llm: Any
    local_llm: Any
    cache: CallCache
    rl: Any                                   # the rules_lite module (or anything with its evaluate / overall)
    reqs: list[dict]
    baseline_dir: Path | None = None          # None: no baseline readings (no cache read at all)
    baseline_model: str = ""
    salt: str = ""
    ablate: frozenset[str] = frozenset()

    def __post_init__(self):
        unknown = set(self.ablate) - ABLATIONS
        if unknown:
            raise ValueError(f"unknown ablation(s) {sorted(unknown)}; known: {sorted(ABLATIONS)}")
        object.__setattr__(self, "ablate", frozenset(self.ablate))


# ------------------------------------------------------------------------------------------------ helpers
def _handover(state: CaseState, agent: str, to: str, t0: float, **summary) -> None:
    state["trace"].handover(agent, to, summary, seconds=time.perf_counter() - t0)


def _readable_pages(state: CaseState) -> dict[str, dict]:
    """page id -> page, readable pages only, in `recs` order (the first page of a duplicated id wins, as in the reviewer)."""
    out: dict[str, dict] = {}
    for rec in state.get("recs", []):
        if rk.readable(rec):
            out.setdefault(str(rec["page"]["page_id"]), rec["page"])
    return out


def _flags(state: CaseState, **add: int) -> dict[str, int]:
    flags = dict(state.get("flags") or {})
    for k, v in add.items():
        flags[k] = flags.get(k, 0) + v
    return flags


def _notes(state: CaseState, node: str, more: list[str]) -> list[str]:
    return list(state.get("notes") or []) + [f"{node}: {n}" for n in more]


class DuplicateReadingId(AssertionError):
    """Two readings share an id: the grades (keyed by reading id) would silently overwrite each other. A bug, never a page's fault."""


def assert_unique_ids(new: list[dict], taken: set[str]) -> None:
    """The readings of one call must have ids that are unique among themselves and new to `taken`. The specialists mint ids that are
    unique by construction (one call per (iteration, family, page) in iteration 1, one per (iteration, family) in iteration 2, and `n`
    numbers a call's readings per (page, field)); this check makes a break of that rule fail loudly instead of losing a grade. It is a
    real check, not an `assert` statement, so `python -O` cannot strip it. The message carries counts only."""
    ids = [r["id"] for r in new]
    clashes = len(ids) - len(set(ids)) + len(set(ids) & taken)
    if clashes:
        raise DuplicateReadingId(f"{clashes} duplicate reading id(s) in one specialist call")


# ------------------------------------------------------------------------------------------------ nodes
def validate_sources(state: CaseState, ag: Agents) -> dict:
    """Source Validator: `run_kaggle.triage` (roles and the injection quarantine), the cache-only baseline readings, and the optional
    local injection second opinion (off by default; a count in the trace only, it never changes `recs`)."""
    t0 = time.perf_counter()
    recs = rk.triage(state["case"], state["pages"])
    base = baseline_mod.baseline_readings(ag.baseline_model, recs, ag.baseline_dir) if ag.baseline_dir is not None else {}
    second = router.injection_second_opinion(ag.local_llm, recs)
    roles = Counter(r["role"] for r in recs)
    _handover(state, "validate_sources", "plan", t0, pages=len(recs), readable=sum(rk.readable(r) for r in recs),
              official=roles["official"], registry=roles["registry"], ignored=roles["ignored"],
              quarantined=sum(1 for r in recs if r["quarantine"]), baseline_pages=len(base), injection_second_opinion=second)
    return {"recs": recs, "baseline": base, "flags": {**{k: 0 for k in FLAG_KEYS}, **(state.get("flags") or {})}}


def plan(state: CaseState, ag: Agents) -> dict:
    """Planner (main model, page metadata only): family -> page ids."""
    t0 = time.perf_counter()
    routes, notes = planner.plan_routes(ag.main_llm, ag.cache, state["case"], state["recs"], ag.salt)
    failed = int("planner failed" in notes)
    _handover(state, "plan", "route_pages", t0, families={f: len(ids) for f, ids in routes.items()}, failed=failed,
              notes=len(notes))
    return {"plan": routes, "notes": _notes(state, "plan", notes), "flags": _flags(state, planner_failures=failed)}


def route_pages(state: CaseState, ag: Agents) -> dict:
    """Page Router (local model, page text): page id -> families. Ablated: every readable page gets every family, no call."""
    t0 = time.perf_counter()
    notes: list[str] = []
    if "router" in ag.ablate:
        tags, failures = {pid: list(FAMILIES) for pid in _readable_pages(state)}, 0
    else:
        tags, failures = router.tag_pages(ag.local_llm, ag.cache, state["recs"], notes)
    _handover(state, "route_pages", "review_plan", t0, pages=len(tags), tags=sum(len(f) for f in tags.values()),
              fallbacks=failures, ablated="router" in ag.ablate)
    return {"router_tags": tags, "notes": _notes(state, "route_pages", notes), "flags": _flags(state, router_fallbacks=failures)}


def review_plan(state: CaseState, ag: Agents) -> dict:
    """Plan Reviewer (deterministic): the union of the planner's and the router's routes, with its fixes; the route recall against
    the baseline's (page, field) pairs goes to the trace."""
    t0 = time.perf_counter()
    routes, fixes = reviewer.review(state.get("plan") or {}, state.get("router_tags") or {}, state["recs"])
    covered, total, missed = reviewer.route_recall(routes, baseline_mod.route_baseline_fields(state.get("baseline") or {}, state["recs"]))
    _handover(state, "review_plan", "specialists", t0, routes={f: len(ids) for f, ids in routes.items()}, fixes=len(fixes),
              recall_covered=covered, recall_total=total, recall_missed=missed)
    return {"routes": routes, "review_fixes": fixes}


def specialists(state: CaseState, ag: Agents) -> dict:
    """Field Specialists (reader model). Iteration 1: `read_page` once per (family, routed page). Iteration 2: `read_case` once per
    family over its routed pages for the target fields of the family (a family with no target field or no page is skipped)."""
    t0 = time.perf_counter()
    iteration = state["iteration"]
    pages = _readable_pages(state)
    routes = state.get("routes") or {}
    targets = set(state.get("targets") or [])
    readings = list(state.get("readings") or [])
    taken = {r["id"] for r in readings}
    notes: list[str] = []
    calls = failures = new_count = 0
    failed_pages: list[str] = []

    def take(new: list[dict], call_notes: list[str], pids: list[str], family: str) -> None:
        nonlocal calls, failures, new_count
        calls += 1
        assert_unique_ids(new, taken)
        taken.update(r["id"] for r in new)
        readings.extend(new)
        new_count += len(new)
        if any(n.startswith("failed") for n in call_notes):
            failures += 1
            failed_pages.extend(p for p in pids if p not in failed_pages)
        notes.extend(f"{family}: {safe_id(pids[0]) if len(pids) == 1 else 'case read'}: {n}" for n in call_notes)

    for family, fields in FAMILIES.items():
        fam_pages = [pid for pid in routes.get(family, []) if pid in pages]
        if iteration == 1:
            for pid in fam_pages:
                new, call_notes = specialist_agents.read_page(ag.reader_llm, ag.cache, family, pages[pid], iteration=1, salt=ag.salt)
                take(new, call_notes, [pid], family)
        else:
            want = [f for f in fields if f in targets]
            if not want or not fam_pages:
                continue
            new, call_notes = specialist_agents.read_case(ag.reader_llm, ag.cache, family, [pages[p] for p in fam_pages], want, ag.salt)
            take(new, call_notes, fam_pages, family)

    order = list(pages)
    failed_pages.sort(key=lambda p: order.index(p) if p in order else len(order))
    notes += [f"failed page {safe_id(p)}" for p in failed_pages]
    _handover(state, "specialists", "grade_evidence", t0, iteration=iteration, calls=calls, readings=new_count,
              failures=failures, failed_pages=[safe_id(p) for p in failed_pages], targets=sorted(targets) if iteration > 1 else [])
    return {"readings": readings, "notes": _notes(state, "specialists", notes),
            "flags": _flags(state, specialist_failures=failures)}


def grade_evidence(state: CaseState, ag: Agents) -> dict:
    """Evidence Grader (local model) over the readings not graded yet. Ablated ("B without the Evidence Grader"): no reading gets a
    grade and no call is made, so no field becomes a target through a grade and the Decision Maker sees every candidate as "unsure"
    (no information), never as a "supported" the grader did not give. The grounding check is part of the grader, so it is off too."""
    t0 = time.perf_counter()
    grades = dict(state.get("grades") or {})
    new = [r for r in state.get("readings") or [] if r["id"] not in grades]
    notes: list[str] = []
    if "grader" in ag.ablate:
        got, failures = {}, 0
    else:
        got, failures = grader.grade_all(ag.local_llm, ag.cache, new, _readable_pages(state), notes)
    grades.update(got)
    _handover(state, "grade_evidence", "evaluate", t0, iteration=state["iteration"], graded=len(got),
              grades=dict(sorted(Counter(got.values()).items())), failures=failures, ablated="grader" in ag.ablate)
    return {"grades": grades, "notes": _notes(state, "grade_evidence", notes), "flags": _flags(state, grader_failures=failures)}


def _targets(state: CaseState) -> list[str]:
    """Fields worth a targeted re-read: any reading of the field graded unsupported or unsure, or any (page, field) whose iteration-1
    value differs from the baseline's. The difference is `collect_conflicts` over the iteration-1 readings and the baseline, which
    compares the two reading-sets with the signature `decision_maker.value_key` is built on, exactly as `decide_conflicts` will; a
    (page, field) only one side has is never a conflict, so never a difference."""
    grades = state.get("grades") or {}
    readings = state.get("readings") or []
    fields = {r["field"] for r in readings if grades.get(r["id"]) in ("unsupported", "unsure")}
    first = [r for r in readings if r.get("iteration") == 1]
    fields |= {c["field"] for c in decision_maker.collect_conflicts(first, state.get("baseline") or {}, {})}
    return sorted(fields)


def evaluate(state: CaseState, ag: Agents) -> dict:
    """Evaluator: continue (iteration + 1, `targets` = the fields to re-read) while iteration < MAX_ITERATIONS, iteration 2 is not
    ablated and there is a target; otherwise stop with `targets` = [] (nothing pending). The graph continues iff `targets` is set."""
    t0 = time.perf_counter()
    iteration = state["iteration"]
    found = _targets(state)
    go = iteration < MAX_ITERATIONS and "iteration2" not in ag.ablate and bool(found)
    _handover(state, "evaluate", "specialists" if go else "decide_conflicts", t0, iteration=iteration, targets=found,
              decision="continue" if go else "stop")
    return {"iteration": iteration + 1, "targets": found} if go else {"targets": []}


def next_after_evaluate(state: CaseState) -> str:
    return "specialists" if state.get("targets") else "decide_conflicts"


def decide_conflicts(state: CaseState, ag: Agents) -> dict:
    """Decision Maker (local model): for each (page, field) where iteration 1, iteration 2 and the baseline disagree, choose one
    candidate or abstain, seeing the page. Ablated: every decision is None (abstain onto the default) and no call is made."""
    t0 = time.perf_counter()
    pages = _readable_pages(state)
    conflicts = decision_maker.collect_conflicts(state.get("readings") or [], state.get("baseline") or {}, state.get("grades") or {})
    decisions: dict[str, int | None] = {}
    notes: list[str] = []
    for c in conflicts:
        if "decision_maker" in ag.ablate:
            decisions[c["id"]] = None
            continue
        decisions[c["id"]] = decision_maker.decide(ag.local_llm, ag.cache, c, pages.get(c["page_id"]), notes)
    failures = len(notes)                        # decide adds a note only for a failed call or an invalid index (both abstain)
    _handover(state, "decide_conflicts", "merge", t0, conflicts=len(conflicts),
              chosen=sum(1 for d in decisions.values() if d is not None), abstained=sum(1 for d in decisions.values() if d is None),
              failures=failures, ablated="decision_maker" in ag.ablate)
    return {"conflicts": conflicts, "decisions": decisions, "notes": _notes(state, "decide_conflicts", notes),
            "flags": _flags(state, decision_failures=failures)}


def merge(state: CaseState, ag: Agents) -> dict:
    """The readings that stand (`decision_maker.apply`), then the baseline's merge through the adapter into rules_lite facts."""
    t0 = time.perf_counter()
    chosen = decision_maker.apply(state.get("readings") or [], state.get("conflicts") or [], state.get("decisions") or {})
    facts, notes, same_date = adapter.facts_for_case(state["case"], state["recs"], chosen)
    _handover(state, "merge", "verdict", t0, groups=len(chosen), facts=sorted(facts), notes=len(notes), same_date_conflicts=len(same_date))
    return {"facts": facts, "notes": _notes(state, "merge", notes + same_date)}


def verdict(state: CaseState, ag: Agents) -> dict:
    """rules_lite on the merged facts, called exactly as `run_kaggle.assemble_case` calls it."""
    t0 = time.perf_counter()
    case = state["case"]
    v = ag.rl.evaluate(state["facts"], ag.reqs, case["tier"], int(case["users"]), date.fromisoformat(case["as_of"]))
    overall = ag.rl.overall(v, ag.reqs)
    _handover(state, "verdict", "audit", t0, requirements=len(v), verdicts=dict(sorted(Counter(v.values()).items())), overall=overall)
    return {"verdicts": v, "overall": overall}


def audit(state: CaseState, ag: Agents) -> dict:
    """Finalises the trace with the run's counts. No file IO (writing artefacts is the runner's job)."""
    t0 = time.perf_counter()
    readings = state.get("readings") or []
    _handover(state, "audit", "end", t0, iterations=state["iteration"], readings=len(readings),
              by_iteration=dict(sorted(Counter(str(r["iteration"]) for r in readings).items())),
              grades=dict(sorted(Counter((state.get("grades") or {}).values()).items())), conflicts=len(state.get("conflicts") or []),
              decided=sum(1 for d in (state.get("decisions") or {}).values() if d is not None),
              flags=dict(state.get("flags") or {}), notes=len(state.get("notes") or []))
    return {}


NODES = {"validate_sources": validate_sources, "plan": plan, "route_pages": route_pages, "review_plan": review_plan,
         "specialists": specialists, "grade_evidence": grade_evidence, "evaluate": evaluate, "decide_conflicts": decide_conflicts,
         "merge": merge, "verdict": verdict, "audit": audit}
