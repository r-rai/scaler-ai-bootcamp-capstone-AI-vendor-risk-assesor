"""The `--estimate` report of `python -m vendor_verdict run` (moved out of runner.py unchanged, ledger ruling R40, so the runner stays
under its size limit): calls and tokens per role before a run, with no model call and no file write. runner.py re-exports
`estimate` and `format_estimate`."""
from __future__ import annotations

from pathlib import Path

from assessor.tracing import estimate_tokens
from eval import run_kaggle as rk

from . import baseline as baseline_mod
from .agents import planner as planner_agent
from .agents import specialists as specialist_agents
from .config import FAMILIES, GRADER_FIELDS, LOCAL_SCOPE
from .nodes import Agents, _readable_pages
from .prompts import PLANNER_SYSTEM, ROUTER_SYSTEM, SPECIALIST_SYSTEM, page_user_prompt, version


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
    lines.append("  quota window [paid]: its size is unknown to the runner; the user reports that earlier passes used under half of "
                 "a 5-hour window (their statement, unmeasured); a quota stop is resumable (the same command again; finished calls "
                 "are cached)")
    return lines

