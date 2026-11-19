"""The Plan Reviewer: combines the planner's routes and the router's tags into the final family -> page ids routes, deterministically
(no LLM), and measures how much of what the baseline read would have been routed.

Rules, in this order:
  1. Only readable pages (`run_kaggle.readable`) are routed: a quarantined or ignored page, or an id that is not a page of this case,
     is dropped.
  2. A page's routes are the union of the planner's and the router's.
  3. A registry page always goes to `certifications` and only there (a registry speaks only about certificates, as in
     `kaggle_policy.check_holders`).
  4. A readable official page that has no route at all goes to every family (recall first).

Every change a rule makes is described in `fixes`. Fixes carry page ids (only id-shaped values) and family names from the fixed list,
never page text, titles, URLs or model output.
"""
from __future__ import annotations

from eval import run_kaggle as rk

from ..config import FAMILIES, FIELD_FAMILY
from ..safe import family_name, safe_id


def _names(families) -> str:
    return ", ".join(f for f in FAMILIES if f in families)


def review(plan: dict[str, list[str]], router_tags: dict[str, list[str]], recs: list[dict]) -> tuple[dict[str, list[str]], list[str]]:
    """(family -> page ids, fixes). Every family key is present (possibly an empty list); ids follow the order of `recs`."""
    fixes: list[str] = []
    recs_by_id: dict[str, dict] = {}
    for r in recs:
        recs_by_id.setdefault(str(r["page"]["page_id"]), r)
    order = {pid: i for i, pid in enumerate(recs_by_id)}

    proposed: dict[str, set[str]] = {}                  # page id -> families the planner or the router named (rule 2)
    for fam, ids in plan.items():
        if fam not in FAMILIES:
            fixes.append(f"dropped {family_name(fam)} from the planner's routes")
            continue
        for pid in ids or []:
            proposed.setdefault(str(pid), set()).add(fam)
    for pid, fams in router_tags.items():
        for fam in fams or []:
            if isinstance(fam, str) and fam in FAMILIES:
                proposed.setdefault(str(pid), set()).add(fam)
            else:
                fixes.append(f"{safe_id(pid)}: dropped {family_name(fam)} from the router's tags")
        proposed.setdefault(str(pid), set())

    chosen: dict[str, set[str]] = {}
    for pid in sorted(proposed, key=lambda p: (order.get(p, len(order)), p)):
        fams = proposed[pid]
        if pid not in recs_by_id:                       # rule 1
            if fams:
                fixes.append(f"{safe_id(pid)}: not a page of this case, removed from {_names(fams)}")
        elif not rk.readable(recs_by_id[pid]):
            if fams:
                fixes.append(f"{safe_id(pid)}: not readable, removed from {_names(fams)}")
        else:
            chosen[pid] = set(fams)

    for pid, rec in recs_by_id.items():
        if not rk.readable(rec):
            continue
        fams = chosen.setdefault(pid, set())
        if rec["role"] == "registry":                   # rule 3
            removed, added = fams - {"certifications"}, "certifications" not in fams
            if removed or added:
                parts = ([f"removed from {_names(removed)}"] if removed else []) + (["added to certifications"] if added else [])
                fixes.append(f"{safe_id(pid)}: registry page, kept to certifications only ({'; '.join(parts)})")
            chosen[pid] = {"certifications"}
        elif rec["role"] == "official" and not fams:    # rule 4
            fixes.append(f"{safe_id(pid)}: official page with no route, sent to every family")
            chosen[pid] = set(FAMILIES)

    routes = {f: [pid for pid in recs_by_id if f in chosen.get(pid, ())] for f in FAMILIES}
    return routes, fixes


def route_recall(routes: dict[str, list[str]], baseline_fields: dict[str, set[str]]) -> tuple[int, int, list[str]]:
    """The R-route measurement: of the (page, field) pairs the baseline read a value from, how many did the routes send to the field's
    family? Returns (covered, total, missed "page_id:field" ids, sorted). A field outside the 8 has no family, so it can never be
    covered: it is counted as missed, never left out of the total."""
    routed = {fam: set(ids) for fam, ids in routes.items()}
    covered, total, missed = 0, 0, []
    for pid, fields in baseline_fields.items():
        for fld in fields:
            total += 1
            fam = FIELD_FAMILY.get(fld)
            if fam is not None and pid in routed.get(fam, ()):
                covered += 1
            else:
                missed.append(f"{pid}:{fld}")
    return covered, total, sorted(missed)
