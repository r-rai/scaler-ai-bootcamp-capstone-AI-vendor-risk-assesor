"""The adapter from specialist readings onto the existing evidence merge (eval/kaggle_policy): the graph decides WHICH readings
stand, and the submitted baseline's normalise -> holder check -> newest-page-wins merge turns them into rules_lite facts unchanged.

Representation: one reading is one `Claim`. A list-valued field (security_certifications, subprocessors, security_incidents) is read
one item per reading; its Claim carries that single item and `kp.normalise_page_claims` wraps it into a one-item list, the same
statement the baseline makes with one list. Downstream already combines several claims of one field on one page (`_merge_certs`
and `_merge_incidents` loop over every claim, `build_facts.page_value` unions sub-processors by name, `check_holders` filters each
claim), so the facts are identical to the baseline's; the only difference is that a malformed item is dropped alone instead of
taking its page's whole list with it. A reading whose value is already a list (e.g. [] for "no incidents") passes through as is.

`evidence_level` is not derived here: `kp.build_facts` sets it on the merged certificates (`kp.is_documented`: a certified
"SOC 2 Type II" or "ISO 27001" with issuer and date) and rules_lite reads it only from that fact; `Claim.evidence_level` is not read.

Notes are sanitised: kaggle_policy's own notes can quote a page value or a holder name, so only the page id, the field name and a
fixed reason reach the caller (no page text may reach notes, logs or traces).
"""
from __future__ import annotations

import copy
from datetime import date

from pydantic import ValidationError

from assessor.schemas import Claim
from eval import kaggle_policy as kp
from eval import run_kaggle as rk

from .config import FIELD_FAMILY


def _field_of(note: str) -> str:
    head = note.split(":", 1)[0].strip()
    return head if head in FIELD_FAMILY else "a field"


def _holder_note(page_id: str, note: str) -> str:
    if "names no holder" in note:
        return f"{page_id}: registry entry names no holder"
    return f"{page_id}: certificate held by another entity than the legal entity"


def to_page_claims(readings: list[dict], rec: dict, legal_entity: str) -> tuple[kp.PageClaims, list[str]]:
    """One page's chosen readings -> its PageClaims after normalisation and the registry holder check, plus the notes of both."""
    page_id = str(rec["page"]["page_id"])
    claims, notes = [], []
    for r in readings:
        if str(r.get("page_id")) != page_id:
            notes.append(f"{page_id}: dropped a reading of another page ({_field_of(str(r.get('field')))})")
            continue
        value = copy.deepcopy(r["value"])            # normalise and the holder check edit values in place; readings stay as read
        try:
            claims.append(Claim(field=r["field"], value=value, display=str(value)[:120], quote=r["quote"]))
        except ValidationError:
            notes.append(f"{page_id}: dropped {_field_of(str(r.get('field')) + ':')}: not a fact field")
    kept, why = kp.normalise_page_claims(claims)
    notes += [f"{page_id}: dropped {_field_of(w)} (normalise)" for w in why]
    pc = kp.PageClaims(page_id, rec["role"], kp.page_date(rec["page"]), kept)
    notes += [_holder_note(page_id, n) for n in kp.check_holders(pc, legal_entity)]
    return pc, notes


def facts_for_case(case: dict, recs: list[dict], chosen: dict[tuple[str, str], list[dict]]) -> tuple[dict, list[str], list[str]]:
    """(facts as rules_lite gets them, notes, conflicts). Mirrors run_kaggle.assemble_case: readable pages in `recs` order,
    `kp.build_facts` with its default strategies, then `run_kaggle.strip_internal`."""
    as_of = date.fromisoformat(case["as_of"])
    by_page: dict[str, list[dict]] = {}
    for (page_id, _field), readings in chosen.items():
        by_page.setdefault(str(page_id), []).extend(readings)
    items, notes = [], []
    for rec in recs:
        if not rk.readable(rec):
            continue
        pc, more = to_page_claims(by_page.get(str(rec["page"]["page_id"]), []), rec, case["legal_entity"])
        notes += more
        items.append(pc)
    merged = kp.build_facts(items, as_of)
    notes += [f"{_field_of(n)}: left unknown by the merge" for n in merged.notes]
    conflicts = [f"{_field_of(c)}: same-date pages disagree" for c in merged.conflicts]
    return rk.strip_internal(merged.facts), notes, conflicts
