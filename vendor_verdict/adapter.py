"""The adapter from specialist readings onto the existing evidence merge (eval/kaggle_policy): the graph decides WHICH readings
stand, and the submitted baseline's normalise -> holder check -> newest-page-wins merge turns them into rules_lite facts unchanged.

Representation: one reading is one `Claim`. A list-valued field (security_certifications, subprocessors, security_incidents) is read
one item per reading; its Claim carries that single item and `kp.normalise_page_claims` wraps it into a one-item list, the same
statement the baseline makes with one list. Downstream already combines several claims of one field on one page (`_merge_certs`
and `_merge_incidents` loop over every claim, `build_facts.page_value` unions sub-processors by name, `check_holders` filters each
claim), so the facts are identical to the baseline's (except for the silence guards below); the only other difference is that a
malformed item is dropped alone instead of taking its page's whole list with it. A reading whose value is already a list (e.g. [] for "no incidents") passes through as is.

`evidence_level` is not derived here: `kp.build_facts` sets it on the merged certificates (`kp.is_documented`: a certified
"SOC 2 Type II" or "ISO 27001" with issuer and date) and rules_lite reads it only from that fact; `Claim.evidence_level` is not read.

A Claim's display is the reading's `display` (the specialist's own short statement), else its quote, clipped to 300 characters like
`claims.normalise_claim` clips it: the absence-statement guard there reads it and drops `sso_saml="none"` / `dpa_available="no"`
read from silence.

Silence guards (GATE-P): two PASS-making values have no such guard downstream, so the adapter applies the same published rule (rule 3:
a field the sources do not state is unknown) with the same test (`claims.is_absence_statement` on the same display) after
normalisation: `trains_on_customer_data="no"` and a `security_incidents` value of [] whose display (else quote) reports that something
is not mentioned are dropped. Only these exact values are guarded: a real incident item is never dropped by its display. This is the
one place where the facts may differ from the baseline path's, on purpose (a reading resting on silence is UNKNOWN, never a PASS).

Notes are sanitised: kaggle_policy's own notes can quote a page value or a holder name, so only the page id, the reading id (when it
is id-shaped), the field name and a fixed reason word reach the caller, never a quote, a value or a display (no page text may reach
notes, logs or traces). Reason words, for the Evaluator and the Auditor to count: `invalid_value`, `absence_statement`,
`derived_region`, `other_page`, `registry_no_holder`, `registry_other_holder` (per reading), `incident_without_severity` (per field:
the merge leaves the field unknown) and `page_not_in_case` (one count note per case for readings citing a page outside `recs`).
The reasons are found on deep copies by the same kaggle_policy functions; the facts come from the real calls, unchanged.
"""
from __future__ import annotations

import copy
import re
from datetime import date

from pydantic import ValidationError

from assessor.claims import is_absence_statement
from assessor.schemas import Claim
from eval import kaggle_policy as kp
from eval import run_kaggle as rk

from .config import FIELD_FAMILY

DISPLAY_CLIP = 300            # claims.normalise_claim clips display to 300 characters: the guard must see the same text
_ECHO_READING_ID = re.compile(r"[A-Za-z0-9_.:-]{1,120}")      # reading_id() joins validated parts; anything else could be page text
_ABSENCE_REASON = "absence of a mention"                       # the start of claims.normalise_claim's absence-guard reason


def _field_of(note: str) -> str:
    head = note.split(":", 1)[0].strip()
    return head if head in FIELD_FAMILY else "a field"


def _rid(reading: dict) -> str:
    text = str(reading.get("id"))
    return text if _ECHO_READING_ID.fullmatch(text) else f"<{len(text)} chars>"


def _from_silence(c: Claim) -> bool:
    """A normalised PASS-making value with no downstream absence guard, whose display reports a missing mention (rule 3)."""
    negative = (c.field == "trains_on_customer_data" and c.value == "no") or (c.field == "security_incidents" and c.value == [])
    return negative and is_absence_statement(c.display)


def _normalise_reason(probe: Claim) -> str | None:
    """Why kp.normalise_page_claims drops this claim on its own (None: it survives alone). Run on a deep copy."""
    kept, why = kp.normalise_page_claims([probe])
    if kept:
        return None
    return "absence_statement" if any(_ABSENCE_REASON in w for w in why) else "invalid_value"


def to_page_claims(readings: list[dict], rec: dict, legal_entity: str) -> tuple[kp.PageClaims, list[str]]:
    """One page's chosen readings -> its PageClaims after normalisation, the silence guards and the registry holder check, plus one
    id-only note per dropped reading (page id, reading id, field, reason word)."""
    page_id = str(rec["page"]["page_id"])
    pairs: list[tuple[Claim, dict]] = []
    notes = []
    for r in readings:
        field = _field_of(str(r.get("field")) + ":")
        if str(r.get("page_id")) != page_id:
            notes.append(f"{page_id}: {_rid(r)}: dropped {field}: other_page")
            continue
        value = copy.deepcopy(r["value"])            # normalise and the holder check edit values in place; readings stay as read
        try:
            # display = the model's own wording (else the quote), so the absence guards can drop a value read from silence
            # ("SSO is not mentioned"), exactly as claims.normalise_claim does on the baseline path
            display = str(r.get("display") or r.get("quote") or "")[:DISPLAY_CLIP]
            pairs.append((Claim(field=r["field"], value=value, display=display, quote=r["quote"]), r))
        except ValidationError:
            notes.append(f"{page_id}: {_rid(r)}: dropped {field}: invalid_value")
    claims = [c for c, _ in pairs]
    probes = [c.model_copy(deep=True) for c in claims]   # the reasons are found on copies; the facts come from the real call below
    kept, _why = kp.normalise_page_claims(claims)
    survivors = {id(c) for c in kept}
    guarded = {id(c) for c in kept if _from_silence(c)}
    kept = [c for c in kept if id(c) not in guarded]
    pc = kp.PageClaims(page_id, rec["role"], kp.page_date(rec["page"]), kept)
    for (c, r), probe in zip(pairs, probes):
        field = _field_of(c.field + ":")
        if id(c) in guarded:
            notes.append(f"{page_id}: {_rid(r)}: dropped {field}: absence_statement")
        elif id(c) not in survivors:
            # dropped on its own by normalise, else only next to the page's other claims: the sub-processor-derived region rule
            notes.append(f"{page_id}: {_rid(r)}: dropped {field}: {_normalise_reason(probe) or 'derived_region'}")
        elif rec["role"] == "registry" and c.field == "security_certifications":
            one = kp.PageClaims(page_id, "registry", pc.date, [c.model_copy(deep=True)])
            for n in kp.check_holders(one, legal_entity):
                notes.append(f"{page_id}: {_rid(r)}: dropped {field}: " + ("registry_no_holder (registry entry names no holder)"
                             if "names no holder" in n else
                             "registry_other_holder (certificate held by another entity than the legal entity)"))
    kp.check_holders(pc, legal_entity)       # the real filter; its notes (which can name a holder) are replaced by the ones above
    return pc, notes


def _merge_note(note: str) -> str:
    reason = ": incident_without_severity" if "no stated severity" in note else ""
    return f"{_field_of(note)}: left unknown by the merge{reason}"


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
    in_case = {str(rec["page"]["page_id"]) for rec in recs}
    outside = sum(len(rs) for pid, rs in by_page.items() if pid not in in_case)
    if outside:
        notes.append(f"{outside} reading(s) cite a page that is not a page of this case: not merged (page_not_in_case)")
    merged = kp.build_facts(items, as_of)
    notes += [_merge_note(n) for n in merged.notes]
    conflicts = [f"{_field_of(c)}: same-date pages disagree" for c in merged.conflicts]
    return rk.strip_internal(merged.facts), notes, conflicts
