"""The four Field Specialists: one LLM call reads one page (iteration 1) or several pages (iteration 2, a targeted re-read) for the
fields of one family and returns reading dicts (see vendor_verdict.state).

Trust rules:
- The model's reply is parsed leniently (`SpecialistRaw`) and every item is validated on its own, so one malformed reading never
  loses the page's good ones.
- A reading's page id is never taken on trust: a single-page read forces the page's own id; a case read keeps only ids of the
  pages it was given (a `page_id:` line could also appear inside page text).
- Notes carry field names, page ids and exception class names only, never a quote, a value or page text.
"""
from __future__ import annotations

import json
import re
import time

from pydantic import BaseModel, ValidationError

from assessor.llm import LLMError
from assessor.tracing import BudgetExceeded, estimate_tokens

from ..cache import CallCache
from ..config import FAMILIES, FIELD_FAMILY
from ..prompts import SPECIALIST_SYSTEM, page_user_prompt, version
from ..schemas import FieldReading, SpecialistRaw
from ..state import reading_id

_sleep = time.sleep          # module-level so a test can stand in for the rate-limit wait
_ATTEMPTS = 5


def payload(user: str, family: str) -> str:
    """The cache payload of one call: the user message plus the family."""
    return f"{user}\x00family:{family}"


def _safe_id(page_id) -> str:
    """A page id fit for a note: a model-written id could carry page text, so only id characters survive, truncated."""
    return re.sub(r"[^A-Za-z0-9_.:-]", "?", str(page_id))[:40]


def _field_name(field) -> str:
    return field if field in FIELD_FAMILY else "an unknown field"


def _call(llm, cache: CallCache, family: str, user: str, salt: str) -> tuple[list | None, list[str]]:
    """(raw reading items, notes). None = the call failed (never cached). A cache hit makes no call."""
    system = SPECIALIST_SYSTEM[family]
    key = cache.key(f"specialist:{family}", version(system), llm.model, payload(user, family), salt)
    rec = cache.get(key)
    if rec is not None and isinstance(rec.get("readings"), list):
        return rec["readings"], []
    res, err = None, None
    for attempt in range(_ATTEMPTS):
        try:
            res = llm.structured(system, user, SpecialistRaw, stage=f"specialist:{family}")
            break
        except BudgetExceeded:
            raise
        except LLMError as exc:
            err = exc
            msg = str(exc)
            if "429" not in msg and "rate limit" not in msg.lower():
                break                                   # only a rate limit is worth waiting for (Extractor.run pattern)
            if attempt < _ATTEMPTS - 1:
                _sleep(min(60, 4 * 3 ** attempt))       # 4, 12, 36, 60 s
    if res is None:
        return None, [f"failed: {type(err).__name__}"]
    raw = [r.model_dump(mode="json") if isinstance(r, BaseModel) else r for r in res.readings]
    raw = json.loads(json.dumps(raw, default=str))      # JSON-safe and detached from the model object
    # ~4 chars/token estimates, as run_kaggle.Extractor.run records them, so a cache-only run can still report tokens
    cache.put(key, {"model": llm.model, "salt": salt, "readings": raw,
                    "tokens_in": estimate_tokens(system + user), "tokens_out": estimate_tokens(json.dumps(raw))})
    return raw, []


def _validate(raw: list, family: str, iteration: int, allowed_ids: set[str], fields: set[str],
              forced_id: str | None) -> tuple[list[dict], list[str]]:
    readings, notes, counts = [], [], {}
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            notes.append(f"dropped reading {i}: not an object")
            continue
        item = dict(item)
        if item.get("page_id") is None:
            item["page_id"] = ""
        if forced_id is not None:
            item["page_id"] = forced_id
        try:
            r = FieldReading.model_validate(item)
        except ValidationError as exc:
            notes.append(f"dropped reading {i} ({_field_name(item.get('field'))}): {type(exc).__name__}")
            continue
        if r.field not in FIELD_FAMILY:
            notes.append(f"dropped reading {i}: {_field_name(r.field)}")
            continue
        if r.field not in FAMILIES[family]:
            notes.append(f"dropped reading {i} ({r.field}): outside family {family}")
            continue
        if r.field not in fields:
            notes.append(f"dropped reading {i} ({r.field}): not a requested field")
            continue
        if not r.page_id or r.page_id not in allowed_ids:
            notes.append(f"dropped reading {i} ({r.field}): page id {_safe_id(r.page_id)!r} is not one of the pages read")
            continue
        n = counts.get((r.page_id, r.field), 0)
        counts[(r.page_id, r.field)] = n + 1
        readings.append({"id": reading_id(iteration, family, r.page_id, r.field, n), "iteration": iteration, "family": family,
                         "page_id": r.page_id, "field": r.field, "value": r.value, "quote": r.quote})
    return readings, notes


def read_page(llm, cache: CallCache, family: str, page: dict, iteration: int = 1, salt: str = "") -> tuple[list[dict], list[str]]:
    """One page, the family's fields. Every reading gets the page's own id, whatever the model returned."""
    raw, notes = _call(llm, cache, family, page_user_prompt(page), salt)
    if raw is None:
        return [], notes
    pid = str(page["page_id"])
    readings, more = _validate(raw, family, iteration, {pid}, set(FAMILIES[family]), forced_id=pid)
    return readings, notes + more


def case_user_prompt(pages: list[dict], fields: list[str]) -> str:
    """Several pages in one message. Each page's id is on its own line OUTSIDE the untrusted wrapper, then the wrapped page."""
    head = "Report only these fields: " + ", ".join(fields) + "\n\n"
    return head + "\n\n".join(f"page_id: {p['page_id']}\n{page_user_prompt(p)}" for p in pages)


def read_case(llm, cache: CallCache, family: str, pages: list[dict], fields: list[str], salt: str = "") -> tuple[list[dict], list[str]]:
    """Iteration 2: one call over several pages for the given fields of the family. A reading must cite one of `pages`."""
    bad = [f for f in fields if f not in FAMILIES[family]]
    if bad or not fields:
        raise ValueError(f"fields {bad or fields} are not fields of family {family}")
    raw, notes = _call(llm, cache, family, case_user_prompt(pages, fields), salt)
    if raw is None:
        return [], notes
    readings, more = _validate(raw, family, 2, {str(p["page_id"]) for p in pages}, set(fields), forced_id=None)
    return readings, notes + more
