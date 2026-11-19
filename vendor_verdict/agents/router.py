"""The Page Router (one local-model call per readable page: which families does this page cover?) and the Source Validator's optional
local injection flag.

The router runs on the local llama.cpp model (Qwen3-4B, about 45 calls a minute). It is recall-first by construction:
- a call that fails, times out or returns unusable JSON never stops the case: the page is tagged with EVERY family and the failure is
  counted (the caller reports local fallbacks separately; they are never hidden);
- if "router" is not in `config.LOCAL_SCOPE` the role is switched off: every page is tagged with every family, no call is made and
  nothing counts as a failure;
- only readable pages are ever sent to a model; a quarantined or ignored page never leaves this process.

The caller passes a local-only model (`LLM.local_only`): a local model with a cloud fallback would send page text to the cloud.
Calls are made one after another; capping concurrent local calls (`config.LOCAL_SLOTS`) across cases is the runner's job.

Notes carry page ids (only id-shaped values), family names from the fixed list and exception CLASS names, never page text, titles,
URLs, model output or exception messages.
"""
from __future__ import annotations

import json

from assessor.guardrails import classify_injection
from assessor.llm import LLMError
from assessor.tracing import BudgetExceeded, estimate_tokens
from eval import run_kaggle as rk

from .. import config
from ..cache import CallCache
from ..config import FAMILIES
from ..prompts import ROUTER_SYSTEM, page_user_prompt, version
from ..safe import safe_id
from ..schemas import RouterOut


def _ask(local_llm, cache: CallCache, page: dict) -> tuple[list | None, str]:
    """(raw families, exception class name). None = the call failed (never cached). A cache hit makes no call."""
    user = page_user_prompt(page)
    key = cache.key("router", version(ROUTER_SYSTEM), local_llm.model, user)
    rec = cache.get(key)
    if rec is not None and isinstance(rec.get("families"), list):
        return rec["families"], ""
    try:
        # max_repairs=0: the local server answers in grammar-constrained JSON, so a repair round only costs time
        res = local_llm.structured(ROUTER_SYSTEM, user, RouterOut, stage="router", max_repairs=0)
    except BudgetExceeded:
        raise
    except Exception as exc:                            # noqa: BLE001  any failure of one page maps to recall-first, and is counted
        return None, type(exc).__name__
    raw = json.loads(json.dumps(res.families, default=str))
    cache.put(key, {"model": local_llm.model, "families": raw,
                    "tokens_in": estimate_tokens(ROUTER_SYSTEM + user), "tokens_out": estimate_tokens(json.dumps(raw))})
    return raw, ""


def tag_pages(local_llm, cache: CallCache, recs: list[dict], notes: list[str] | None = None) -> tuple[dict[str, list[str]], int]:
    """(page id -> families, failures) for the readable pages of `recs`. `notes`, when given, collects the id-only notes.
    A tag list may be empty (the page covers no family); the reviewer's rule 4 still routes an official page then."""
    notes = notes if notes is not None else []
    pages = [r["page"] for r in recs if rk.readable(r)]
    if "router" not in config.LOCAL_SCOPE:
        return {str(p["page_id"]): list(FAMILIES) for p in pages}, 0
    tags: dict[str, list[str]] = {}
    failures = 0
    for page in pages:
        pid = str(page["page_id"])
        raw, err = _ask(local_llm, cache, page)
        if raw is None:
            failures += 1
            tags[pid] = list(FAMILIES)
            notes.append(f"{safe_id(pid)}: router failed: {err}, tagged with every family")
            continue
        known: set[str] = set()
        for f in raw:
            if isinstance(f, str) and f in FAMILIES:
                known.add(f)
            else:
                notes.append(f"{safe_id(pid)}: dropped an unknown family")
        tags[pid] = [f for f in FAMILIES if f in known]
    return tags, failures


class _LocalCallFailed(Exception):
    """A local call failed. Deliberately not an LLMError: `classify_injection` answers an LLMError with its rule classifier and writes
    `str(exc)` to the tracer, and an LLMError message can carry the model's raw output."""


class _LocalOnly:
    """`classify_injection` swaps any LLM that is not in mock mode for `LLM.small()`, the configured small provider (a cloud model
    when the main provider is one). This stand-in reports mock mode so the classifier uses it as given, and forwards every call to the
    local model, so the second opinion never leaves the machine. A failed call is re-raised as `_LocalCallFailed` (class name only), so
    nothing the model wrote reaches the tracer and the page counts as not flagged."""
    mock_mode = True

    def __init__(self, llm):
        self._llm = llm
        self.budget, self.tracer = getattr(llm, "budget", None), getattr(llm, "tracer", None)

    @property
    def model(self):
        return self._llm.model

    def structured(self, *args, **kwargs):
        kwargs.setdefault("max_repairs", 0)             # grammar-constrained JSON on the local server: a repair round only costs time
        try:
            return self._llm.structured(*args, **kwargs)
        except LLMError as exc:
            raise _LocalCallFailed(type(exc).__name__) from None


def injection_second_opinion(local_llm, recs: list[dict]) -> int:
    """How many readable pages the local model (with the existing layered classifier) would flag as injection. Off unless "injection"
    is in `config.LOCAL_SCOPE` (then it returns 0 without any call). Trace use only: it never changes `recs`, and a page whose check
    fails counts as not flagged."""
    if "injection" not in config.LOCAL_SCOPE:
        return 0
    llm = _LocalOnly(local_llm)
    flagged = 0
    for r in recs:
        if not rk.readable(r):
            continue
        try:
            if classify_injection(str(r["page"].get("text") or ""), llm).is_injection:
                flagged += 1
        except BudgetExceeded:
            raise
        except Exception:                               # noqa: BLE001  a failed check is "not flagged", never a stopped case
            continue
    return flagged
