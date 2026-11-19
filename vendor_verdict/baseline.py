"""The submitted baseline's page readings, recovered from its extraction cache, as reading dicts (see vendor_verdict.state), and the
pre-registered denominator of the page-routing check.

`baseline_readings` NEVER calls a model: it reads `run_kaggle.Extractor` in cache-only mode. How the Extractor behaves, and what is
done about it:
- on a cache miss it returns `{"status": "not_cached", ...}` and neither raises nor calls the model; such a page has no key in the result;
- its constructor runs `mkdir`, so a missing cache directory is answered with `{}` BEFORE the Extractor is built (nothing is created);
- `system or EXTRACT_SYSTEM`: an empty prompt text would silently select the CURRENT prompt, so an empty `system` is refused;
- a hit reads the JSON file and indexes `claims`, `in` and `out`: a corrupt or partial file would raise, so each page is guarded and a
  bad file means no key for that page, never a crash;
- the stand-in model refuses every call and carries `model`, because the model name is part of the cache key.

Only readable pages are read (a quarantined or ignored page never feeds anything downstream). Claims go through
`kaggle_policy.normalise_page_claims` exactly as the baseline's own merge does, and only the 8 family fields stay (`legal_entity` and
anything else is dropped). A list-valued claim stays ONE reading whose value is the whole normalised list, the statement the baseline
made. A page that has a cache entry but no usable claim has a key with an empty list; a page with no entry has no key.

Readings hold page-derived text (value, quote, display): they must never be printed, logged or traced.
"""
from __future__ import annotations

import copy
import functools
from pathlib import Path

from assessor.schemas import Claim
from eval import kaggle_policy as kp
from eval import run_kaggle as rk

from . import config
from .config import FIELD_FAMILY
from .state import reading_id


class _NoCallLLM:
    """The model of a cache-only Extractor: it only has to carry the name that is part of the cache key. Any call is a bug."""

    def __init__(self, model: str):
        self.model = model

    def structured(self, *args, **kwargs):
        raise RuntimeError("baseline_readings never calls a model")


@functools.lru_cache(maxsize=1)
def baseline_prompt() -> str:
    """The submitted baseline's extraction prompt, recovered byte for byte from git (`config.BASELINE_PROMPT`). Recovered once per
    process: it runs `git show` and executes an old source file. A missing commit or a drifted hash raises (nothing is cached then)."""
    return kp.prompt_text_from_git(*config.BASELINE_PROMPT)


def _page_readings(page_id: str, claims: list[dict]) -> list[dict]:
    parsed: list[Claim] = []
    for raw in claims:
        try:
            parsed.append(Claim.model_validate(raw))
        except Exception:                                  # noqa: BLE001  one malformed cached claim never costs the page its other claims
            continue
    kept, _ = kp.normalise_page_claims(parsed)
    counts: dict[str, int] = {}
    out = []
    for c in kept:
        if c.field not in FIELD_FAMILY:
            continue
        n = counts.get(c.field, 0)
        counts[c.field] = n + 1
        family = FIELD_FAMILY[c.field]
        out.append({"id": reading_id(0, family, page_id, c.field, n), "iteration": 0, "family": family, "page_id": page_id,
                    "field": c.field, "value": copy.deepcopy(c.value), "quote": c.quote, "display": c.display})
    return out


def baseline_readings(llm_model: str, recs: list[dict], cache_dir: Path, system: str | None = None) -> dict[str, list[dict]]:
    """page id -> the baseline's readings of that page (iteration 0). Never calls a model. `system` is the prompt text the baseline
    ran with (tests inject one); the default is `baseline_prompt()`."""
    if system is not None and not system:
        raise ValueError("an empty prompt text would select the current prompt, not the baseline's")
    system = baseline_prompt() if system is None else system
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():                             # Extractor.__init__ would create it
        return {}
    extractor = rk.Extractor(_NoCallLLM(llm_model), cache_dir, True, system=system)
    out: dict[str, list[dict]] = {}
    for rec in recs:
        if not rk.readable(rec):
            continue
        page_id = str(rec["page"]["page_id"])
        try:
            res = extractor.run(rec["page"])
            if res.get("status") != "ok" or not isinstance(res.get("claims"), list):
                continue
            out[page_id] = _page_readings(page_id, res["claims"])
        except Exception:                                  # noqa: BLE001  a corrupt or partial cache file means no key for this page
            continue
    return out


def route_baseline_fields(baseline: dict[str, list[dict]], recs: list[dict]) -> dict[str, set[str]]:
    """The pre-registered denominator of the page-routing check: page id -> the family fields the baseline read on it.
    Only readable pages count (role not "ignored", no quarantine hit); a "registry" page contributes only security_certifications; only
    the 8 family fields count; a page with no remaining field is omitted."""
    out: dict[str, set[str]] = {}
    for rec in recs:
        if not rk.readable(rec):
            continue
        page_id = str(rec["page"]["page_id"])
        fields = {r["field"] for r in baseline.get(page_id, []) if isinstance(r.get("field"), str) and r["field"] in FIELD_FAMILY}
        if rec["role"] == "registry":
            fields &= {"security_certifications"}
        if fields:
            out[page_id] = fields
    return out
