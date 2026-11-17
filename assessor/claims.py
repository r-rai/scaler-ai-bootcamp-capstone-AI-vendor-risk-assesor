"""
claims.py — the shape of a claim's value is a contract too.

`Claim.value` is whatever JSON the extraction model returned. The matcher is deterministic code that
expects exact shapes (a list of certificate objects, a region dict, an enum). A model that answers
`["AWS", "OpenAI"]` where a list of objects is expected must not crash the run, and must not be "fixed"
into something it did not say.

`normalise_claim()` is the trust boundary between the model and the rules:
  * restructure only: trim, canonical spelling ("SOC2 Type 2" -> "SOC 2 Type II"), "30 days" -> 30
  * never invent: a missing status, category, severity or date is NOT defaulted. Defaulting a sub-processor's
    category to "other" would let an undisclosed model provider pass; defaulting a certificate's status to
    "certified" would turn a mention into evidence.
  * anything that cannot be read safely is rejected with a reason. The field then has no fact, the matcher
    says UNKNOWN, and the question goes to the vendor. UNKNOWN beats a guess.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from .schemas import Claim

CERT_STATUSES = {"certified", "in_progress", "planned"}
SUBPROCESSOR_CATEGORIES = {"hosting", "model_provider", "email", "analytics", "support", "other"}
ENUMS: dict[str, set[str]] = {
    "trains_on_customer_data": {"no", "no_on_paid_plans", "opt_out_available", "yes"},
    "sso_saml": {"all_plans", "paid_plans", "enterprise_only", "none"},
    "dpa_available": {"yes", "on_request", "no"},
}
# The enum value that means "the vendor does not offer this". Only an explicit statement may produce it.
NEGATIVE_VALUES = {"sso_saml": "none", "dpa_available": "no"}
SEVERITY_MAP = {"major": "major", "critical": "major", "high": "major", "severe": "major",
                "minor": "minor", "low": "minor", "medium": "minor", "availability": "minor"}


class Malformed(ValueError):
    pass


# "No SSO mentioned on the page" says what the PAGE lacks, not what the VENDOR lacks. Treating it as a negative
# fact is a guess. Matched on the claim's own words so the same test works at extraction time (normalise_claim)
# and at verdict time (matcher.rule_enum_verdict, which also runs on saved profiles). It must not catch real
# negatives such as "does not support SSO" or "SSO is not available".
_ABSENCE = re.compile(
    r"\b(?:no|not|without|nothing|never)\b[^.;]{0,40}?\b(?:mention(?:s|ed|ing)?|stated?|specif(?:y|ied)|listed?|"
    r"disclos(?:e|ed|es)|document(?:ed|ation)?|found|described?|reference[sd]?|indicat(?:e|ed))\b"
    r"|\bno (?:information|details?|data|evidence|indication)\b"
    r"|\bunable to (?:find|confirm|verify)\b|\bcould(?: not|n't) (?:be )?(?:find|found|confirm(?:ed)?|verif(?:y|ied))\b"
    r"|\bdoes(?:n't| not) (?:say|state|mention|specify|list|disclose)\b", re.I)


def is_absence_statement(text: Any) -> bool:
    """True when `text` reports that something was not found or not mentioned, rather than that it does not exist."""
    return bool(_ABSENCE.search(str(text or "")))


def _iso(value: Any) -> Optional[str]:
    """A valid YYYY-MM-DD string, or None. Never raises."""
    if isinstance(value, date):
        return value.isoformat()
    m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})", str(value or ""))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
    except ValueError:
        return None


def _text(value: Any, limit: int = 300) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def canonical_cert_name(name: str) -> str:
    n = re.sub(r"[\s\-_/]+", " ", name.lower()).strip()
    if re.search(r"\bsoc ?2\b", n):
        if re.search(r"type ?(ii|2)\b", n):
            return "SOC 2 Type II"
        if re.search(r"type ?(i|1)\b", n):
            return "SOC 2 Type I"
        return "SOC 2"
    if re.search(r"\biso( iec)? ?27001\b", n):
        return "ISO 27001"
    return _text(name, 80)


def _certifications(value: Any) -> list[dict]:
    if not isinstance(value, list):
        raise Malformed("expected a list of certificate objects")
    out = []
    for item in value:
        if not isinstance(item, dict) or not _text(item.get("name")):
            raise Malformed("a certificate entry is not an object with a name")
        status = _text(item.get("status")).lower().replace(" ", "_").replace("-", "_")
        if status not in CERT_STATUSES:
            raise Malformed(f"certificate status {item.get('status')!r} is not one of {sorted(CERT_STATUSES)}")
        out.append({"name": canonical_cert_name(str(item["name"])), "status": status, "date": _iso(item.get("date")),
                    "valid_until": _iso(item.get("valid_until")), "issuer": _text(item.get("issuer"), 120) or None})
    if not out:
        raise Malformed("empty certificate list")
    return out


def _regions(value: Any) -> dict:
    if not isinstance(value, dict):
        raise Malformed("expected {available: [...], planned: [...]}: a bare list does not say which regions exist today")
    out = {}
    for key in ("available", "planned"):
        v = value.get(key) or []
        if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
            raise Malformed(f"'{key}' is not a list of region names")
        out[key] = [_text(x, 60) for x in v if _text(x)]
    if not out["available"] and not out["planned"]:
        raise Malformed("no region named")
    return out


def _subprocessors(value: Any) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise Malformed("expected a non-empty list of sub-processor objects")
    out = []
    for item in value:
        if not isinstance(item, dict) or not _text(item.get("name")):
            raise Malformed("a sub-processor entry is not an object with a name (a bare name has no category, "
                            "so a model provider could not be recognised)")
        cat = _text(item.get("category")).lower().replace(" ", "_").replace("-", "_")
        if cat not in SUBPROCESSOR_CATEGORIES:
            raise Malformed(f"sub-processor category {item.get('category')!r} is not one of {sorted(SUBPROCESSOR_CATEGORIES)}")
        out.append({"name": _text(item["name"], 120), "purpose": _text(item.get("purpose"), 200), "category": cat,
                    "country": _text(item.get("country"), 60) or None})
    return out


def _int_days(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if 0 <= value < 100_000 else None
    m = re.match(r"^\s*(\d{1,5})\s*(days?)?\s*$", str(value))
    return int(m.group(1)) if m else None


def _retention(value: Any) -> dict:
    if not isinstance(value, dict):
        raise Malformed("expected {default_days, deletion_on_request_days}")
    out = {"default_days": _int_days(value.get("default_days")), "deletion_on_request_days": _int_days(value.get("deletion_on_request_days"))}
    if out["default_days"] is None and out["deletion_on_request_days"] is None:
        raise Malformed("no retention period in days")
    return out


def _incidents(value: Any) -> list[dict]:
    if not isinstance(value, list):
        raise Malformed("expected a list of incident objects")
    out = []
    for item in value:
        if not isinstance(item, dict):
            raise Malformed("an incident entry is not an object")
        when = _iso(item.get("date"))
        sev = SEVERITY_MAP.get(_text(item.get("severity")).lower())
        if not when or not sev:
            # The matcher's look-back and post-mortem rule need both. An incident without them cannot be judged,
            # and silently skipping it could hide a major breach.
            raise Malformed("an incident has no valid date or severity")
        out.append({"date": when, "summary": _text(item.get("summary"), 300), "severity": sev,
                    "postmortem_published": item.get("postmortem_published") is True,
                    "postmortem_url": _text(item.get("postmortem_url"), 500) or None})
    return out


def _entity(value: Any) -> dict:
    if not isinstance(value, dict) or not _text(value.get("name")):
        raise Malformed("expected {name, country, hq_city}")
    return {"name": _text(value["name"], 160), "country": _text(value.get("country"), 60) or None,
            "hq_city": _text(value.get("hq_city"), 80) or None}


def _enum(field: str, value: Any) -> str:
    v = _text(value).lower().replace(" ", "_").replace("-", "_")
    if v not in ENUMS[field]:
        raise Malformed(f"{value!r} is not one of {sorted(ENUMS[field])}")
    return v


_SHAPES = {
    "security_certifications": _certifications, "data_regions": _regions, "subprocessors": _subprocessors,
    "data_retention": _retention, "security_incidents": _incidents, "legal_entity": _entity,
}


def normalise_claim(claim: Claim) -> tuple[Optional[Claim], str]:
    """Return (claim with a validated value, "") or (None, reason it was rejected)."""
    try:
        if claim.field in ENUMS:
            claim.value = _enum(claim.field, claim.value)
        else:
            claim.value = _SHAPES[claim.field](claim.value)
    except Malformed as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 — anything unexpected is also "cannot be read safely"
        return None, f"unreadable value ({type(exc).__name__})"
    claim.display = _text(claim.display, 300) or _text(claim.value, 300)
    claim.quote = _text(claim.quote, 600)
    if claim.field in NEGATIVE_VALUES and claim.value == NEGATIVE_VALUES[claim.field] and is_absence_statement(claim.display):
        return None, "absence of a mention is not evidence that the vendor lacks it (UNKNOWN beats a guess)"
    return claim, ""
