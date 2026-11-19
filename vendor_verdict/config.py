"""Constants shared by every vendor_verdict module. Nothing here reads the environment or the competition files."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_kaggle"                       # gitignored competition data
CACHE_DIR = DATA_DIR / "cache" / "agents"             # default root of the agent call cache (vendor_verdict.cache.CallCache)

# The 8 rules_lite fields, grouped by the specialist that reads them. Each field belongs to exactly one family.
FAMILIES: dict[str, tuple[str, ...]] = {
    "certifications": ("security_certifications",),
    "data": ("data_regions", "data_retention", "dpa_available"),
    "ai": ("trains_on_customer_data", "subprocessors"),
    "access_incidents": ("sso_saml", "security_incidents"),
}
FIELD_FAMILY: dict[str, str] = {field: family for family, fields in FAMILIES.items() for field in fields}

MAX_ITERATIONS = 2                                    # evaluator loop: the first read plus one targeted re-read

# Agent roles that run on the local llama.cpp model. Value confirmed by the user after the Task 0 spike; "injection" is deliberately
# absent (screening stays off the local model).
LOCAL_SCOPE: frozenset[str] = frozenset({"router", "grader", "decision_maker"})
LOCAL_SLOTS = 2                                       # slots on the local server: the cap on concurrent local calls

# Fields the grader checks: all 8 by default (ledger ruling R2); Task 0/5 may narrow it.
GRADER_FIELDS: frozenset[str] = frozenset(FIELD_FAMILY)

# Holdout rows whose labels were exposed in an earlier session: excluded from every holdout rule and gate.
EXPOSED_HOLDOUT_ROWS: frozenset[str] = frozenset({
    "c0060__SEC-4.2", "c0312__SEC-4.2-T2", "c0678__SEC-4.2-T2", "c0062__DP-3.1",
    "c0288__SEC-7.1", "c0426__SEC-4.2", "c0512__SEC-9.3",
})

# The extraction prompt of the submitted baseline: (git commit that holds it, its short content hash).
BASELINE_PROMPT: tuple[str, str] = ("e42203d", "9566ad4b6b95")
