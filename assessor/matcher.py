"""
matcher.py — the deterministic core: policy requirement × vendor fact → PASS / FAIL / UNKNOWN / NEEDS_REVIEW.

THIS FILE HAS NO LLM IN IT, ON PURPOSE.
The research layer may use a model to READ pages. The verdict is computed by code you can unit-test,
so that two runs on the same evidence always give the same verdict, and so that a reviewer can point at
the exact line of policy and the exact line of evidence behind every cell of the report.

COURSE MAP
  Deep Search course  Module 6  (from evidence to a clean structured result; provenance on every claim)
                      Module 10 (programmatic metrics: the verdicts here are what eval/run_eval.py scores)
  Production RAG      Module 9  ("Refusing": UNKNOWN beats a guess)

Rules are declared in data/requirements.yaml; each `rule:` name maps to one function below.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

import yaml

from .claims import is_absence_statement
from .config import REQUIREMENTS_FILE
from .errors import record_error
from .policy_rag import PolicyLibrary
from .schemas import (AssessmentRequest, Overall, Requirement, RequirementResult, Verdict, VendorFact, VendorProfile)

TIER_RANK = {"T1": 1, "T2": 2, "T3": 3}

# Which questionnaire item (data/policies/security_questionnaire_template.md) to send when a rule is not PASS.
QUESTION_FOR_CLAUSE = {
    "SEC-4.2": "Q-SEC-4.2", "DP-3.1": "Q-DP-3.1", "DP-3.2": "Q-DP-3.1", "AI-2.1": "Q-AI-2.1", "AI-3.4": "Q-AI-3.4",
    "DP-5.4": "Q-DP-5.4", "SEC-7.1": "Q-SEC-7.1", "SEC-9.3": "Q-SEC-9.3", "DP-2.2": "Q-DP-2.2", "DP-4.1": "Q-DP-4.1",
}


def load_requirements(path: Path = REQUIREMENTS_FILE) -> list[Requirement]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Requirement.model_validate(r) for r in data["requirements"]]


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("/", " ").replace("-", " ").split())


def _months_between(a: date, b: date) -> float:
    return abs((a - b).days) / 30.44


# ---------------------------------------------------------------------------
# Rule functions. Signature: (req, fact, ctx) -> (verdict, reason)
# ctx carries as_of, request, library — only approved_for_tier uses the library.
# ---------------------------------------------------------------------------

def rule_certification_valid(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or not fact.value:
        return "UNKNOWN", "No security certification could be established from public sources."
    accepted = {_norm(a) for a in req.params.get("accepted", [])}
    max_age = int(req.params.get("max_age_months", 15))
    require_documented = bool(req.params.get("require_documented", True))
    as_of: date = ctx["as_of"]

    certs = list(fact.value)
    matches = [c for c in certs if _norm(c.get("name", "")) in accepted]
    generic_soc2 = [c for c in certs if _norm(c.get("name", "")) == "soc 2" and c.get("status") == "certified"]
    certified = [c for c in matches if c.get("status") == "certified"]

    def is_current(c: dict) -> bool:
        vu = c.get("valid_until")
        if vu:
            return date.fromisoformat(str(vu)) >= as_of
        d = c.get("date")
        return bool(d) and _months_between(as_of, date.fromisoformat(str(d))) <= max_age

    current = [c for c in certified if is_current(c)]
    if current:
        names = "; ".join(f"{c['name']} ({c.get('date') or 'date n/a'}{', valid to ' + str(c['valid_until']) if c.get('valid_until') else ''})" for c in current)
        if require_documented and fact.evidence_level != "documented":
            return "NEEDS_REVIEW", f"Vendor states {names}, but only as a claim. Request the report or certificate before relying on it."
        return "PASS", f"Current, independently issued: {names}."
    if certified:
        names = "; ".join(f"{c['name']} ({c.get('date') or 'undated'})" for c in certified)
        return "NEEDS_REVIEW", f"{names} is older than {max_age} months or undated. Request the current report."
    if generic_soc2:
        return "NEEDS_REVIEW", "Vendor says 'SOC 2' without stating Type I or Type II, a date or an auditor. A statement is not a report — request it."
    if matches:
        names = "; ".join(f"{c['name']} ({c.get('status')})" for c in matches)
        return "FAIL", f"Only {names}. A certification that is in progress or planned does not satisfy the clause."
    names = "; ".join(str(c.get("name")) for c in certs) or "none"
    return "FAIL", f"No accepted certification found (saw: {names}). Accepted: {', '.join(req.params.get('accepted', []))}."


# ISO 3166 codes the extractor may return instead of names ("IN" for India). Only used for whole-value codes.
_ISO_REGION = {"in": "india", "us": "united states", "gb": "united kingdom", "uk": "united kingdom", "de": "germany",
               "fr": "france", "ie": "ireland", "sg": "singapore", "au": "australia", "jp": "japan", "ca": "canada",
               "cn": "china", "ae": "united arab emirates", "sa": "saudi arabia", "br": "brazil"}


def _region_hit(wanted: str, value: str) -> bool:
    """Does a stated storage region (any wording) satisfy a wanted region name?

    Matches the exact name, the ISO code ("IN"), or the name as a whole word inside a longer description
    ("India (AWS Mumbai, ap-south-1)"). Not a substring match, so "Indiana" is not India.
    """
    w, v = _norm(wanted), _norm(value)
    if not w or not v:
        return False
    if v == w or _ISO_REGION.get(v) == w:
        return True
    return re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", v) is not None


def rule_region_includes(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or not isinstance(fact.value, dict):
        return "UNKNOWN", "Storage regions could not be established from public sources."
    wanted = [_norm(x) for x in req.params.get("any_of", [])]
    available = {_norm(x) for x in fact.value.get("available", [])}
    planned = {_norm(x) for x in fact.value.get("planned", [])}
    hit = sorted({w for w in wanted if any(_region_hit(w, a) for a in available)})
    if hit:
        return "PASS", f"Customer content can be stored in {', '.join(hit).title()}. Regions: {fact.display}."
    planned_hit = sorted({w for w in wanted if any(_region_hit(w, p) for p in planned)})
    if planned_hit and not req.params.get("allow_planned", False):
        return "FAIL", f"{', '.join(planned_hit).title()} is only planned, which does not count. Available today: {', '.join(sorted(available)).title() or 'none stated'}."
    if available:
        return "FAIL", f"Storage regions are {', '.join(sorted(available)).title()}; none of the required regions ({', '.join(req.params.get('any_of', []))})."
    return "UNKNOWN", "The vendor does not state where content is stored."


def rule_value_known(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None:
        return "UNKNOWN", "Not stated by the vendor; record it in the review file."
    if isinstance(fact.value, dict) and not fact.value.get("available"):
        return "UNKNOWN", "Not stated by the vendor; record it in the review file."
    return "PASS", f"Recorded: {fact.display}."


def rule_enum_verdict(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or fact.value in (None, ""):
        return "UNKNOWN", "Not stated in the vendor's public material."
    v = str(fact.value)
    p = req.params
    if v in p.get("fail", []) and is_absence_statement(fact.display):
        # "No SSO mentioned on the page" describes the page, not the vendor. Absence is not evidence (UNKNOWN beats a guess).
        return "UNKNOWN", f"The source only does not mention it ({fact.display}). Absence of a mention is not evidence; ask the vendor."
    decisive = p.get("decisive_levels")
    if decisive and fact.evidence_level not in decisive and (v in p.get("pass", []) or v in p.get("fail", [])):
        # Who said it matters for a blocker: a forum post, review or third-party write-up may not decide PASS or FAIL.
        return "NEEDS_REVIEW", (f"{fact.display}. Only {fact.evidence_level} evidence (a decision here needs "
                                f"{' or '.join(decisive)}); confirm with the vendor's own terms.")
    if v in p.get("pass", []):
        return "PASS", fact.display
    if v in p.get("review", []):
        return "NEEDS_REVIEW", f"{fact.display}. Acceptable only with the conditions in {req.clause}; record how they are met."
    if v in p.get("fail", []):
        return "FAIL", fact.display
    return "UNKNOWN", f"Unrecognised value {v!r}; check the extraction."


# A sub-processor that runs or serves a language model is a model provider whatever category the extractor gave it
# ("hosting" for Baseten serving LLMs). Tight on purpose: bare "AI" would flag every vendor.
_MODEL_PROVIDER_PURPOSE = re.compile(
    r"\b(?:large language models?|language models?|llms?|foundation models?|generative ai|model (?:hosting|serving|inference)|"
    r"(?:hosting|serving|running) (?:of )?(?:ai |ml |machine learning )?models?)\b", re.I)
_MODEL_PROVIDER_NAME = re.compile(
    r"\b(?:openai|anthropic|cohere|mistral|gemini|vertex ai|bedrock|hugging ?face|baseten|fireworks|together ai|groq|replicate)\b", re.I)


def _is_model_provider(item: dict) -> bool:
    if item.get("category") == "model_provider":
        return True
    return bool(_MODEL_PROVIDER_PURPOSE.search(str(item.get("purpose", ""))) or _MODEL_PROVIDER_NAME.search(str(item.get("name", ""))))


def rule_list_disclosed(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or not fact.value:
        return "UNKNOWN", "No sub-processor list could be found. Ask the vendor for it."
    flag_cats = set(req.params.get("review_if_category", []))
    flagged = [i for i in fact.value if i.get("category") in flag_cats or _is_model_provider(i)]
    if flagged:
        names = ", ".join(f"{i['name']} ({i.get('country', '?')})" for i in flagged)
        return "NEEDS_REVIEW", f"{len(fact.value)} sub-processors disclosed. Model provider(s) present: {names} — confirm AI-2.1 applies to them too."
    if req.params.get("require_documented", False) and fact.evidence_level != "documented":
        # A list quoted from an FAQ, trust summary or marketing page may be one product's subset, not the published list.
        return "NEEDS_REVIEW", (f"{len(fact.value)} sub-processor(s) found, but only as a {fact.evidence_level} statement, not on the "
                                "vendor's published sub-processor list. Confirm the list is complete.")
    return "PASS", f"{len(fact.value)} sub-processors disclosed; no third-party model provider."


def rule_retention_within(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or not isinstance(fact.value, dict):
        return "UNKNOWN", "Retention and deletion terms not stated."
    max_days = int(req.params.get("max_days", 30))
    default_days = fact.value.get("default_days")
    on_request = fact.value.get("deletion_on_request_days")
    if default_days is not None and int(default_days) <= max_days:
        return "PASS", f"Default retention {default_days} days (limit {max_days}). {fact.display}."
    if default_days is None and on_request is not None and int(on_request) <= max_days:
        return "NEEDS_REVIEW", f"Default retention not stated; deletion on request within {on_request} days. Record the commitment in the contract."
    if default_days is not None and on_request is not None and int(on_request) <= max_days:
        return "NEEDS_REVIEW", f"Default retention {default_days} days exceeds {max_days}, but deletion on request within {on_request} days is offered. Record the commitment in the contract."
    if default_days is not None and on_request is None:
        return "NEEDS_REVIEW", f"Default retention {default_days} days exceeds {max_days} and no deletion-on-request commitment was found. Ask the vendor."
    return "FAIL", f"Default retention {default_days} days and deletion on request {on_request} days both exceed {max_days}."


def rule_no_unresolved_incident(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or fact.value is None:
        return "UNKNOWN", "Incident history could not be established. Ask the vendor (and keep rumours out, SEC-9.4)."
    as_of: date = ctx["as_of"]
    lookback = as_of - timedelta(days=int(req.params.get("lookback_months", 24)) * 30)
    majors = {s.lower() for s in req.params.get("major_severities", ["major", "critical"])}
    recent = [i for i in fact.value if i.get("date") and date.fromisoformat(str(i["date"])) >= lookback]
    major = [i for i in recent if str(i.get("severity", "")).lower() in majors]
    without = [i for i in major if not i.get("postmortem_published")]
    if without:
        s = "; ".join(f"{i['date']}: {i['summary']}" for i in without)
        return "FAIL", f"Major incident without a published post-mortem: {s}."
    if major:
        s = "; ".join(f"{i['date']}: {i['summary']}" for i in major)
        return "NEEDS_REVIEW", f"Major incident(s) with a published post-mortem: {s}. Information Security must read the post-mortem before approval."
    minor = len(recent) - len(major)
    return "PASS", f"No major security incident in the look-back window." + (f" {minor} minor/availability incident(s) noted." if minor else "")


def rule_approved_for_tier(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    library: PolicyLibrary = ctx["library"]
    request: AssessmentRequest = ctx["request"]
    entry = library.approved_vendor_entry(request.vendor, request.user_groups)
    if entry == "NO_ACCESS":
        return "UNKNOWN", "You do not have access to the Approved Vendor List. Ask Procurement for this vendor's approval status at this tier."
    if entry is None:
        return "NEEDS_REVIEW", "Not on the Approved Vendor List. This assessment is the input to that review; Procurement records the outcome (PROC-3.1)."
    if entry["status"] == "rejected":
        return "FAIL", f"Vendor was rejected on {entry.get('date')}: {entry.get('notes')}"
    if entry["status"] == "in_progress":
        return "NEEDS_REVIEW", f"A review is already in progress ({entry.get('notes')}). Do not start a second one; attach this assessment to it."
    approved = entry.get("tier", "")
    if TIER_RANK.get(approved, 0) >= TIER_RANK.get(request.tier, 3):
        return "PASS", f"Approved for {approved} on {entry.get('approved_on')} (plan: {entry.get('plan')}). {entry.get('notes')}"
    return "NEEDS_REVIEW", f"Approved for {approved} only (on {entry.get('approved_on')}); this request is for {request.tier}. Approval must be extended. {entry.get('notes')}"


RULES: dict[str, Callable[[Requirement, Optional[VendorFact], dict], tuple[Verdict, str]]] = {
    "certification_valid": rule_certification_valid,
    "region_includes": rule_region_includes,
    "value_known": rule_value_known,
    "enum_verdict": rule_enum_verdict,
    "list_disclosed": rule_list_disclosed,
    "retention_within": rule_retention_within,
    "no_unresolved_incident": rule_no_unresolved_incident,
    "approved_for_tier": rule_approved_for_tier,
}


class Matcher:
    def __init__(self, library: PolicyLibrary, requirements: list[Requirement] | None = None):
        self.library = library
        self.requirements = requirements or load_requirements()
        unknown = {r.rule for r in self.requirements} - set(RULES)
        if unknown:
            raise ValueError(f"requirements.yaml uses rules with no implementation: {sorted(unknown)}")

    def evaluate(self, profile: VendorProfile, request: AssessmentRequest) -> list[RequirementResult]:
        ctx = {"as_of": profile.as_of, "request": request, "library": self.library}
        results: list[RequirementResult] = []
        for req in self.requirements:
            fact = profile.get(req.field)
            if not req.applies(request.tier, request.users):
                verdict, reason = "NOT_APPLICABLE", f"Applies to tiers {', '.join(req.applies_when.tiers)}" + (
                    f" or {req.applies_when.or_min_users}+ users" if req.applies_when.or_min_users else "") + f"; this request is {request.tier}, {request.users} users."
            else:
                try:
                    verdict, reason = RULES[req.rule](req, fact, ctx)
                except Exception as exc:  # noqa: BLE001
                    # A rule that cannot read its evidence must not take the whole assessment down, and must not
                    # guess. The requirement is reported as not established and the defect is logged for a fix.
                    record_error("matcher:rule", exc, {"requirement": req.id, "rule": req.rule, "vendor": request.vendor})
                    verdict, reason = "UNKNOWN", ("The evidence found for this requirement could not be interpreted, "
                                                  "so it is treated as not established. Ask the vendor.")
            q_id = None
            if verdict in ("UNKNOWN", "NEEDS_REVIEW"):
                q_id = QUESTION_FOR_CLAUSE.get(req.clause)
            results.append(RequirementResult(
                requirement_id=req.id, title=req.title, clause=req.clause, policy_doc=req.policy_doc,
                severity=req.severity, verdict=verdict, reason=reason,
                evidence_urls=list(fact.sources) if fact else [],
                evidence_as_of=fact.as_of if fact else None,
                evidence_level=fact.evidence_level if fact else None,
                vendor_question_id=q_id))
        return results

    @staticmethod
    def overall(results: list[RequirementResult]) -> tuple[Overall, str]:
        blockers = [r for r in results if r.verdict == "FAIL" and r.severity == "blocker"]
        if blockers:
            return "BLOCKED", "Blocking requirement(s) failed: " + ", ".join(f"{r.requirement_id} ({r.title})" for r in blockers) + "."
        majors = [r for r in results if r.verdict == "FAIL" and r.severity == "major"]
        open_items = [r for r in results if r.verdict in ("UNKNOWN", "NEEDS_REVIEW")]
        if majors or open_items:
            parts = []
            if majors:
                parts.append("failed major requirement(s) need an approver exception: " + ", ".join(r.requirement_id for r in majors))
            if open_items:
                parts.append(f"{len(open_items)} item(s) need evidence or a human decision: " + ", ".join(r.requirement_id for r in open_items))
            return "CONDITIONAL", "; ".join(parts).capitalize() + "."
        return "READY_FOR_APPROVAL", "Every applicable requirement passed on documented evidence. A human approver still decides (AI-5.1)."
