"""Guards that turn a live-run guess into UNKNOWN / NEEDS_REVIEW (live accuracy session, 2026-11-17).

Each test uses synthetic facts and states the principle it protects; none is keyed on a gold-set vendor.
"""

from __future__ import annotations

import pytest

from assessor.claims import is_absence_statement, normalise_claim
from assessor.matcher import Matcher
from assessor.policy_rag import PolicyLibrary
from assessor.schemas import AssessmentRequest, Claim
from assessor.vendor_research import guess_source_type
from tests.conftest import fact, profile


@pytest.fixture(scope="module")
def matcher():
    return Matcher(PolicyLibrary())


def _verdicts(matcher, **facts):
    req = AssessmentRequest(question="q", vendor="TestVendor", tier="T3", users=40, user_groups=["public", "employees", "security"])
    return {r.requirement_id: r for r in matcher.evaluate(profile(**facts), req)}


# --- absence of a mention is not evidence -------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "No SAML/OIDC SSO mentioned on the page", "SSO is not mentioned", "The page does not mention single sign-on",
    "Not stated in the documentation", "No information about SSO was found", "SSO support could not be confirmed",
])
def test_absence_phrases_are_detected(text):
    assert is_absence_statement(text)


@pytest.mark.parametrize("text", [
    "SAML SSO is not available on any plan", "The vendor does not support SSO", "SSO is offered on Enterprise only",
    "No SSO is offered", "Single sign-on is unavailable",
])
def test_real_negatives_are_not_absence(text):
    assert not is_absence_statement(text)


def test_matcher_turns_absence_fail_into_unknown(matcher):
    f = fact("sso_saml", "none", "claimed")
    f.display = "No SAML/OIDC SSO mentioned on the page"
    r = _verdicts(matcher, sso_saml=f)["SEC-7.1"]
    assert r.verdict == "UNKNOWN"


def test_matcher_keeps_an_explicit_no_sso_as_fail(matcher):
    f = fact("sso_saml", "none", "claimed")
    f.display = "SSO is not available on any plan"
    assert _verdicts(matcher, sso_saml=f)["SEC-7.1"].verdict == "FAIL"


def test_claim_normaliser_rejects_negative_value_built_on_absence():
    c = Claim(field="sso_saml", value="none", display="No SSO mentioned on the page", evidence_level="claimed")
    out, why = normalise_claim(c)
    assert out is None and "absence" in why


def test_claim_normaliser_keeps_explicit_negative_and_positive():
    ok = Claim(field="sso_saml", value="none", display="SSO is not offered", evidence_level="claimed")
    assert normalise_claim(ok)[0] is not None
    pos = Claim(field="sso_saml", value="enterprise_only", display="SAML SSO is not mentioned elsewhere but offered on Enterprise", evidence_level="claimed")
    assert normalise_claim(pos)[0] is not None   # only the "vendor lacks it" value is guarded


# --- who said it: a blocker cannot be decided by a third-party write-up ------------------------------------------

@pytest.mark.parametrize("level", ["third_party", "rumour"])
def test_training_claim_from_weak_source_cannot_pass_or_fail(matcher, level):
    for value in ("no", "yes"):
        r = _verdicts(matcher, trains_on_customer_data=fact("trains_on_customer_data", value, level))["AI-2.1"]
        assert r.verdict == "NEEDS_REVIEW", (level, value)


def test_training_claim_from_the_vendor_still_decides(matcher):
    assert _verdicts(matcher, trains_on_customer_data=fact("trains_on_customer_data", "no", "documented"))["AI-2.1"].verdict == "PASS"
    assert _verdicts(matcher, trains_on_customer_data=fact("trains_on_customer_data", "yes", "claimed"))["AI-2.1"].verdict == "FAIL"


def test_decisive_levels_only_apply_where_declared(matcher):
    # SEC-7.1 declares no decisive_levels: a third-party SSO statement still decides, as before.
    assert _verdicts(matcher, sso_saml=fact("sso_saml", "all_plans", "third_party"))["SEC-7.1"].verdict == "PASS"


@pytest.mark.parametrize("host, expected", [
    ("https://community.vendor.example/t/is-my-data-used", "forum"),
    ("https://forum.vendor.example/x", "forum"),
    ("https://www.vendor.example/trust", "official"),
    ("https://vendor.example/legal/dpa", "official"),
    ("https://trust.vendor.example/", "official"),
])
def test_vendor_hosted_forum_is_not_the_vendor_speaking(host, expected):
    assert guess_source_type(host, "vendor.example") == expected


# --- sub-processors: model providers and partial lists -----------------------------------------------------------

def _subs(*items, level="documented"):
    return fact("subprocessors", [dict(i) for i in items], level)


HOST = {"name": "CloudHost", "purpose": "Infrastructure hosting", "category": "hosting", "country": "India"}
LLM_HOST = {"name": "Serving Co", "purpose": "Compute infrastructure for serving large language models", "category": "hosting", "country": "USA"}


def test_documented_plain_list_passes(matcher):
    assert _verdicts(matcher, subprocessors=_subs(HOST))["AI-3.4"].verdict == "PASS"


def test_llm_host_tagged_as_hosting_is_still_a_model_provider(matcher):
    r = _verdicts(matcher, subprocessors=_subs(HOST, LLM_HOST))["AI-3.4"]
    assert r.verdict == "NEEDS_REVIEW" and "Serving Co" in r.reason


def test_named_model_provider_is_flagged_whatever_its_category(matcher):
    openai = {"name": "OpenAI, L.L.C.", "purpose": "Text generation", "category": "other", "country": "USA"}
    assert _verdicts(matcher, subprocessors=_subs(HOST, openai))["AI-3.4"].verdict == "NEEDS_REVIEW"


def test_ordinary_ai_wording_does_not_flag(matcher):
    ai_analytics = {"name": "Metrics Inc", "purpose": "Product analytics for our AI features", "category": "analytics", "country": "USA"}
    assert _verdicts(matcher, subprocessors=_subs(HOST, ai_analytics))["AI-3.4"].verdict == "PASS"


def test_list_quoted_from_a_claim_is_not_the_published_list(matcher):
    r = _verdicts(matcher, subprocessors=_subs(HOST, level="claimed"))["AI-3.4"]
    assert r.verdict == "NEEDS_REVIEW" and "complete" in r.reason
