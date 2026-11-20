"""Guess fix before the final run (reports/guess_fix_20261120/preregistration.md). Two PASS paths that rested on a weak reading:

* AI-3.4: a list is only evidence of completeness when it comes from a page that IS a sub-processor list. A one-entry "list" quoted from a
  terms page, labelled `documented` because it is the vendor's own legal page, must not PASS.
* DP-2.2: "a DPA is signed when required" is conditional availability, i.e. `on_request`, not an unconditional `yes`.

Both are matcher-side (so saved profiles are rescored), switched on by a requirement parameter, and only ever turn a PASS into a review.
Synthetic facts only; nothing here is keyed on a gold-set vendor.
"""
import pytest

from assessor.matcher import Matcher, load_requirements
from assessor.schemas import AssessmentRequest, VendorFact
from tests.conftest import profile

LIST = [{"name": "Acme Cloud", "purpose": "hosting", "category": "hosting", "country": "US"},
        {"name": "MailCo", "purpose": "email", "category": "email", "country": "US"}]


def _req(rid, **params):
    base = next(r for r in load_requirements() if r.id == rid)
    keep = {k: v for k, v in base.params.items() if k not in ("require_list_page", "conditional_review")}
    keep.update(params)
    return base.model_copy(update={"params": keep})


def _verdict(library, req, field, fact):
    request = AssessmentRequest(question="q", vendor="TestVendor", tier="T3", users=40, user_groups=["public", "employees", "security"])
    (res,) = Matcher(library, requirements=[req]).evaluate(profile(**{field: fact}), request)
    return res


def _subs(url, value=LIST, level="documented"):
    return VendorFact(field="subprocessors", value=value, display=f"{len(value)} sub-processors", confidence=0.8, evidence_level=level, sources=[url])


def _dpa(display, value="yes", level="claimed"):
    return VendorFact(field="dpa_available", value=value, display=display, confidence=0.5, evidence_level=level, sources=["https://v.example/x"])


# ----------------------------------------------------------------------------------------------------------------- AI-3.4
@pytest.mark.parametrize("url", [
    "https://acme.example/legal/subprocessors", "https://acme.example/legal/sub-processors", "https://acme.example/Sub_Processors",
    "https://acme.example/dpa#subprocessors", "https://acme.example/legal?page=sub-processor-list", "https://SUBPROCESSORS.acme.example/",
])
def test_a_list_from_a_sub_processor_page_still_passes(library, url):
    assert _verdict(library, _req("AI-3.4", require_list_page=True), "subprocessors", _subs(url)).verdict == "PASS"


@pytest.mark.parametrize("url", ["https://acme.example/terms/supplemental-terms", "https://acme.example/privacy", "https://trust.acme.example/",
                                 "https://acme.example/security#processors"])
def test_a_list_from_any_other_page_is_not_known_to_be_complete(library, url):
    r = _verdict(library, _req("AI-3.4", require_list_page=True), "subprocessors", _subs(url, LIST[:1]))
    assert r.verdict == "NEEDS_REVIEW" and "not a published sub-processor list" in r.reason and url.split("//")[1][:20] in r.reason
    assert r.vendor_question_id == "Q-AI-3.4"


def test_one_listing_source_among_several_is_enough(library):
    f = _subs("https://acme.example/privacy")
    f.sources.append("https://acme.example/legal/subprocessors")
    assert _verdict(library, _req("AI-3.4", require_list_page=True), "subprocessors", f).verdict == "PASS"


def test_without_the_parameter_nothing_changes(library):
    assert _verdict(library, _req("AI-3.4"), "subprocessors", _subs("https://acme.example/terms/supplemental-terms", LIST[:1])).verdict == "PASS"


def test_the_earlier_guards_still_come_first(library):
    req = _req("AI-3.4", require_list_page=True)
    provider = [{"name": "OpenAI", "purpose": "LLM", "category": "model_provider", "country": "US"}]
    assert "Model provider" in _verdict(library, req, "subprocessors", _subs("https://acme.example/terms", provider)).reason
    assert "only as a claimed statement" in _verdict(library, req, "subprocessors", _subs("https://acme.example/terms", LIST, "claimed")).reason


def test_an_empty_list_stays_unknown(library):
    assert _verdict(library, _req("AI-3.4", require_list_page=True), "subprocessors", _subs("https://acme.example/terms", [])).verdict == "UNKNOWN"


# ----------------------------------------------------------------------------------------------------------------- DP-2.2
@pytest.mark.parametrize("display", [
    "A DPA is signed when required", "We sign a DPA if required", "A DPA is available upon request", "DPA available on request",
    "A DPA can be requested from our team", "DPA may be requested", "A DPA is provided where applicable", "Provided on demand",
    "A DPA is available if needed", "We sign DPAs when requested",
])
def test_a_conditional_dpa_is_a_review_not_a_pass(library, display):
    r = _verdict(library, _req("DP-2.2", conditional_review=True), "dpa_available", _dpa(display))
    assert r.verdict == "NEEDS_REVIEW" and "conditional" in r.reason and r.vendor_question_id == "Q-DP-2.2"


@pytest.mark.parametrize("display", [
    "Lorikeet signs Data Processing Agreements.", "Freshworks publishes a Data Processing Agreement for customers",
    "A DPA is included in the terms for every customer", "Under GDPR Article 28 a DPA is required, and Acme publishes one",
    "The DPA, as required by law, is part of the contract", "A Data Processing Addendum is published.",
])
def test_an_unconditional_dpa_still_passes(library, display):
    assert _verdict(library, _req("DP-2.2", conditional_review=True), "dpa_available", _dpa(display)).verdict == "PASS"


def test_the_conditional_guard_is_off_without_its_parameter_and_leaves_other_values_alone(library):
    assert _verdict(library, _req("DP-2.2"), "dpa_available", _dpa("A DPA is signed when required")).verdict == "PASS"
    req = _req("DP-2.2", conditional_review=True)
    assert _verdict(library, req, "dpa_available", _dpa("A DPA is available on request", "on_request")).verdict == "NEEDS_REVIEW"
    assert _verdict(library, req, "dpa_available", _dpa("We do not offer a DPA", "no")).verdict == "FAIL"


def test_a_refusal_that_mentions_requests_is_still_a_fail(library):
    """The guard only demotes a PASS; "no DPA, not even on request" is a `no` and stays a FAIL."""
    assert _verdict(library, _req("DP-2.2", conditional_review=True), "dpa_available", _dpa("No DPA is offered, not even on request", "no")).verdict == "FAIL"


def test_the_shipped_requirements_switch_both_guards_on():
    assert next(r for r in load_requirements() if r.id == "AI-3.4").params.get("require_list_page") is True
    assert next(r for r in load_requirements() if r.id == "DP-2.2").params.get("conditional_review") is True
