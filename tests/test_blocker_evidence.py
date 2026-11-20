"""`decisive_levels` on a requirement: only evidence of those levels may turn a rule into PASS or FAIL.

Principle (the same one 842bbc6 applied to AI-2.1, extended): a blocker must not be decided by evidence weaker than the
requirement asks for. Weaker evidence gives NEEDS_REVIEW, never a guess in either direction. The mechanism lives in
Matcher.evaluate so it applies to every rule. Whether a requirement turns it on is a data decision in requirements.yaml.

Synthetic facts only; nothing here is keyed on a gold-set vendor. The requirements are built explicitly so these tests
do not depend on which requirements the shipped file has switched on.
"""
import pytest

from assessor.matcher import Matcher, load_requirements
from assessor.schemas import AssessmentRequest
from tests.conftest import fact, profile

REGIONS_IN = {"available": ["US", "India"], "planned": []}
REGIONS_OUT = {"available": ["US", "Germany"], "planned": []}


def _req(rid, decisive=None):
    base = next(r for r in load_requirements() if r.id == rid)
    params = {k: v for k, v in base.params.items() if k != "decisive_levels"}
    if decisive is not None:
        params["decisive_levels"] = decisive
    return base.model_copy(update={"params": params})


def _result(library, requirement, **facts):
    request = AssessmentRequest(question="q", vendor="TestVendor", tier="T3", users=40, user_groups=["public", "employees", "security"])
    (res,) = Matcher(library, requirements=[requirement]).evaluate(profile(**facts), request)
    return res


@pytest.mark.parametrize("regions, what", [(REGIONS_IN, "PASS"), (REGIONS_OUT, "FAIL")])
def test_a_claimed_fact_cannot_decide_a_blocker_that_asks_for_documented_evidence(library, regions, what):
    r = _result(library, _req("DP-3.1", ["documented"]), data_regions=fact("data_regions", regions, "claimed"))
    assert r.verdict == "NEEDS_REVIEW", f"a claimed fact must not give {what}"
    assert "Only claimed evidence" in r.reason and "documented" in r.reason and r.vendor_question_id


@pytest.mark.parametrize("regions, expected", [(REGIONS_IN, "PASS"), (REGIONS_OUT, "FAIL")])
def test_documented_evidence_still_decides(library, regions, expected):
    r = _result(library, _req("DP-3.1", ["documented"]), data_regions=fact("data_regions", regions, "documented"))
    assert r.verdict == expected


@pytest.mark.parametrize("regions, expected", [(REGIONS_IN, "PASS"), (REGIONS_OUT, "FAIL")])
def test_without_the_parameter_nothing_changes(library, regions, expected):
    r = _result(library, _req("DP-3.1"), data_regions=fact("data_regions", regions, "claimed"))
    assert r.verdict == expected


def test_not_established_stays_not_established_and_is_never_turned_into_a_review_by_the_demotion(library):
    req = _req("DP-3.1", ["documented"])
    assert _result(library, req).verdict == "UNKNOWN"                                              # no fact at all
    assert _result(library, req, data_regions=fact("data_regions", {"available": [], "planned": []}, "claimed")).verdict == "UNKNOWN"


def test_a_demoted_blocker_is_a_conditional_not_a_block(library):
    request = AssessmentRequest(question="q", vendor="TestVendor", tier="T3", users=40, user_groups=["public", "employees", "security"])
    matcher = Matcher(library, requirements=[_req("DP-3.1", ["documented"])])
    results = matcher.evaluate(profile(data_regions=fact("data_regions", REGIONS_OUT, "claimed")), request)
    assert matcher.overall(results)[0] == "CONDITIONAL"


def test_the_mechanism_is_not_specific_to_the_region_rule(library):
    """Same switch on the incident rule: an empty incident list on claimed evidence must not PASS a blocker."""
    claimed = fact("security_incidents", [], "claimed")
    assert _result(library, _req("SEC-9.3"), security_incidents=claimed).verdict == "PASS"
    r = _result(library, _req("SEC-9.3", ["documented"]), security_incidents=claimed)
    assert r.verdict == "NEEDS_REVIEW" and "Only claimed evidence" in r.reason
