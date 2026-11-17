"""Regression for the live finding: the extractor returned 'IN' / 'India (AWS Mumbai, ap-south-1)' and the matcher
compared exact strings, so DP-3.1 FAILed vendors that do store data in India (found by eval/run_live_real.py)."""
import pytest

from assessor.matcher import RULES, _region_hit, load_requirements
from assessor.schemas import VendorFact


@pytest.mark.parametrize("value,hit", [
    ("India", True), ("india", True), ("IN", True), ("in", True),
    ("India (AWS Mumbai, ap-south-1)", True), ("Mumbai, India", True), ("Asia-Pacific (India)", True),
    ("Indiana", False), ("Indianapolis", False), ("US", False), ("EU", False), ("United States", False),
    ("Singapore", False), ("", False),
])
def test_region_hit(value, hit):
    assert _region_hit("India", value) is hit


def _verdict(available, planned=()):
    req = next(r for r in load_requirements() if r.id == "DP-3.1")
    fact = VendorFact(field="data_regions", value={"available": list(available), "planned": list(planned)},
                      display=", ".join(available), confidence=0.9, evidence_level="documented")
    return RULES[req.rule](req, fact, {})[0]


def test_dp31_pass_for_code_and_description():
    assert _verdict(["US", "EU", "IN"]) == "PASS"
    assert _verdict(["India (AWS Mumbai, ap-south-1)"]) == "PASS"


def test_dp31_still_fails_without_india_and_when_only_planned():
    assert _verdict(["US", "EU"]) == "FAIL"
    assert _verdict(["Indiana, US"]) == "FAIL"
    assert _verdict(["US"], planned=["India"]) == "FAIL"      # planned does not count (allow_planned: false)
