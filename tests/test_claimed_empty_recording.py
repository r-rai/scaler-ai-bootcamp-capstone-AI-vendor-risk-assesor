"""SEC-9.3: the research layer records what the incident search covered ALSO on a fact a page produced with an empty list.

Pre-registration: reports/sec93_claimed_20261120/preregistration.md (section 2 "Recording rules", tests R1-R11). Before this, the recorder
returned early whenever a `security_incidents` fact existed, so the results of an incident search that DID run (two searches per vendor in
the final run) were thrown away and a vendor's own "no incidents disclosed" could pass over them. The recorder only RECORDS: the fact's value,
level, confidence, sources and display stay what the page said; the matcher decides (tests/test_claimed_empty_incidents.py).
Synthetic data only. Uses the stubbed web of tests/test_negative_search.py.
"""
import json

import pytest

from assessor import fetching
from assessor.config import SETTINGS
from assessor.matcher import Matcher, load_requirements
from assessor.schemas import AssessmentRequest, VendorFact, VendorProfile
from assessor.tracing import Budget, Tracer
from assessor.vendor_research import VendorResearcher, _Run
from tests import test_research_loop as loop
from tests.test_negative_search import ACME_HITS, IncidentWeb

SECURITY_URL = f"https://{loop.DOM}/security"


@pytest.fixture()
def acme(monkeypatch, tmp_path):
    """The stubbed web of test_negative_search.py (incident queries answered with `hits`), one researcher per call."""
    def build(hits, fail=False, fail_first=False):
        web = IncidentWeb(hits, fail, fail_first)
        monkeypatch.setattr(SETTINGS, "search_provider", "ddgs")
        monkeypatch.setattr(SETTINGS, "pages_per_question", 5)
        monkeypatch.setattr(fetching, "search", web.search)
        monkeypatch.setattr(fetching, "fetch", web.fetch)
        tracer = Tracer("t", root=tmp_path)
        r = loop.LiveResearcher(budget=Budget(max_seconds=0), tracer=tracer)
        r.memory.path = tmp_path / "memory.json"
        r.memory.data = r.memory._read()
        return r, web, tracer
    return build


@pytest.fixture()
def says_no_incidents(monkeypatch):
    """The vendor's own security page says 'No incidents disclosed' (an empty list, claimed)."""
    monkeypatch.setitem(loop.CLAIMS, SECURITY_URL, loop.CLAIMS[SECURITY_URL] + [dict(
        field="security_incidents", evidence_level="claimed", display="No incidents disclosed", value=[])])


def _verdict(library, prof):
    request = AssessmentRequest(question="q", vendor="Acme AI", tier="T3", users=40, user_groups=["public", "employees", "security"])
    req = next(r for r in load_requirements() if r.id == "SEC-9.3")
    (res,) = Matcher(library, requirements=[req]).evaluate(prof, request)
    return res


def _own_memory(researcher, tmp_path, name):
    """The `acme` fixture gives every researcher the same memory file: a second run would skip what the first one flagged as bad."""
    researcher.memory.path = tmp_path / name
    researcher.memory.data = researcher.memory._read()


def _bare_profile(level="claimed", value=None, domain=loop.DOM) -> VendorProfile:
    prof = VendorProfile(vendor="Acme AI", official_domain=domain, as_of="2026-11-01")
    prof.facts["security_incidents"] = VendorFact(field="security_incidents", value=[] if value is None else value, display="No incidents disclosed",
                                                  confidence=0.475, evidence_level=level, sources=[SECURITY_URL])
    return prof


def _run_with(results=ACME_HITS, queries=("Acme AI security incident",), failed=0, official=True) -> _Run:
    run = _Run()
    run.incident_queries.extend(queries)
    run.incident_results.extend(dict(r) for r in results)
    run.incident_failed = failed
    if official:
        run.clean["p1"] = {"id": "p1", "url": SECURITY_URL, "source_type": "official"}
    return run


# ----------------------------------------------------------------------------------------------- R1: the live path attaches it

def test_a_fact_a_page_produced_with_an_empty_list_gets_the_search_coverage(acme, says_no_incidents):
    r, web, tracer = acme(ACME_HITS)
    f = r.research("Acme AI", loop.DOM).facts["security_incidents"]
    assert f.value == [] and f.evidence_level == "claimed" and f.display == "No incidents disclosed" and f.sources == [SECURITY_URL]
    cov = f.coverage
    assert cov is not None and len(cov.queries) == 2 and sorted(cov.queries) == sorted(set(web.incident_queries))
    assert {x["url"] for x in cov.results} == {h["url"] for h in ACME_HITS} and set(cov.results[0]) == {"url", "title", "snippet"}
    assert cov.official_domain == loop.DOM and cov.official_page_read is True and cov.failed_queries == 0


def test_the_record_changes_nothing_the_page_said(acme, says_no_incidents, monkeypatch, tmp_path):
    r, web, tracer = acme(ACME_HITS)
    with_record = r.research("Acme AI", loop.DOM).facts["security_incidents"].model_dump(exclude={"coverage"})
    r2, web2, tracer2 = acme(ACME_HITS)
    _own_memory(r2, tmp_path, "m2.json")
    monkeypatch.setattr(VendorResearcher, "_record_incident_search", lambda self, profile, run, vendor: None)
    without = r2.research("Acme AI", loop.DOM).facts["security_incidents"]
    assert without.coverage is None and with_record == without.model_dump(exclude={"coverage"})


# ----------------------------------------------------------------------------------------------- R2-R5: the cases the `searched` fact refuses

def test_it_is_recorded_when_no_result_names_the_vendor(acme, says_no_incidents, library):
    """The `searched` fact is not recorded then; for an existing page fact that refusal would leave the old hole open."""
    generic = [{"url": f"https://news.example/g{i}", "title": t, "snippet": ""} for i, t in enumerate(["Top breaches of 2026", "How to respond to a breach"])]
    r, web, tracer = acme(generic)
    prof = r.research("Acme AI", loop.DOM)
    assert prof.facts["security_incidents"].coverage is not None
    assert _verdict(library, prof).verdict == "NEEDS_REVIEW"


def test_it_is_recorded_when_every_incident_search_failed(acme, says_no_incidents, library):
    r, web, tracer = acme(ACME_HITS, fail=True)
    prof = r.research("Acme AI", loop.DOM)
    cov = prof.facts["security_incidents"].coverage
    assert cov is not None and cov.queries == [] and cov.failed_queries >= 1
    assert _verdict(library, prof).verdict == "NEEDS_REVIEW"


def test_it_is_recorded_when_the_incident_search_returned_nothing(acme, says_no_incidents, library):
    r, web, tracer = acme([])
    prof = r.research("Acme AI", loop.DOM)
    cov = prof.facts["security_incidents"].coverage
    assert cov is not None and cov.results == []
    assert _verdict(library, prof).verdict == "NEEDS_REVIEW"


def test_it_is_recorded_without_a_confirmed_official_domain(acme, library):
    r, web, tracer = acme(ACME_HITS)
    prof = _bare_profile(domain=None)
    r._record_incident_search(prof, _run_with(official=False), "Acme AI")
    cov = prof.facts["security_incidents"].coverage
    assert cov is not None and cov.official_domain is None and cov.official_page_read is False
    assert _verdict(library, prof).verdict == "NEEDS_REVIEW"


def test_it_is_recorded_when_no_incident_search_ran_at_all(acme, library):
    r, web, tracer = acme(ACME_HITS)
    prof = _bare_profile()
    r._record_incident_search(prof, _run_with(results=[], queries=(), official=True), "Acme AI")
    cov = prof.facts["security_incidents"].coverage
    assert cov is not None and cov.queries == [] and cov.results == []
    assert _verdict(library, prof).verdict == "NEEDS_REVIEW"


# ----------------------------------------------------------------------------------------------- R6-R7: where it is NOT recorded

def test_nothing_is_recorded_on_the_offline_path(acme):
    """Offline there is no search: `coverage is None` is how the matcher knows, and the fictional corpus keeps its behaviour."""
    r, web, tracer = acme(ACME_HITS)
    r.offline_search = True
    prof = _bare_profile()
    r._record_incident_search(prof, _run_with(), "Acme AI")
    assert prof.facts["security_incidents"].coverage is None


def test_a_fact_that_lists_incidents_is_left_alone(acme):
    r, web, tracer = acme(ACME_HITS)
    prof = _bare_profile(level="documented", value=[{"date": "2026-08-01", "summary": "x", "severity": "major", "postmortem_published": False}])
    r._record_incident_search(prof, _run_with(), "Acme AI")
    assert prof.facts["security_incidents"].coverage is None


def test_a_searched_fact_is_not_recorded_twice_and_a_second_call_changes_nothing(acme):
    r, web, tracer = acme(ACME_HITS)
    prof = VendorProfile(vendor="Acme AI", official_domain=loop.DOM, as_of="2026-11-01")
    run = _run_with()
    r._record_incident_search(prof, run, "Acme AI")
    first = prof.facts["security_incidents"].model_dump_json()
    assert prof.facts["security_incidents"].evidence_level == "searched"
    r._record_incident_search(prof, run, "Acme AI")
    assert prof.facts["security_incidents"].model_dump_json() == first
    claimed = _bare_profile()
    r._record_incident_search(claimed, run, "Acme AI")
    once = claimed.facts["security_incidents"].model_dump_json()
    r._record_incident_search(claimed, _run_with(results=[{"url": "https://x.example/1", "title": "Acme hacked", "snippet": ""}]), "Acme AI")
    assert claimed.facts["security_incidents"].model_dump_json() == once      # the first record stands


# ----------------------------------------------------------------------------------------------- R8-R10: memory, cost, audit

def test_vendor_memory_never_sees_the_coverage_on_a_page_fact(acme, says_no_incidents):
    r, web, tracer = acme(ACME_HITS)
    r.research("Acme AI", loop.DOM)
    stored = json.dumps(r.memory.data)
    assert "coverage" not in stored and "Acme AI launches an assistant" not in stored and "Industry breach roundup" not in stored


def test_recording_costs_no_search_and_no_model_call(acme, says_no_incidents, monkeypatch, tmp_path):
    r, web, tracer = acme(ACME_HITS)
    r.research("Acme AI", loop.DOM)
    r2, web2, tracer2 = acme(ACME_HITS)
    _own_memory(r2, tmp_path, "m2.json")
    monkeypatch.setattr(VendorResearcher, "_record_incident_search", lambda self, profile, run, vendor: None)
    r2.research("Acme AI", loop.DOM)
    assert sorted(web.queries) == sorted(web2.queries) and sorted(web.fetched) == sorted(web2.fetched)     # searches run in parallel: order varies
    assert r.budget.tool_calls == r2.budget.tool_calls and r.budget.llm_calls == r2.budget.llm_calls


def test_a_saved_profile_keeps_the_coverage_and_gives_the_same_verdict(acme, says_no_incidents, library):
    r, web, tracer = acme(ACME_HITS)
    prof = r.research("Acme AI", loop.DOM)
    reloaded = VendorProfile.model_validate_json(prof.model_dump_json())
    assert reloaded.facts["security_incidents"].coverage == prof.facts["security_incidents"].coverage
    assert _verdict(library, reloaded).verdict == _verdict(library, prof).verdict


# ----------------------------------------------------------------------------------------------- R11: end to end

def test_end_to_end_a_claimed_no_incidents_passes_over_clean_results_and_not_over_a_hit(acme, says_no_incidents, library):
    r, web, tracer = acme(ACME_HITS)
    clean = _verdict(library, r.research("Acme AI", loop.DOM))
    assert clean.verdict == "PASS" and clean.evidence_level == "claimed" and "search" in clean.reason
    hit = ACME_HITS + [{"url": "https://news.example/x", "title": "Acme AI confirms data breach", "snippet": ""}]
    r2, web2, tracer2 = acme(hit)
    flagged = _verdict(library, r2.research("Acme AI", loop.DOM))
    assert flagged.verdict == "NEEDS_REVIEW" and "Acme AI confirms data breach" in flagged.reason
