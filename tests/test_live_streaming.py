"""The streaming live research loop, its deadline, and the guards against a stalled network.

These run against a stubbed web with real threads and short real timeouts (a few seconds in total): the property
under test is timing, so it cannot be faked with a clock.
"""
import threading
import time

import pytest

from assessor import fetching, rag_core
from assessor.config import SETTINGS
from assessor.schemas import Claim, VendorProfile
from assessor.tracing import Budget, Tracer
from assessor.vendor_research import VendorResearcher, _Run

DOM = "streamco.example"
SECURITY, PRICING, BLANK = f"https://{DOM}/security", f"https://{DOM}/pricing", f"https://{DOM}/blank"
TEXT = {
    SECURITY: "StreamCo holds ISO 27001 certification for its platform and publishes its audit scope. " * 3,
    PRICING: "Single sign-on with SAML is included on the Business plan and above for all customers. " * 3,
    BLANK: "",                                           # a page with no readable text: triage records a drop for it
}
CLAIMS = {
    SECURITY: dict(field="security_certifications", evidence_level="documented", display="ISO 27001",
                   value=[{"name": "ISO 27001", "status": "certified", "date": "2026-09-01"}]),
    PRICING: dict(field="sso_saml", evidence_level="documented", display="paid plans", value="paid_plans"),
}


class Web:
    def __init__(self, slow_fetch: dict[str, float] | None = None, slow_search_s: float = 0.0):
        self.slow_fetch = slow_fetch or {}
        self.slow_search_s = slow_search_s
        self.log: list[tuple[str, str]] = []          # (what, detail) in the order it happened

    def search(self, query, limit, *, refresh=False, budget=None, run_id=""):
        if budget is not None:
            budget.charge_tool("search", "stub")
        if "SSO" in query and self.slow_search_s:
            time.sleep(self.slow_search_s)             # a slow engine
            self.log.append(("search_done", query))
            return []
        urls = [SECURITY, BLANK] if "SOC 2" in query else [PRICING] if "SSO" in query else []
        return [{"url": u, "title": u, "snippet": "StreamCo", "rank": i, "query": query, "provider": "stub",
                 "retrieved_at": "2026-11-01T00:00:00+00:00"} for i, u in enumerate(urls, 1)]

    def fetch(self, url, *, refresh=False, vendor="", run_id=""):
        fetching.check_url(url, resolve=False)
        if url in self.slow_fetch:
            time.sleep(self.slow_fetch[url])
        if url not in TEXT:
            raise fetching.FetchError("HTTP 404")
        return {"url": url, "final_url": url, "text": TEXT[url], "hidden_text": "", "title": "StreamCo",
                "published_date": "2026-09-01", "fetched_at": "2026-11-01T00:00:00+00:00", "from_cache": False}


def build(monkeypatch, tmp_path, web: Web, max_seconds: float):
    monkeypatch.setattr(SETTINGS, "search_provider", "ddgs")
    monkeypatch.setattr(fetching, "search", web.search)
    monkeypatch.setattr(fetching, "fetch", web.fetch)
    tracer = Tracer("t", root=tmp_path)

    class R(VendorResearcher):
        def extract_claims(self, page):
            web.log.append(("extract", page["url"]))
            claim = CLAIMS.get(page["url"])
            return [Claim.model_validate(claim)] if claim else []

    r = R(budget=Budget(max_seconds=max_seconds), tracer=tracer)
    r.memory.path = tmp_path / "memory.json"
    r.memory.data = r.memory._read()
    return r, tracer


def test_a_page_is_read_while_a_slow_search_is_still_running(monkeypatch, tmp_path):
    """Streaming: the security page must be extracted before the slow SSO search returns, not after all searches."""
    web = Web(slow_search_s=2.5)
    r, tracer = build(monkeypatch, tmp_path, web, max_seconds=0)
    try:
        r.research("StreamCo", DOM)
    finally:
        tracer.close()
    order = [w for w, d in web.log if (w == "extract" and d == SECURITY) or w == "search_done"]
    assert order[0] == "extract", f"the page was read only after the slow search finished: {web.log}"


def test_deadline_keeps_what_finished_and_drops_what_did_not(monkeypatch, tmp_path):
    """A hung download must not cost the whole run: facts from pages already read survive, the late page never
    enters the evidence, and the report says research stopped early."""
    web = Web(slow_fetch={PRICING: 9.0})
    r, tracer = build(monkeypatch, tmp_path, web, max_seconds=5.0)
    started = time.monotonic()
    try:
        prof = r.research("StreamCo", DOM, refresh=True)
    finally:
        tracer.close()
    elapsed = time.monotonic() - started
    assert elapsed < 8.5, f"research ran {elapsed:.1f}s: the deadline did not cut off the hung download"
    assert "security_certifications" in prof.facts                     # read before the deadline: kept
    assert "sso_saml" not in prof.facts                                # still downloading at the deadline: not used
    assert not any(w == "extract" and d == PRICING for w, d in web.log)
    assert any("stopped early" in n and "time limit" in n for n in prof.notes)
    assert [e for e in tracer.events if e["stage"] == "deadline"]


def test_a_page_that_arrives_after_the_deadline_cannot_touch_the_run(monkeypatch, tmp_path):
    """`run.closed` exists so that work still in flight when the deadline passes cannot mutate the run (pages seen, drops,
    rejected/quarantined lists) while the report is being built. A late page with no text is the observable case:
    triage would record a drop for it."""
    web = Web(slow_fetch={BLANK: 6.0})
    r, tracer = build(monkeypatch, tmp_path, web, max_seconds=4.0)
    try:
        profile = VendorProfile(vendor="StreamCo", official_domain=DOM, as_of=r.as_of)
        run = _Run(refresh=True)
        r.live_iteration(r.make_plan("StreamCo", DOM).sub_questions, "StreamCo", DOM, 1, profile, run)
        before = (sorted(run.pages), list(run.page_drops), list(profile.rejected), list(profile.quarantined))
        assert run.closed
        time.sleep(3.5)                                                # the hung download now completes inside its thread
    finally:
        tracer.close()
    after = (sorted(run.pages), list(run.page_drops), list(profile.rejected), list(profile.quarantined))
    assert after == before, "a late page changed the run after the deadline"
    assert BLANK not in run.pages


# ---- a hung search call -------------------------------------------------------------------------------------

def test_call_with_deadline_returns_raises_and_times_out():
    assert fetching._call_with_deadline(lambda a, b: a + b, 1.0, 1, 2) == 3
    with pytest.raises(ValueError):
        fetching._call_with_deadline(lambda: (_ for _ in ()).throw(ValueError("boom")), 1.0)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        fetching._call_with_deadline(lambda: time.sleep(5), 0.3)
    assert time.monotonic() - started < 1.5


def test_a_hung_search_provider_falls_over_to_the_next_one(monkeypatch):
    monkeypatch.setattr(fetching, "SEARCH_HARD_CAP_S", 0.3)
    monkeypatch.setattr(fetching, "search_order", lambda: ["ddgs", "tavily"])
    monkeypatch.setattr(fetching, "_PROVIDERS", {
        "ddgs": lambda q, n: time.sleep(5) or [],
        "tavily": lambda q, n: [{"url": "https://x.example/a", "title": "A", "snippet": "s"}],
    })
    started = time.monotonic()
    hits = fetching.search("anything", 3, refresh=True)
    assert hits and hits[0]["provider"] == "tavily"
    assert time.monotonic() - started < 3.0


# ---- a stalled model download -------------------------------------------------------------------------------

class FakeModel:
    """Cached: loads with local_files_only. Not cached: 'downloads' for as long as the test's event is unset."""
    stall = threading.Event()
    cached = False
    downloads = 0

    def __init__(self, **kwargs):
        if kwargs.get("local_files_only"):
            if not FakeModel.cached:
                raise ValueError("not in the local cache")
            return
        FakeModel.downloads += 1
        FakeModel.stall.wait(10)


@pytest.fixture()
def fake_model(monkeypatch):
    FakeModel.stall = threading.Event()
    FakeModel.cached, FakeModel.downloads = False, 0
    monkeypatch.setattr(rag_core, "_LOAD_FAILED", {})
    monkeypatch.setattr(SETTINGS, "model_download", True)
    monkeypatch.setattr(SETTINGS, "model_load_timeout_s", 0.4)
    yield FakeModel
    FakeModel.stall.set()


def test_a_stalled_download_raises_instead_of_hanging(fake_model):
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        rag_core.load_fastembed(fake_model, model_name="m")
    assert time.monotonic() - started < 2.0


def test_a_failed_download_is_not_retried_straight_away(fake_model):
    with pytest.raises(TimeoutError):
        rag_core.load_fastembed(fake_model, model_name="m")
    started = time.monotonic()
    with pytest.raises(RuntimeError):
        rag_core.load_fastembed(fake_model, model_name="m")
    assert time.monotonic() - started < 0.2 and fake_model.downloads == 1


def test_cached_models_load_without_any_download(fake_model):
    fake_model.cached = True
    assert isinstance(rag_core.load_fastembed(fake_model, model_name="m"), FakeModel)
    assert fake_model.downloads == 0


def test_downloads_can_be_disabled(fake_model, monkeypatch):
    monkeypatch.setattr(SETTINGS, "model_download", False)
    with pytest.raises(RuntimeError):
        rag_core.load_fastembed(fake_model, model_name="m")
    assert fake_model.downloads == 0
