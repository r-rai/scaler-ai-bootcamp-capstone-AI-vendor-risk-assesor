"""Every refused or failed fetch must be visible in the trace, official-site probes included.

Found by the OpenAI live check (2026-11-20): 8 Firecrawl credits were counted but only 4 fetches were traced, because a failed
probe (a guessed official URL such as /security) was dropped silently. A paid call that leaves no trace cannot be audited.
A probe failure is its own stage, so `fetch_failed` keeps meaning "a page a search returned could not be read".
"""
import pytest

from assessor import fetching
from assessor.fetching import FetchError
from assessor.tracing import Tracer
from assessor.vendor_research import VendorResearcher, _Run

URL = "https://trust.acme-ai.example/security"


def _failing_fetch(message):
    def fetch(url, **kw):
        raise FetchError(message)

    return fetch


def _try(monkeypatch, message, *, probe):
    monkeypatch.setattr(fetching, "fetch", _failing_fetch(message))
    r = VendorResearcher(tracer=Tracer("probe-test"))
    page = r._fetch_page({"url": URL, "title": ""}, "Acme AI", "acme-ai.example", _Run(), probe=probe)
    return page, [e for e in r.tracer.events if e["stage"] in ("probe_failed", "fetch_failed")]


@pytest.mark.parametrize("message", ["HTTP 404", "HTTP 403 (firecrawl: the site still answered HTTP 404)"])
def test_a_failed_probe_is_traced_with_its_error(monkeypatch, message):
    page, events = _try(monkeypatch, message, probe=True)
    assert page is None
    assert [(e["stage"], e["url"], e["error"]) for e in events] == [("probe_failed", URL, message)]


def test_a_failed_search_hit_is_still_a_fetch_failed_and_not_a_probe_failure(monkeypatch):
    page, events = _try(monkeypatch, "HTTP 403", probe=False)
    assert page is None
    assert [(e["stage"], e["url"]) for e in events] == [("fetch_failed", URL)]
