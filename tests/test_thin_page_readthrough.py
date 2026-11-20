"""Retrieval fix 2026-11-20 (reports/retrieval_fix_20261120/preregistration.md).

1. A guessed-path probe never spends a Firecrawl credit, even when the site refuses the bot.
2. A direct download that SUCCEEDED but is a near-empty shell (JS-only page) on the vendor's own domain is re-read through
   Firecrawl once; the longer text wins; any failure keeps the original page.
Nothing touches the network: the direct download and httpx.post are faked.
"""
import socket

import httpx
import pytest

from assessor import costs, fetching
from assessor.config import SETTINGS
from assessor.fetching import FetchError
from assessor.tracing import Budget
from tests.test_firecrawl_fallback import API, DOMAIN, KEY, URL, FakeFirecrawl, _payload

SHELL = b"<html><head><title>Trust</title></head><body><div id='root'>Loading</div></body></html>"
FULL_HTML = "<html><body><p>" + "We hold a SOC 2 Type II report dated 2026-01-10. " * 20 + "</p></body></html>"
LONG_PAGE = b"<html><body><p>" + b"Real security page text. " * 40 + b"</p></body></html>"


@pytest.fixture
def fc(monkeypatch):
    monkeypatch.setattr(SETTINGS, "firecrawl_api_key", KEY)
    monkeypatch.setattr(SETTINGS, "firecrawl_enabled", True)
    monkeypatch.setattr(SETTINGS, "fetcher_url", "")
    monkeypatch.setattr(costs, "firecrawl_credits_this_month", lambda: 0)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(fetching, "_download", lambda url: (SHELL, "text/html", url))
    fake = FakeFirecrawl(payload=_payload(FULL_HTML))
    monkeypatch.setattr(httpx, "post", fake)
    return fake


def test_a_probe_refused_by_the_site_never_calls_firecrawl(fc, monkeypatch):
    def refused(url):
        raise FetchError("HTTP 403", status=403)
    monkeypatch.setattr(fetching, "_download", refused)
    with pytest.raises(FetchError):
        fetching.fetch(URL, official_domain=DOMAIN, budget=Budget(), probe=True)
    assert fc.calls == []


def test_a_non_probe_refusal_still_uses_firecrawl(fc, monkeypatch):
    def refused(url):
        raise FetchError("HTTP 403", status=403)
    monkeypatch.setattr(fetching, "_download", refused)
    assert fetching.fetch(URL, official_domain=DOMAIN, budget=Budget())["via"] == "firecrawl"


def test_a_thin_official_page_is_read_through_firecrawl_once(fc):
    budget = Budget()
    page = fetching.fetch(URL, official_domain=DOMAIN, budget=budget)
    assert page["via"] == "firecrawl" and "SOC 2 Type II" in page["text"]
    assert len(fc.calls) == 1 and budget.tools["fetch:firecrawl"] == 1


def test_a_probe_that_returns_a_thin_200_is_not_read_through_firecrawl(fc):
    page = fetching.fetch(URL, official_domain=DOMAIN, budget=Budget(), probe=True)
    assert fc.calls == [] and "via" not in page


@pytest.mark.parametrize("why, arrange", [
    ("other domain", lambda m, fc: None),
    ("long page", lambda m, fc: m.setattr(fetching, "_download", lambda u: (LONG_PAGE, "text/html", u))),
    ("pdf", lambda m, fc: (m.setattr(fetching, "_download", lambda u: (b"%PDF-1.4", "application/pdf", u)),
                           m.setattr(fetching, "_parse", lambda *a: {"text": "x", "hidden_text": "", "title": "t", "published_date": None}))),
    ("no key", lambda m, fc: m.setattr(SETTINGS, "firecrawl_api_key", "")),
    ("sandbox", lambda m, fc: m.setattr(SETTINGS, "fetcher_url", "http://fetcher:9000")),
])
def test_no_call_when_the_thin_page_rule_does_not_apply(fc, monkeypatch, why, arrange):
    arrange(monkeypatch, fc)
    url = "https://some-blog.example/post" if why == "other domain" else URL
    if why == "sandbox":
        monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(500, json={}, request=httpx.Request("POST", "http://x")))
        with pytest.raises(FetchError):
            fetching.fetch(url, official_domain=DOMAIN, budget=Budget())
        return
    page = fetching.fetch(url, official_domain=DOMAIN, budget=Budget())
    assert fc.calls == [] and page.get("via") != "firecrawl", why


def test_cap_reached_or_no_budget_keeps_the_original_page(fc):
    b = Budget()
    b.tools.update({"fetch:firecrawl": SETTINGS.firecrawl_max_per_run})
    page = fetching.fetch(URL, official_domain=DOMAIN, budget=b)
    assert fc.calls == [] and "via" not in page
    assert "via" not in fetching.fetch(URL, official_domain=DOMAIN, budget=None)


def test_a_firecrawl_failure_or_shorter_answer_keeps_the_original_page(fc, monkeypatch):
    monkeypatch.setattr(httpx, "post", FakeFirecrawl(status=500, payload={}))
    page = fetching.fetch(URL, official_domain=DOMAIN, budget=Budget())
    assert "via" not in page and "Loading" in page["text"]
    monkeypatch.setattr(httpx, "post", FakeFirecrawl(payload=_payload("<html><body>x</body></html>")))
    assert "via" not in fetching.fetch(URL, official_domain=DOMAIN, budget=Budget(), refresh=True)
