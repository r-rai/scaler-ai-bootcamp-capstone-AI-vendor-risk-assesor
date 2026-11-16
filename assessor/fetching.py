"""
fetching.py — how the agent touches the outside world: web search and page download.

Everything that comes back from here is UNTRUSTED. This module's job is to make fetching safe for the
machine it runs on; deciding whether the content can be believed is vendor_research.py's job.

Safety rules for a download (they apply natively and inside the sandboxed fetcher service):
  * http/https only, default ports only
  * the host must resolve to public addresses: loopback, private ranges, link-local (cloud metadata at
    169.254.169.254) and other reserved ranges are refused. This is the SSRF guard: a search result or a
    redirect must never make the server call its own network.
  * every redirect hop is validated again; at most 5 hops
  * response size is capped while streaming; only text, HTML and PDF content types are read
  * a short timeout, and a pause between requests to the same host

Where the download runs:
  * VRA_FETCHER_URL set  → the sandboxed `fetcher` container downloads and parses the page. That container has
    no API keys and no data volumes, so a parser exploit or a malicious page has nothing to steal.
  * otherwise            → in this process, with the same guards.

Search: Tavily and DuckDuckGo, with the other one as fallback when the first is rate-limited or down.
Results and pages are cached (assessor/cache.py); a re-review (`refresh=True`) bypasses the cache.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urljoin, urlsplit

from .config import SETTINGS

USER_AGENT = "vendor-risk-assessor/2.0 (security due-diligence research)"
ALLOWED_PORTS = {None, 80, 443}
ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/pdf")
MAX_REDIRECTS = 5
MAX_TEXT_CHARS = 20_000
HOST_MIN_INTERVAL_S = 0.25
SEARCH_TIMEOUT_S = 10

_host_last: dict[str, float] = {}
_host_lock = threading.Lock()


class UnsafeURL(ValueError):
    """The URL must not be fetched (SSRF guard)."""


class FetchError(RuntimeError):
    pass


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def check_url(url: str, *, resolve: bool = True) -> str:
    """Return the URL if it is safe to request, else raise UnsafeURL with the reason."""
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise UnsafeURL(f"malformed URL: {exc}") from None
    if parts.scheme not in ("http", "https"):
        raise UnsafeURL(f"scheme {parts.scheme or '(none)'!r} is not allowed")
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeURL("no host")
    if parts.username or parts.password:
        raise UnsafeURL("credentials in URL are not allowed")
    try:
        port = parts.port
    except ValueError:
        raise UnsafeURL("invalid port") from None
    if port not in ALLOWED_PORTS:
        raise UnsafeURL(f"port {port} is not allowed")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home.arpa")):
        raise UnsafeURL(f"host {host!r} is internal")
    addrs: list[str] = []
    try:
        addrs = [str(ipaddress.ip_address(host))]          # an IP literal (also catches decimal / hex forms via getaddrinfo below)
    except ValueError:
        if resolve:
            try:
                addrs = sorted({ai[4][0] for ai in socket.getaddrinfo(host, port or (443 if parts.scheme == "https" else 80),
                                                                      proto=socket.IPPROTO_TCP)})
            except socket.gaierror:
                raise UnsafeURL(f"host {host!r} does not resolve") from None
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if getattr(ip, "ipv4_mapped", None):
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise UnsafeURL(f"host {host!r} resolves to a non-public address ({ip})")
    return url


def _pace(host: str) -> None:
    """Be polite: space out requests to the same host. Each caller reserves the next free slot."""
    with _host_lock:
        now = time.monotonic()
        slot = max(now, _host_last.get(host, 0.0) + HOST_MIN_INTERVAL_S)
        _host_last[host] = slot
    if slot > now:
        time.sleep(min(slot - now, 3.0))


def _download(url: str) -> tuple[bytes, str, str]:
    """GET with manual, validated redirects and a streaming size cap. Returns (body, content_type, final_url)."""
    import httpx

    current = url
    started = time.monotonic()
    timeout = httpx.Timeout(SETTINGS.fetch_timeout_s, connect=5.0)
    with httpx.Client(follow_redirects=False, timeout=timeout, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf;q=0.9,text/plain;q=0.8"}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            check_url(current)
            _pace(host_of(current))
            with client.stream("GET", current) as r:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    current = urljoin(current, r.headers["location"])
                    continue
                if r.status_code >= 400:
                    raise FetchError(f"HTTP {r.status_code}")
                ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and not ctype.startswith(ALLOWED_TYPES):
                    raise FetchError(f"content type {ctype!r} is not read")
                declared = r.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > SETTINGS.fetch_max_bytes:
                    raise FetchError(f"response larger than {SETTINGS.fetch_max_bytes} bytes")      # fail before downloading
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > SETTINGS.fetch_max_bytes:
                        raise FetchError(f"response larger than {SETTINGS.fetch_max_bytes} bytes")
                    if time.monotonic() - started > SETTINGS.fetch_timeout_s * 2:
                        raise FetchError("download took too long")                                   # slow-drip responses
                return bytes(buf), ctype, current
    raise FetchError("too many redirects")


def _parse(body: bytes, ctype: str, url: str) -> dict[str, Any]:
    """Main text, title and publication date. Parsing untrusted bytes is the risky step, which is why the
    fetcher container exists."""
    title, published, text = url, None, ""
    if ctype == "application/pdf" or body[:5] == b"%PDF-":
        try:
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(body))
            text = "\n".join((page.extract_text() or "") for page in reader.pages[:40])
            meta = reader.metadata or {}
            title = str(meta.get("/Title") or url)
            raw = str(meta.get("/ModDate") or meta.get("/CreationDate") or "")
            m = re.match(r"D:(\d{4})(\d{2})(\d{2})", raw)
            published = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
        except ImportError:
            raise FetchError("PDF found but pypdf is not installed (pip install pypdf)") from None
        except Exception as exc:  # noqa: BLE001 — a malformed PDF is just an unreadable page
            raise FetchError(f"unreadable PDF: {type(exc).__name__}") from None
    else:
        html, hidden = _split_hidden(body.decode("utf-8", errors="replace"))
        try:
            import trafilatura

            text = trafilatura.extract(html, include_comments=False) or ""
            meta = trafilatura.extract_metadata(html)
            if meta:
                title = getattr(meta, "title", None) or url
                published = getattr(meta, "date", None)
        except ImportError:
            pass
        if not text:
            stripped = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", stripped)).strip()
        return {"text": text[:MAX_TEXT_CHARS], "hidden_text": hidden[:4000], "title": str(title)[:300], "published_date": _iso_date(published)}
    return {"text": text[:MAX_TEXT_CHARS], "hidden_text": "", "title": str(title)[:300], "published_date": _iso_date(published)}


_HIDDEN_XPATH = (
    "//*[contains(translate(@style, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ ', 'abcdefghijklmnopqrstuvwxyz'), 'display:none')"
    " or contains(translate(@style, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ ', 'abcdefghijklmnopqrstuvwxyz'), 'visibility:hidden')"
    " or contains(translate(@style, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ ', 'abcdefghijklmnopqrstuvwxyz'), 'font-size:0')"
    " or contains(translate(@style, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ ', 'abcdefghijklmnopqrstuvwxyz'), 'opacity:0')"
    " or @hidden or @aria-hidden='true']"
)


def _split_hidden(html: str) -> tuple[str, str]:
    """Separate what a browser shows from what it hides (HTML comments, display:none, aria-hidden).

    Hidden text is where instructions aimed at machines are planted. It is removed from the page before the
    main text is extracted, so the extractor never sees it as content, and returned on its own so the
    injection scanner still does.
    """
    try:
        import lxml.html
        from lxml import etree

        root = lxml.html.fromstring(html)
        hidden: list[str] = []
        for node in root.xpath("//comment()"):
            hidden.append(node.text or "")
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)
        for node in root.xpath(_HIDDEN_XPATH):
            hidden.append(node.text_content() or "")
            node.drop_tree()
        return etree.tostring(root, encoding="unicode", method="html"), re.sub(r"\s+", " ", " ".join(hidden)).strip()
    except Exception:  # noqa: BLE001 — lxml missing or unparseable markup: fall back to patterns
        hidden_text = " ".join(re.findall(r"(?s)<!--(.*?)-->", html))
        hidden_text += " " + " ".join(re.findall(
            r"(?is)<[^>]+(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|\shidden[\s=>])[^>]*>(.*?)</", html))
        visible = re.sub(r"(?s)<!--.*?-->", " ", html)
        return visible, re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", hidden_text)).strip()


def _iso_date(value: Any) -> Optional[str]:
    if not value:
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value))
    if not m:
        return None
    try:
        datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    return m.group(0)


def safe_fetch(url: str) -> dict[str, Any]:
    """Download and parse one page in this process. Raises UnsafeURL or FetchError."""
    body, ctype, final_url = _download(url)
    page = _parse(body, ctype, final_url)
    page.update({"url": url, "final_url": final_url, "content_type": ctype, "sha256": hashlib.sha256(body).hexdigest(),
                 "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return page


def _clean_remote_page(data: Any, url: str) -> dict[str, Any]:
    """The fetcher's answer crosses a trust boundary too: keep only the expected fields, typed and size-capped."""
    if not isinstance(data, dict):
        raise FetchError("fetcher returned an unexpected payload")
    return {
        "url": url,
        "final_url": str(data.get("final_url") or url)[:2000],
        "text": str(data.get("text") or "")[:MAX_TEXT_CHARS],
        "hidden_text": str(data.get("hidden_text") or "")[:4000],
        "title": str(data.get("title") or url)[:300],
        "published_date": _iso_date(data.get("published_date")),
        "content_type": str(data.get("content_type") or "")[:80],
        "sha256": str(data.get("sha256") or "")[:64],
        "fetched_at": str(data.get("fetched_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"))[:40],
    }


def fetch(url: str, *, refresh: bool = False, vendor: str = "", run_id: str = "") -> dict[str, Any]:
    """Cache → sandboxed fetcher (if configured) → local guarded download. Adds `from_cache`."""
    from .cache import get_cache

    cache = get_cache()
    if not refresh:
        hit = cache.get("page", url, run_id=run_id)
        if hit:
            return {**hit, "from_cache": True}
    check_url(url, resolve=not SETTINGS.fetcher_url)     # with a fetcher, DNS is resolved inside the sandbox
    if SETTINGS.fetcher_url:
        import httpx

        try:
            r = httpx.post(SETTINGS.fetcher_url.rstrip("/") + "/fetch", json={"url": url}, timeout=SETTINGS.fetch_timeout_s + 10)
        except httpx.HTTPError as exc:
            raise FetchError(f"fetcher unreachable: {type(exc).__name__}") from None
        if r.status_code == 422:
            raise UnsafeURL(str(r.json().get("detail", "refused by fetcher"))[:200])
        if r.status_code != 200:
            raise FetchError(str(r.json().get("detail", f"fetcher HTTP {r.status_code}"))[:200])
        page = _clean_remote_page(r.json(), url)
    else:
        page = safe_fetch(url)
    if page["text"].strip():
        cache.put("page", url, page, ttl=SETTINGS.page_cache_ttl_s, tag=vendor)
    return {**page, "from_cache": False}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _search_ddgs(query: str, limit: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("pip install ddgs") from exc
    try:
        client = DDGS(timeout=SEARCH_TIMEOUT_S)
    except TypeError:          # older ddgs without the timeout argument
        client = DDGS()
    return [{"url": r["href"], "title": r.get("title", ""), "snippet": (r.get("body") or "")[:400]}
            for r in client.text(query, max_results=limit) if r.get("href")]


def _search_tavily(query: str, limit: int) -> list[dict]:
    if not SETTINGS.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise RuntimeError("pip install tavily-python") from exc
    res = TavilyClient(api_key=SETTINGS.tavily_api_key).search(query, max_results=limit)
    return [{"url": r["url"], "title": r.get("title", ""), "snippet": (r.get("content") or "")[:400],
             "published_date": _iso_date(r.get("published_date"))} for r in res.get("results", []) if r.get("url")]


_PROVIDERS = {"ddgs": _search_ddgs, "tavily": _search_tavily}


def search_order() -> list[str]:
    """The configured provider first, then the other one as a fallback if it is usable."""
    first = SETTINGS.search_provider
    order = [first] if first in _PROVIDERS else []
    for other in ("tavily", "ddgs"):
        if other not in order and (other != "tavily" or SETTINGS.tavily_api_key):
            order.append(other)
    return order


def search(query: str, limit: int, *, refresh: bool = False, budget: Any = None, run_id: str = "") -> list[dict]:
    """Typed hits: {url, title, snippet, rank, query, provider, retrieved_at}. Raises RuntimeError if every provider fails."""
    from .cache import get_cache, sha

    cache = get_cache()
    key = sha(SETTINGS.search_provider, query, str(limit))
    if not refresh:
        hit = cache.get("search", key, run_id=run_id)
        if hit is not None:
            return hit
    errors: list[str] = []
    for provider in search_order():
        for attempt in range(2):
            if attempt and errors and "Timeout" in errors[-1]:
                break                      # a timeout is not worth a second wait: move to the next provider
            try:
                if budget is not None:
                    budget.charge_tool("search", provider)
                rows = _PROVIDERS[provider](query, limit)
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                hits = [{**r, "rank": i, "query": query, "provider": provider, "retrieved_at": now} for i, r in enumerate(rows, 1)]
                if hits:
                    cache.put("search", key, hits, ttl=SETTINGS.search_cache_ttl_s)
                return hits
            except Exception as exc:  # noqa: BLE001 — rate limit, network, missing package: back off, then try the next provider
                errors.append(f"{provider}: {type(exc).__name__}: {str(exc)[:80]}")
                if attempt == 0:
                    time.sleep(1.0)
    raise RuntimeError("; ".join(errors) or "no search provider is configured")
