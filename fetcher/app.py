"""
fetcher/app.py — the sandbox for untrusted web content.

Downloading and parsing arbitrary HTML and PDF is the riskiest thing this system does: the bytes come from
whoever a search engine returned, and parsers have bugs. This tiny service does only that, in a container
that has NO API keys, NO data volumes, a read-only filesystem, no Linux capabilities and its own network
(see docker-compose.yml). If a page exploits the parser, there is nothing here to steal and nowhere to go.

The API container calls POST /fetch over an internal network and treats the answer as untrusted data again
(assessor/fetching._clean_remote_page). The same SSRF rules apply inside: public addresses only.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Belt and braces: even if a secret were passed to this container by mistake, drop it before handling any page.
for _name in list(os.environ):
    if any(k in _name.upper() for k in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
        os.environ.pop(_name, None)
os.environ.setdefault("VRA_CACHE", "0")
os.environ.pop("VRA_FETCHER_URL", None)       # this IS the fetcher: never forward

from assessor.fetching import FetchError, UnsafeURL, safe_fetch  # noqa: E402

app = FastAPI(title="Vendor Risk Assessor — sandboxed fetcher", docs_url=None, redoc_url=None, openapi_url=None)


class FetchBody(BaseModel):
    url: str = Field(min_length=8, max_length=2000)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/fetch")
def fetch(body: FetchBody):
    try:
        return safe_fetch(body.url)
    except UnsafeURL as exc:
        raise HTTPException(422, f"refused: {exc}") from None
    except FetchError as exc:
        raise HTTPException(502, str(exc)) from None
    except Exception as exc:  # noqa: BLE001 — never echo parser internals to the caller
        raise HTTPException(502, f"fetch failed: {type(exc).__name__}") from None
