"""
app/api.py — the assessor as a web service (FastAPI).

COURSE MAP  Production RAG course Module 4 (entry points and validation), Module 13 (the production app:
api.py / worker.py pattern).

What makes this a service rather than a script:
  * Identity is resolved on the server from a token (assessor/auth.py). In prod mode every endpoint except
    /health and /ready needs one, and `user_groups` in a request body is ignored.
  * Decisions go through assessor/approvals.py: named approver with the right role, requester cannot approve,
    T3 needs two signatures, BLOCKED cannot be approved. Every decision is in the hash-chained audit log.
  * Long assessments run on a bounded worker pool backed by the `jobs` table; jobs interrupted by a restart
    are re-run once on startup instead of being lost.
  * Heavy components (policy index, embedder, reranker) are built once per process, not once per request.
  * Per-caller rate limit, input size limits, request IDs, and errors that never leak a stack trace.
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import Field

from assessor import observability as obs
from assessor.approvals import ApprovalError, load_report, record_decision
from assessor.auth import LIMITER, AuthError, Identity, authenticate, extract_token
from assessor.config import RUNS_DIR, SETTINGS
from assessor.costs import SpendCapExceeded, check_spend_caps, cost_overview
from assessor.db import (
    audit, count_jobs, create_job, get_assessment, get_audit_log, get_job, get_recent_metrics,
    get_unfinished_jobs, get_vendor_history, list_assessments, save_usage, update_job, verify_audit_chain,
)
from assessor.errors import record_error
from assessor.graph import Pipeline, new_state, shared_components
from assessor.report import to_markdown
from assessor.schemas import BaseModel
from assessor.tracing import Budget, Tracer

log = obs.get_logger("api")
MAX_JOB_ATTEMPTS = 2
_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=max(1, SETTINGS.max_concurrent_jobs), thread_name_prefix="assess")
    return _POOL


@asynccontextmanager
async def lifespan(_: FastAPI):
    obs.setup_logging()
    obs.setup_tracing("vendor-risk-api")
    recovered = recover_jobs()
    log.info("startup", extra={"env": SETTINGS.env, "llm_provider": SETTINGS.llm_provider,
                               "search_provider": SETTINGS.search_provider, "jobs_recovered": recovered})
    yield
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Nimbu Mart Vendor Risk Assessor",
    version="2.0.0",
    description="Capstone project by Shivank Agrawal (Future with Shivank). Assessments are inputs to a human decision.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request models: validated and size-limited at the door
# ---------------------------------------------------------------------------

class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=SETTINGS.max_question_chars)
    user_groups: list[str] = Field(default_factory=lambda: ["public", "employees"], max_length=10)


class AssessBody(BaseModel):
    question: str = Field(min_length=3, max_length=SETTINGS.max_question_chars)
    vendor: str = Field(min_length=1, max_length=120)
    official_domain: Optional[str] = Field(default=None, max_length=253, pattern=r"^[A-Za-z0-9.-]+$")
    tier: Literal["T1", "T2", "T3"] = "T3"
    users: int = Field(default=1, ge=1, le=1_000_000)
    user_groups: list[str] = Field(default_factory=lambda: ["public", "employees"], max_length=10)
    requested_by: str = Field(default="api", max_length=120)
    refresh: bool = Field(default=False, description="Re-review: ignore cached searches and pages")


class ApproveBody(BaseModel):
    decision: Literal["approve", "reject"]
    comment: str = Field(default="", max_length=2000)
    role: Optional[str] = Field(default=None, description="Which of your approver roles you sign in (only needed if you hold several)")
    approver: Optional[str] = Field(default=None, description="Ignored: the approver is the authenticated caller")


# ---------------------------------------------------------------------------
# Cross-cutting: request id, metrics, error shapes
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex[:16]
    request.state.request_id = rid
    with obs.request_scope(rid, request.method, request.url.path) as scope:
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 — never leak a stack trace to a caller
            record_error(f"api:{request.url.path}", exc, {"request_id": rid})
            scope["status"] = 500
            log.error("unhandled error", extra={"request_id": rid, "error": type(exc).__name__})
            response = JSONResponse({"detail": "Internal error. Quote the request id to the operator.", "request_id": rid}, status_code=500)
        scope["status"] = response.status_code
        scope["route"] = getattr(request.scope.get("route"), "path", request.url.path)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(AuthError)
async def _auth_error(request: Request, exc: AuthError):
    audit("auth.failed", "unknown", request.url.path, {"reason": exc.message, "client": request.client.host if request.client else ""})
    obs.count_guardrail("auth", "rejected")
    return JSONResponse({"detail": exc.message}, status_code=exc.status)


@app.exception_handler(ApprovalError)
async def _approval_error(_: Request, exc: ApprovalError):
    return JSONResponse({"detail": exc.message}, status_code=exc.status)


def identify(x_api_token: Optional[str], authorization: Optional[str], fallback_groups: list[str] | None = None,
             cost: float = 1.0) -> Identity:
    ident = authenticate(extract_token(x_api_token, authorization), fallback_groups or ["public", "employees"])
    if not LIMITER.allow(ident.user, cost):
        obs.count_guardrail("rate_limit", "rejected")
        raise HTTPException(429, "Rate limit exceeded. Try again in a minute.")
    return ident


def require_privileged(ident: Identity) -> None:
    if SETTINGS.prod and not ident.privileged:
        raise HTTPException(403, "Security or Procurement membership is required.")


def _readable_report(run_id: str, ident: Identity) -> dict[str, Any]:
    rep = get_assessment(run_id)
    if rep is None:
        raise HTTPException(404, "run not found")
    if not ident.can_read_run(rep.get("request", {}).get("requested_by", "")):
        raise HTTPException(404, "run not found")   # 404, not 403: do not confirm the run exists
    return rep


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------

def _run(question: str, ident: Identity, *, tracer: Tracer, plain: bool = False, **state_kw: Any) -> tuple[dict, Budget]:
    budget = Budget()
    pipe = Pipeline(tracer=tracer, budget=budget, **shared_components())
    s = new_state(question, user_groups=list(ident.groups), **state_kw)
    try:
        s = pipe.run_plain(s) if plain else pipe.run(s)
    finally:
        save_usage(tracer.run_id, budget.calls, route=str(s.get("route") or ""), tools=budget.tools)
        obs.observe_run(s, budget)
    return s, budget


@app.get("/health")
def health():
    """Liveness: the process is up. No dependencies are checked."""
    return {"ok": True, "env": SETTINGS.env, "llm_provider": SETTINGS.llm_provider, "search_provider": SETTINGS.search_provider}


@app.get("/ready")
def ready():
    """Readiness: database writable, policy index loaded, model endpoints reachable."""
    checks = obs.readiness_checks()
    ok = all(c["ok"] or not c["required"] for c in checks.values())
    return JSONResponse({"ok": ok, "checks": checks}, status_code=200 if ok else 503)


@app.post("/ask")
def ask(body: AskBody, x_api_token: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)):
    ident = identify(x_api_token, authorization, body.user_groups)
    tracer = Tracer("ask", enabled=False)
    s, _ = _run(body.question, ident, tracer=tracer, plain=True, requested_by=ident.user)
    if s["route"] == "assessment":
        raise HTTPException(400, "This question needs a full assessment: call POST /assess with vendor, tier and users.")
    return {
        "route": s["route"],
        "answer": s["answer"],
        "user_groups": list(ident.groups),
        "cached": bool(s.get("cache_hit")),
        "citations": [c.model_dump() for c in (s["policy_answer"].citations if s.get("policy_answer") else [])],
    }


def _assess(body: AssessBody, ident: Identity, requested_by: str) -> tuple[dict, str]:
    tracer = Tracer("api")
    try:
        s, _ = _run(body.question, ident, tracer=tracer, tier=body.tier, users=body.users, vendor=body.vendor,
                    domain=body.official_domain, requested_by=requested_by, refresh=body.refresh)
    finally:
        tracer.close()
    return s, tracer.run_id


def _execute_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job or job["status"] not in ("queued", "processing"):
        return
    update_job(job_id, "processing", bump_attempts=True)
    try:
        req = json.loads(job["request_json"] or "{}")
        body = AssessBody.model_validate(req["body"])
        ident = Identity(req["user"], tuple(req["groups"]), tuple(req.get("roles", [])), True)
        check_spend_caps()
        s, run_id = _assess(body, ident, req["requested_by"])
        update_job(job_id, "completed", run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        record_error("api:/assess(job)", exc, {"job_id": job_id, "vendor": job.get("vendor")})
        update_job(job_id, "failed", error=str(exc)[:300])
    finally:
        obs.set_queue_depth(count_jobs())


def recover_jobs() -> int:
    """On startup: re-run jobs a previous process left unfinished (once), fail the ones that already used their retry."""
    n = 0
    for job in get_unfinished_jobs():
        if job["attempts"] >= MAX_JOB_ATTEMPTS or not job["request_json"]:
            update_job(job["job_id"], "failed", error="interrupted by a restart; resubmit the request")
            continue
        update_job(job["job_id"], "queued")
        _pool().submit(_execute_job, job["job_id"])
        n += 1
    return n


@app.post("/assess")
def assess(
    body: AssessBody,
    async_job: bool = Query(default=False, description="If true, enqueue job and return job_id immediately"),
    x_api_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
):
    ident = identify(x_api_token, authorization, body.user_groups, cost=5.0)
    # The requester recorded on the report is the authenticated user; a free-text name is only kept for anonymous dev calls.
    req_by = ident.user if ident.authenticated else body.requested_by
    try:
        check_spend_caps()
    except SpendCapExceeded as exc:
        raise HTTPException(429, str(exc)) from exc
    if async_job:
        job_id = f"job-{uuid.uuid4().hex[:10]}"
        create_job(job_id, body.vendor, owner=ident.user, request={
            "body": body.model_dump(), "user": ident.user, "groups": list(ident.groups), "roles": list(ident.roles),
            "requested_by": req_by})
        _pool().submit(_execute_job, job_id)
        obs.set_queue_depth(count_jobs())
        return {"job_id": job_id, "status": "queued", "vendor": body.vendor}

    s, run_id = _assess(body, ident, req_by)
    if s.get("report") is None:
        return {"route": s["route"], "answer": s["answer"], "run_id": run_id}
    return {"run_id": run_id, "report": s["report"].model_dump(mode="json")}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, x_api_token: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)):
    ident = identify(x_api_token, authorization, cost=0.2)
    job = get_job(job_id)
    if not job or not ident.can_read_run(job.get("owner", "")):
        raise HTTPException(404, "job not found")
    job.pop("request_json", None)
    if job.get("run_id") and job["status"] == "completed":
        job["report"] = get_assessment(job["run_id"])
    return job


@app.get("/runs")
def runs(status: Optional[str] = None, limit: int = Query(default=50, ge=1, le=200),
         x_api_token: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)):
    ident = identify(x_api_token, authorization, cost=0.2)
    return [r for r in list_assessments(limit, status=status) if ident.can_read_run(r["requested_by"])]


@app.get("/runs/{run_id}")
def run_report(run_id: str, x_api_token: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)):
    return _readable_report(run_id, identify(x_api_token, authorization, cost=0.2))


@app.get("/runs/{run_id}/report.md", response_class=PlainTextResponse)
def report_md(run_id: str, x_api_token: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)):
    _readable_report(run_id, identify(x_api_token, authorization, cost=0.2))
    return to_markdown(load_report(run_id))


@app.get("/runs/{run_id}/trace")
def run_trace(run_id: str, x_api_token: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)):
    _readable_report(run_id, identify(x_api_token, authorization, cost=0.2))
    p = RUNS_DIR / run_id / "trace.jsonl"
    if not p.is_file():
        raise HTTPException(404, "trace not found")
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


@app.post("/approve/{run_id}")
def approve(run_id: str, body: ApproveBody, x_api_token: Optional[str] = Header(default=None),
            authorization: Optional[str] = Header(default=None)):
    """Record one human decision. The approver is the authenticated caller, never a name in the body."""
    ident = identify(x_api_token, authorization)
    if not ident.authenticated:
        raise HTTPException(401, "A decision needs an authenticated, named approver (AI-5.1). Present an API token.")
    approval = record_decision(run_id, user=ident.user, roles=ident.roles, decision=body.decision, comment=body.comment,
                               method="api", role=body.role)
    obs.count_decision(body.decision, approval.status)
    return approval.model_dump(mode="json")


@app.get("/vendors/{vendor}/history")
def vendor_history(vendor: str, x_api_token: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)):
    require_privileged(identify(x_api_token, authorization, cost=0.2))
    return get_vendor_history(vendor)


@app.get("/stats")
def stats(limit: int = Query(default=20, ge=1, le=200), x_api_token: Optional[str] = Header(default=None),
          authorization: Optional[str] = Header(default=None)):
    """Latency and cost across recent runs (JSON). Prometheus scrapes /metrics instead."""
    require_privileged(identify(x_api_token, authorization, cost=0.2))
    return {**get_recent_metrics(limit=limit), "jobs": count_jobs()}


@app.get("/costs")
def costs(days: int = Query(default=30, ge=1, le=365), x_api_token: Optional[str] = Header(default=None),
          authorization: Optional[str] = Header(default=None)):
    """The cost dashboard's data: spend per day, model, stage and run; local vs cloud; budget caps."""
    require_privileged(identify(x_api_token, authorization, cost=0.2))
    return cost_overview(days)


@app.get("/audit")
def audit_log(limit: int = Query(default=100, ge=1, le=1000), action: Optional[str] = None, subject: Optional[str] = None,
              x_api_token: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)):
    require_privileged(identify(x_api_token, authorization, cost=0.2))
    return {"chain": verify_audit_chain(), "entries": get_audit_log(limit, action=action, subject=subject)}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus exposition format. Aggregate counters only: no questions, vendors or user names."""
    body, content_type = obs.prometheus_payload()
    return PlainTextResponse(body, media_type=content_type)
