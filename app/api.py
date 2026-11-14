"""
app/api.py — the assessor as a web service (FastAPI).

COURSE MAP  Production RAG course Module 4 (entry points and validation), Module 13 (the production app:
api.py / worker.py pattern).

Milestone 5 complete:
  * Server-side caller authentication (`X-API-Token` / `Authorization: Bearer`) deriving `user_groups` on the server
    (never trusting client-supplied groups when a token is presented)
  * Async background job queue (`POST /assess?async_job=true` + `GET /jobs/{job_id}`)
  * Persistent storage in SQLite (`runs/assessments.db`) with `/vendors/{vendor}/history` (SEC-12.1 diff)
    and `/metrics` (p50/p95 latency and cost across runs)
"""

from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import Field

from assessor.config import RUNS_DIR, SETTINGS
from assessor.db import (
    create_job, get_assessment, get_job, get_recent_metrics, get_vendor_history,
    save_assessment, update_approval_in_db, update_job,
)
from assessor.graph import Pipeline, new_state
from assessor.report import to_markdown
from assessor.schemas import AssessmentReport, BaseModel
from assessor.tracing import Budget, Tracer

app = FastAPI(
    title="Nimbu Mart Vendor Risk Assessor",
    version="1.0.0",
    description="Capstone project by Shivank Agrawal (Future with Shivank). Assessments are inputs to a human decision.",
)

# Server-side token-to-identity directory (Milestone 5: derive user_groups from verified caller identity)
TOKEN_DIRECTORY: dict[str, dict[str, Any]] = {
    "tok-public": {"user": "guest", "groups": ["public"]},
    "tok-employee": {"user": "ravi.k@nimbumart.example", "groups": ["public", "employees"]},
    "tok-security": {"user": "sec.lead@nimbumart.example", "groups": ["public", "employees", "security"]},
    "tok-procurement": {"user": "proc.lead@nimbumart.example", "groups": ["public", "employees", "procurement"]},
    "tok-dpo": {"user": "dpo@nimbumart.example", "groups": ["public", "employees", "security", "procurement"]},
}


def resolve_caller_groups(
    fallback_groups: list[str],
    x_api_token: Optional[str] = None,
    authorization: Optional[str] = None,
) -> tuple[str, list[str]]:
    token = x_api_token
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token:
        entry = TOKEN_DIRECTORY.get(token)
        if not entry:
            raise HTTPException(401, "Invalid API token")
        return entry["user"], list(entry["groups"])
    return "api", list(fallback_groups)


class AskBody(BaseModel):
    question: str
    user_groups: list[str] = Field(default_factory=lambda: ["public", "employees"])


class AssessBody(BaseModel):
    question: str
    vendor: str
    official_domain: Optional[str] = None
    tier: str = "T3"
    users: int = 1
    user_groups: list[str] = Field(default_factory=lambda: ["public", "employees"])
    requested_by: str = "api"


class ApproveBody(BaseModel):
    decision: str  # approve | reject
    approver: str
    comment: str = ""


@app.get("/health")
def health():
    return {"ok": True, "llm_provider": SETTINGS.llm_provider, "search_provider": SETTINGS.search_provider}


@app.post("/ask")
def ask(
    body: AskBody,
    x_api_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
):
    caller, groups = resolve_caller_groups(body.user_groups, x_api_token, authorization)
    pipe = Pipeline(tracer=None)
    s = pipe.run_plain(new_state(body.question, user_groups=groups, requested_by=caller))
    if s["route"] == "assessment":
        raise HTTPException(400, "This question needs a full assessment: call POST /assess with vendor, tier and users.")
    return {
        "route": s["route"],
        "answer": s["answer"],
        "user_groups": groups,
        "citations": [c.model_dump() for c in (s["policy_answer"].citations if s.get("policy_answer") else [])],
    }


def _execute_assessment_job(job_id: str, body: AssessBody, groups: list[str], caller: str) -> None:
    update_job(job_id, "processing")
    tracer = Tracer("api")
    pipe = Pipeline(tracer=tracer, budget=Budget())
    s = new_state(
        body.question, user_groups=groups, tier=body.tier, users=body.users,
        vendor=body.vendor, domain=body.official_domain, requested_by=caller,
    )
    try:
        s = pipe.run(s)
        update_job(job_id, "completed", run_id=tracer.run_id)
    except Exception as exc:  # noqa: BLE001
        update_job(job_id, "failed", run_id=tracer.run_id, error=str(exc)[:300])
    finally:
        tracer.close()


@app.post("/assess")
def assess(
    body: AssessBody,
    background_tasks: BackgroundTasks,
    async_job: bool = Query(default=False, description="If true, enqueue job and return job_id immediately"),
    x_api_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
):
    caller, groups = resolve_caller_groups(body.user_groups, x_api_token, authorization)
    req_by = caller if caller != "api" else body.requested_by
    if async_job:
        job_id = f"job-{uuid.uuid4().hex[:10]}"
        create_job(job_id, body.vendor)
        background_tasks.add_task(_execute_assessment_job, job_id, body, groups, req_by)
        return {"job_id": job_id, "status": "queued", "vendor": body.vendor}

    tracer = Tracer("api")
    pipe = Pipeline(tracer=tracer, budget=Budget())
    s = new_state(
        body.question, user_groups=groups, tier=body.tier, users=body.users,
        vendor=body.vendor, domain=body.official_domain, requested_by=req_by,
    )
    try:
        s = pipe.run(s)
    finally:
        tracer.close()
    if s.get("report") is None:
        return {"route": s["route"], "answer": s["answer"], "run_id": tracer.run_id}
    return {"run_id": tracer.run_id, "report": s["report"].model_dump(mode="json")}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.get("run_id") and job["status"] == "completed":
        job["report"] = get_assessment(job["run_id"])
    return job


@app.get("/vendors/{vendor}/history")
def vendor_history(vendor: str):
    return get_vendor_history(vendor)


@app.get("/metrics")
def metrics(limit: int = 20):
    return get_recent_metrics(limit=limit)


def _run_dir(run_id: str) -> Path:
    p = RUNS_DIR / run_id
    if not p.exists() or ".." in run_id:
        raise HTTPException(404, "run not found")
    return p


@app.post("/approve/{run_id}")
def approve(run_id: str, body: ApproveBody):
    d = _run_dir(run_id)
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    if body.decision == "approve" and rep["overall"] == "BLOCKED":
        raise HTTPException(409, "A BLOCKED assessment cannot be approved. Resolve the blocking requirement or change the data tier.")
    if not body.approver.strip():
        raise HTTPException(400, "A named human approver is required (AI-5.1).")
    rep["approval"].update(
        {
            "status": "approved" if body.decision == "approve" else "rejected",
            "decided_by": body.approver.strip(),
            "decided_on": date.today().isoformat(),
            "comment": body.comment,
        }
    )
    (d / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    # Re-render report.md so the Markdown file also reflects the recorded human decision
    rep_obj = AssessmentReport.model_validate(rep)
    (d / "report.md").write_text(to_markdown(rep_obj), encoding="utf-8")
    save_assessment(rep_obj)
    update_approval_in_db(run_id, rep["approval"])
    with open(d / "trace.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"stage": "approval", **rep["approval"]}) + "\n")
    return rep["approval"]


@app.get("/runs/{run_id}/report.md", response_class=PlainTextResponse)
def report_md(run_id: str):
    return (_run_dir(run_id) / "report.md").read_text(encoding="utf-8")
