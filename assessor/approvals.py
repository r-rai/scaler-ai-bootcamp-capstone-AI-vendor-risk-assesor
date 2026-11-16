"""
approvals.py — the human gate, in one place.

The API, the UI and the CLI all record decisions through `record_decision()`, so the rules cannot drift:

  * a BLOCKED assessment cannot be approved
  * the approver is a named, authenticated person holding an approver role for the tier (AI-5.1)
  * the requester cannot approve their own request (separation of duties)
  * T3 needs Head of Information Security AND DPO: two signatures from two different people
  * one rejection is final; a final decision cannot be changed (start a new assessment instead)
  * the decision and its audit record are written in one transaction

Nothing here calls a model. The system records what a person decided; it never decides.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .auth import APPROVERS
from .config import RUNS_DIR
from .db import get_assessment, save_decision
from .errors import record_error
from .report import to_markdown
from .schemas import Approval, ApprovalDecision, AssessmentReport


class ApprovalError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def apply_decision(rep: AssessmentReport, *, user: str, roles: Iterable[str], decision: str, comment: str = "",
                   method: str = "api", role: Optional[str] = None) -> ApprovalDecision:
    """Validate one decision against the rules and apply it to `rep.approval` in memory. Raises ApprovalError."""
    user = (user or "").strip()
    decision = {"approved": "approve", "rejected": "reject"}.get(decision, decision)
    if decision not in ("approve", "reject"):
        raise ApprovalError(400, "decision must be 'approve' or 'reject'.")
    if not user:
        raise ApprovalError(400, "A named human approver is required (AI-5.1).")
    ap = rep.approval
    required = list(ap.approver_roles) or list(APPROVERS.get(rep.request.tier, []))
    ap.approver_roles = required
    if ap.status in ("approved", "rejected"):
        raise ApprovalError(409, f"This assessment is already {ap.status}. Run a new assessment to revisit it.")
    if decision == "approve" and rep.overall == "BLOCKED":
        raise ApprovalError(409, "A BLOCKED assessment cannot be approved. Resolve the blocking requirement or change the data tier.")
    if user == rep.request.requested_by:
        raise ApprovalError(403, "The requester cannot decide their own request (separation of duties).")
    if any(d.user == user for d in ap.decisions):
        raise ApprovalError(409, f"{user} has already signed this assessment. Each approver role needs a different person.")
    held = [r for r in required if r in set(roles)]
    if role is not None:
        held = [r for r in held if r == role]
    if not held:
        raise ApprovalError(403, f"{user} does not hold an approver role for {rep.request.tier} (needs: {', '.join(required)}).")
    open_roles = [r for r in held if r in ap.pending_roles]
    if not open_roles:
        raise ApprovalError(409, f"The role {held[0]} has already signed.")

    d = ApprovalDecision(user=user, role=open_roles[0], decision=decision,  # type: ignore[arg-type]
                         decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         comment=(comment or "").strip()[:2000], method=method)
    ap.decisions.append(d)
    if decision == "reject":
        ap.status = "rejected"
    elif not ap.pending_roles:
        ap.status = "approved"
    if ap.status in ("approved", "rejected"):
        ap.decided_by = ", ".join(f"{x.user} ({x.role})" for x in ap.decisions)
        ap.decided_on = date.today()
    if d.comment:
        ap.comment = d.comment
    return d


def load_report(run_id: str, db_path: Path | None = None) -> AssessmentReport:
    """The database is the source of truth. A run folder from before the database existed is imported once."""
    data = get_assessment(run_id, db_path)
    if data is None:
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            raise ApprovalError(404, "run not found")
        legacy = RUNS_DIR / run_id / "report.json"
        if not legacy.is_file():
            raise ApprovalError(404, "run not found")
        data = json.loads(legacy.read_text(encoding="utf-8"))
    return AssessmentReport.model_validate(data)


def export_run_files(rep: AssessmentReport) -> None:
    """Refresh the human-readable copies in the run folder. Best effort: the database already holds the truth."""
    try:
        d = RUNS_DIR / rep.run_id
        if not d.is_dir():
            return
        (d / "report.json").write_text(json.dumps(rep.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "report.md").write_text(to_markdown(rep), encoding="utf-8")
        last = rep.approval.decisions[-1] if rep.approval.decisions else None
        with open(d / "trace.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "stage": "approval",
                                 "status": rep.approval.status, **(last.model_dump() if last else {})}) + "\n")
    except Exception as exc:  # noqa: BLE001
        record_error("approvals:export", exc, {"run_id": rep.run_id})


def record_decision(run_id: str, *, user: str, roles: Iterable[str], decision: str, comment: str = "",
                    method: str = "api", role: Optional[str] = None, db_path: Path | None = None) -> Approval:
    """Load, validate, persist (with audit) and export one human decision. Raises ApprovalError."""
    rep = load_report(run_id, db_path)
    d = apply_decision(rep, user=user, roles=roles, decision=decision, comment=comment, method=method, role=role)
    save_decision(rep, user, {"decision": d.decision, "role": d.role, "method": method, "comment": d.comment,
                              "status_after": rep.approval.status, "overall": rep.overall,
                              "vendor": rep.request.vendor, "tier": rep.request.tier}, db_path)
    export_run_files(rep)
    return rep.approval
