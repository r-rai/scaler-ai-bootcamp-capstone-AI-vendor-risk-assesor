"""
db.py — persistence and observability store for Milestone 5 (Production RAG Module 12 & 13).

Uses Python's built-in `sqlite3` (`runs/assessments.db`) by default so it works on Windows WSL2 and CI
with zero external database setup, while storing:
- Full assessment reports and approval decisions (surviving server restarts)
- Background job statuses for `/assess` queue workers
- Vendor assessment history and SEC-12.1 change diffs (`/vendors/{name}/history`)
- Observability metrics (p50/p95 latency and cost across the last N runs)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import RUNS_DIR
from .schemas import AssessmentReport

DB_PATH = RUNS_DIR / "assessments.db"


def _conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assessments (
            run_id TEXT PRIMARY KEY,
            vendor TEXT NOT NULL,
            tier TEXT NOT NULL,
            users INTEGER NOT NULL,
            requested_by TEXT NOT NULL,
            overall TEXT NOT NULL,
            overall_reason TEXT NOT NULL,
            approval_status TEXT NOT NULL,
            decided_by TEXT,
            decided_on TEXT,
            cost_usd REAL NOT NULL,
            llm_calls INTEGER NOT NULL,
            duration_s REAL NOT NULL,
            created_at TEXT NOT NULL,
            report_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            vendor TEXT NOT NULL,
            run_id TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def save_assessment(rep: AssessmentReport, db_path: Path = DB_PATH) -> None:
    if not rep.run_id:
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = json.dumps(rep.model_dump(mode="json"), ensure_ascii=False)
    with _conn(db_path) as c:
        c.execute(
            """
            INSERT OR REPLACE INTO assessments (
                run_id, vendor, tier, users, requested_by, overall, overall_reason,
                approval_status, decided_by, decided_on, cost_usd, llm_calls, duration_s,
                created_at, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rep.run_id,
                rep.request.vendor,
                rep.request.tier,
                rep.request.users,
                rep.request.requested_by,
                rep.overall,
                rep.overall_reason,
                rep.approval.status,
                rep.approval.decided_by,
                str(rep.approval.decided_on) if rep.approval.decided_on else None,
                rep.cost_usd,
                rep.llm_calls,
                rep.duration_s,
                now,
                payload,
            ),
        )


def update_approval_in_db(run_id: str, approval: dict[str, Any], db_path: Path = DB_PATH) -> None:
    with _conn(db_path) as c:
        row = c.execute("SELECT report_json FROM assessments WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return
        rep = json.loads(row["report_json"])
        rep["approval"] = approval
        c.execute(
            """
            UPDATE assessments
               SET approval_status = ?, decided_by = ?, decided_on = ?, report_json = ?
             WHERE run_id = ?
            """,
            (
                approval.get("status", "pending"),
                approval.get("decided_by"),
                approval.get("decided_on"),
                json.dumps(rep, ensure_ascii=False),
                run_id,
            ),
        )


def get_assessment(run_id: str, db_path: Path = DB_PATH) -> Optional[dict[str, Any]]:
    with _conn(db_path) as c:
        row = c.execute("SELECT report_json FROM assessments WHERE run_id = ?", (run_id,)).fetchone()
        return json.loads(row["report_json"]) if row else None


def get_vendor_history(vendor: str, db_path: Path = DB_PATH) -> dict[str, Any]:
    """Return past assessments for `vendor` and compute SEC-12.1 verdict/fact diffs between the latest two."""
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM assessments WHERE lower(vendor) = lower(?) ORDER BY created_at DESC, run_id DESC",
            (vendor,),
        ).fetchall()
    runs = []
    for r in rows:
        rep = json.loads(r["report_json"])
        runs.append(
            {
                "run_id": r["run_id"],
                "created_at": r["created_at"],
                "tier": r["tier"],
                "overall": r["overall"],
                "approval_status": r["approval_status"],
                "decided_by": r["decided_by"],
                "verdicts": {res["requirement_id"]: res["verdict"] for res in rep.get("results", [])},
                "facts": {k: v.get("display") for k, v in rep.get("profile", {}).get("facts", {}).items()},
            }
        )
    diffs: list[str] = []
    if len(runs) >= 2:
        latest, prev = runs[0], runs[1]
        for rid, v_now in latest["verdicts"].items():
            v_old = prev["verdicts"].get(rid)
            if v_old and v_old != v_now:
                diffs.append(f"verdict {rid}: {v_old} -> {v_now}")
        for fk, f_now in latest["facts"].items():
            f_old = prev["facts"].get(fk)
            if f_old and f_old != f_now:
                diffs.append(f"fact {fk}: '{f_old}' -> '{f_now}'")
    return {"vendor": vendor, "count": len(runs), "changes_since_previous": diffs, "history": runs}


def get_recent_metrics(limit: int = 20, db_path: Path = DB_PATH) -> dict[str, Any]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT run_id, vendor, tier, overall, approval_status, cost_usd, llm_calls, duration_s, created_at "
            "FROM assessments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    items = [dict(r) for r in rows]
    durations = sorted(float(r["duration_s"]) for r in items)
    costs = [float(r["cost_usd"]) for r in items]

    def _pct(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
        return round(vals[idx], 3)

    return {
        "runs_counted": len(items),
        "p50_duration_s": _pct(durations, 0.50),
        "p95_duration_s": _pct(durations, 0.95),
        "total_cost_usd": round(sum(costs), 4),
        "avg_cost_usd": round(sum(costs) / max(1, len(costs)), 4),
        "recent_runs": items,
    }


def create_job(job_id: str, vendor: str, db_path: Path = DB_PATH) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn(db_path) as c:
        c.execute(
            "INSERT OR REPLACE INTO jobs (job_id, status, vendor, run_id, error, created_at, updated_at) VALUES (?, 'queued', ?, NULL, NULL, ?, ?)",
            (job_id, vendor, now, now),
        )


def update_job(job_id: str, status: str, *, run_id: Optional[str] = None, error: Optional[str] = None, db_path: Path = DB_PATH) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn(db_path) as c:
        c.execute(
            "UPDATE jobs SET status = ?, run_id = COALESCE(?, run_id), error = ?, updated_at = ? WHERE job_id = ?",
            (status, run_id, error, now, job_id),
        )


def get_job(job_id: str, db_path: Path = DB_PATH) -> Optional[dict[str, Any]]:
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None
