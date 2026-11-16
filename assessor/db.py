"""
db.py — the system of record (Production RAG Module 12 & 13).

SQLite (`runs/assessments.db`, WAL mode) holds everything that must survive a restart and be answerable
to an auditor:

- assessments   full report JSON + approval state. This is the SINGLE source of truth; the files in a run
                folder (report.json / report.md) are exports written from it.
- jobs          the async `/assess` queue, with the request payload so an interrupted job can be re-run
- audit_log     append-only, hash-chained record of who did what (assessments, decisions, auth failures,
                guardrail refusals, cache purges). Each row stores the SHA-256 of the previous row, so an
                edit or deletion anywhere breaks the chain (`verify_audit_chain`, scripts/verify_audit.py).
- llm_usage     one row per LLM call: the data behind the cost dashboard
- eval_history  one row per evaluation run, so quality drift is visible over time

SQLite is the right size here: one machine, a handful of concurrent reviewers, no extra service to run.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .config import RUNS_DIR
from .errors import mask_secrets, record_error
from .schemas import AssessmentReport

DB_PATH = RUNS_DIR / "assessments.db"
GENESIS_HASH = "0" * 64

_INIT_LOCK = threading.Lock()
_INITIALISED: set[str] = set()

# Ordered migrations. Never edit an applied one; append a new entry.
MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (1, "baseline tables", [
        """CREATE TABLE IF NOT EXISTS assessments (
            run_id TEXT PRIMARY KEY, vendor TEXT NOT NULL, tier TEXT NOT NULL, users INTEGER NOT NULL,
            requested_by TEXT NOT NULL, overall TEXT NOT NULL, overall_reason TEXT NOT NULL,
            approval_status TEXT NOT NULL, decided_by TEXT, decided_on TEXT, cost_usd REAL NOT NULL,
            llm_calls INTEGER NOT NULL, duration_s REAL NOT NULL, created_at TEXT NOT NULL, report_json TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY, status TEXT NOT NULL, vendor TEXT NOT NULL, run_id TEXT, error TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    ]),
    (2, "job ownership and replay, audit log, usage, eval history", [
        "ALTER TABLE jobs ADD COLUMN owner TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE jobs ADD COLUMN request_json TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
        """CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
            subject TEXT NOT NULL, details TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL)""",
        """CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
           BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END""",
        """CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
           BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END""",
        """CREATE TABLE IF NOT EXISTS llm_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, run_id TEXT NOT NULL, route TEXT NOT NULL DEFAULT '',
            stage TEXT NOT NULL, role TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, usd REAL NOT NULL, avoided_usd REAL NOT NULL,
            local INTEGER NOT NULL, cached INTEGER NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS llm_usage_ts ON llm_usage (ts)",
        "CREATE INDEX IF NOT EXISTS llm_usage_run ON llm_usage (run_id)",
        """CREATE TABLE IF NOT EXISTS eval_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, gold_set TEXT NOT NULL, llm_provider TEXT NOT NULL,
            passed INTEGER NOT NULL, metrics TEXT NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS assessments_vendor ON assessments (vendor, created_at)",
    ]),
    (3, "tool usage (search / fetch calls, for free-tier quota tracking)", [
        """CREATE TABLE IF NOT EXISTS tool_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, run_id TEXT NOT NULL, tool TEXT NOT NULL,
            provider TEXT NOT NULL, calls INTEGER NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS tool_usage_ts ON tool_usage (ts)",
    ]),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(db_path: Path | None) -> Path:
    return Path(db_path) if db_path is not None else DB_PATH


def _migrate(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, description TEXT NOT NULL, applied_at TEXT NOT NULL)")
    applied = {r[0] for r in conn.execute("SELECT version FROM schema_version")}
    for version, description, statements in MIGRATIONS:
        if version in applied:
            continue
        for stmt in statements:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():   # re-running an ADD COLUMN is harmless
                    raise
        conn.execute("INSERT INTO schema_version (version, description, applied_at) VALUES (?, ?, ?)", (version, description, _now()))
    conn.commit()


@contextmanager
def _db(db_path: Path | None = None, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
    """One short-lived connection per operation: commit on success, roll back on error, always close.

    `immediate=True` takes the write lock up front (BEGIN IMMEDIATE) so a read-then-write sequence, such as
    extending the audit hash chain, cannot interleave with another writer.
    """
    path = _path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        key = str(path.resolve())
        if key not in _INITIALISED:
            with _INIT_LOCK:
                if key not in _INITIALISED:
                    conn.execute("PRAGMA journal_mode=WAL")
                    _migrate(conn)
                    _INITIALISED.add(key)
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Audit log (append-only, hash-chained)
# ---------------------------------------------------------------------------

def _audit_hash(prev_hash: str, ts: str, actor: str, action: str, subject: str, details: str) -> str:
    payload = json.dumps([prev_hash, ts, actor, action, subject, details], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audit_insert(c: sqlite3.Connection, actor: str, action: str, subject: str, details: dict[str, Any] | None) -> str:
    ts = _now()
    det = mask_secrets(json.dumps(details or {}, ensure_ascii=False, sort_keys=True, default=str))
    row = c.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    prev = row["hash"] if row else GENESIS_HASH
    h = _audit_hash(prev, ts, actor, action, subject, det)
    c.execute("INSERT INTO audit_log (ts, actor, action, subject, details, prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (ts, actor, action, subject, det, prev, h))
    return h


def audit(action: str, actor: str, subject: str = "", details: dict[str, Any] | None = None, *,
          db_path: Path | None = None, strict: bool = False) -> None:
    """Append one audit record. `strict=True` raises on failure (used where the action must not happen unaudited)."""
    try:
        with _db(db_path, immediate=True) as c:
            _audit_insert(c, actor or "unknown", action, subject, details)
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        record_error("db:audit", exc, {"action": action, "subject": subject})


def verify_audit_chain(db_path: Path | None = None) -> dict[str, Any]:
    """Recompute every hash. Returns {"ok", "rows", "first_bad_id", "problem"}."""
    with _db(db_path) as c:
        rows = c.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
    prev = GENESIS_HASH
    for r in rows:
        if r["prev_hash"] != prev:
            return {"ok": False, "rows": len(rows), "first_bad_id": r["id"], "problem": "previous-hash link broken (row removed or reordered)"}
        if _audit_hash(prev, r["ts"], r["actor"], r["action"], r["subject"], r["details"]) != r["hash"]:
            return {"ok": False, "rows": len(rows), "first_bad_id": r["id"], "problem": "row content does not match its hash (row edited)"}
        prev = r["hash"]
    return {"ok": True, "rows": len(rows), "first_bad_id": None, "problem": ""}


def get_audit_log(limit: int = 100, *, action: str | None = None, subject: str | None = None,
                  db_path: Path | None = None) -> list[dict[str, Any]]:
    q, args = "SELECT * FROM audit_log", []
    conds = []
    if action:
        conds.append("action = ?")
        args.append(action)
    if subject:
        conds.append("subject = ?")
        args.append(subject)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    with _db(db_path) as c:
        rows = c.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["details"] = json.loads(d["details"] or "{}")
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Assessments
# ---------------------------------------------------------------------------

def _upsert_assessment(c: sqlite3.Connection, rep: AssessmentReport) -> None:
    payload = json.dumps(rep.model_dump(mode="json"), ensure_ascii=False)
    # ON CONFLICT keeps the original created_at: recording a decision must not reorder the vendor's history.
    c.execute(
        """
        INSERT INTO assessments (run_id, vendor, tier, users, requested_by, overall, overall_reason, approval_status,
                                 decided_by, decided_on, cost_usd, llm_calls, duration_s, created_at, report_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            overall = excluded.overall, overall_reason = excluded.overall_reason,
            approval_status = excluded.approval_status, decided_by = excluded.decided_by,
            decided_on = excluded.decided_on, report_json = excluded.report_json
        """,
        (rep.run_id, rep.request.vendor, rep.request.tier, rep.request.users, rep.request.requested_by, rep.overall,
         rep.overall_reason, rep.approval.status, rep.approval.decided_by,
         str(rep.approval.decided_on) if rep.approval.decided_on else None, rep.cost_usd, rep.llm_calls,
         rep.duration_s, _now(), payload),
    )


def save_assessment(rep: AssessmentReport, db_path: Path | None = None, *, audit_actor: str | None = None) -> None:
    """Store a report. With `audit_actor`, the audit record is written in the same transaction."""
    if not rep.run_id:
        return
    with _db(db_path, immediate=True) as c:
        _upsert_assessment(c, rep)
        if audit_actor is not None:
            _audit_insert(c, audit_actor, "assessment.created", rep.run_id, {
                "vendor": rep.request.vendor, "tier": rep.request.tier, "overall": rep.overall,
                "verdicts": {r.requirement_id: r.verdict for r in rep.results},
                "provenance": rep.provenance})


def save_decision(rep: AssessmentReport, actor: str, details: dict[str, Any], db_path: Path | None = None) -> None:
    """Persist an approval decision and its audit record atomically: neither exists without the other."""
    with _db(db_path, immediate=True) as c:
        _upsert_assessment(c, rep)
        _audit_insert(c, actor, f"approval.{details.get('decision', 'decision')}", rep.run_id, details)


def update_approval_in_db(run_id: str, approval: dict[str, Any], db_path: Path | None = None) -> None:
    """Kept for callers that only hold the approval dict. Prefer approvals.record_decision()."""
    with _db(db_path, immediate=True) as c:
        row = c.execute("SELECT report_json FROM assessments WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return
        rep = json.loads(row["report_json"])
        rep["approval"] = approval
        c.execute(
            "UPDATE assessments SET approval_status = ?, decided_by = ?, decided_on = ?, report_json = ? WHERE run_id = ?",
            (approval.get("status", "pending"), approval.get("decided_by"), approval.get("decided_on"),
             json.dumps(rep, ensure_ascii=False), run_id),
        )


def get_assessment(run_id: str, db_path: Path | None = None) -> Optional[dict[str, Any]]:
    with _db(db_path) as c:
        row = c.execute("SELECT report_json FROM assessments WHERE run_id = ?", (run_id,)).fetchone()
        return json.loads(row["report_json"]) if row else None


def list_assessments(limit: int = 50, *, status: str | None = None, db_path: Path | None = None) -> list[dict[str, Any]]:
    q = ("SELECT run_id, vendor, tier, users, requested_by, overall, approval_status, decided_by, decided_on, "
         "cost_usd, duration_s, created_at FROM assessments")
    args: list[Any] = []
    if status:
        q += " WHERE approval_status = ?"
        args.append(status)
    with _db(db_path) as c:
        return [dict(r) for r in c.execute(q + " ORDER BY created_at DESC, run_id DESC LIMIT ?", (*args, limit))]


def get_vendor_history(vendor: str, db_path: Path | None = None) -> dict[str, Any]:
    """Return past assessments for `vendor` and compute SEC-12.1 verdict/fact diffs between the latest two."""
    with _db(db_path) as c:
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


def get_recent_metrics(limit: int = 20, db_path: Path | None = None) -> dict[str, Any]:
    with _db(db_path) as c:
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


# ---------------------------------------------------------------------------
# Jobs (the async /assess queue)
# ---------------------------------------------------------------------------

def create_job(job_id: str, vendor: str, db_path: Path | None = None, *, owner: str = "", request: dict | None = None) -> None:
    now = _now()
    with _db(db_path, immediate=True) as c:
        c.execute(
            "INSERT OR REPLACE INTO jobs (job_id, status, vendor, run_id, error, created_at, updated_at, owner, request_json, attempts) "
            "VALUES (?, 'queued', ?, NULL, NULL, ?, ?, ?, ?, 0)",
            (job_id, vendor, now, now, owner, json.dumps(request or {}, ensure_ascii=False)),
        )


def update_job(job_id: str, status: str, *, run_id: Optional[str] = None, error: Optional[str] = None,
               db_path: Path | None = None, bump_attempts: bool = False) -> None:
    with _db(db_path, immediate=True) as c:
        c.execute(
            "UPDATE jobs SET status = ?, run_id = COALESCE(?, run_id), error = ?, updated_at = ?, "
            "attempts = attempts + ? WHERE job_id = ?",
            (status, run_id, error, _now(), 1 if bump_attempts else 0, job_id),
        )


def get_job(job_id: str, db_path: Path | None = None) -> Optional[dict[str, Any]]:
    with _db(db_path) as c:
        row = c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def get_failed_jobs(limit: int = 50, db_path: Path | None = None) -> list[dict[str, Any]]:
    if not _path(db_path).exists():
        return []
    with _db(db_path) as c:
        rows = c.execute("SELECT * FROM jobs WHERE status = 'failed' ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_unfinished_jobs(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Jobs a previous process left queued or mid-run (it crashed or was restarted)."""
    with _db(db_path) as c:
        rows = c.execute("SELECT * FROM jobs WHERE status IN ('queued', 'processing') ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]


def count_jobs(db_path: Path | None = None) -> dict[str, int]:
    with _db(db_path) as c:
        return {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")}


# ---------------------------------------------------------------------------
# LLM usage (cost dashboard) and evaluation history
# ---------------------------------------------------------------------------

def save_usage(run_id: str, calls: list[dict], *, route: str = "", tools: dict[str, int] | None = None,
               db_path: Path | None = None) -> None:
    """Persist one run's LLM calls (and tool-call counts, keyed "tool:provider") for the cost dashboard."""
    if not calls and not tools:
        return
    ts = _now()
    try:
        with _db(db_path, immediate=True) as c:
            for key, n in (tools or {}).items():
                tool, _, provider = key.partition(":")
                c.execute("INSERT INTO tool_usage (ts, run_id, tool, provider, calls) VALUES (?, ?, ?, ?, ?)",
                          (ts, run_id, tool, provider, int(n)))
            c.executemany(
                "INSERT INTO llm_usage (ts, run_id, route, stage, role, provider, model, input_tokens, output_tokens, usd, "
                "avoided_usd, local, cached) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(ts, run_id, route, x["stage"], x["role"], x["provider"], x["model"], x["input_tokens"], x["output_tokens"],
                  x["usd"], x.get("avoided_usd", 0.0), int(bool(x.get("local"))), int(bool(x.get("cached")))) for x in calls],
            )
    except Exception as exc:  # noqa: BLE001 — usage accounting must never fail a run
        record_error("db:save_usage", exc, {"run_id": run_id})


def spend_since(iso_ts: str, db_path: Path | None = None) -> float:
    with _db(db_path) as c:
        row = c.execute("SELECT COALESCE(SUM(usd), 0) AS s FROM llm_usage WHERE ts >= ?", (iso_ts,)).fetchone()
        return float(row["s"])


def tool_calls_since(iso_ts: str, tool: str, provider: str, db_path: Path | None = None) -> int:
    with _db(db_path) as c:
        row = c.execute("SELECT COALESCE(SUM(calls), 0) AS n FROM tool_usage WHERE ts >= ? AND tool = ? AND provider = ?",
                        (iso_ts, tool, provider)).fetchone()
        return int(row["n"])


def usage_report(days: int = 30, db_path: Path | None = None) -> dict[str, Any]:
    """Everything the cost dashboard shows, from one place."""
    since = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - days * 86400, timezone.utc).isoformat(timespec="seconds")

    def rows(sql: str) -> list[dict]:
        with _db(db_path) as c:
            return [dict(r) for r in c.execute(sql, (since,))]

    agg = "COUNT(*) AS calls, SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, ROUND(SUM(usd), 5) AS usd, ROUND(SUM(avoided_usd), 5) AS avoided_usd"
    per_run = rows("SELECT run_id, MAX(route) AS route, MIN(ts) AS ts, COUNT(*) AS calls, ROUND(SUM(usd), 5) AS usd, "
                   "ROUND(SUM(avoided_usd), 5) AS avoided_usd FROM llm_usage WHERE ts >= ? GROUP BY run_id ORDER BY ts DESC")
    costs = sorted(r["usd"] for r in per_run if r["route"] == "assessment") or sorted(r["usd"] for r in per_run)
    total = rows(f"SELECT {agg}, SUM(local) AS local_calls, SUM(cached) AS cached_calls FROM llm_usage WHERE ts >= ?")[0]
    return {
        "days": days,
        "total": {k: (v or 0) for k, v in total.items()},
        "by_day": rows(f"SELECT substr(ts, 1, 10) AS day, {agg} FROM llm_usage WHERE ts >= ? GROUP BY day ORDER BY day"),
        "by_model": rows(f"SELECT provider, model, MAX(local) AS local, {agg} FROM llm_usage WHERE ts >= ? GROUP BY provider, model ORDER BY usd DESC"),
        "by_stage": rows(f"SELECT stage, {agg} FROM llm_usage WHERE ts >= ? GROUP BY stage ORDER BY usd DESC"),
        "per_run": per_run[:50],
        "avg_cost_per_run": round(sum(costs) / len(costs), 5) if costs else 0.0,
        "p95_cost_per_run": round(costs[min(len(costs) - 1, int(round(0.95 * (len(costs) - 1))))], 5) if costs else 0.0,
    }


def save_eval_run(gold_set: str, llm_provider: str, passed: bool, metrics: dict[str, Any], db_path: Path | None = None) -> None:
    try:
        with _db(db_path, immediate=True) as c:
            c.execute("INSERT INTO eval_history (ts, gold_set, llm_provider, passed, metrics) VALUES (?, ?, ?, ?, ?)",
                      (_now(), gold_set, llm_provider, int(passed), json.dumps(metrics, default=str)))
    except Exception as exc:  # noqa: BLE001
        record_error("db:save_eval_run", exc, {"gold_set": gold_set})


def get_eval_history(limit: int = 30, db_path: Path | None = None) -> list[dict[str, Any]]:
    with _db(db_path) as c:
        rows = c.execute("SELECT * FROM eval_history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "metrics": json.loads(r["metrics"])} for r in rows]
