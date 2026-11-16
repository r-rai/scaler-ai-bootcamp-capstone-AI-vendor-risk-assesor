"""
costs.py — spend across runs: caps and the cost dashboard's numbers.

`Budget` (tracing.py) limits ONE assessment. This module answers the questions a budget owner asks
across all of them: what did we spend today and this month, on which model and pipeline stage, how much
did the local model and the caches save, and are we near a cap?

Caps (0 disables): VRA_DAILY_USD and VRA_MONTHLY_USD. When a cap is reached, new live runs are refused
with a clear message; offline/mock runs are never blocked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import SETTINGS
from .db import spend_since, tool_calls_since, usage_report

WARN_AT = 0.8   # fraction of a cap at which the dashboard and alerts start warning


class SpendCapExceeded(RuntimeError):
    pass


def _day_start() -> str:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")


def _month_start() -> str:
    return datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")


def spend_summary(db_path: Path | None = None) -> dict[str, Any]:
    today, month = spend_since(_day_start(), db_path), spend_since(_month_start(), db_path)

    def cap(spent: float, limit: float) -> dict[str, Any]:
        used = spent / limit if limit else 0.0
        return {"limit_usd": limit, "used_fraction": round(used, 4),
                "state": "off" if not limit else "exceeded" if used >= 1 else "warning" if used >= WARN_AT else "ok"}

    tavily = tool_calls_since(_month_start(), "search", "tavily", db_path)
    return {
        "today_usd": round(today, 5),
        "month_usd": round(month, 5),
        "per_run_cap_usd": SETTINGS.max_usd,
        "daily_cap": cap(today, SETTINGS.daily_usd),
        "monthly_cap": cap(month, SETTINGS.monthly_usd),
        "tavily_credits_used": tavily,
        "tavily_credits_quota": SETTINGS.tavily_monthly_credits,
    }


def check_spend_caps(db_path: Path | None = None) -> None:
    """Raise SpendCapExceeded if a daily or monthly cap is reached. A no-op in mock mode and when no cap is set."""
    if SETTINGS.llm_provider == "mock" or not (SETTINGS.daily_usd or SETTINGS.monthly_usd):
        return
    s = spend_summary(db_path)
    if s["daily_cap"]["state"] == "exceeded":
        raise SpendCapExceeded(f"Daily LLM spend cap reached (${s['today_usd']:.2f} of ${SETTINGS.daily_usd:.2f}). "
                               "New live assessments resume tomorrow (UTC) or when VRA_DAILY_USD is raised.")
    if s["monthly_cap"]["state"] == "exceeded":
        raise SpendCapExceeded(f"Monthly LLM spend cap reached (${s['month_usd']:.2f} of ${SETTINGS.monthly_usd:.2f}). "
                               "Raise VRA_MONTHLY_USD to continue.")


def cost_overview(days: int = 30, db_path: Path | None = None) -> dict[str, Any]:
    """One payload for the cost dashboard (Streamlit tab and GET /costs)."""
    rep = usage_report(days, db_path)
    total = rep["total"]
    calls = total.get("calls", 0) or 0
    rep["summary"] = spend_summary(db_path)
    rep["shares"] = {
        "local_call_share": round((total.get("local_calls", 0) or 0) / calls, 4) if calls else 0.0,
        "cached_call_share": round((total.get("cached_calls", 0) or 0) / calls, 4) if calls else 0.0,
    }
    return rep
