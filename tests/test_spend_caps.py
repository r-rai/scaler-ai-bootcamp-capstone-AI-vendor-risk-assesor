"""Daily and monthly spend caps (VRA_DAILY_USD / VRA_MONTHLY_USD): new live runs are refused at the cap.

They were documented and wired into the CLI, API and UI but had no test. These use a throwaway database and
a non-mock provider name; no model is called.
"""
from datetime import datetime, timedelta, timezone

import pytest

from assessor.config import SETTINGS
from assessor.costs import SpendCapExceeded, check_spend_caps, spend_summary
from assessor.db import save_usage


def usage(usd: float) -> dict:
    return dict(stage="extract", role="extract", provider="minimax", model="m", input_tokens=10, output_tokens=10, usd=usd)


@pytest.fixture
def live(monkeypatch):
    """Pretend a live provider is configured, with no caps unless the test sets them."""
    monkeypatch.setattr(SETTINGS, "llm_provider", "minimax")
    monkeypatch.setattr(SETTINGS, "daily_usd", 0.0)
    monkeypatch.setattr(SETTINGS, "monthly_usd", 0.0)
    return monkeypatch


def backdate(db_path, days: int) -> None:
    import sqlite3

    ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as c:
        c.execute("UPDATE llm_usage SET ts = ?", (ts,))


def test_no_cap_means_no_limit(live, tmp_path):
    db = tmp_path / "a.db"
    save_usage("r1", [usage(1000.0)], db_path=db)
    check_spend_caps(db)                                  # must not raise


def test_daily_cap_refuses_at_the_limit_and_not_below(live, tmp_path):
    db = tmp_path / "a.db"
    live.setattr(SETTINGS, "daily_usd", 1.0)
    save_usage("r1", [usage(0.60)], db_path=db)
    check_spend_caps(db)                                  # 60% used: allowed
    assert spend_summary(db)["daily_cap"]["state"] == "ok"
    save_usage("r2", [usage(0.25)], db_path=db)
    assert spend_summary(db)["daily_cap"]["state"] == "warning"     # 85%: warns, still allowed
    check_spend_caps(db)
    save_usage("r3", [usage(0.15)], db_path=db)           # exactly 100%
    with pytest.raises(SpendCapExceeded, match="Daily"):
        check_spend_caps(db)


def test_monthly_cap_counts_earlier_days_but_the_daily_cap_does_not(live, tmp_path):
    db = tmp_path / "a.db"
    save_usage("r1", [usage(5.0)], db_path=db)
    now = datetime.now(timezone.utc)
    days_into_month = now.day - 1
    if days_into_month < 1:
        pytest.skip("first day of the month: yesterday is in the previous month")
    backdate(db, 1)                                       # spent yesterday, same month
    live.setattr(SETTINGS, "daily_usd", 1.0)
    check_spend_caps(db)                                  # nothing today: the daily cap is not hit
    live.setattr(SETTINGS, "monthly_usd", 5.0)
    with pytest.raises(SpendCapExceeded, match="Monthly"):
        check_spend_caps(db)


def test_spend_from_last_month_is_not_counted(live, tmp_path):
    db = tmp_path / "a.db"
    save_usage("r1", [usage(99.0)], db_path=db)
    backdate(db, 62)
    live.setattr(SETTINGS, "monthly_usd", 5.0)
    live.setattr(SETTINGS, "daily_usd", 5.0)
    check_spend_caps(db)


def test_mock_mode_is_never_blocked(live, tmp_path):
    db = tmp_path / "a.db"
    save_usage("r1", [usage(50.0)], db_path=db)
    live.setattr(SETTINGS, "daily_usd", 1.0)
    live.setattr(SETTINGS, "llm_provider", "mock")
    check_spend_caps(db)


def test_api_answers_429_when_a_cap_is_reached(live, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import app.api as api

    def refuse():
        raise SpendCapExceeded("Daily LLM spend cap reached (test).")

    live.setattr(api, "check_spend_caps", refuse)
    api.LIMITER.per_minute = 0
    try:
        with TestClient(api.app) as c:
            r = c.post("/assess", json={"question": "Can support use LedgerLens AI with customer tickets?",
                                        "vendor": "LedgerLens AI", "tier": "T3", "users": 35},
                       headers={"X-API-Token": "tok-employee"})
    finally:
        api.LIMITER.per_minute = None
    assert r.status_code == 429 and "cap reached" in r.text
