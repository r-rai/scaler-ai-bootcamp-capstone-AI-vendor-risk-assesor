"""Why a research run stopped must be reported truthfully.

Measured on the live run of 2026-11-20 (reports/live_real_20261120_head): four of the five runs that the report
called "budget: 24s left of the 90s time limit: not enough for another pass" had ALREADY used both passes
(VRA_MAX_ITERATIONS=2). The time check ran after the Plan Evaluator had said "stop: iteration cap" and replaced that
reason, so the report said "Research stopped early" for a run that had finished its work, and the BUDGET_STOPS
metric counted it. Only a run that could have gone on (the evaluator said "continue") is stopped by the clock.
"""
import pytest

from assessor import fetching
from assessor.config import SETTINGS
from assessor.tracing import Budget, Tracer
from tests.test_research_loop import DOM, LiveResearcher, StubWeb, stages


@pytest.fixture()
def acme(monkeypatch, tmp_path):
    """The stubbed-web researcher of test_research_loop (instant search and fetch), built here so no fixture is imported."""
    web = StubWeb()
    monkeypatch.setattr(SETTINGS, "search_provider", "ddgs")
    monkeypatch.setattr(SETTINGS, "pages_per_question", 5)
    monkeypatch.setattr(fetching, "search", web.search)
    monkeypatch.setattr(fetching, "fetch", web.fetch)
    tracer = Tracer("t", root=tmp_path)
    r = LiveResearcher(budget=Budget(max_seconds=0), tracer=tracer)
    r.memory.path = tmp_path / "memory.json"
    r.memory.data = r.memory._read()
    yield r, web, tracer
    tracer.close()


def _clock_left(researcher, seconds):
    """A 90 s deadline with `seconds` left whenever the loop asks (the stubbed web answers instantly)."""
    researcher.budget.max_seconds = 90.0
    researcher._seconds_left = lambda: seconds


def test_an_iteration_cap_is_not_reported_as_a_time_stop(acme):
    r, web, tracer = acme
    r.max_iterations = 1
    _clock_left(r, 10.0)                       # below MIN_SECONDS_FOR_ITERATION, but there is no further pass to refuse
    prof = r.research("Acme AI", DOM)
    assert prof.iterations == 1
    assert prof.stop_reason.startswith("reached 1 iteration")
    assert not stages(tracer, "budget_stop")
    assert not any("stopped early" in n for n in prof.notes)


def test_a_run_that_could_continue_is_still_stopped_by_the_clock(acme):
    r, web, tracer = acme
    r.max_iterations = 2
    _clock_left(r, 10.0)                       # the evaluator wants a second pass; 10 s is not enough for it
    prof = r.research("Acme AI", DOM)
    assert prof.iterations == 1
    assert prof.stop_reason.startswith("budget:") and "not enough for another pass" in prof.stop_reason
    assert [e["reason"] for e in stages(tracer, "budget_stop")] == [prof.stop_reason]
    assert any("stopped early" in n for n in prof.notes)


def test_enough_time_lets_the_second_pass_run(acme):
    r, web, tracer = acme
    r.max_iterations = 2
    _clock_left(r, 60.0)
    prof = r.research("Acme AI", DOM)
    assert prof.iterations == 2
    assert not stages(tracer, "budget_stop")
