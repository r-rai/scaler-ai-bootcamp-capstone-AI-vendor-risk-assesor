"""Tail latency of the page-reading model: one duplicate request for a call that is slow, never for one that failed.

Measured on 283 extract calls of the live runs of 2026-11-16..20 (MiniMax-M3.1-Flash): p50 5 s, p90 16 s, p99 49 s, and the slow
ones were NOT long generations (they ran at 4-50 tokens/s where fast ones run at about 109): a pass of the research loop
waited for a single straggler for 14-22 s in 5 of 10 runs. Lowering the timeout instead of hedging would have lost pages
(simulated: 0.5-3.7% of calls fail outright against 0.08% now), so a call that is still running after `LLM_HEDGE_AFTER_S` gets
ONE more request and the first good answer wins.

The rules these tests pin down:
* extract role and a cloud provider only (the local server has one GPU; main-role calls are legitimately long)
* at most one duplicate per call, and it has no SDK retries of its own (otherwise one call becomes four requests)
* a failure BEFORE the hedge time is raised as before: the hedge is for slow calls, not a hidden retry
* the abandoned request is paid for by the provider, so it is charged to the Budget as an ESTIMATE and traced
* the loser's failure does not count against the circuit breaker, and no thread outlives the run
* the whole call is bounded even if the transport never answers
"""
import threading
import time
from types import SimpleNamespace

import pytest

from assessor import llm as llm_mod
from assessor.config import SETTINGS
from assessor.llm import LLM
from assessor.tracing import Budget, Tracer

CFG = {"provider": "minimax", "model": "m", "base_url": "u", "api_key": "k"}
MSGS = [{"role": "user", "content": "x"}]
HEDGE_S = 0.1


class StatusError(Exception):
    def __init__(self, msg="boom", status_code=None):
        super().__init__(msg)
        self.status_code = status_code


def answer(text="{}", tin=100, tout=20):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout))


class ScriptedClient:
    """The n-th create() call sleeps script[n][0] seconds, then returns script[n][1] or raises it if it is an exception."""

    def __init__(self, script):
        self.script = list(script)
        self.options: list[dict] = []
        self._lock = threading.Lock()
        self.chat = self
        self.completions = self

    def with_options(self, **opts):
        return _Bound(self, opts)

    def create(self, **kwargs):
        return self._run({}, kwargs)

    def _run(self, opts, kwargs):
        with self._lock:
            n = len(self.options)
            self.options.append(opts)
            delay, outcome = self.script[n] if n < len(self.script) else (0.0, "extra")
        time.sleep(delay)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def calls(self) -> int:
        return len(self.options)


class _Bound:
    def __init__(self, client, opts):
        self.client, self.opts = client, opts
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return self.client._run(self.opts, kwargs)


def make_llm(monkeypatch, client, *, role="extract", max_seconds=90.0, elapsed=0.0, tracer=None):
    obj = LLM.__new__(LLM)
    obj.role, obj.tracer = role, tracer
    obj.budget = Budget(max_seconds=max_seconds)
    obj.mock_mode, obj.local, obj.model, obj.provider, obj.targets = False, False, "m", "minimax", [CFG]
    monkeypatch.setattr(type(obj.budget), "elapsed_s", property(lambda self: elapsed))
    monkeypatch.setattr(llm_mod, "_client", lambda cfg: client)
    llm_mod.reset_breakers()
    return obj


@pytest.fixture(autouse=True)
def hedge_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "llm_hedge_after_s", HEDGE_S)
    yield
    llm_mod.reset_breakers()


def timed(fn):
    t0 = time.monotonic()
    out = fn()
    return out, time.monotonic() - t0


# ---------------------------------------------------------------------------------------------------- the point of it

def test_a_stalled_first_attempt_is_beaten_by_the_hedge(monkeypatch):
    client = ScriptedClient([(1.5, "slow"), (0.0, "fast")])
    obj = make_llm(monkeypatch, client)
    out, took = timed(lambda: obj._create(CFG, MSGS))
    assert out == "fast" and client.calls == 2
    assert took < 0.8            # the straggler would have taken 1.5 s


def test_a_call_that_answers_before_the_hedge_time_sends_no_second_request(monkeypatch):
    client = ScriptedClient([(0.0, "ok"), (0.0, "never")])
    obj = make_llm(monkeypatch, client)
    assert obj._create(CFG, MSGS) == "ok"
    time.sleep(HEDGE_S * 3)      # a late timer would fire here
    assert client.calls == 1


def test_the_first_attempt_still_wins_if_it_finishes_before_the_hedge(monkeypatch):
    client = ScriptedClient([(0.3, "first"), (1.5, "second")])
    obj = make_llm(monkeypatch, client)
    out, took = timed(lambda: obj._create(CFG, MSGS))
    assert out == "first" and client.calls == 2 and took < 0.8


def test_at_most_one_hedge_per_call(monkeypatch):
    client = ScriptedClient([(0.7, "a"), (0.7, "b"), (0.0, "c")])
    obj = make_llm(monkeypatch, client)
    assert obj._create(CFG, MSGS) == "a"
    time.sleep(0.4)
    assert client.calls == 2


def test_the_hedge_has_no_sdk_retries_and_the_first_attempt_keeps_its_own(monkeypatch):
    client = ScriptedClient([(0.5, "slow"), (0.0, "fast")])
    obj = make_llm(monkeypatch, client)
    obj._create(CFG, MSGS)
    first, hedge = client.options
    assert first["max_retries"] == obj._call_limits(CFG)[1] and hedge["max_retries"] == 0
    assert first["timeout"] == hedge["timeout"] <= SETTINGS.llm_small_timeout_s


# ------------------------------------------------------------------------------------------------------ what it is not

def test_a_failure_before_the_hedge_time_is_raised_not_duplicated(monkeypatch):
    monkeypatch.setattr(SETTINGS, "llm_hedge_after_s", 0.5)
    client = ScriptedClient([(0.0, StatusError("overloaded", 503)), (0.0, "would-hide-the-failure")])
    obj = make_llm(monkeypatch, client)
    with pytest.raises(StatusError, match="overloaded"):
        obj._create(CFG, MSGS)
    time.sleep(0.2)
    assert client.calls == 1


def test_a_first_attempt_that_fails_after_the_hedge_started_waits_for_the_hedge(monkeypatch):
    client = ScriptedClient([(0.25, StatusError("timed out")), (0.2, "ok")])
    obj = make_llm(monkeypatch, client)
    out, took = timed(lambda: obj._create(CFG, MSGS))
    assert out == "ok" and took < 0.7


def test_when_both_attempts_fail_the_first_error_is_raised(monkeypatch):
    client = ScriptedClient([(0.3, StatusError("first")), (0.3, StatusError("second"))])
    obj = make_llm(monkeypatch, client)
    with pytest.raises(StatusError, match="first"):
        obj._create(CFG, MSGS)
    assert client.calls == 2


@pytest.mark.parametrize("role", ["small", "main"])
def test_only_the_extract_role_is_hedged(monkeypatch, role):
    client = ScriptedClient([(0.4, "slow"), (0.0, "fast")])
    obj = make_llm(monkeypatch, client, role=role)
    assert obj._create(CFG, MSGS) == "slow" and client.calls == 1


def test_a_local_server_is_never_hedged(monkeypatch):
    client = ScriptedClient([(0.4, "slow"), (0.0, "fast")])
    obj = make_llm(monkeypatch, client)
    assert obj._create({**CFG, "local": True}, MSGS) == "slow" and client.calls == 1


def test_a_zero_setting_switches_hedging_off(monkeypatch):
    monkeypatch.setattr(SETTINGS, "llm_hedge_after_s", 0.0)
    client = ScriptedClient([(0.4, "slow"), (0.0, "fast")])
    obj = make_llm(monkeypatch, client)
    assert obj._create(CFG, MSGS) == "slow" and client.calls == 1


def test_no_hedge_when_a_duplicate_could_not_finish_before_the_deadline(monkeypatch):
    client = ScriptedClient([(0.4, "slow"), (0.0, "fast")])
    obj = make_llm(monkeypatch, client, max_seconds=90.0, elapsed=85.0)      # 5 s left
    assert obj._create(CFG, MSGS) == "slow" and client.calls == 1


def test_a_transport_that_never_answers_cannot_hold_the_call_forever(monkeypatch):
    monkeypatch.setattr(SETTINGS, "llm_small_timeout_s", 0.2)
    client = ScriptedClient([(10.0, "never"), (10.0, "never")])
    obj = make_llm(monkeypatch, client, max_seconds=0.0)
    with pytest.raises(TimeoutError):
        obj._create(CFG, MSGS)
    # the stuck attempts are daemon threads: they cannot keep the process alive after the run
    assert all(t.daemon for t in threading.enumerate() if t.name.startswith("llm-attempt"))


# ------------------------------------------------------------------------------------------------- audit and accounting

def test_the_abandoned_request_is_charged_as_an_estimate_and_traced(monkeypatch, tmp_path):
    tracer = Tracer("t", root=tmp_path)
    client = ScriptedClient([(0.8, answer(tin=100, tout=20)), (0.0, answer(tin=100, tout=20))])
    obj = make_llm(monkeypatch, client, tracer=tracer)
    obj.chat("sys", "user", stage="extract#0", json_mode=True)
    hedge = [e for e in tracer.events if e["stage"] == "llm_hedge"]
    assert len(hedge) == 1 and hedge[0]["winner"] == "hedge" and hedge[0]["estimated"] is True
    assert hedge[0]["llm_stage"] == "extract#0" and hedge[0]["abandoned_input_tokens"] == 100
    assert obj.budget.llm_calls == 2 and obj.budget.input_tokens == 200      # winner + the estimated abandoned request
    done = [e for e in tracer.events if e["stage"] == "llm"]
    assert len(done) == 1 and done[0]["duration_ms"] < 700                   # the call's own latency is what the caller waited
    tracer.close()


def test_an_unhedged_call_is_charged_once_and_traces_no_hedge(monkeypatch, tmp_path):
    tracer = Tracer("t", root=tmp_path)
    obj = make_llm(monkeypatch, ScriptedClient([(0.0, answer())]), tracer=tracer)
    obj.chat("sys", "user", stage="extract#0", json_mode=True)
    assert obj.budget.llm_calls == 1 and not [e for e in tracer.events if e["stage"] == "llm_hedge"]
    tracer.close()


def test_the_losers_failure_does_not_count_against_the_circuit_breaker(monkeypatch):
    """Recorded call by call: a later success resets the counter, so checking the counter at the end would hide a stray failure."""
    seen = []
    monkeypatch.setattr(llm_mod, "_breaker_record", lambda provider, ok: seen.append((provider, ok)))
    client = ScriptedClient([(0.4, answer()), (0.0, StatusError("hedge failed"))])
    obj = make_llm(monkeypatch, client)
    obj.chat("sys", "user", stage="extract#0", json_mode=True)
    assert seen == [("minimax", True)]


def test_a_call_that_fails_on_every_attempt_counts_once_against_the_breaker(monkeypatch):
    client = ScriptedClient([(0.3, StatusError("a")), (0.3, StatusError("b"))])
    obj = make_llm(monkeypatch, client)
    with pytest.raises(llm_mod.LLMError):
        obj.chat("sys", "user", stage="extract#0", json_mode=True)
    assert llm_mod._BREAKERS["minimax"]["fails"] == 1


def test_defaults_and_environment_overrides(monkeypatch):
    """10 s: about the p85 of an extract call, so roughly one call in four gets a duplicate (simulated, not tuned on any verdict)."""
    from assessor.config import Settings

    for name in ("LLM_HEDGE_AFTER_S", "VRA_LLM_PARALLEL"):
        monkeypatch.delenv(name, raising=False)
    fresh = Settings()
    assert fresh.llm_hedge_after_s == 10.0 and fresh.llm_parallel == 6
    monkeypatch.setenv("LLM_HEDGE_AFTER_S", "0")
    monkeypatch.setenv("VRA_LLM_PARALLEL", "3")
    changed = Settings()
    assert changed.llm_hedge_after_s == 0.0 and changed.llm_parallel == 3


def test_the_reading_slots_follow_their_own_setting_not_the_fetch_pool(monkeypatch):
    """VRA_MAX_PARALLEL also sizes the search and download pools (Tavily and site concurrency); reading has its own knob."""
    from assessor.vendor_research import VendorResearcher

    monkeypatch.setattr(SETTINGS, "llm_parallel", 3)
    monkeypatch.setattr(SETTINGS, "max_parallel", 9)
    assert VendorResearcher()._llm_slots._value == 3
    monkeypatch.setattr(SETTINGS, "llm_parallel", 7)
    monkeypatch.setattr(SETTINGS, "max_parallel", 2)
    assert VendorResearcher()._llm_slots._value == 7


def test_every_provider_call_goes_through_the_create_seam(monkeypatch):
    """tests/test_egress_pipeline.py proves that restricted text reaches no cloud call by replacing `LLM._create`. That proof only
    holds while `_complete` reaches the provider through that one method, hedge or no hedge."""
    seen = []
    monkeypatch.setattr(LLM, "_create", lambda self, cfg, messages, **kw: seen.append(cfg["provider"]) or answer())

    def forbidden(self, *a, **k):
        raise AssertionError("a provider call bypassed LLM._create")

    monkeypatch.setattr(LLM, "_send", forbidden)
    monkeypatch.setattr(LLM, "_create_once", forbidden)
    obj = make_llm(monkeypatch, ScriptedClient([]))
    obj.chat("sys", "user", stage="extract#0", json_mode=True)
    assert seen == ["minimax"] and obj.budget.llm_calls == 1


def test_a_hedged_call_does_not_leak_its_accounting_into_the_next_call(monkeypatch, tmp_path):
    tracer = Tracer("t", root=tmp_path)
    client = ScriptedClient([(0.6, answer()), (0.0, answer()), (0.0, answer())])    # call 1 is hedged; call 2 answers at once
    obj = make_llm(monkeypatch, client, tracer=tracer)
    obj.chat("sys", "user", stage="extract#0", json_mode=True)
    obj.chat("sys", "user", stage="extract#0", json_mode=True)
    assert len([e for e in tracer.events if e["stage"] == "llm_hedge"]) == 1
    assert obj.budget.llm_calls == 3             # call 1 + its abandoned request + call 2
    tracer.close()

