"""A stalled model call must not hold the run past its deadline, and a transient failure must not turn into an
unbounded-thinking retry.

Measured live (Zoho Desk Zia, MiniMax): the same 3.3k-token page took 6 s with reasoning_effort=low and 48 s
without it. The old fallback dropped reasoning_effort after ANY exception, so one timeout became a 45-82 s call.
"""
import pytest

from assessor import llm as llm_mod
from assessor.config import SETTINGS
from assessor.llm import LLM
from assessor.tracing import Budget


class StatusError(Exception):
    def __init__(self, status_code=None, msg="boom"):
        super().__init__(msg)
        self.status_code = status_code


class FakeClient:
    """Records the options and parameters of every call; raises what the script says, then answers."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls: list[dict] = []
        self.options: dict = {}
        self.chat = self
        self.completions = self

    def with_options(self, **opts):
        self.options = opts
        return self

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def make_llm(monkeypatch, client, role="extract", max_seconds=90.0, elapsed=0.0):
    obj = LLM.__new__(LLM)
    obj.role, obj.tracer = role, None
    obj.budget = Budget(max_seconds=max_seconds)
    monkeypatch.setattr(type(obj.budget), "elapsed_s", property(lambda self: elapsed))
    monkeypatch.setattr(llm_mod, "_client", lambda cfg: client)
    return obj


CFG = {"provider": "minimax", "model": "m", "base_url": "u", "api_key": "k"}
MSGS = [{"role": "user", "content": "x"}]


def test_a_timeout_is_not_retried_without_reasoning_effort(monkeypatch):
    client = FakeClient([StatusError(None, "Request timed out.")])
    obj = make_llm(monkeypatch, client)
    with pytest.raises(StatusError):
        obj._create(CFG, MSGS, json_mode=True)
    assert len(client.calls) == 1
    assert client.calls[0].get("reasoning_effort") == SETTINGS.llm_small_reasoning_effort


@pytest.mark.parametrize("status", [429, 500, 503])
def test_server_side_failures_are_not_retried_bare(monkeypatch, status):
    client = FakeClient([StatusError(status)])
    obj = make_llm(monkeypatch, client)
    with pytest.raises(StatusError):
        obj._create(CFG, MSGS, json_mode=True)
    assert len(client.calls) == 1


def test_a_provider_that_rejects_a_parameter_still_gets_the_bare_retry(monkeypatch):
    client = FakeClient([StatusError(400, "unknown parameter reasoning_effort")])
    obj = make_llm(monkeypatch, client)
    assert obj._create(CFG, MSGS, json_mode=True) == "ok"
    assert len(client.calls) == 2
    assert "reasoning_effort" in client.calls[0] and "reasoning_effort" not in client.calls[1]
    assert "response_format" not in client.calls[1]


def test_the_call_timeout_never_exceeds_the_time_left(monkeypatch):
    client = FakeClient([])
    obj = make_llm(monkeypatch, client, role="extract", max_seconds=90.0, elapsed=80.0)   # 10 s left
    obj._create(CFG, MSGS)
    assert client.options["timeout"] <= 10.0
    assert client.options["max_retries"] == 0           # no time for a second attempt


def test_extract_calls_are_capped_below_the_general_timeout(monkeypatch):
    obj = make_llm(monkeypatch, FakeClient([]), role="extract", max_seconds=90.0, elapsed=0.0)
    timeout, retries = obj._call_limits(CFG)
    assert timeout <= SETTINGS.llm_small_timeout_s < SETTINGS.llm_timeout_s
    assert retries <= 2
    main = make_llm(monkeypatch, FakeClient([]), role="main", max_seconds=0.0)
    assert main._call_limits(CFG) == (SETTINGS.llm_timeout_s, 2)


def test_retries_shrink_so_that_all_attempts_fit_in_the_time_left(monkeypatch):
    obj = make_llm(monkeypatch, FakeClient([]), role="extract", max_seconds=90.0, elapsed=40.0)   # 50 s left
    timeout, retries = obj._call_limits(CFG)
    assert (retries + 1) * timeout <= 50.0


def test_a_local_model_is_never_retried_by_the_sdk(monkeypatch):
    obj = make_llm(monkeypatch, FakeClient([]), role="small")
    assert obj._call_limits({**CFG, "local": True})[1] == 0


def test_main_role_keeps_its_sdk_retries_inside_a_normal_budget(monkeypatch):
    """Trimming retries to fit the deadline is for the research roles only: a main-role call with a 90 s budget must
    still retry a 429 or a dropped connection."""
    obj = make_llm(monkeypatch, FakeClient([]), role="main", max_seconds=90.0, elapsed=0.0)
    assert obj._call_limits(CFG) == (SETTINGS.llm_timeout_s, 2)
    near_end = make_llm(monkeypatch, FakeClient([]), role="main", max_seconds=90.0, elapsed=85.0)
    timeout, retries = near_end._call_limits(CFG)
    assert timeout <= 5.0 and retries == 2           # the attempt is still bounded by the time left
