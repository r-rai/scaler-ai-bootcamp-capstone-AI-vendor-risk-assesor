"""Data-egress rule at pipeline level: restricted policy text must reach no cloud model through ANY call in the policy path
(answer generation, relevance grader, hallucination grader), including when the local model is down and the small role
would normally fall back to the cloud.

test_egress.py covers answer generation alone; the graders were a separate call site that this file closes.
"""
import json
from types import SimpleNamespace

import pytest

from assessor import llm as llm_mod
from assessor.config import LOCAL_PROVIDERS, SETTINGS
from assessor.graph import Pipeline, new_state
from assessor.llm import LLM
from assessor.tracing import Budget, Tracer

RESTRICTED_Q = "Which tier is Parakeet Translate approved for?"
OPEN_Q = "Where must customer personal data be stored?"
SECURITY = ["public", "employees", "security"]
RESTRICTED_MARKERS = ("AVL-", "Parakeet")          # entries of the Approved Vendor List (Security / Procurement only)


class Wire:
    """Replaces the network. Records every call per provider; can make the local server look down."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.local_down = False

    def create(self, llm, cfg, messages, **kwargs):
        provider = cfg["provider"]
        self.calls.append((provider, "\n".join(m["content"] for m in messages if isinstance(m.get("content"), str))))
        if self.local_down and provider in LOCAL_PROVIDERS:
            raise ConnectionError("local server is down")
        body = json.dumps({"verdict": True, "reason": "supported by the sources"})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body), finish_reason="stop")], usage=None)

    def cloud_calls(self):
        return [(p, t) for p, t in self.calls if p not in LOCAL_PROVIDERS]


@pytest.fixture()
def wire(monkeypatch, tmp_path):
    w = Wire()
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-not-real")
    monkeypatch.setattr(SETTINGS, "llm_provider", "minimax")
    monkeypatch.setattr(SETTINGS, "llm_fallback", True)
    monkeypatch.setattr(LLM, "_create", lambda self, cfg, messages, **kw: w.create(self, cfg, messages, **kw))
    llm_mod.reset_breakers()
    yield w
    llm_mod.reset_breakers()


def run_policy_qa(library, question, groups):
    tracer = Tracer("egress", enabled=False)
    pipe = Pipeline(tracer=tracer, budget=Budget(max_usd=5, max_seconds=0), library=library)
    out = pipe.policy_qa(new_state(question, user_groups=groups))
    return out["policy_answer"], tracer


def leaks(wire):
    return [(p, m) for p, text in wire.cloud_calls() for m in RESTRICTED_MARKERS if m in text.split("Question:")[0]]


@pytest.mark.parametrize("small_provider", ["llamacpp", ""])       # "" = LLM_SMALL_PROVIDER unset: the small role IS the cloud model
@pytest.mark.parametrize("local_down", [False, True])
def test_restricted_text_reaches_no_cloud_call_anywhere_in_the_policy_path(library, wire, monkeypatch, small_provider, local_down):
    monkeypatch.setattr(SETTINGS, "llm_small_provider", small_provider)
    wire.local_down = local_down
    pa, _ = run_policy_qa(library, RESTRICTED_Q, SECURITY)
    assert pa.citations and any(h.doc == "approved_vendor_list.md" for h in pa.citations), "the question must retrieve the restricted list"
    assert wire.cloud_calls() == [] or not leaks(wire), f"restricted policy text was sent to a cloud model: {leaks(wire)[:1]}"
    assert not [t for _, t in wire.cloud_calls() if "AVL-" in t]


def test_without_a_local_model_restricted_answers_are_graded_by_rules_not_the_cloud(library, wire, monkeypatch):
    monkeypatch.setattr(SETTINGS, "llm_small_provider", "")
    pa, _ = run_policy_qa(library, RESTRICTED_Q, SECURITY)
    assert [c for c in wire.cloud_calls() if "grade" in c[1].lower() or "Sources:" in c[1] or "Context:" in c[1]] == []
    assert pa.grader_relevant is not None and pa.grader_grounded is not None       # still graded, by the rule graders


def test_the_guard_is_not_vacuous_ordinary_clauses_still_use_the_cloud(library, wire, monkeypatch):
    """If this fails the tests above would pass for the wrong reason (nothing was ever sent anywhere)."""
    monkeypatch.setattr(SETTINGS, "llm_small_provider", "")
    pa, _ = run_policy_qa(library, OPEN_Q, ["public", "employees"])
    assert not pa.abstained
    assert wire.cloud_calls(), "an ordinary policy question should still reach the configured models"


def test_local_only_llm_has_no_cloud_target(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-not-real")
    monkeypatch.setattr(SETTINGS, "llm_provider", "minimax")
    monkeypatch.setattr(SETTINGS, "llm_small_provider", "llamacpp")
    assert [t["provider"] for t in LLM.small().targets] == ["llamacpp", "minimax"]       # the normal small role may fall back
    assert [t["provider"] for t in LLM.local_only().targets] == ["llamacpp"]
    monkeypatch.setattr(SETTINGS, "llm_small_provider", "")
    assert LLM.local_only().mock_mode                                                   # nothing local configured: rules
