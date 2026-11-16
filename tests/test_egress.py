"""Data-egress rule: policy text restricted to Security / Procurement is never sent to a cloud model.

It may be answered by a model on this machine (llamacpp, ollama) or extractively with no model at all.
"""
from types import SimpleNamespace

SECURITY = ["public", "employees", "security"]
EMPLOYEE = ["public", "employees"]
RESTRICTED_Q = "Parakeet Translate approved tier"
OPEN_Q = "Where must customer personal data be stored?"


class SpyLLM:
    """Looks like a live LLM wrapper; records what it was asked to generate."""

    mock_mode = False

    def __init__(self, provider: str):
        self.provider = provider
        self.prompts: list[str] = []
        self.events: list[dict] = []
        self.tracer = SimpleNamespace(event=lambda name, **kw: self.events.append({"name": name, **kw}))

    def chat(self, system: str, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return "PROC-1.1: Parakeet Translate is approved for T2 only.  (source: Approved Vendor List, v2026.1)"


def test_restricted_clause_is_not_sent_to_a_cloud_model(library):
    llm = SpyLLM("minimax")
    pa = library.answer(RESTRICTED_Q, SECURITY, llm=llm)
    assert llm.prompts == [], "restricted policy text was sent to a cloud provider"
    assert pa.mode == "extractive" and not pa.abstained
    blocked = [e for e in llm.events if e["name"] == "egress_blocked"]
    assert blocked and blocked[0]["provider"] == "minimax"


def test_restricted_clause_may_go_to_a_model_on_this_machine(library):
    llm = SpyLLM("llamacpp")
    pa = library.answer(RESTRICTED_Q, SECURITY, llm=llm)
    assert llm.prompts and "Parakeet" in llm.prompts[0]
    assert pa.mode == "llm"
    assert not [e for e in llm.events if e["name"] == "egress_blocked"]


def test_unrestricted_clauses_may_still_use_the_cloud_model(library):
    llm = SpyLLM("minimax")
    pa = library.answer(OPEN_Q, EMPLOYEE, llm=llm)
    assert llm.prompts, "the rule must not disable generation for ordinary policy questions"
    assert pa.mode == "llm"
    assert not [e for e in llm.events if e["name"] == "egress_blocked"]


def test_an_employee_never_reaches_the_restricted_text_at_all(library):
    llm = SpyLLM("minimax")
    library.answer(RESTRICTED_Q, EMPLOYEE, llm=llm)
    for prompt in llm.prompts:                    # the prompt repeats the question; the sources must hold nothing restricted
        sources = prompt.split("Question:")[0]
        assert "AVL-" not in sources and "Parakeet" not in sources          # the list's own entries
