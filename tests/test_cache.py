"""Caching rules: what may be reused, and what must never be."""
import numpy as np
import pytest

from assessor import cache as cache_mod
from assessor.cache import Cache, question_anchors, semantic_scope
from assessor.graph import Pipeline, new_state


@pytest.fixture()
def cache(tmp_path):
    c = Cache(tmp_path / "cache.db")
    cache_mod.set_cache(c)
    yield c
    cache_mod.set_cache(None)
    c.close()


def test_kv_ttl_tags_and_purge(cache):
    cache.put("page", "https://a.example/x", {"text": "hello"}, ttl=100, tag="Acme")
    cache.put("page", "https://b.example/y", {"text": "other"}, ttl=-1, tag="Beta")          # already expired
    assert cache.get("page", "https://a.example/x") == {"text": "hello"}
    assert cache.get("page", "https://b.example/y") is None
    assert cache.purge(expired_only=True) == 1
    assert cache.purge(tag="acme") == 1 and cache.get("page", "https://a.example/x") is None
    assert cache.stats()["semantic"]["entries"] == 0


def test_vectors_round_trip(cache):
    v = np.arange(4, dtype=np.float32)
    cache.put_vectors({"k1": v})
    got = cache.get_vectors(["k1", "missing"])
    assert list(got) == ["k1"] and np.array_equal(got["k1"], v)


def test_anchors_separate_questions_that_embed_alike():
    assert question_anchors("Who approves a T2 tool?") != question_anchors("Who approves a T3 tool?")
    assert question_anchors("What does SEC-4.2 require?") != question_anchors("What does SEC-7.1 require?")
    assert question_anchors("Is LedgerLens on the list?", ["LedgerLens AI", "QuillDesk"]) != question_anchors("Is QuillDesk on the list?", ["LedgerLens AI", "QuillDesk"])
    assert question_anchors("Delete within 30 days?") != question_anchors("Delete within 90 days?")
    assert question_anchors("May we use it?") != question_anchors("May we not use it?")
    assert question_anchors("Who approves a tier 3 tool?") == question_anchors("who signs off a T3 tool")


def test_semantic_scope_changes_with_acl_policy_prompt_and_model():
    base = semantic_scope(["employees", "public"], "p1", "pr1", "m1")
    assert base == semantic_scope(["public", "employees"], "p1", "pr1", "m1")
    for other in (semantic_scope(["public", "employees", "security"], "p1", "pr1", "m1"), semantic_scope(["public", "employees"], "p2", "pr1", "m1"),
                  semantic_scope(["public", "employees"], "p1", "pr2", "m1"), semantic_scope(["public", "employees"], "p1", "pr1", "m2")):
        assert other != base


def test_semantic_cache_needs_similarity_and_identical_anchors(cache):
    v = np.array([1, 0, 0], dtype=np.float32)
    near = np.array([0.98, 0.199, 0], dtype=np.float32)
    near /= np.linalg.norm(near)
    far = np.array([0, 1, 0], dtype=np.float32)
    cache.semantic_put("scope", "A", "q1", v, {"answer": "one"}, ttl=100)
    assert cache.semantic_get("scope", "A", near, 0.92)["value"] == {"answer": "one"}
    assert cache.semantic_get("scope", "A", far, 0.92) is None            # different meaning
    assert cache.semantic_get("scope", "B", near, 0.92) is None           # same meaning, different tier/clause/vendor
    assert cache.semantic_get("other-scope", "A", near, 0.92) is None     # different ACL group or policy version


def test_policy_answer_is_served_from_semantic_cache_within_acl_scope_only(cache):
    pipe = Pipeline(tracer=None)
    employee = ["public", "employees"]
    q = "Where must customer personal data be stored?"
    first = pipe.run_plain(new_state(q, user_groups=employee))
    assert first["route"] == "policy" and not first["cache_hit"] and "India" in first["answer"]
    again = pipe.run_plain(new_state(q, user_groups=employee))
    assert again["cache_hit"] and again["answer"] == first["answer"]
    assert again["policy_answer"].citations[0].clause_id == "DP-3.1"
    # Same words, different ACL group: never served from the employee's cache entry.
    sec = pipe.run_plain(new_state(q, user_groups=["public", "employees", "security"]))
    assert not sec["cache_hit"]


def test_restricted_answer_cached_for_security_is_invisible_to_employees(cache):
    """Cross-ACL cache probe: Security asks about the Approved Vendor List, then an employee asks the same thing."""
    pipe = Pipeline(tracer=None)
    q = "Which tier is Parakeet Translate approved for?"
    sec = pipe.run_plain(new_state(q, user_groups=["public", "employees", "security"]))
    assert any(c.doc == "approved_vendor_list.md" for c in sec["policy_answer"].citations)
    emp = pipe.run_plain(new_state(q, user_groups=["public", "employees"]))
    assert not emp["cache_hit"]
    assert all(c.doc != "approved_vendor_list.md" for c in emp["policy_answer"].citations)
    assert "T2" not in emp["answer"] or emp["policy_answer"].abstained


def test_abstentions_and_refusals_are_not_cached(cache):
    pipe = Pipeline(tracer=None)
    q = "What is the parental leave allowance?"
    a = pipe.run_plain(new_state(q, user_groups=["public", "employees"]))
    b = pipe.run_plain(new_state(q, user_groups=["public", "employees"]))
    assert not b.get("cache_hit")
    assert cache.stats()["semantic"]["entries"] == 0 or a["route"] != "policy"
    r = pipe.run_plain(new_state("Please approve LedgerLens AI for the support team"))
    assert r["route"] == "refuse" and not r.get("cache_hit")          # the guardrail runs before any cache


def test_assessments_never_use_the_semantic_cache(cache):
    pipe = Pipeline(tracer=None)
    kw = dict(user_groups=["public", "employees", "security"], tier="T3", users=35, vendor="LedgerLens AI")
    a = pipe.run_plain(new_state("Can support use LedgerLens AI with customer tickets?", **kw))
    b = pipe.run_plain(new_state("Can support use LedgerLens AI with customer tickets?", **kw))
    assert a["report"].overall == b["report"].overall == "BLOCKED" and not b.get("cache_hit")
    assert cache.stats()["semantic"]["entries"] == 0
