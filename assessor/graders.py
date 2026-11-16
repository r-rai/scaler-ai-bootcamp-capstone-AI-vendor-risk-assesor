"""
graders.py — the two checks before an answer leaves the system, plus a self-correction retry loop.

COURSE MAP
  Agentic RAG guide  "agents.py → Answer Grader" (is the retrieved text relevant to the question?) and
                     "Hallucination Grader" (is every statement in the answer supported by the sources?).
  Production RAG     Module 9 "Reasoning, self-checks and guardrails"; Module 10 "Answer metrics" (faithfulness).

Milestone 3 complete:
  * Ports the Answer Grader and Hallucination Grader prompts and calls them via `LLM.small().structured()`
  * Implements the retry loop:
    - If `relevant == False` and `library` is passed, widens retrieval (`k = 8`) with query expansion
    - If `grounded == False`, regenerates once with a strict quote-only prompt; if still ungrounded, abstains cleanly
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .llm import LLM, LLMError
from .policy_rag import PolicyLibrary, tokenize
from .rag_core import expand_concepts
from .schemas import BaseModel, PolicyAnswer


class GradeResult(BaseModel):
    verdict: bool
    reason: str = ""


RELEVANCE_SYSTEM = """You are the Answer Grader (strict relevance inspector).
Read the user's question and the retrieved policy text.
Judge ONLY one thing: does the retrieved context contain the specific rules or facts needed to answer what was asked?
Return JSON: {"verdict": true|false, "reason": "short explanation"}."""

GROUNDED_SYSTEM = """You are the Hallucination Grader (strict fact-checker and auditor).
Compare the draft answer against the source policy text it came from.
Decide whether EVERY factual claim, number, tier, role, or clause ID in the answer is directly supported by the source text.
An answer that invents clauses, numbers, roles, or promises not present in the sources MUST receive verdict=false.
Return JSON: {"verdict": true|false, "reason": "short explanation"}."""


def _grader_llm(llm: Optional[LLM], local_only: bool) -> LLM:
    if llm is None or llm.mock_mode:
        return llm or LLM()
    if local_only:                                    # restricted policy text: never a cloud model, even as a fallback
        return LLM.local_only(budget=llm.budget, tracer=llm.tracer)
    return LLM.small(budget=llm.budget, tracer=llm.tracer)


def _grade(grader_llm: LLM, local_only: bool, system: str, user: str, stage: str, mock) -> GradeResult:
    try:
        return grader_llm.structured(system, user, GradeResult, stage=stage, mock=mock)
    except LLMError:
        if not local_only:
            raise
        # The local model is down and the text may not go anywhere else: fall back to the rule grader.
        return LLM(provider="mock", budget=grader_llm.budget, tracer=grader_llm.tracer).structured(
            system, user, GradeResult, stage=stage, mock=mock)


def grade_relevance(question: str, context: str, llm: Optional[LLM] = None, *, local_only: bool = False) -> GradeResult:
    def mock() -> dict:
        q_toks = set(tokenize(question))
        c_toks = set(tokenize(context))
        q_concepts = set(expand_concepts(q_toks))
        c_concepts = set(expand_concepts(c_toks))
        overlap = len(q_toks & c_toks) + len(q_concepts & c_concepts)
        needed = max(1, len(q_toks) // 4)
        return {
            "verdict": overlap >= needed,
            "reason": f"overlap={overlap} (needed>={needed}) across terms and semantic concepts",
        }

    return _grade(_grader_llm(llm, local_only), local_only, RELEVANCE_SYSTEM,
                  f"Question: {question}\n\nContext:\n{context[:6000]}", "grade_relevance", mock)


def grade_grounded(answer: str, sources: str, llm: Optional[LLM] = None, *, local_only: bool = False) -> GradeResult:
    def mock() -> dict:
        src = set(tokenize(sources))
        body = re.sub(r"^[A-Z]{1,5}(?:-[\dA-Z]+)+(?:\.\d+)*:\s*", "", answer)
        body = re.sub(r"\(source:.*?\)\s*$", "", body).strip()
        # Check for invented clause IDs not in sources
        cited_clauses = set(re.findall(r"\b[A-Z]{2,5}-\d+(?:\.\d+)*\b", answer))
        src_clauses = set(re.findall(r"\b[A-Z]{2,5}-\d+(?:\.\d+)*\b", sources))
        if cited_clauses - src_clauses:
            return {"verdict": False, "reason": f"invented clause ID(s): {sorted(cited_clauses - src_clauses)}"}
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\s*\|\s*", body) if len(tokenize(s)) >= 3]
        unsupported = [s for s in sentences if len(set(tokenize(s)) & src) / max(1, len(set(tokenize(s)))) < 0.55]
        return {
            "verdict": not unsupported,
            "reason": ("all sentences supported" if not unsupported else f"unsupported claim: {unsupported[0][:80]}"),
        }

    return _grade(_grader_llm(llm, local_only), local_only, GROUNDED_SYSTEM,
                  f"Sources:\n{sources[:8000]}\n\nAnswer:\n{answer}", "grade_grounded", mock)


def grade_policy_answer(
    pa: PolicyAnswer,
    llm: Optional[LLM] = None,
    *,
    library: Optional[PolicyLibrary] = None,
    user_groups: Iterable[str] = ("public", "employees"),
) -> PolicyAnswer:
    if pa.abstained or not pa.citations:
        return pa

    # Restricted clauses (Security / Procurement only) may be graded by a model on this machine or by rules, never by a cloud model.
    def restricted() -> bool:
        return any(not (set(h.access) & PolicyLibrary.PUBLIC_GROUPS) for h in pa.citations)

    ctx = "\n\n".join(f"{h.clause_id} {h.heading}\n{h.text}" for h in pa.citations)
    rel = grade_relevance(pa.question, ctx, llm, local_only=restricted())
    pa.grader_relevant = rel.verdict

    # Retry 1: if relevance failed and we have the library, broaden search (k=8 + expanded query)
    if not pa.grader_relevant and library is not None:
        expanded_q = pa.question + " " + " ".join(expand_concepts(tokenize(pa.question)))
        wider_hits = library.search(expanded_q, user_groups, k=8)
        if wider_hits:
            pa.citations = wider_hits
            ctx = "\n\n".join(f"{h.clause_id} {h.heading}\n{h.text}" for h in pa.citations)
            pa.answer = library._extractive_answer(pa.question, wider_hits)
            pa.grader_relevant = grade_relevance(pa.question, ctx, llm, local_only=restricted()).verdict

    grd = grade_grounded(pa.answer, ctx, llm, local_only=restricted())
    pa.grader_grounded = grd.verdict

    # Retry 2: if hallucination grader caught unsupported claims, try strict extractive regeneration once
    if not pa.grader_grounded and library is not None and pa.citations:
        strict_ans = library._extractive_answer(pa.question, pa.citations)
        if grade_grounded(strict_ans, ctx, llm, local_only=restricted()).verdict and pa.mode == "llm":
            pa.answer = strict_ans
            pa.grader_grounded = True

    if not pa.grader_grounded:
        pa.abstained = True
        pa.answer = (
            "I found related policy text but could not produce an answer that is fully supported by it. "
            "Please read the cited clauses directly."
        )
    return pa
