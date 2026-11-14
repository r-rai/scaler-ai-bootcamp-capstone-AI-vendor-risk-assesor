"""
router.py — decide which line a question takes: policy library, vendor research, full assessment, or refuse.

COURSE MAP
  Agentic RAG guide  "agents.py → The Router" and "tasks.py → Task 1: route".
  Milestone 3 complete:
  * Ports the Router agent's backstory and instructions (including the critical "semantic meaning, not just
    keyword matching" instruction, toggleable via `semantic_prompt=False` for the course ablation exercise)
  * Calls `LLM.small().structured()` so malformed JSON is automatically repaired
  * Includes a high-precision semantic rule mock for offline runs
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .llm import LLM
from .schemas import RouteDecision

ROUTER_BACKSTORY_SEMANTIC = (
    "You are an expert Router for Nimbu Mart's Vendor Risk Assistant. "
    "You make routing decisions based on the semantic MEANING and intent of the user's question, "
    "not just on matching surface keywords."
)

ROUTER_BACKSTORY_BROKEN = (
    "You are a keyword router. Route based only on exact surface words: if a vendor name appears, "
    "always pick 'vendor' regardless of whether the user is asking about internal policy or suitability."
)

ROUTER_RULES = """Classify the user's question into exactly one of these four routes:
- "policy": asks what OUR internal policy, rules, clauses, approval tiers, incident runbooks, or Approved Vendor List say
  (even if a vendor is named, e.g. 'Which tier is Parakeet approved for?' or 'Is LedgerLens on the approved list?').
- "vendor": asks a purely factual question about a specific vendor/tool's public posture (certifications, data regions,
  SSO, incidents, DPA, training policy) WITHOUT asking whether Nimbu Mart can/should use or onboard it.
- "assessment": asks whether a team CAN, MAY, SHOULD, or is ALLOWED to use/adopt/onboard a vendor/tool, OR asks why/whether
  a vendor's posture makes it acceptable for our use (e.g., 'Is LedgerLens SOC 2 so we can use it?'). Needs BOTH policy + vendor facts.
- "refuse": asks you to approve/sign off/bypass checks yourself (only humans approve under AI-5.1), or is completely unrelated
  to vendors, tools, security, or company policy.
Return JSON matching {"route": "policy"|"vendor"|"assessment"|"refuse", "vendor": "<name or null>", "reason": "..."}."""


ASSESS_RE = re.compile(
    r"\b(can|may|should|could|allowed|ok(ay)? to|safe to|assess|review|evaluate|check|onboard|suitab(le|ility)|ready for)\b.*"
    r"\b(use|using|adopt|buy|onboard|vendor|tool|team|support|marketing|product|tickets|chats|docs)\b|"
    r"\bso we can use\b|\bfor our (support|marketing|product|engineering|team)\b|"
    r"\b(assess|review|evaluate|check)\s+[A-Z]",
    re.I,
)
POLICY_RE = re.compile(
    r"\b(our policy|our policies|clause|which tier is .* approved|on the approved vendor list|already approved|"
    r"who approves|who signs off|required by our|must customer|dpdp|cert-in|offboarding|destruction certificate|"
    r"within how many (hours|days)|what does (sec|dp|ai|proc|ir|off)-)\b",
    re.I,
)
REFUSE_APPROVE_RE = re.compile(
    r"\b(please approve|mark .* approved|sign off on|override .* blocker|skip the (checks|review)|just approve)\b",
    re.I,
)
UNRELATED_RE = re.compile(
    r"\b(weather|biryani|cricket|recipe|movie|stock price|flight to|train ticket)\b",
    re.I,
)


def guess_vendor(question: str, known_vendors: Iterable[str]) -> Optional[str]:
    q = question.lower()
    for v in sorted(known_vendors, key=len, reverse=True):
        if v.lower() in q:
            return v
        # Also match the first distinctive word (e.g., "LedgerLens", "Parakeet", "QuillDesk")
        first = v.split()[0].lower()
        if len(first) >= 5 and re.search(rf"\b{re.escape(first)}\b", q):
            return v
    return None


def decide(
    question: str,
    known_vendors: Iterable[str],
    llm: Optional[LLM] = None,
    *,
    semantic_prompt: bool = True,
) -> RouteDecision:
    vendors_list = list(known_vendors)
    vendor = guess_vendor(question, vendors_list)

    def mock() -> dict:
        if not semantic_prompt:
            # Simulates the "Break the Router on purpose" exercise: naive keyword matching
            if vendor:
                return {"route": "vendor", "vendor": vendor, "reason": "Keyword match on vendor name (semantic routing disabled)."}
            return {"route": "policy", "vendor": None, "reason": "No vendor keyword matched."}

        if REFUSE_APPROVE_RE.search(question):
            return {"route": "refuse", "vendor": vendor, "reason": "Only a human approver can approve a vendor or waive checks (AI-5.1)."}
        if UNRELATED_RE.search(question) and not vendor:
            return {"route": "refuse", "vendor": None, "reason": "Question is unrelated to vendor risk or company policy."}
        # Internal status / policy check even if a vendor is named
        if POLICY_RE.search(question) and not re.search(r"\b(can|may|should) (we|our|the)\b", question, re.I):
            return {"route": "policy", "vendor": vendor, "reason": "Asks what OUR internal policy or Approved Vendor List says."}
        if vendor and ASSESS_RE.search(question):
            return {"route": "assessment", "vendor": vendor, "reason": "Asks whether a named tool may be used or assessed against policy."}
        if vendor:
            return {"route": "vendor", "vendor": vendor, "reason": "Factual question about a named vendor's public security posture."}
        return {"route": "policy", "vendor": None, "reason": "Internal policy question."}

    active_llm = LLM.small(budget=llm.budget, tracer=llm.tracer) if (llm and not llm.mock_mode) else (llm or LLM())
    system_prompt = f"{ROUTER_BACKSTORY_SEMANTIC if semantic_prompt else ROUTER_BACKSTORY_BROKEN}\n\n{ROUTER_RULES}"
    return active_llm.structured(
        system_prompt,
        f"Known vendors: {', '.join(vendors_list)}\nQuestion: {question}",
        RouteDecision,
        stage="route",
        mock=mock,
    )
