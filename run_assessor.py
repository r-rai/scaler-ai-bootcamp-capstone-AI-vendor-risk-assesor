"""
run_assessor.py — the command line for the AI Vendor Risk Assessor (capstone by Shivank Agrawal).

Runs OFFLINE by default (no key, no internet) against the fictional web in data/vendor_web/corpus.json.

Examples
  python run_assessor.py                                   # the default demo: LedgerLens AI, T3, 35 users
  python run_assessor.py "Can we use Parakeet Translate for support chats?" --tier T3 --users 60
  python run_assessor.py "Can the product team use Parakeet Translate for roadmap docs?" --tier T2 --users 8
  python run_assessor.py "Where must customer personal data be stored?"            # a policy question
  python run_assessor.py "Does QuillDesk have SOC 2?"                               # a vendor fact question
  python run_assessor.py "..." --groups public,employees                            # outside Security: AVL hidden
  python run_assessor.py "..." --interactive --approver "A. Sharma"                 # record a human decision
  python run_assessor.py "..." --engine plain                                       # skip LangGraph

Every assessment writes runs/<timestamp>-assess/{report.md, report.json, profile.json, trace.jsonl, vendor_questions.txt}.
"""

from __future__ import annotations

import argparse
import sys

from assessor import observability as obs
from assessor.config import SETTINGS
from assessor.costs import SpendCapExceeded, check_spend_caps
from assessor.db import save_usage
from assessor.errors import record_error
from assessor.graph import Pipeline, new_state
from assessor.tracing import Budget, Tracer

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows terminals
except Exception:  # noqa: BLE001
    pass

DEFAULT_Q = "Can our support team use LedgerLens AI to summarise customer tickets?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Nimbu Mart AI Vendor Risk Assessor")
    ap.add_argument("question", nargs="?", default=DEFAULT_Q)
    ap.add_argument("--vendor", help="Vendor name (otherwise detected from the question)")
    ap.add_argument("--domain", help="Vendor's official domain, e.g. ledgerlens.example (identity gate)")
    ap.add_argument("--tier", default="T3", choices=["T1", "T2", "T3"], help="Highest data tier the tool will touch")
    ap.add_argument("--users", type=int, default=35)
    ap.add_argument("--groups", default="public,employees,security", help="Comma-separated user groups (ACL)")
    ap.add_argument("--requested-by", default="cli")
    ap.add_argument("--interactive", action="store_true", help="Ask for a human decision on the terminal")
    ap.add_argument("--decision", choices=["approve", "reject", "pending"], help="Record a decision without prompting")
    ap.add_argument("--approver", help="Name recorded with the decision")
    ap.add_argument("--approver-role", help="Approver role the named person signs in (AI-5.1), e.g. 'DPO'. T3 needs two signatures.")
    ap.add_argument("--engine", default="auto", choices=["auto", "plain", "langgraph"])
    ap.add_argument("--max-usd", type=float, default=None, help="Hard budget for live LLM calls")
    ap.add_argument("--refresh", action="store_true", help="Re-review: ignore cached searches and pages")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.quiet:
        SETTINGS.verbose = False
    try:
        check_spend_caps()
    except SpendCapExceeded as exc:
        print(f"Refused: {exc}")
        return 2
    budget = Budget(max_usd=a.max_usd) if a.max_usd is not None else Budget()
    tracer = Tracer("assess")
    print(f"LLM_PROVIDER={SETTINGS.llm_provider}  SEARCH_PROVIDER={SETTINGS.search_provider}  run={tracer.run_id}")

    pipe = Pipeline(tracer=tracer, budget=budget)
    state = new_state(a.question, user_groups=[g.strip() for g in a.groups.split(",") if g.strip()], tier=a.tier,
                      users=a.users, vendor=a.vendor, domain=a.domain, requested_by=a.requested_by,
                      interactive=a.interactive, approver=a.approver, decision=a.decision,
                      approver_role=a.approver_role, refresh=a.refresh)
    try:
        state = pipe.run(state, engine=a.engine)
    except Exception as exc:  # noqa: BLE001
        record_error("cli:run_assessor", exc, {"question": a.question, "vendor": a.vendor, "tier": a.tier, "run_id": tracer.run_id})
        raise
    finally:
        tracer.close()
        save_usage(tracer.run_id, budget.calls, route=str(state.get("route") or ""), tools=budget.tools)
        obs.observe_run(state, budget)

    print("\n" + "=" * 78)
    print(state["answer"])
    if state.get("report") is not None:
        print("=" * 78)
        print(f"Files: {tracer.dir}/report.md · report.json · profile.json · trace.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
