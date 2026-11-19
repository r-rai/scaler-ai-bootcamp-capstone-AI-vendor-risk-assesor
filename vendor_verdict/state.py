"""The shared state of one graph run (one case). Both engines (plain Python and LangGraph) pass this dict between nodes.

A reading is one value a specialist read from one page:
    {"id": str, "iteration": int, "family": str, "page_id": str, "field": str, "value": Any, "quote": str, "display": str}
and its id is built by `reading_id`, so every module that mints or looks up a reading uses the same format. `display` is the
specialist's own short statement ("" when it gave none); like `quote` it is page-derived text, read only by the adapter's absence
guards, and must never reach a trace, a note or the console."""
from __future__ import annotations

from typing import TypedDict

from .trace import Trace


def reading_id(iteration: int, family: str, page_id: str, field: str, n: int) -> str:
    """`n` numbers the readings of one (iteration, family, page, field) in the order the specialist returned them."""
    return f"{iteration}:{family}:{page_id}:{field}:{n}"


class CaseState(TypedDict, total=False):
    """Keys are filled by later tasks; each comment gives the shape only (ledger rulings R1, R3), the owning task defines the rest."""
    case: dict                          # the case row (cases.csv)
    pages: list[dict]                   # the raw eligible pages of the case (R1)
    recs: list[dict]                    # from run_kaggle.triage
    plan: dict[str, list[str]]          # family -> page ids (R3)
    router_tags: dict[str, list[str]]   # page id -> families (R3)
    routes: dict[str, list[str]]
    review_fixes: list[str]
    readings: list[dict]                # reading dicts, see the module docstring
    grades: dict[str, str]
    baseline: dict[str, list[dict]]
    conflicts: list[dict]
    decisions: dict[str, int | None]
    facts: dict
    verdicts: dict[str, str]
    overall: str
    iteration: int
    targets: list[str]
    notes: list[str]
    flags: dict[str, int]
    trace: Trace                        # the handover trace of this run
