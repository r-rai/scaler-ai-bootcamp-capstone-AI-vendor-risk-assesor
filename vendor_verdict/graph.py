"""The two engines of the per-case graph: plain Python (`run_plain`) and LangGraph (`build_langgraph`). Both run the SAME node
functions (vendor_verdict.nodes) over the same edges, read from one table (`NODE_ORDER` plus the one conditional edge after
`evaluate`), and both apply a node's partial update by replacing keys, so they give the same state, verdicts and handover order.

    validate_sources -> plan -> route_pages -> review_plan -> specialists -> grade_evidence -> evaluate
      -> (continue: specialists | stop: decide_conflicts) -> merge -> verdict -> audit

The LangGraph engine is compiled WITHOUT a checkpointer: the state holds a `Trace` (a live object with a lock), and the runner needs
none. Neither engine writes a file; `mermaid` only returns text.
"""
from __future__ import annotations

from .config import MAX_ITERATIONS
from .nodes import NODES, Agents, next_after_evaluate
from .state import CaseState
from .trace import Trace

NODE_ORDER: list[str] = ["validate_sources", "plan", "route_pages", "review_plan", "specialists", "grade_evidence", "evaluate",
                         "decide_conflicts", "merge", "verdict", "audit"]
LOOP_NODE = "evaluate"
LOOP_BRANCHES = {"specialists": "specialists", "decide_conflicts": "decide_conflicts"}
_MAX_STEPS = len(NODE_ORDER) + 3 * MAX_ITERATIONS      # the whole chain plus every loop pass the evaluator may take, with room


def new_state(case: dict, pages: list[dict]) -> CaseState:
    """The seed state of one case (ledger ruling R1): the case, its raw pages, iteration 1, a fresh trace and empty accumulators."""
    return CaseState(case=case, pages=pages, recs=[], plan={}, router_tags={}, routes={}, review_fixes=[], readings=[], grades={},
                     baseline={}, conflicts=[], decisions={}, facts={}, verdicts={}, iteration=1, targets=[], notes=[], flags={},
                     trace=Trace(str(case.get("case_id", ""))))


def _next(node: str, state: CaseState) -> str | None:
    if node == LOOP_NODE:
        return LOOP_BRANCHES[next_after_evaluate(state)]
    i = NODE_ORDER.index(node)
    return NODE_ORDER[i + 1] if i + 1 < len(NODE_ORDER) else None


def run_plain(state: CaseState, ag: Agents) -> CaseState:
    """Run one case in plain Python. The given state is updated in place and returned."""
    node: str | None = NODE_ORDER[0]
    steps = 0
    while node is not None:
        steps += 1
        if steps > _MAX_STEPS:
            raise RuntimeError("the graph did not reach audit: the evaluator loop did not stop")
        state.update(NODES[node](state, ag))
        node = _next(node, state)
    return state


def build_langgraph(ag: Agents):
    """The same nodes and edges as a compiled LangGraph `StateGraph` (no checkpointer), with the same step bound as `run_plain`
    (LangGraph's `recursion_limit` counts node runs exactly as `run_plain` counts steps)."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(CaseState)
    for name in NODE_ORDER:
        g.add_node(name, lambda s, fn=NODES[name]: fn(s, ag))
    g.add_edge(START, NODE_ORDER[0])
    for a, b in zip(NODE_ORDER, NODE_ORDER[1:]):
        if a != LOOP_NODE:
            g.add_edge(a, b)
    g.add_conditional_edges(LOOP_NODE, next_after_evaluate, LOOP_BRANCHES)
    g.add_edge(NODE_ORDER[-1], END)
    return g.compile().with_config(recursion_limit=_MAX_STEPS)


def mermaid(ag: Agents) -> str:
    """The Mermaid drawing of the LangGraph engine (text only; writing graph.mmd is the runner's job)."""
    return build_langgraph(ag).get_graph().draw_mermaid()
