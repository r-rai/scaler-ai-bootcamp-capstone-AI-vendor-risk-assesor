"""
app/ui.py — Streamlit reviewer workstation.

Run:   streamlit run app/ui.py

  * Sign in with an API token; ACL groups and approver roles come from the token (assessor/auth.py).
    In dev mode a demo identity can be picked instead; it can ask and assess but never approve.
  * Live node-by-node progress, the review file, and a Verdict → Evidence → Trace explorer.
  * Approval queue and decision form. Decisions go through assessor/approvals.py: the right role for the tier,
    not the requester, two signatures for T3, never for a BLOCKED outcome.
  * Vendor history with SEC-12.1 diffs.
  * Cost dashboard (spend per day / model / stage, local vs cloud, cache savings, caps) and operations view
    (latency, cache contents, evaluation history, audit-chain status). Works without Docker or Grafana.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from assessor import observability as obs  # noqa: E402
from assessor.approvals import ApprovalError, load_report, record_decision  # noqa: E402
from assessor.auth import DEMO_TOKENS, AuthError, Identity, authenticate  # noqa: E402
from assessor.cache import get_cache  # noqa: E402
from assessor.config import RUNS_DIR, SETTINGS  # noqa: E402
from assessor.costs import SpendCapExceeded, check_spend_caps, cost_overview  # noqa: E402
from assessor.db import (  # noqa: E402
    get_audit_log, get_eval_history, get_recent_metrics, get_vendor_history, list_assessments, save_usage,
    verify_audit_chain,
)
from assessor.graph import Pipeline, new_state, shared_components  # noqa: E402
from assessor.report import to_markdown  # noqa: E402
from assessor.tracing import Budget, Tracer  # noqa: E402

st.set_page_config(page_title="Vendor Risk Assessor — Nimbu Mart", layout="wide")
obs.setup_logging()


@st.cache_resource(show_spinner="Loading the policy index and models…")
def components() -> dict:
    return shared_components()


def sign_in() -> Identity | None:
    with st.sidebar:
        st.header("Who you are")
        token = st.text_input("API token", type="password", help="Groups and approver roles come from the token, not from this form.")
        if token:
            try:
                ident = authenticate(token.strip())
            except AuthError as exc:
                st.error(exc.message)
                return None
            st.success(f"{ident.user}")
            st.caption(f"Groups: {', '.join(ident.groups)}  \nApprover roles: {', '.join(ident.roles) or 'none'}")
            return ident
        if SETTINGS.prod:
            st.info("Sign in with your API token to continue.")
            return None
        st.caption("Dev mode: pick a demo identity. Demo identities cannot record decisions; sign in with a demo token "
                   f"({', '.join(k for k, v in DEMO_TOKENS.items() if v['roles'])}) to approve.")
        preset = st.selectbox("Demo identity", ["Security Engineer", "Procurement Lead", "General Employee", "Guest"])
        groups = {"Security Engineer": ("public", "employees", "security"), "Procurement Lead": ("public", "employees", "procurement"),
                  "General Employee": ("public", "employees"), "Guest": ("public",)}[preset]
        return Identity(f"demo:{preset}", groups, (), False)


ident = sign_in()
st.title("AI Vendor Risk Assessor")
st.caption(f"Nimbu Mart · env: {SETTINGS.env} · LLM: {SETTINGS.llm_provider}"
           + (f" (small: {SETTINGS.llm_small_provider})" if SETTINGS.llm_small_provider else "")
           + f" · search: {SETTINGS.search_provider} · an input to the review, not an approval (AI-5.1)")
if ident is None:
    st.stop()

with st.sidebar:
    st.header("Request")
    tier = st.selectbox("Highest data tier", ["T1", "T2", "T3"], index=2)
    users = st.number_input("Number of users", 1, 10000, 35)
    vendor = st.text_input("Vendor (optional, otherwise detected)", "")
    domain = st.text_input("Official domain (optional)", "")
    refresh = st.checkbox("Re-review: ignore cached pages", value=False)


def show_report(rep, events: list | None = None, where: str = "assess") -> None:
    colour = {"BLOCKED": "red", "CONDITIONAL": "orange", "READY_FOR_APPROVAL": "green"}[rep.overall]
    st.markdown(f"### :{colour}[{rep.overall}] — {rep.overall_reason}")
    st.dataframe([{
        "Requirement": r.title, "Clause": r.clause, "Verdict": r.verdict, "Severity": r.severity, "Reason": r.reason,
        "Evidence": ", ".join(r.evidence_urls[:2]) or ("policy library" if r.requirement_id == "PROC-1.1" else "—"),
        "As of": str(r.evidence_as_of or ""),
    } for r in rep.results], width="stretch", hide_index=True)

    col_q, col_gate = st.columns([3, 2])
    with col_q:
        if rep.open_questions:
            st.subheader("Questions for the vendor (NM-SEC-Q)")
            for i, oq in enumerate(rep.open_questions, 1):
                st.markdown(f"**{i}. {oq['id']}** ({oq['clause']}) — {oq['question']}")
    with col_gate:
        decision_form(rep, where)

    with st.expander("Trace & provenance"):
        st.markdown("**Quarantined (prompt injection):** " + (", ".join(rep.profile.quarantined) or "None"))
        st.markdown("**Rejected (identity gate):** " + (", ".join(rep.profile.rejected) or "None"))
        st.markdown("**Dropped (rumours / low authority / empty):** " + ("; ".join(rep.profile.dropped) or "None"))
        st.json(rep.provenance, expanded=False)
        if events is None:
            p = RUNS_DIR / rep.run_id / "trace.jsonl"
            events = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.is_file() else []
        st.json(events, expanded=False)
    with st.expander("Full review file (Markdown)"):
        st.markdown(to_markdown(rep))
    st.caption(f"Run {rep.run_id} · {rep.llm_calls} LLM calls · est. ${rep.cost_usd:.4f} · {rep.duration_s}s")


def decision_form(rep, where: str) -> None:
    ap = rep.approval
    st.subheader("Human approval gate (AI-5.1)")
    st.caption(f"Required for {rep.request.tier}: **{', '.join(ap.approver_roles)}**")
    st.write(f"Status: **{ap.status}**" + (f" · still required: {', '.join(ap.pending_roles)}" if ap.status == "pending" and ap.decisions else ""))
    for d in ap.decisions:
        st.write(f"- {d.decided_at}: {d.decision} by {d.user} as {d.role}" + (f" — {d.comment}" if d.comment else ""))
    if ap.status in ("approved", "rejected"):
        return
    my_roles = [r for r in ap.pending_roles if r in ident.roles]
    if not ident.authenticated:
        st.info("Sign in with an API token to record a decision.")
        return
    if not my_roles:
        st.info("You do not hold an open approver role for this assessment.")
        return
    with st.form(f"gate-{where}-{rep.run_id}"):
        options = ["reject"] if rep.overall == "BLOCKED" else ["approve", "reject"]
        if rep.overall == "BLOCKED":
            st.warning("Outcome is BLOCKED: approval is disabled until the blocking requirements are resolved.")
        role = st.selectbox("Signing as", my_roles)
        choice = st.selectbox("Decision", options)
        comment = st.text_area("Decision notes", value="")
        if st.form_submit_button(f"Record decision as {ident.user}"):
            try:
                new = record_decision(rep.run_id, user=ident.user, roles=ident.roles, decision=choice, comment=comment,
                                      method="ui", role=role)
                obs.count_decision(choice, new.status)
                st.success(f"Recorded. Status is now {new.status}.")
                st.session_state.pop("last_state", None)
                st.session_state["open_run"] = rep.run_id
                st.rerun()
            except ApprovalError as exc:
                st.error(exc.message)


tab_assess, tab_queue, tab_history, tab_cost, tab_ops = st.tabs(
    ["Assess / ask policy", "Approval queue", "Vendor history (SEC-12.1)", "Cost", "Operations"])

with tab_assess:
    q = st.text_input("Question", "Can our support team use LedgerLens AI to summarise customer tickets?", max_chars=SETTINGS.max_question_chars)
    if st.button("Run", type="primary"):
        try:
            check_spend_caps()
        except SpendCapExceeded as exc:
            st.error(str(exc))
            st.stop()
        SETTINGS.verbose = False
        tracer = Tracer("ui")
        box = st.status("Starting pipeline…", expanded=True)
        budget = Budget()
        pipe = Pipeline(tracer=tracer, budget=budget, **components(),
                        on_step=lambda stage, data: box.write(f"✓ **{stage}** — `{json.dumps(data, default=str)[:140]}`"))
        s = new_state(q, user_groups=list(ident.groups), tier=tier, users=int(users), vendor=vendor or None,
                      domain=domain or None, requested_by=ident.user, refresh=refresh)
        try:
            s = pipe.run(s)
        except Exception as exc:  # noqa: BLE001
            box.update(label=f"Failed ({tracer.run_id})", state="error")
            st.error(f"The run failed: {type(exc).__name__}. It was logged for the operator (run {tracer.run_id}).")
            from assessor.errors import record_error

            record_error("ui:run", exc, {"run_id": tracer.run_id})
            st.stop()
        finally:
            tracer.close()
            save_usage(tracer.run_id, budget.calls, route=str(s.get("route") or ""), tools=budget.tools)
            obs.observe_run(s, budget)
        box.update(label=f"Completed ({tracer.run_id})", state="complete", expanded=False)
        st.session_state["last_state"] = s
        st.session_state["last_events"] = list(tracer.events)
        st.session_state.pop("open_run", None)

    s = st.session_state.get("last_state")
    if st.session_state.get("open_run"):
        show_report(load_report(st.session_state["open_run"]))
    elif s:
        if s.get("report") is None:
            st.subheader(f"Route: {s['route']}" + (" · served from the semantic cache" if s.get("cache_hit") else ""))
            st.markdown(s["answer"])
        else:
            show_report(load_report(s["report"].run_id) if s["report"].run_id else s["report"], st.session_state.get("last_events"))

with tab_queue:
    st.subheader("Assessments waiting for a decision")
    pending = [r for r in list_assessments(100, status="pending") if ident.can_read_run(r["requested_by"])]
    if not pending:
        st.write("Nothing is waiting.")
    else:
        st.dataframe(pending, width="stretch", hide_index=True)
        pick = st.selectbox("Open a run", [r["run_id"] for r in pending], format_func=lambda rid: next(
            f"{r['vendor']} · {r['tier']} · {r['overall']} · {rid}" for r in pending if r["run_id"] == rid))
        if pick:
            show_report(load_report(pick), where="queue")

with tab_history:
    st.subheader("Vendor assessment history and SEC-12.1 change diff")
    if SETTINGS.prod and not ident.privileged:
        st.info("History is available to Security and Procurement.")
    else:
        hist_vendor = st.text_input("Vendor name", value="LedgerLens AI")
        if st.button("Load history"):
            h = get_vendor_history(hist_vendor)
            st.write(f"Assessments recorded for **{h['vendor']}**: {h['count']}")
            if h["changes_since_previous"]:
                st.info("Changes since the previous assessment: " + "; ".join(h["changes_since_previous"]))
            st.json(h["history"], expanded=False)

with tab_cost:
    st.subheader("LLM cost")
    if SETTINGS.prod and not ident.privileged:
        st.info("Cost figures are available to Security and Procurement.")
    else:
        days = st.select_slider("Window (days)", [1, 7, 30, 90], value=30)
        c = cost_overview(days)
        sm, tot = c["summary"], c["total"]
        k = st.columns(5)
        k[0].metric("Spend today", f"${sm['today_usd']:.4f}")
        k[1].metric("Spend this month", f"${sm['month_usd']:.4f}")
        k[2].metric("Avg cost / run", f"${c['avg_cost_per_run']:.4f}", help=f"p95 ${c['p95_cost_per_run']:.4f}")
        k[3].metric("Cloud cost avoided", f"${tot.get('avoided_usd') or 0:.4f}", help="By the local model and by caches, priced at the cloud model's rate")
        k[4].metric("Local / cached calls", f"{c['shares']['local_call_share']:.0%} / {c['shares']['cached_call_share']:.0%}")
        for label, cap in (("Daily cap", sm["daily_cap"]), ("Monthly cap", sm["monthly_cap"])):
            if cap["state"] != "off":
                st.progress(min(1.0, cap["used_fraction"]), text=f"{label}: {cap['used_fraction']:.0%} of ${cap['limit_usd']:.2f} ({cap['state']})")
        st.caption(f"Per-assessment cap ${sm['per_run_cap_usd']:.2f} · Tavily searches this month: "
                   f"{sm['tavily_credits_used']} of {sm['tavily_credits_quota']} free credits")
        if c["by_day"]:
            st.markdown("**Spend per day (USD)**")
            st.bar_chart({r["day"]: r["usd"] for r in c["by_day"]})
        left, right = st.columns(2)
        with left:
            st.markdown("**By model**")
            st.dataframe(c["by_model"], width="stretch", hide_index=True)
        with right:
            st.markdown("**By pipeline stage**")
            st.dataframe(c["by_stage"], width="stretch", hide_index=True)
        st.markdown("**Per run**")
        st.dataframe(c["per_run"], width="stretch", hide_index=True)

with tab_ops:
    st.subheader("Operations")
    if SETTINGS.prod and not ident.privileged:
        st.info("The operations view is available to Security and Procurement.")
    else:
        m = get_recent_metrics(limit=50)
        chain = verify_audit_chain()
        k = st.columns(4)
        k[0].metric("Runs tracked", m["runs_counted"])
        k[1].metric("p50 latency (s)", m["p50_duration_s"])
        k[2].metric("p95 latency (s)", m["p95_duration_s"], help="Target: under 90 s per live assessment")
        k[3].metric("Audit chain", "intact" if chain["ok"] else "TAMPERED", help=f"{chain['rows']} records")
        if not chain["ok"]:
            st.error(f"Audit log verification failed at record {chain['first_bad_id']}: {chain['problem']}")
        st.markdown("**Readiness**")
        st.dataframe([{"check": name, **v} for name, v in obs.readiness_checks().items()], width="stretch", hide_index=True)
        st.markdown("**Caches**")
        st.dataframe([{"cache": name, **v} for name, v in get_cache().stats().items()] or [{"cache": "disabled"}],
                     width="stretch", hide_index=True)
        ev = get_eval_history(10)
        if ev:
            st.markdown("**Evaluation history** (quality drift)")
            st.dataframe([{"when": e["ts"], "gold set": e["gold_set"], "llm": e["llm_provider"], "passed": bool(e["passed"]),
                           **{k2: v for k2, v in e["metrics"].items() if isinstance(v, (int, float))}} for e in ev],
                         width="stretch", hide_index=True)
        st.markdown("**Recent runs**")
        st.dataframe(m["recent_runs"], width="stretch", hide_index=True)
        with st.expander("Audit log (latest 50)"):
            st.dataframe([{**e, "details": json.dumps(e["details"])[:200]} for e in get_audit_log(50)],
                         width="stretch", hide_index=True)
