"""
app/ui.py — Streamlit reviewer workstation (Milestone 5).

Run:   streamlit run app/ui.py          (pip install streamlit)

Features:
  * Streams live node-by-node progress (guard_input → route → vendor_research → match → questionnaire → human_gate → report)
  * Interactive Verdict → Evidence → Trace explorer so a reviewer can inspect every page accepted, rejected, or quarantined
  * Human Approval / Rejection gate form (blocks approval when Outcome is BLOCKED, records decision to report.json + SQLite)
  * Vendor History & SEC-12.1 diff tab + Observability Metrics tab (p50/p95 latency & cost)
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from assessor.config import RUNS_DIR, SETTINGS  # noqa: E402
from assessor.db import get_recent_metrics, get_vendor_history, save_assessment, update_approval_in_db  # noqa: E402
from assessor.graph import Pipeline, new_state  # noqa: E402
from assessor.report import to_markdown  # noqa: E402
from assessor.tracing import Budget, Tracer  # noqa: E402

st.set_page_config(page_title="Vendor Risk Assessor — Nimbu Mart", layout="wide")
st.title("AI Vendor Risk Assessor")
st.caption(
    f"Nimbu Mart · LLM: {SETTINGS.llm_provider} · search: {SETTINGS.search_provider} · "
    "an input to the review, not an approval (AI-5.1)"
)

with st.sidebar:
    st.header("Reviewer Context")
    role_preset = st.selectbox(
        "Caller Identity (Server ACL)",
        ["Security Engineer (security)", "Procurement Lead (procurement)", "General Employee (employees)", "Guest (public)"],
    )
    preset_groups = {
        "Security Engineer (security)": ["public", "employees", "security"],
        "Procurement Lead (procurement)": ["public", "employees", "procurement"],
        "General Employee (employees)": ["public", "employees"],
        "Guest (public)": ["public"],
    }[role_preset]
    groups = st.multiselect("Effective ACL Groups", ["public", "employees", "security", "procurement"], default=preset_groups)
    tier = st.selectbox("Highest data tier", ["T1", "T2", "T3"], index=2)
    users = st.number_input("Number of users", 1, 10000, 35)
    vendor = st.text_input("Vendor (optional, otherwise detected)", "")
    domain = st.text_input("Official domain (optional)", "")

tab_assess, tab_history, tab_metrics = st.tabs(["Assess / Ask Policy", "Vendor History (SEC-12.1)", "Observability & Cost"])

with tab_assess:
    q = st.text_input("Question", "Can our support team use LedgerLens AI to summarise customer tickets?")
    if st.button("Run Assessment / Query", type="primary"):
        SETTINGS.verbose = False
        tracer = Tracer("ui")
        progress_box = st.status("Starting pipeline…", expanded=True)

        def on_step(stage: str, data: dict):
            progress_box.write(f"✓ **{stage}** — `{json.dumps(data, default=str)[:140]}`")

        pipe = Pipeline(tracer=tracer, budget=Budget(), on_step=on_step)
        s = pipe.run(
            new_state(
                q, user_groups=groups, tier=tier, users=int(users),
                vendor=vendor or None, domain=domain or None, requested_by=role_preset,
            )
        )
        tracer.close()
        progress_box.update(label=f"Completed ({tracer.run_id})", state="complete", expanded=False)
        st.session_state["last_state"] = s
        st.session_state["last_events"] = list(tracer.events)

    s = st.session_state.get("last_state")
    if s:
        rep = s.get("report")
        if rep is None:
            st.subheader(f"Route: {s['route']}")
            st.markdown(s["answer"])
        else:
            colour = {"BLOCKED": "red", "CONDITIONAL": "orange", "READY_FOR_APPROVAL": "green"}[rep.overall]
            st.markdown(f"### :{colour}[{rep.overall}] — {rep.overall_reason}")
            rows = [
                {
                    "Requirement": r.title,
                    "Clause": r.clause,
                    "Verdict": r.verdict,
                    "Severity": r.severity,
                    "Reason": r.reason,
                    "Evidence": ", ".join(r.evidence_urls[:2]) or ("policy library" if r.requirement_id == "PROC-1.1" else "—"),
                    "As of": str(r.evidence_as_of or ""),
                }
                for r in rep.results
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

            col_q, col_gate = st.columns([3, 2])
            with col_q:
                if rep.open_questions:
                    st.subheader("Questions for the vendor (NM-SEC-Q)")
                    for i, oq in enumerate(rep.open_questions, 1):
                        st.markdown(f"**{i}. {oq['id']}** ({oq['clause']}) — {oq['question']}")
            with col_gate:
                st.subheader("Human Approval Gate (AI-5.1)")
                st.caption(f"Required approvers for {rep.request.tier}: **{', '.join(rep.approval.approver_roles)}**")
                st.write(f"Current status: **{rep.approval.status}**")
                with st.form("human_gate_form"):
                    approver_name = st.text_input("Approver Full Name", value="")
                    allowed_decisions = ["reject"] if rep.overall == "BLOCKED" else ["approve", "reject"]
                    if rep.overall == "BLOCKED":
                        st.warning("Outcome is BLOCKED: approval is disabled until blocking requirements are resolved.")
                    decision_choice = st.selectbox("Decision", allowed_decisions)
                    comment_txt = st.text_area("Decision Notes", value="")
                    submitted = st.form_submit_button("Record Human Decision")
                    if submitted:
                        if not approver_name.strip():
                            st.error("Please enter the approver's name.")
                        else:
                            rep.approval.status = "approved" if decision_choice == "approve" else "rejected"
                            rep.approval.decided_by = approver_name.strip()
                            rep.approval.decided_on = date.today()
                            rep.approval.comment = comment_txt.strip()
                            run_dir = RUNS_DIR / rep.run_id
                            if run_dir.exists():
                                (run_dir / "report.json").write_text(
                                    json.dumps(rep.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8"
                                )
                                (run_dir / "report.md").write_text(to_markdown(rep), encoding="utf-8")
                            save_assessment(rep)
                            update_approval_in_db(rep.run_id, rep.approval.model_dump(mode="json"))
                            st.success(f"Recorded {rep.approval.status} by {rep.approval.decided_by}.")

            with st.expander("Interactive Trace & Provenance Tree"):
                st.markdown("**Quarantined (Prompt Injection):** " + (", ".join(rep.profile.quarantined) or "None"))
                st.markdown("**Rejected (Identity Gate):** " + (", ".join(rep.profile.rejected) or "None"))
                st.markdown("**Dropped (Rumours / Low Authority):** " + ("; ".join(rep.profile.dropped) or "None"))
                st.json(st.session_state.get("last_events", []))

            with st.expander("Full Review File (Markdown)"):
                st.markdown(to_markdown(rep))

            st.caption(
                f"Run {rep.run_id} · {rep.llm_calls} LLM calls · est. ${rep.cost_usd:.4f} · "
                f"{rep.duration_s}s · approvers: {', '.join(rep.approval.approver_roles)}"
            )

with tab_history:
    st.subheader("Vendor Assessment History & SEC-12.1 Change Diff")
    hist_vendor = st.text_input("Vendor Name", value="LedgerLens AI")
    if st.button("Load History"):
        h = get_vendor_history(hist_vendor)
        st.write(f"Total assessments recorded for **{h['vendor']}**: {h['count']}")
        if h["changes_since_previous"]:
            st.info("Changes since previous assessment: " + "; ".join(h["changes_since_previous"]))
        st.json(h["history"])

with tab_metrics:
    st.subheader("Last 20 Runs — Latency & Cost Budget")
    m = get_recent_metrics(limit=20)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs Tracked", m["runs_counted"])
    c2.metric("p50 Latency (s)", m["p50_duration_s"])
    c3.metric("p95 Latency (s)", m["p95_duration_s"])
    c4.metric("Total Cost (USD)", f"${m['total_cost_usd']:.4f}")
    if m["recent_runs"]:
        st.dataframe(m["recent_runs"], use_container_width=True, hide_index=True)
