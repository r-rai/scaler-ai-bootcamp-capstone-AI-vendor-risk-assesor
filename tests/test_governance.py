"""Governance controls: identity, the approval rules, the audit chain, budgets and the live-mode crash regression.

Each test names the rule it protects. None of them needs a model or the network.
"""
import json
import sqlite3
import time

import pytest

from assessor import db
from assessor.approvals import ApprovalError, apply_decision, record_decision
from assessor.auth import APPROVERS, AuthError, RateLimiter, TokenDirectory, authenticate, hash_token
from assessor.config import SETTINGS
from assessor.graph import Pipeline, new_state
from assessor.schemas import Approval, AssessmentReport, AssessmentRequest, VendorProfile
from assessor.tracing import Budget, BudgetExceeded, Tracer
from assessor.vendor_research import VendorResearcher

from conftest import AS_OF

SEC = ("sec.lead@nimbumart.example", ["Head of Information Security"])
DPO = ("dpo@nimbumart.example", ["DPO"])


def make_report(overall="CONDITIONAL", tier="T3", requested_by="ravi.k@nimbumart.example", run_id="run-1") -> AssessmentReport:
    return AssessmentReport(
        request=AssessmentRequest(question="q", vendor="V", tier=tier, users=5, requested_by=requested_by),
        profile=VendorProfile(vendor="V", as_of=AS_OF), results=[], overall=overall, overall_reason="r",
        approval=Approval(status="pending", approver_roles=list(APPROVERS[tier])), run_id=run_id)


# --- approval rules (AI-5.1) ---------------------------------------------------------------------

def test_t3_needs_both_signatures_from_two_people():
    rep = make_report()
    apply_decision(rep, user=SEC[0], roles=SEC[1], decision="approve")
    assert rep.approval.status == "pending" and rep.approval.pending_roles == ["DPO"]
    apply_decision(rep, user=DPO[0], roles=DPO[1], decision="approve")
    assert rep.approval.status == "approved"
    assert SEC[0] in rep.approval.decided_by and DPO[0] in rep.approval.decided_by


def test_one_person_cannot_sign_both_roles():
    rep = make_report()
    both = ["Head of Information Security", "DPO"]
    apply_decision(rep, user="x@nimbumart.example", roles=both, decision="approve")
    with pytest.raises(ApprovalError) as e:
        apply_decision(rep, user="x@nimbumart.example", roles=both, decision="approve")
    assert e.value.status == 409 and rep.approval.status == "pending"


def test_blocked_cannot_be_approved_but_can_be_rejected():
    rep = make_report(overall="BLOCKED")
    with pytest.raises(ApprovalError) as e:
        apply_decision(rep, user=SEC[0], roles=SEC[1], decision="approve")
    assert e.value.status == 409
    apply_decision(rep, user=SEC[0], roles=SEC[1], decision="reject")
    assert rep.approval.status == "rejected"


def test_requester_cannot_decide_own_request():
    rep = make_report(requested_by=SEC[0])
    with pytest.raises(ApprovalError) as e:
        apply_decision(rep, user=SEC[0], roles=SEC[1], decision="approve")
    assert e.value.status == 403


def test_approver_must_hold_a_role_for_the_tier():
    rep = make_report()
    with pytest.raises(ApprovalError) as e:
        apply_decision(rep, user="team.lead@nimbumart.example", roles=["Team lead"], decision="approve")
    assert e.value.status == 403
    with pytest.raises(ApprovalError):
        apply_decision(rep, user="", roles=SEC[1], decision="approve")


def test_a_final_decision_cannot_be_changed():
    rep = make_report(tier="T2")
    apply_decision(rep, user=SEC[0], roles=SEC[1], decision="approve")
    assert rep.approval.status == "approved"
    with pytest.raises(ApprovalError):
        apply_decision(rep, user=DPO[0], roles=["Head of Information Security"], decision="reject")


def test_decision_is_persisted_with_an_audit_record(tmp_path):
    path = tmp_path / "a.db"
    rep = make_report(tier="T2")
    db.save_assessment(rep, path, audit_actor="ravi")
    ap = record_decision("run-1", user=SEC[0], roles=SEC[1], decision="approve", comment="ok", db_path=path)
    assert ap.status == "approved"
    stored = db.get_assessment("run-1", path)
    assert stored["approval"]["status"] == "approved" and stored["approval"]["decisions"][0]["user"] == SEC[0]
    actions = [e["action"] for e in db.get_audit_log(db_path=path)]
    assert actions == ["approval.approve", "assessment.created"]


# --- audit log ---------------------------------------------------------------------------------------

def test_audit_chain_detects_edits_and_deletions(tmp_path):
    path = tmp_path / "a.db"
    for i in range(5):
        db.audit("test.event", "alice", f"s{i}", {"n": i}, db_path=path)
    assert db.verify_audit_chain(path) == {"ok": True, "rows": 5, "first_bad_id": None, "problem": ""}

    conn = sqlite3.connect(path)
    with pytest.raises(sqlite3.DatabaseError):          # the table itself refuses edits
        conn.execute("UPDATE audit_log SET actor = 'mallory' WHERE id = 3")
    conn.execute("DROP TRIGGER audit_log_no_update")     # an attacker with file access removes the guard...
    conn.execute("UPDATE audit_log SET actor = 'mallory' WHERE id = 3")
    conn.commit()
    res = db.verify_audit_chain(path)                    # ...and the hash chain still shows it
    assert not res["ok"] and res["first_bad_id"] == 3

    conn.execute("UPDATE audit_log SET actor = 'alice' WHERE id = 3")
    conn.execute("DROP TRIGGER audit_log_no_delete")
    conn.execute("DELETE FROM audit_log WHERE id = 2")
    conn.commit()
    conn.close()
    res = db.verify_audit_chain(path)
    assert not res["ok"] and res["first_bad_id"] == 3 and "link" in res["problem"]


def test_audit_details_are_secret_masked(tmp_path):
    path = tmp_path / "a.db"
    db.audit("test.event", "alice", "s", {"note": "key sk-abcdefghijklmnop1234"}, db_path=path)
    assert "sk-abcdefghijklmnop1234" not in json.dumps(db.get_audit_log(db_path=path))


def test_approval_keeps_the_original_created_at(tmp_path):
    path = tmp_path / "a.db"
    rep = make_report(tier="T2")
    db.save_assessment(rep, path)
    with db._db(path) as c:
        c.execute("UPDATE assessments SET created_at = '2026-01-01T00:00:00+00:00'")
    record_decision("run-1", user=SEC[0], roles=SEC[1], decision="approve", db_path=path)
    assert db.list_assessments(db_path=path)[0]["created_at"].startswith("2026-01-01")


# --- identity ---------------------------------------------------------------------------------------------

def test_token_directory_stores_hashes_and_reloads(tmp_path, monkeypatch):
    f = tmp_path / "tokens.json"
    f.write_text(json.dumps({"tokens": [{"sha256": hash_token("s3cret-token"), "user": "a@x.example",
                                         "groups": ["employees", "security"], "roles": ["DPO"]}]}))
    d = TokenDirectory(str(f), include_demo=False)
    ident = d.lookup("s3cret-token")
    assert ident.user == "a@x.example" and "public" in ident.groups and ident.roles == ("DPO",) and ident.privileged
    assert d.lookup("tok-security") is None             # demo tokens are not loaded
    assert "s3cret-token" not in f.read_text()
    time.sleep(0.01)
    f.write_text(json.dumps({"tokens": []}))            # revoke
    import os
    os.utime(f, (time.time() + 5, time.time() + 5))
    assert d.lookup("s3cret-token") is None


def test_prod_mode_requires_a_token_and_ignores_body_groups(monkeypatch):
    monkeypatch.setattr(SETTINGS, "env", "prod")
    d = TokenDirectory(include_demo=False)
    with pytest.raises(AuthError) as e:
        authenticate(None, ["public", "employees", "security"], d)
    assert e.value.status == 401
    with pytest.raises(AuthError):
        authenticate("tok-security", directory=d)       # a demo token is not valid in prod


def test_dev_mode_anonymous_caller_has_no_approver_roles():
    ident = authenticate(None, ["public", "employees"])
    assert not ident.authenticated and ident.roles == ()
    with pytest.raises(AuthError):
        authenticate("not-a-token")


def test_rate_limiter_blocks_bursts_per_caller():
    rl = RateLimiter(per_minute=3)
    assert [rl.allow("a") for _ in range(4)] == [True, True, True, False]
    assert rl.allow("b")


# --- budget ---------------------------------------------------------------------------------------------

def test_budget_limits_and_stage_breakdown():
    b = Budget(max_usd=1.0, max_tokens=1000, max_seconds=0, max_tool_calls=2)
    b.charge(100, 50, stage="route#0", role="small", model="m")
    b.charge_tool("search", "ddgs")
    assert b.exhausted() is None and b.by_stage["route"]["llm_calls"] == 1 and b.tools == {"search:ddgs": 1}
    b.charge_tool("fetch")
    assert "tool-call" in b.exhausted()
    with pytest.raises(BudgetExceeded):
        b.check()
    b2 = Budget(max_usd=0.000001)
    with pytest.raises(BudgetExceeded):
        b2.charge(10_000, 10_000)


def test_local_and_cached_calls_cost_nothing_and_record_savings():
    b = Budget()
    b.charge(1_000_000, 0, local=True, provider="llamacpp", role="small")
    b.charge(1_000_000, 0, cached=True)
    assert b.spent_usd == 0 and b.saved_usd > 0 and [c["local"] for c in b.calls] == [True, False]


def test_run_ids_are_unique_within_one_second(tmp_path):
    ids = {Tracer("x", root=tmp_path, enabled=False).run_id for _ in range(50)}
    assert len(ids) == 50


# --- regression: the live-mode ZeroDivisionError --------------------------------------------------------

class _EmptyPageResearcher(VendorResearcher):
    def gather(self, vendor, official_domain, **kw):
        pages = super().gather(vendor, official_domain, **kw)
        blank = dict(pages[0], id="W98", url="https://ledgerlens.example/js-only", text="", mock_claims=[])
        blank2 = dict(pages[0], id="W99", url="https://ledgerlens.example/blocked", text="   ", mock_claims=[])
        return [blank, *pages, blank2]


def test_pages_without_text_are_dropped_not_crashed_on():
    prof = _EmptyPageResearcher().research("LedgerLens AI")
    assert any("js-only" in d and "no readable text" in d for d in prof.dropped)
    assert prof.facts["security_certifications"].value          # the rest of the research is unaffected


def test_report_carries_provenance_and_cli_gate_uses_the_same_rules():
    pipe = Pipeline(tracer=None)
    s = pipe.run_plain(new_state("Can our support team use Parakeet Translate for chat?", tier="T3", users=5,
                                 user_groups=["public", "employees", "security"], vendor="Parakeet Translate",
                                 decision="approve", approver="sec.lead", approver_role="Head of Information Security"))
    rep = s["report"]
    assert rep.overall != "BLOCKED"
    assert rep.approval.status == "pending" and rep.approval.pending_roles == ["DPO"]   # one signature is not enough for T3
    pv = rep.provenance
    assert pv["policy_corpus"] and pv["requirements"] and pv["prompts"] and pv["models"]["main"] == "mock"
    assert "## 7. Provenance" in s["answer"] and "still required: DPO" in s["answer"]
