"""The HTTP surface: identity on the server, ownership of runs, the approval endpoint, jobs and metrics."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from assessor.config import SETTINGS  # noqa: E402

EMP = {"X-API-Token": "tok-employee"}
SEC = {"X-API-Token": "tok-security"}
DPO = {"X-API-Token": "tok-dpo"}
PARAKEET_T3 = {"question": "Can our support team use Parakeet Translate for chat?", "vendor": "Parakeet Translate", "tier": "T3", "users": 5}
LEDGER_T3 = {"question": "Can support use LedgerLens AI with customer tickets?", "vendor": "LedgerLens AI", "tier": "T3", "users": 35}


@pytest.fixture(scope="module")
def client():
    import app.api as api

    api.LIMITER.per_minute = 0          # rate limiting is tested separately
    with TestClient(api.app) as c:
        yield c
    api.LIMITER.per_minute = None


@pytest.fixture(scope="module")
def parakeet_run(client):
    r = client.post("/assess", json=PARAKEET_T3, headers=EMP)
    assert r.status_code == 200, r.text
    return r.json()["run_id"]


def test_health_and_ready(client):
    assert client.get("/health").json()["ok"] is True
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["checks"]["policy_index"]["ok"]
    assert r.headers["X-Request-ID"]


def test_groups_come_from_the_token_not_the_body(client):
    # An employee claims Security membership in the body: the Approved Vendor List must stay invisible.
    body = {"question": "Which tier is Parakeet Translate approved for?", "user_groups": ["public", "employees", "security"]}
    r = client.post("/ask", json=body, headers=EMP)
    assert r.status_code == 200
    assert r.json()["user_groups"] == ["public", "employees"]
    assert all(c["doc"] != "approved_vendor_list.md" for c in r.json()["citations"])
    r = client.post("/ask", json=body, headers=SEC)
    assert any(c["doc"] == "approved_vendor_list.md" for c in r.json()["citations"])


def test_invalid_token_is_rejected_and_audited(client):
    assert client.post("/ask", json={"question": "Who approves a T3 tool?"}, headers={"X-API-Token": "nope"}).status_code == 401
    entries = client.get("/audit", headers=SEC).json()
    assert entries["chain"]["ok"] and any(e["action"] == "auth.failed" for e in entries["entries"])


def test_input_is_validated(client):
    assert client.post("/assess", json={**PARAKEET_T3, "tier": "T9"}, headers=EMP).status_code == 422
    assert client.post("/assess", json={**PARAKEET_T3, "users": 0}, headers=EMP).status_code == 422
    assert client.post("/ask", json={"question": "x" * (SETTINGS.max_question_chars + 1)}, headers=EMP).status_code == 422
    assert client.post("/assess", json={**PARAKEET_T3, "official_domain": "evil.example/../x"}, headers=EMP).status_code == 422


def test_requester_is_the_authenticated_user(client, parakeet_run):
    rep = client.get(f"/runs/{parakeet_run}", headers=EMP).json()
    assert rep["request"]["requested_by"] == "ravi.k@nimbumart.example"
    assert rep["approval"]["status"] == "pending" and rep["provenance"]["policy_corpus"]


def test_another_employee_cannot_read_the_run_but_security_can(client, parakeet_run):
    assert client.get(f"/runs/{parakeet_run}", headers={"X-API-Token": "tok-teamlead"}).status_code == 404
    assert client.get(f"/runs/{parakeet_run}", headers=SEC).status_code == 200
    assert "Vendor risk assessment" in client.get(f"/runs/{parakeet_run}/report.md", headers=SEC).text
    assert client.get("/runs/../../etc/passwd", headers=SEC).status_code == 404


def test_approval_needs_identity_role_and_two_signatures(client, parakeet_run):
    url = f"/approve/{parakeet_run}"
    assert client.post(url, json={"decision": "approve"}).status_code == 401                      # anonymous
    assert client.post(url, json={"decision": "approve", "approver": "Head of Security"}, headers=EMP).status_code == 403  # requester, no role
    assert client.post(url, json={"decision": "approve"}, headers={"X-API-Token": "tok-procurement"}).status_code == 403   # no approver role
    r = client.post(url, json={"decision": "approve", "comment": "evidence reviewed"}, headers=SEC)
    assert r.status_code == 200 and r.json()["status"] == "pending"                                 # 1 of 2
    assert client.post(url, json={"decision": "approve"}, headers=SEC).status_code == 409           # same person again
    r = client.post(url, json={"decision": "approve"}, headers=DPO)
    assert r.status_code == 200 and r.json()["status"] == "approved"
    assert client.post(url, json={"decision": "reject"}, headers=DPO).status_code == 409            # final
    md = client.get(f"/runs/{parakeet_run}/report.md", headers=SEC).text
    assert "sec.lead@nimbumart.example as Head of Information Security" in md
    audit = client.get("/audit", params={"subject": parakeet_run}, headers=SEC).json()
    assert [e["action"] for e in audit["entries"]].count("approval.approve") == 2 and audit["chain"]["ok"]


def test_blocked_run_cannot_be_approved(client):
    run_id = client.post("/assess", json=LEDGER_T3, headers=EMP).json()["run_id"]
    r = client.post(f"/approve/{run_id}", json={"decision": "approve"}, headers=SEC)
    assert r.status_code == 409 and "BLOCKED" in r.json()["detail"]
    assert client.post(f"/approve/{run_id}", json={"decision": "reject"}, headers=SEC).json()["status"] == "rejected"


def test_async_job_runs_and_is_owner_scoped(client):
    import time

    job = client.post("/assess", params={"async_job": "true"}, json=PARAKEET_T3, headers=EMP).json()
    for _ in range(100):
        j = client.get(f"/jobs/{job['job_id']}", headers=EMP).json()
        if j["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert j["status"] == "completed" and j["report"]["overall"] and "request_json" not in j
    assert client.get(f"/jobs/{job['job_id']}", headers={"X-API-Token": "tok-teamlead"}).status_code == 404


def test_interrupted_jobs_are_recovered_once(client):
    import app.api as api
    from assessor import db

    req = {"body": PARAKEET_T3, "user": "ravi.k@nimbumart.example", "groups": ["public", "employees"], "roles": [], "requested_by": "ravi.k@nimbumart.example"}
    db.create_job("job-interrupted", "Parakeet Translate", owner="ravi.k@nimbumart.example", request=req)
    db.update_job("job-interrupted", "processing", bump_attempts=True)          # the process died here
    db.create_job("job-exhausted", "Parakeet Translate", owner="x", request=req)
    db.update_job("job-exhausted", "processing", bump_attempts=True)
    db.update_job("job-exhausted", "processing", bump_attempts=True)
    assert api.recover_jobs() == 1
    api._pool().submit(lambda: None).result()
    import time
    for _ in range(100):
        if db.get_job("job-interrupted")["status"] == "completed":
            break
        time.sleep(0.1)
    assert db.get_job("job-interrupted")["status"] == "completed"
    assert db.get_job("job-exhausted")["status"] == "failed"


def test_rate_limit_returns_429(client):
    import app.api as api

    api.LIMITER.per_minute = 2
    api.LIMITER._buckets.clear()
    try:
        codes = [client.get("/runs", headers={"X-API-Token": "tok-public"}).status_code for _ in range(15)]
    finally:
        api.LIMITER.per_minute = 0
    assert 429 in codes and codes[0] == 200


def test_metrics_are_prometheus_text_without_content(client, parakeet_run):
    text = client.get("/metrics").text
    assert "vra_runs" in text and "vra_verdicts" in text and "vra_http_request_seconds" in text
    assert "Parakeet" not in text and "nimbumart" not in text      # no vendor names or users in labels
    assert "vra_audit_chain_ok 1" in text.replace(".0", "")


def test_costs_and_stats_endpoints(client, parakeet_run):
    c = client.get("/costs", headers=SEC).json()
    assert c["total"]["calls"] > 0 and "summary" in c and c["by_stage"]
    assert client.get("/stats", headers=SEC).json()["runs_counted"] >= 1


def test_prod_mode_closes_every_door(client, monkeypatch):
    monkeypatch.setattr(SETTINGS, "env", "prod")
    import assessor.auth as auth

    monkeypatch.setattr(auth, "DIRECTORY", auth.TokenDirectory(include_demo=False))
    for method, url in (("get", "/runs"), ("get", "/stats"), ("get", "/costs"), ("get", "/audit"), ("get", "/vendors/x/history")):
        assert getattr(client, method)(url).status_code == 401, url
    assert client.post("/ask", json={"question": "Who approves a T3 tool?"}).status_code == 401
    assert client.post("/assess", json=PARAKEET_T3, headers=SEC).status_code == 401      # demo tokens are dead in prod
    assert client.get("/health").status_code == 200
