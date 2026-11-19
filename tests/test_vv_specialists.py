"""vendor_verdict Field Specialists and the adapter onto the existing evidence merge (eval/kaggle_policy.build_facts).

Invented data only: no competition file is read and no LLM is called (stubs stand in for it)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from assessor.llm import LLMError
from assessor.schemas import Claim
from assessor.tracing import BudgetExceeded
from vendor_verdict.adapter import facts_for_case, to_page_claims
from vendor_verdict.agents import specialists
from vendor_verdict.agents.specialists import read_case, read_page
from vendor_verdict.cache import CallCache
from vendor_verdict.prompts import SPECIALIST_SYSTEM, page_user_prompt
from vendor_verdict.schemas import FieldReading, SpecialistOut, SpecialistRaw
from vendor_verdict.state import reading_id

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run_kaggle as rk  # noqa: E402

PAGE = {"page_id": "p1", "url": "https://acme.example/trust", "title": "Trust", "published_date": "2026-01-01",
        "last_updated": "2026-02-01", "text": "Acme invented trust page. " * 5}
PAGE2 = {"page_id": "p2", "url": "https://acme.example/sso", "title": "SSO", "published_date": "2026-03-01",
         "last_updated": "2026-03-02", "text": "Acme invented SSO page. " * 5}
REC_OFFICIAL = {"page": PAGE, "role": "official", "why": "", "quarantine": [], "extract": None}


class StubLLM:
    model = "stub-model"

    def __init__(self, out):
        self.out, self.calls = out, []

    def structured(self, system, user, schema, stage=""):
        self.calls.append((system, user, schema))
        return schema.model_validate(self.out.model_dump())


class RaisingLLM:
    model = "stub-model"

    def __init__(self, exc=None):
        self.exc, self.calls = exc or LLMError("provider error 500"), 0

    def structured(self, system, user, schema, stage=""):
        self.calls += 1
        raise self.exc


def reading(field, value, page_id="p1", quote="q", n=0, family="x"):
    return {"id": reading_id(1, family, page_id, field, n), "iteration": 1, "family": family, "page_id": page_id,
            "field": field, "value": value, "quote": quote}


# ------------------------------------------------------------------------------------------------ brief tests (verbatim)
def test_reading_outside_family_is_dropped_with_note(tmp_path):
    out = SpecialistOut(readings=[FieldReading(page_id="p1", field="sso_saml", value="all_plans", quote="q"),
                                  FieldReading(page_id="p1", field="dpa_available", value="yes", quote="q")])
    r, notes = read_page(StubLLM(out), CallCache(tmp_path), "data", PAGE)
    assert [x["field"] for x in r] == ["dpa_available"] and any("sso_saml" in n for n in notes)


def test_second_read_hits_cache_without_call(tmp_path):
    llm = StubLLM(SpecialistOut(readings=[])); c = CallCache(tmp_path)
    read_page(llm, c, "ai", PAGE); read_page(llm, c, "ai", PAGE)
    assert len(llm.calls) == 1


def test_llm_error_returns_no_readings_and_a_failed_note(tmp_path):
    r, notes = read_page(RaisingLLM(), CallCache(tmp_path), "ai", PAGE)
    assert r == [] and notes[0].startswith("failed")


def test_invalid_enum_value_is_dropped_by_normalise_not_crash():
    pc, notes = to_page_claims([reading("trains_on_customer_data", "sometimes")], REC_OFFICIAL, "Acme Inc.")
    assert pc.claims == [] and notes


def test_adapter_matches_baseline_claim_path():
    # same statements as Claim objects through run_kaggle.assemble_case vs readings through facts_for_case -> identical facts
    assert facts_for_case(CASE, RECS, CHOSEN)[0] == assemble_case_facts_from_claims(CASE, RECS, CLAIMS)


def test_case_read_drops_item_citing_foreign_page(tmp_path):
    out = SpecialistOut(readings=[FieldReading(page_id="pX", field="sso_saml", value="none", quote="q")])
    r, notes = read_case(StubLLM(out), CallCache(tmp_path), "access_incidents", [PAGE], ["sso_saml"])
    assert r == [] and any("pX" in n for n in notes)


# ------------------------------------------------------------------------------------------------ equivalence fixture
CASE = {"case_id": "c9999", "vendor": "Acme", "legal_entity": "Acme Inc.", "official_domain": "acme.example", "as_of": "2026-11-01",
        "tier": "T2", "users": "50", "question": "Can we use Acme?"}
_OLD = {"page_id": "c9999_p1", "url": "https://acme.example/security", "source_type": "official", "last_updated": "2026-01-10"}
_NEW = {"page_id": "c9999_p2", "url": "https://trust.acme.example/", "source_type": "official", "last_updated": "2026-06-15"}
_REG = {"page_id": "c9999_p3", "url": "https://registry.example/acme", "source_type": "regulator", "last_updated": "2026-05-01"}
_BAD = {"page_id": "c9999_p4", "url": "https://acme.example/legal", "source_type": "official", "last_updated": "2026-04-01"}
_ROLES = {"c9999_p1": "official", "c9999_p2": "official", "c9999_p3": "registry", "c9999_p4": "official"}

# page_id -> [(field, [items] for a list field | value for a scalar field)]; the same invented statements feed both paths
_STATEMENTS = {
    "c9999_p1": [("security_certifications", [{"name": "SOC 2", "status": "certified", "date": None, "valid_until": None, "issuer": None, "holder": None}]),
                 ("sso_saml", "paid_plans"),
                 ("trains_on_customer_data", "yes"),
                 ("subprocessors", [{"name": "OldHost", "category": "hosting", "country": "Germany"}])],
    "c9999_p2": [("security_certifications", [{"name": "SOC 2 Type II", "status": "certified", "date": "2026-03-31", "valid_until": None,
                                               "issuer": "Invented Audit LLP", "holder": "Acme"}]),
                 ("sso_saml", "enterprise_only"),
                 ("trains_on_customer_data", "no_on_paid_plans"),
                 ("dpa_available", "on_request"),
                 ("data_regions", {"available": ["United States", "Germany"], "planned": ["India"]}),
                 ("data_retention", {"default_days": 30, "deletion_on_request_days": 7}),
                 ("subprocessors", [{"name": "CloudCo", "category": "hosting", "country": "United States"},
                                    {"name": "ModelCo", "category": "model_provider", "country": "United States"},
                                    {"name": "MailCo", "category": "email", "country": "Ireland"}]),
                 ("security_incidents", [{"date": "2026-02-02", "summary": "Invented outage", "severity": "major", "postmortem_published": True},
                                         {"date": "2025-07-07", "summary": "Invented minor issue", "severity": "minor", "postmortem_published": False}])],
    "c9999_p3": [("security_certifications", [{"name": "ISO/IEC 27001", "status": "certified", "date": "2025-12-01", "valid_until": "2028-11-30",
                                               "issuer": "Invented Cert Body", "holder": "ACME, Inc"},
                                              {"name": "ISO 27701", "status": "certified", "date": "2025-12-01", "valid_until": "2028-11-30",
                                               "issuer": "Invented Cert Body", "holder": "Acme Holdings Ltd"}])],
    "c9999_p4": [("dpa_available", "sometimes")],            # an invalid enum: dropped on both paths
}
_FAMILY = {"security_certifications": "certifications", "data_regions": "data", "data_retention": "data", "dpa_available": "data",
           "trains_on_customer_data": "ai", "subprocessors": "ai", "sso_saml": "access_incidents", "security_incidents": "access_incidents"}
_LIST = {"security_certifications", "subprocessors", "security_incidents"}


def _page(meta):
    return {**meta, "title": "t", "published_date": meta["last_updated"], "text": "invented page text " * 5}


def _claims_of(pid):          # the baseline shape: one Claim per field, a list field holds all of its items
    return [Claim(field=f, value=v, display=str(v)[:120], quote="q") for f, v in _STATEMENTS[pid]]


def _chosen():                # the specialist shape: one reading per list item
    out = {}
    for pid, stmts in _STATEMENTS.items():
        for f, v in stmts:
            items = v if f in _LIST else [v]
            out[(pid, f)] = [reading(f, item, page_id=pid, n=i, family=_FAMILY[f]) for i, item in enumerate(items)]
    return out


RECS = [{"page": _page(m), "role": _ROLES[m["page_id"]], "why": "", "quarantine": [], "extract": None} for m in (_OLD, _NEW, _REG, _BAD)]
CLAIMS = {pid: _claims_of(pid) for pid in _STATEMENTS}
CHOSEN = _chosen()


def assemble_case_facts_from_claims(case, recs, claims):
    recs = [{**r, "extract": {"status": "ok", "claims": [c.model_dump(mode="json") for c in claims.get(r["page"]["page_id"], [])]}} for r in recs]
    rl = SimpleNamespace(evaluate=lambda *a: {}, overall=lambda *a: "CONDITIONAL")
    return rk.assemble_case(case, recs, [], rl)["facts"]


def test_equivalence_fixture_is_non_trivial():
    facts = facts_for_case(CASE, RECS, CHOSEN)[0]
    certs = {c["name"]: c for c in facts["security_certifications"]["value"]}
    assert set(certs) == {"SOC 2 Type II", "ISO 27001"}                  # older official "SOC 2" superseded; foreign holder dropped
    assert certs["ISO 27001"]["valid_until"] == "2028-11-30" and "holder" not in certs["ISO 27001"]
    assert facts["security_certifications"]["evidence_level"] == "documented"
    assert facts["sso_saml"] == {"value": "enterprise_only"}             # the newer official page wins
    assert facts["trains_on_customer_data"] == {"value": "no_on_paid_plans"}
    assert [s["name"] for s in facts["subprocessors"]["value"]] == ["CloudCo", "ModelCo", "MailCo"]
    assert len(facts["security_incidents"]["value"]) == 2
    assert facts["dpa_available"] == {"value": "on_request"}             # the invalid enum on the other page never took part


def test_adapter_returns_notes_and_conflicts():
    _, notes, conflicts = facts_for_case(CASE, RECS, CHOSEN)
    assert any("c9999_p4" in n and "dpa_available" in n for n in notes)
    assert any("c9999_p3" in n and "held by" in n for n in notes)
    assert conflicts == []


def test_same_date_disagreement_is_a_conflict():
    recs = [{**RECS[0], "page": {**RECS[0]["page"], "page_id": "a"}}, {**RECS[0], "page": {**RECS[0]["page"], "page_id": "b"}}]
    chosen = {("a", "sso_saml"): [reading("sso_saml", "none", page_id="a")], ("b", "sso_saml"): [reading("sso_saml", "all_plans", page_id="b")]}
    facts, _, conflicts = facts_for_case(CASE, recs, chosen)
    assert "sso_saml" not in facts and any("sso_saml" in c for c in conflicts)


def test_claimed_only_certificate_is_claimed():
    chosen = {("c9999_p1", "security_certifications"): CHOSEN[("c9999_p1", "security_certifications")]}
    facts = facts_for_case(CASE, RECS[:1], chosen)[0]
    assert facts["security_certifications"]["evidence_level"] == "claimed"


def test_one_bad_certificate_does_not_drop_the_others():
    chosen = {("c9999_p2", "security_certifications"): [
        reading("security_certifications", {"name": "SOC 2 Type II", "status": "certified", "date": "2026-03-31", "issuer": "X"}, page_id="c9999_p2"),
        reading("security_certifications", {"name": "ISO 27001", "status": "rumoured"}, page_id="c9999_p2", n=1)]}
    facts, notes, _ = facts_for_case(CASE, [RECS[1]], chosen)
    assert [c["name"] for c in facts["security_certifications"]["value"]] == ["SOC 2 Type II"]
    assert any("security_certifications" in n for n in notes)


def test_unreadable_page_is_skipped():
    rec = {**RECS[1], "quarantine": ["hit"]}
    facts, _, _ = facts_for_case(CASE, [rec], {k: v for k, v in CHOSEN.items() if k[0] == "c9999_p2"})
    assert facts == {}


def test_adapter_does_not_mutate_the_readings():
    before = repr(CHOSEN)
    facts_for_case(CASE, RECS, CHOSEN)
    assert repr(CHOSEN) == before


def test_registry_page_speaks_only_about_certificates():
    rec = RECS[2]
    pc, _ = to_page_claims([reading("sso_saml", "all_plans", page_id="c9999_p3"),
                            reading("security_certifications", {"name": "ISO 27001", "status": "certified", "holder": "Acme Inc."},
                                    page_id="c9999_p3")], rec, "Acme Inc.")
    assert [c.field for c in pc.claims] == ["security_certifications"] and pc.role == "registry" and pc.date == "2026-05-01"


def test_no_incidents_statement_passes_through_as_an_empty_list():
    facts = facts_for_case(CASE, [RECS[1]], {("c9999_p2", "security_incidents"): [reading("security_incidents", [], page_id="c9999_p2")]})[0]
    assert facts["security_incidents"] == {"value": []}


# ------------------------------------------------------------------------------------------------ specialists (rulings)
def test_specialist_raw_accepts_any_objects():
    raw = SpecialistRaw.model_validate({"readings": [{"field": "x"}, {}]})
    assert raw.readings == [{"field": "x"}, {}]


def test_read_page_forces_the_page_id_and_returns_reading_dicts(tmp_path):
    out = SpecialistRaw(readings=[{"page_id": None, "field": "subprocessors", "value": {"name": "A", "category": "hosting"}, "quote": "q1"},
                                  {"page_id": "wrong", "field": "subprocessors", "value": {"name": "B", "category": "email"}, "quote": "q2"},
                                  {"page_id": "", "field": "trains_on_customer_data", "value": "no", "quote": "q3"}])
    llm = StubLLM(out)
    r, notes = read_page(llm, CallCache(tmp_path), "ai", PAGE, iteration=1)
    assert [x["page_id"] for x in r] == ["p1", "p1", "p1"]
    assert [x["id"] for x in r] == ["1:ai:p1:subprocessors:0", "1:ai:p1:subprocessors:1", "1:ai:p1:trains_on_customer_data:0"]
    assert r[0] == {"id": "1:ai:p1:subprocessors:0", "iteration": 1, "family": "ai", "page_id": "p1", "field": "subprocessors",
                    "value": {"name": "A", "category": "hosting"}, "quote": "q1"}
    system, user, schema = llm.calls[0]
    assert system == SPECIALIST_SYSTEM["ai"] and user == page_user_prompt(PAGE) and schema is SpecialistRaw


def test_one_malformed_reading_does_not_lose_the_good_ones(tmp_path):
    out = SpecialistRaw(readings=[{"field": "dpa_available", "value": "yes", "quote": ""},         # empty quote
                                  {"value": "yes", "quote": "q"},                                  # no field
                                  {"field": "dpa_available", "value": "no", "quote": "q"}])
    r, notes = read_page(StubLLM(out), CallCache(tmp_path), "data", PAGE)
    assert [(x["field"], x["value"]) for x in r] == [("dpa_available", "no")]
    assert len([n for n in notes if "ValidationError" in n]) == 2


def test_cache_hit_returns_the_same_result_and_counts_tokens(tmp_path):
    out = SpecialistRaw(readings=[{"field": "dpa_available", "value": "yes", "quote": "q"}, {"field": "sso_saml", "value": "none", "quote": "q"}])
    c = CallCache(tmp_path)
    first = read_page(StubLLM(out), c, "data", PAGE)
    llm2 = StubLLM(SpecialistRaw(readings=[]))
    assert read_page(llm2, c, "data", PAGE) == first and llm2.calls == []
    rec = c.get(c.key("specialist:data", specialists.version(SPECIALIST_SYSTEM["data"]), "stub-model", specialists.payload(page_user_prompt(PAGE), "data")))
    assert rec["readings"][0]["field"] == "dpa_available" and rec["tokens_in"] > 0 and rec["tokens_out"] > 0


def test_salt_draws_a_separate_sample(tmp_path):
    llm = StubLLM(SpecialistOut(readings=[])); c = CallCache(tmp_path)
    read_page(llm, c, "ai", PAGE); read_page(llm, c, "ai", PAGE, salt="s1")
    assert len(llm.calls) == 2


def test_failed_call_is_not_cached(tmp_path):
    c = CallCache(tmp_path)
    read_page(RaisingLLM(), c, "ai", PAGE)
    llm = StubLLM(SpecialistOut(readings=[]))
    read_page(llm, c, "ai", PAGE)
    assert len(llm.calls) == 1


def test_llm_error_note_names_the_class_only(tmp_path):
    r, notes = read_page(RaisingLLM(LLMError("CANARY-PAGE-TEXT")), CallCache(tmp_path), "ai", PAGE)
    assert notes == ["failed: LLMError"]


def test_only_a_rate_limit_is_retried(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(specialists, "_sleep", slept.append)
    llm = RaisingLLM(LLMError("HTTP 429 too many requests"))
    r, notes = read_page(llm, CallCache(tmp_path), "ai", PAGE)
    assert llm.calls == 5 and len(slept) == 4 and r == [] and notes == ["failed: LLMError"]
    llm = RaisingLLM()
    read_page(llm, CallCache(tmp_path), "ai", PAGE)
    assert llm.calls == 1


def test_budget_exceeded_propagates(tmp_path):
    with pytest.raises(BudgetExceeded):
        read_page(RaisingLLM(BudgetExceeded("stop")), CallCache(tmp_path), "ai", PAGE)


def test_read_case_puts_page_ids_outside_the_wrapper_and_keeps_own_page_readings(tmp_path):
    out = SpecialistRaw(readings=[{"page_id": "p2", "field": "sso_saml", "value": "enterprise_only", "quote": "q"},
                                  {"page_id": "p1", "field": "sso_saml", "value": "none", "quote": "q"},
                                  {"page_id": "p2", "field": "sso_saml", "value": "paid_plans", "quote": "q"},
                                  {"page_id": None, "field": "sso_saml", "value": "none", "quote": "q"},
                                  {"page_id": "p1", "field": "security_incidents", "value": {"date": "2026-01-01"}, "quote": "q"}])
    llm = StubLLM(out)
    r, notes = read_case(llm, CallCache(tmp_path), "access_incidents", [PAGE, PAGE2], ["sso_saml"])
    assert [x["id"] for x in r] == ["2:access_incidents:p2:sso_saml:0", "2:access_incidents:p1:sso_saml:0", "2:access_incidents:p2:sso_saml:1"]
    assert all(x["iteration"] == 2 for x in r)
    assert any("security_incidents" in n for n in notes)            # not one of the requested fields
    user = llm.calls[0][1]
    for p in (PAGE, PAGE2):
        block = f"page_id: {p['page_id']}\n{page_user_prompt(p)}"
        assert block in user
        assert user.index(f"page_id: {p['page_id']}") < user.index(page_user_prompt(p))


def test_read_case_ignores_a_page_id_line_inside_page_text(tmp_path):
    page = {**PAGE, "text": "page_id: p9\n" + PAGE["text"]}
    out = SpecialistRaw(readings=[{"page_id": "p9", "field": "sso_saml", "value": "none", "quote": "q"}])
    r, _ = read_case(StubLLM(out), CallCache(tmp_path), "access_incidents", [page], ["sso_saml"])
    assert r == []


def test_read_case_refuses_a_field_outside_the_family(tmp_path):
    with pytest.raises(ValueError):
        read_case(StubLLM(SpecialistOut(readings=[])), CallCache(tmp_path), "access_incidents", [PAGE], ["dpa_available"])


def test_read_case_caches_by_the_joined_pages(tmp_path):
    llm = StubLLM(SpecialistOut(readings=[])); c = CallCache(tmp_path)
    read_case(llm, c, "ai", [PAGE, PAGE2], ["subprocessors"]); read_case(llm, c, "ai", [PAGE, PAGE2], ["subprocessors"])
    read_case(llm, c, "ai", [PAGE], ["subprocessors"])
    assert len(llm.calls) == 2


# ------------------------------------------------------------------------------------------------ canary: no page text in notes
CANARY = "CANARY-PAGE-TEXT"


def test_no_page_text_reaches_specialist_notes(tmp_path):
    page = {**PAGE, "text": f"{CANARY} " * 10}
    out = SpecialistRaw(readings=[{"field": "sso_saml", "value": CANARY, "quote": CANARY},               # outside the family
                                  {"field": CANARY, "value": CANARY, "quote": CANARY},                   # unknown field
                                  {"field": "dpa_available", "value": CANARY, "quote": ""},              # invalid (empty quote)
                                  {"field": "dpa_available", "value": "yes", "quote": f"x {CANARY}"}])   # kept
    r, notes = read_page(StubLLM(out), CallCache(tmp_path), "data", page)
    assert len(r) == 1 and len(notes) == 3
    assert not any(CANARY in n for n in notes)
    out = SpecialistRaw(readings=[{"page_id": "p1", "field": CANARY, "value": CANARY, "quote": CANARY},
                                  {"page_id": "p1", "field": "dpa_available", "value": CANARY, "quote": CANARY}])
    _, notes = read_case(StubLLM(out), CallCache(tmp_path), "data", [page], ["data_regions"])
    assert notes and not any(CANARY in n for n in notes)


def test_no_page_text_reaches_adapter_notes():
    readings = [reading("trains_on_customer_data", CANARY, page_id="c9999_p3", quote=CANARY),
                reading("security_certifications", {"name": "ISO 27001", "status": CANARY, "issuer": CANARY}, page_id="c9999_p3", quote=CANARY),
                reading("security_certifications", {"name": "ISO 27001", "status": "certified", "holder": CANARY}, page_id="c9999_p3", n=1, quote=CANARY),
                reading("security_incidents", {"date": CANARY, "summary": CANARY}, page_id="c9999_p3", quote=CANARY)]
    pc, notes = to_page_claims(readings, RECS[2], "Acme Inc.")
    assert len(notes) >= 3 and not any(CANARY in n for n in notes)
    facts, notes, conflicts = facts_for_case(CASE, [RECS[2]], {("c9999_p3", "x"): readings})
    assert not any(CANARY in n for n in notes + conflicts)
