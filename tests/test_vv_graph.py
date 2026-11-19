"""vendor_verdict agent nodes, the Evaluator loop and the two engines (plain Python and LangGraph).

Invented data only: no competition file is read (nothing under `data_kaggle/`, not even rules_lite: `FakeRL` has its call signature)
and no model is called (stubs stand in for MiniMax and the local model; no call reaches the local server)."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from assessor.llm import LLMError
from assessor.tracing import BudgetExceeded
from eval import run_kaggle as rk
from vendor_verdict import baseline as baseline_mod
from vendor_verdict import nodes
from vendor_verdict.cache import CallCache
from vendor_verdict.config import FAMILIES, FIELD_FAMILY, MAX_ITERATIONS
from vendor_verdict.graph import NODE_ORDER, build_langgraph, mermaid, new_state, run_plain
from vendor_verdict.nodes import Agents
from vendor_verdict.prompts import SPECIALIST_SYSTEM, page_user_prompt
from vendor_verdict.schemas import DecisionOut, GradeOut, PageRoute, PlanOut, RouterOut, SpecialistRaw
from vendor_verdict.state import reading_id

CANARY = "CANARY-PAGE-TEXT"

CASE = {"case_id": "c9001", "vendor": "Acme", "legal_entity": "Acme Widgets Inc.", "official_domain": "acme.example",
        "tier": "T3", "users": "35", "as_of": "2026-11-01", "question": "Can support use Acme with tickets?"}
P1 = {"page_id": "p1", "url": "https://acme.example/trust", "source_type": "official", "title": "Trust", "publisher": "Acme",
      "published_date": "2026-01-01", "last_updated": "2026-03-01",
      "text": "Invented trust page. Single sign-on (SAML) is offered on every plan. We do not train models on customer data. "
              "Customer data is stored in the United States."}
P2 = {"page_id": "p2", "url": "https://acme.example/privacy", "source_type": "official", "title": "Privacy", "publisher": "Acme",
      "published_date": "2026-04-01", "last_updated": "2026-05-01",
      "text": "Invented privacy page. Customer data is deleted after 30 days. A data processing agreement is available on request."}
P3 = {"page_id": "p3", "url": "https://registry.example/acme", "source_type": "regulator", "title": "Registry", "publisher": "Registry",
      "published_date": "2026-02-01", "last_updated": "2026-02-01",
      "text": "Invented registry entry. Acme Widgets Inc. holds SOC 2 Type II, issued 2026-02-01 by Example Audit LLP."}
PAGES = [P1, P2, P3]
BY_ID = {p["page_id"]: p for p in PAGES}

CERT = {"name": "SOC 2 Type II", "status": "certified", "date": "2026-02-01", "valid_until": None, "issuer": "Example Audit LLP",
        "holder": "Acme Widgets Inc."}
# (family, page id) -> the readings the reader stub returns; every quote is a literal substring of its page (grounded)
READ = {
    ("access_incidents", "p1"): [{"field": "sso_saml", "value": "all_plans", "quote": "Single sign-on (SAML) is offered on every plan."}],
    ("ai", "p1"): [{"field": "trains_on_customer_data", "value": "no", "quote": "We do not train models on customer data."}],
    ("data", "p1"): [{"field": "data_regions", "value": {"available": ["United States"], "planned": []},
                      "quote": "Customer data is stored in the United States."}],
    ("data", "p2"): [{"field": "data_retention", "value": {"default_days": 30, "deletion_on_request_days": None},
                      "quote": "Customer data is deleted after 30 days."},
                     {"field": "dpa_available", "value": "on_request", "quote": "A data processing agreement is available on request."}],
    ("certifications", "p3"): [{"field": "security_certifications", "value": CERT, "quote": "Acme Widgets Inc. holds SOC 2 Type II"}],
}
READ_FIELDS = sorted({r["field"] for rs in READ.values() for r in rs})
REQS = [{"id": f"R-{f}", "field": f, "severity": "major"} for f in FIELD_FAMILY]


class FakeRL:
    """A tiny stand-in for rules_lite with its call signature: PASS when the field has a value, else UNKNOWN. Its answer does not
    depend on whether `as_of` is a date or an ISO text."""

    @staticmethod
    def evaluate(facts, requirements, tier, users, as_of):
        assert isinstance(users, int) and str(as_of)[:10] == CASE["as_of"]
        return {r["id"]: ("PASS" if (facts.get(r["field"]) or {}).get("value") not in (None, "", [], {}) else "UNKNOWN")
                for r in requirements}

    @staticmethod
    def overall(verdicts, requirements):
        return "READY_FOR_APPROVAL" if all(v == "PASS" for v in verdicts.values()) else "CONDITIONAL"


class NullCache(CallCache):
    """Never hits, never writes: every test sees every call, and nothing is written to disk."""

    def __init__(self):
        super().__init__(Path("/nonexistent-vv-null-cache"))

    def get(self, key):
        self.misses += 1
        return None

    def put(self, key, record):
        pass


FAMILY_OF_SYSTEM = {v: k for k, v in SPECIALIST_SYSTEM.items()}


def _page_of(user: str) -> str:
    return next(pid for pid, p in BY_ID.items() if f"URL: {p['url']}\n" in user)


class PlannerLLM:
    model = "stub-main"

    def __init__(self, raise_exc=None):
        self.calls, self.raise_exc = [], raise_exc

    def structured(self, system, user, schema, stage="", **kw):
        self.calls.append(stage)
        if self.raise_exc is not None:
            raise self.raise_exc
        assert schema is PlanOut
        return PlanOut(routes=[PageRoute(page_id="p1", families=["access_incidents", "ai", "data"]),
                               PageRoute(page_id="p2", families=["data"]), PageRoute(page_id="p3", families=["certifications"])])


class ReaderLLM:
    """The specialists' model. read_page (iteration 1) -> READ[(family, page)]; read_case (iteration 2) -> the same readings of the
    requested fields for every page given, with `iter2` values overriding by field."""
    model = "stub-reader"

    def __init__(self, fail_pages=(), iter2=None, raise_exc=None, extra=None, iter2_extra=None, fail_iter2=False):
        self.calls, self.fail_pages, self.iter2, self.raise_exc, self.extra = [], set(fail_pages), iter2 or {}, raise_exc, extra or {}
        self.iter2_extra = iter2_extra or {}           # family -> raw items added to every iteration-2 answer (any page_id)
        self.fail_iter2 = fail_iter2                   # every iteration-2 case read fails (iteration 1 succeeds)

    def structured(self, system, user, schema, stage="", **kw):
        assert schema is SpecialistRaw
        family = FAMILY_OF_SYSTEM[system]
        if self.raise_exc is not None:
            raise self.raise_exc
        if user.startswith("Report only these fields: "):
            fields = user.split("\n", 1)[0][len("Report only these fields: "):].split(", ")
            pids = [line[len("page_id: "):] for line in user.split("\n") if line.startswith("page_id: ")]
            self.calls.append({"iteration": 2, "family": family, "pages": pids, "fields": fields})
            if self.fail_iter2 or self.fail_pages & set(pids):
                raise LLMError("provider error 500 " + CANARY)
            out = []
            for pid in pids:
                for r in READ.get((family, pid), []) + self.extra.get((family, pid), []):
                    if r["field"] in fields:
                        out.append({**r, "page_id": pid, "value": self.iter2.get(r["field"], r["value"])})
            return SpecialistRaw(readings=copy.deepcopy(out + self.iter2_extra.get(family, [])))
        pid = _page_of(user)
        self.calls.append({"iteration": 1, "family": family, "pages": [pid], "fields": list(FAMILIES[family])})
        if pid in self.fail_pages:
            raise LLMError("provider error 500 " + CANARY)
        return SpecialistRaw(readings=copy.deepcopy(READ.get((family, pid), []) + self.extra.get((family, pid), [])))


class LocalLLM:
    """Router, grader and decision maker. `raise_on` holds the stages that raise; `families` is the router's answer."""
    model = "stub-local"

    def __init__(self, grade="supported", choice=0, raise_on=(), families=None, exc=None):
        self.grade, self.choice, self.raise_on, self.families = grade, choice, set(raise_on), families
        self.exc = exc or LLMError("local server error " + CANARY)
        self.calls: list[dict] = []

    def structured(self, system, user, schema, stage="", max_repairs=2, **kw):
        self.calls.append({"stage": stage, "user": user})
        if stage in self.raise_on:
            raise self.exc
        if schema is RouterOut:
            return RouterOut(families=list(self.families) if self.families is not None else list(FAMILIES))
        if schema is GradeOut:
            return GradeOut(grade=self.grade)
        if schema is DecisionOut:
            return DecisionOut(choice=self.choice)
        raise AssertionError(f"unexpected schema {schema}")

    def stages(self, stage):
        return [c for c in self.calls if c["stage"] == stage]


def make_ag(grade="supported", choice=0, raise_on=(), reader=None, planner=None, families=None, ablate=(), baseline_dir=None,
            rl=FakeRL, reqs=REQS):
    return Agents(main_llm=planner or PlannerLLM(), reader_llm=reader or ReaderLLM(),
                  local_llm=LocalLLM(grade=grade, choice=choice, raise_on=raise_on, families=families), cache=NullCache(), rl=rl,
                  reqs=reqs, baseline_dir=baseline_dir, baseline_model="stub-baseline", salt="", ablate=frozenset(ablate))


AG = make_ag()
AG_ALWAYS_UNSURE = make_ag(grade="unsure")
AG_ALL_SUPPORTED = make_ag(grade="supported")


def baseline_of(changes=None, drop=()):
    """Invented baseline readings (iteration 0) equal to READ, with `changes` overriding values by field and `drop` fields left out.
    A list-valued field is ONE reading whose value is the whole list, as baseline_readings returns it."""
    out: dict[str, list[dict]] = {}
    for (family, pid), rs in READ.items():
        for n, r in enumerate(rs):
            if r["field"] in drop:
                continue
            value = (changes or {}).get(r["field"], r["value"])
            if r["field"] == "security_certifications" and not isinstance(value, list):
                value = [value]
            out.setdefault(pid, []).append({"id": reading_id(0, family, pid, r["field"], n), "iteration": 0, "family": family,
                                            "page_id": pid, "field": r["field"], "value": copy.deepcopy(value),
                                            "quote": r["quote"], "display": ""})
    return out


@pytest.fixture
def with_baseline(monkeypatch, tmp_path):
    """Patch the cache-only baseline loader; returns a function that installs a given baseline and gives the baseline_dir to use."""
    def install(base):
        calls = []

        def fake(model, recs, cache_dir, system=None):
            calls.append((model, cache_dir))
            return copy.deepcopy(base)
        monkeypatch.setattr(baseline_mod, "baseline_readings", fake)
        return tmp_path, calls
    return install


def agents_of(state):
    return [h["agent"] for h in state["trace"].to_json()["handovers"]]


# ---------------------------------------------------------------- the brief's tests
def test_both_engines_give_identical_verdicts_and_handover_order():
    a, b = run_plain(new_state(CASE, PAGES), AG), build_langgraph(AG).invoke(new_state(CASE, PAGES))
    assert a["verdicts"] == b["verdicts"] and a["overall"] == b["overall"]
    assert [h["agent"] for h in a["trace"].to_json()["handovers"]] == [h["agent"] for h in b["trace"].to_json()["handovers"]]


def test_loop_stops_after_max_iterations():
    s = run_plain(new_state(CASE, PAGES), AG_ALWAYS_UNSURE)
    assert s["iteration"] == MAX_ITERATIONS


def test_no_flags_means_one_iteration():
    assert run_plain(new_state(CASE, PAGES), AG_ALL_SUPPORTED)["iteration"] == 1


def test_iteration2_ablation_runs_once():
    assert run_plain(new_state(CASE, PAGES), replace(AG_ALWAYS_UNSURE, ablate=frozenset({"iteration2"})))["iteration"] == 1


def test_verdicts_come_from_rules_lite_on_merged_facts():
    s = run_plain(new_state(CASE, PAGES), AG)
    assert s["verdicts"] == AG.rl.evaluate(s["facts"], AG.reqs, CASE["tier"], int(CASE["users"]), CASE["as_of"])


def test_mermaid_names_every_agent():
    assert all(n in mermaid(AG) for n in NODE_ORDER)


# ---------------------------------------------------------------- state, order and engines
def test_new_state_seeds_every_key():
    s = new_state(CASE, PAGES)
    assert s["case"] is CASE and s["pages"] is PAGES and s["iteration"] == 1 and s["trace"].case_id == "c9001"
    for k in ("recs", "review_fixes", "readings", "conflicts", "notes", "targets"):
        assert s[k] == []
    for k in ("plan", "router_tags", "routes", "grades", "baseline", "decisions", "facts", "verdicts", "flags"):
        assert s[k] == {}


def test_node_order_is_the_execution_order_and_every_node_hands_over_once():
    assert NODE_ORDER == ["validate_sources", "plan", "route_pages", "review_plan", "specialists", "grade_evidence", "evaluate",
                          "decide_conflicts", "merge", "verdict", "audit"]
    assert agents_of(run_plain(new_state(CASE, PAGES), make_ag())) == NODE_ORDER
    loop = ["specialists", "grade_evidence", "evaluate"]
    assert agents_of(run_plain(new_state(CASE, PAGES), make_ag(grade="unsure"))) == NODE_ORDER[:7] + loop + NODE_ORDER[7:]


def test_langgraph_nodes_are_exactly_node_order():
    g = build_langgraph(AG).get_graph()
    assert set(g.nodes) - {"__start__", "__end__"} == set(NODE_ORDER)


def test_engines_agree_through_the_loop_and_conflicts(with_baseline):
    base_dir, _ = with_baseline(baseline_of(changes={"sso_saml": "none"}))
    ag = make_ag(grade="unsure", baseline_dir=base_dir)
    a, b = run_plain(new_state(CASE, PAGES), ag), build_langgraph(ag).invoke(new_state(CASE, PAGES))
    assert a["iteration"] == b["iteration"] == 2
    assert a["verdicts"] == b["verdicts"] and a["overall"] == b["overall"] and a["facts"] == b["facts"]
    assert a["decisions"] == b["decisions"] and a["decisions"]
    assert agents_of(a) == agents_of(b)
    assert without_trace(a) == without_trace(b)


def without_trace(state):
    return {k: v for k, v in state.items() if k != "trace"}


def test_engines_agree_on_the_whole_state_when_the_planner_routes():
    # the router tags every page with certifications only, so the planner's routes are what reach the other families
    ag = make_ag(grade="unsure", families=["certifications"])
    a, b = run_plain(new_state(CASE, PAGES), ag), build_langgraph(ag).invoke(new_state(CASE, PAGES))
    assert a["plan"] and a["plan"]["data"] == ["p1", "p2"] and a["routes"]["data"] == ["p1", "p2"]
    assert without_trace(a) == without_trace(b) and agents_of(a) == agents_of(b)


def test_pages_survive_triage_and_facts_flow_into_verdicts():
    s = run_plain(new_state(CASE, PAGES), make_ag())
    assert [r["role"] for r in s["recs"]] == ["official", "official", "registry"] and all(rk.readable(r) for r in s["recs"])
    assert set(s["facts"]) == set(READ_FIELDS)
    assert all(s["verdicts"][f"R-{f}"] == "PASS" for f in READ_FIELDS)
    assert s["routes"]["certifications"] == ["p1", "p2", "p3"] and s["routes"]["data"] == ["p1", "p2"]     # registry: certifications only


# ---------------------------------------------------------------- the evaluator loop
def test_always_unsure_rereads_only_target_fields_in_iteration_2():
    reader = ReaderLLM()
    s = run_plain(new_state(CASE, PAGES), make_ag(grade="unsure", reader=reader))
    assert s["iteration"] == MAX_ITERATIONS and s["targets"] == []            # nothing left pending after the stop
    second = [c for c in reader.calls if c["iteration"] == 2]
    assert [c["family"] for c in second] == list(FAMILIES)                     # one read_case per family, in config order
    for c in second:
        assert c["fields"] == [f for f in FAMILIES[c["family"]] if f in READ_FIELDS]       # targets ∩ family, config order
        assert c["pages"] == s["routes"][c["family"]]
    assert {r["iteration"] for r in s["readings"]} == {1, 2}
    assert all(r["id"] in s["grades"] for r in s["readings"])                  # iteration-2 readings are graded too


def test_iteration_2_skips_a_family_without_targets():
    reader = ReaderLLM()
    local = LocalLLM(grade="supported")
    ag = replace(make_ag(reader=reader), local_llm=local)
    # only the sso reading is unsure: patch the stub's answer by the reading's field (the grader prompt starts with the field)
    local.structured = (lambda orig: (lambda system, user, schema, stage="", max_repairs=2, **kw:
                        GradeOut(grade="unsure") if schema is GradeOut and user.startswith("Field: sso_saml\n")
                        else orig(system, user, schema, stage=stage, max_repairs=max_repairs)))(local.structured)
    s = run_plain(new_state(CASE, PAGES), ag)
    second = [c for c in reader.calls if c["iteration"] == 2]
    assert [(c["family"], c["fields"]) for c in second] == [("access_incidents", ["sso_saml"])]
    assert s["iteration"] == 2


def test_all_supported_and_equal_to_the_baseline_is_one_iteration(with_baseline):
    base_dir, calls = with_baseline(baseline_of())
    reader = ReaderLLM()
    s = run_plain(new_state(CASE, PAGES), make_ag(baseline_dir=base_dir, reader=reader))
    assert calls == [("stub-baseline", base_dir)] and s["baseline"]
    assert s["iteration"] == 1 and all(c["iteration"] == 1 for c in reader.calls)
    assert s["conflicts"] == []                    # the cert's single item equals the baseline's whole one-item list


def test_a_value_differing_from_the_baseline_triggers_iteration_2(with_baseline):
    base_dir, _ = with_baseline(baseline_of(changes={"sso_saml": "none"}))
    reader = ReaderLLM()
    s = run_plain(new_state(CASE, PAGES), make_ag(baseline_dir=base_dir, reader=reader))
    assert s["iteration"] == 2
    assert [(c["family"], c["fields"]) for c in reader.calls if c["iteration"] == 2] == [("access_incidents", ["sso_saml"])]


def test_a_field_only_one_side_has_is_not_a_difference(with_baseline):
    extra = {("ai", "p1"): [{"field": "subprocessors", "value": {"name": "Example Cloud", "category": "hosting", "country": "US"},
                             "quote": "We do not train models on customer data."}]}
    base = baseline_of(drop=("sso_saml",))
    base["p2"].append({"id": reading_id(0, "access_incidents", "p2", "security_incidents", 0), "iteration": 0,
                       "family": "access_incidents", "page_id": "p2", "field": "security_incidents", "value": [],
                       "quote": "Customer data is deleted after 30 days.", "display": ""})
    base_dir, _ = with_baseline(base)
    s = run_plain(new_state(CASE, PAGES), make_ag(baseline_dir=base_dir, reader=ReaderLLM(extra=extra)))
    assert s["iteration"] == 1


def test_no_baseline_dir_means_no_baseline_load(with_baseline):
    _, calls = with_baseline(baseline_of(changes={"sso_saml": "none"}))
    s = run_plain(new_state(CASE, PAGES), make_ag(baseline_dir=None))
    assert calls == [] and s["baseline"] == {} and s["iteration"] == 1


def test_an_iteration_2_item_citing_a_page_outside_the_family_routes_is_dropped():
    # p3 (registry) is routed to certifications only; the data specialist's case read cites it anyway
    spoof = [{"page_id": "p3", "field": "dpa_available", "value": "none", "quote": "Invented registry entry."},
             {"page_id": "p9", "field": "dpa_available", "value": "none", "quote": "Invented registry entry."}]
    reader = ReaderLLM(iter2_extra={"data": spoof})
    s = run_plain(new_state(CASE, PAGES), make_ag(grade="unsure", reader=reader))
    data_read = next(c for c in reader.calls if c["iteration"] == 2 and c["family"] == "data")
    assert "p3" not in data_read["pages"] and s["iteration"] == 2
    assert not any(r["field"] == "dpa_available" and r["page_id"] in ("p3", "p9") for r in s["readings"])
    dropped = [n for n in s["notes"] if n.startswith("specialists: data: case read: dropped") and "is not one of the pages read" in n]
    assert len(dropped) == 2
    assert s["facts"]["dpa_available"]["value"] == "on_request"


def test_an_abstained_iteration_conflict_keeps_the_iteration_2_reading():
    s = run_plain(new_state(CASE, PAGES), make_ag(grade="unsure", choice=None, reader=ReaderLLM(iter2={"sso_saml": "none"})))
    assert [c["id"] for c in s["conflicts"]] == ["p1:sso_saml"] and s["decisions"] == {"p1:sso_saml": None}
    assert s["facts"]["sso_saml"]["value"] == "none"


# ---------------------------------------------------------------- decisions
def test_decide_receives_the_page_of_the_conflict(with_baseline):
    base_dir, _ = with_baseline(baseline_of(changes={"sso_saml": "none"}))
    ag = make_ag(baseline_dir=base_dir, choice=1)
    s = run_plain(new_state(CASE, PAGES), ag)
    decisions = ag.local_llm.stages("decision")
    assert len(decisions) == 1 and page_user_prompt(P1) in decisions[0]["user"]
    assert [c["id"] for c in s["conflicts"]] == ["p1:sso_saml"] and s["decisions"] == {"p1:sso_saml": 1}
    assert s["facts"]["sso_saml"]["value"] == "none"                          # the chosen (baseline) candidate stands


# ---------------------------------------------------------------- ablations
def test_router_ablation_tags_every_family_without_a_call():
    ag = make_ag(ablate={"router"}, families=["certifications"])
    s = run_plain(new_state(CASE, PAGES), ag)
    assert ag.local_llm.stages("router") == []
    assert s["router_tags"] == {pid: list(FAMILIES) for pid in BY_ID}
    s2 = run_plain(new_state(CASE, PAGES), make_ag(families=["certifications"]))
    assert s2["router_tags"] == {pid: ["certifications"] for pid in BY_ID}


def test_grader_ablation_gives_no_grade_and_makes_no_call():
    ag = make_ag(grade="unsure", ablate={"grader"})
    s = run_plain(new_state(CASE, PAGES), ag)
    assert ag.local_llm.stages("grader") == [] and s["iteration"] == 1 and s["readings"]
    assert s["grades"] == {}                       # no grade is invented: the Decision Maker would see "unsure", not "supported"


def test_grader_ablation_shows_candidates_to_the_decision_maker_as_unsure(with_baseline):
    base_dir, _ = with_baseline(baseline_of(changes={"sso_saml": "none"}))
    ag = make_ag(baseline_dir=base_dir, ablate={"grader"})
    s = run_plain(new_state(CASE, PAGES), ag)
    assert [c["id"] for c in s["conflicts"]] == ["p1:sso_saml"]
    assert {c["grade"] for c in s["conflicts"][0]["candidates"]} == {"unsure"}


def test_decision_maker_ablation_abstains_without_a_call(with_baseline):
    base_dir, _ = with_baseline(baseline_of(changes={"sso_saml": "none"}))
    ag = make_ag(baseline_dir=base_dir, choice=1, ablate={"decision_maker"})
    s = run_plain(new_state(CASE, PAGES), ag)
    assert ag.local_llm.stages("decision") == []
    assert s["conflicts"] and s["decisions"] == {"p1:sso_saml": None}
    assert s["facts"]["sso_saml"]["value"] == "all_plans"                     # abstain -> the default (the latest iteration)


def test_iteration2_ablation_makes_no_case_read():
    reader = ReaderLLM()
    s = run_plain(new_state(CASE, PAGES), make_ag(grade="unsure", reader=reader, ablate={"iteration2"}))
    assert s["iteration"] == 1 and all(c["iteration"] == 1 for c in reader.calls)


def test_unknown_ablation_is_refused():
    with pytest.raises(ValueError):
        make_ag(ablate={"planner"})


# ---------------------------------------------------------------- failures
def test_local_failure_everywhere_still_gives_verdicts(with_baseline):
    base_dir, _ = with_baseline(baseline_of(changes={"sso_saml": "none"}))
    ag = make_ag(baseline_dir=base_dir, raise_on={"router", "grader", "decision"})
    s = run_plain(new_state(CASE, PAGES), ag)
    assert set(s["verdicts"]) == {r["id"] for r in REQS} and s["overall"]
    assert s["flags"]["router_fallbacks"] == 3
    assert s["flags"]["grader_failures"] == len(ag.local_llm.stages("grader")) > 0
    assert s["flags"]["decision_failures"] == len(ag.local_llm.stages("decision")) == len(s["conflicts"]) > 0
    assert set(s["grades"].values()) == {"unsure"} and s["iteration"] == 2
    assert all(v is None for v in s["decisions"].values())


def test_a_specialist_failure_on_one_page_is_counted_and_the_case_finishes():
    reader = ReaderLLM(fail_pages={"p2"})
    s = run_plain(new_state(CASE, PAGES), make_ag(reader=reader))
    failed_calls = [c for c in reader.calls if "p2" in c["pages"]]
    assert s["flags"]["specialist_failures"] == len(failed_calls) > 0
    assert [n for n in s["notes"] if n.startswith("specialists: failed page ")] == ["specialists: failed page p2"]
    spec = next(h for h in s["trace"].to_json()["handovers"] if h["agent"] == "specialists")
    assert spec["summary"]["failed_pages"] == ["p2"]
    assert "data_retention" not in s["facts"] and s["verdicts"] and s["overall"]


def test_flags_start_at_zero():
    s = run_plain(new_state(CASE, PAGES), make_ag())
    assert s["flags"] == {"planner_failures": 0, "router_fallbacks": 0, "specialist_failures": 0, "case_read_failures": 0,
                          "grader_failures": 0, "decision_failures": 0, "decision_invalid": 0}


def test_planner_failure_is_counted_and_routing_still_covers_the_pages():
    s = run_plain(new_state(CASE, PAGES), make_ag(planner=PlannerLLM(raise_exc=LLMError("provider error 500"))))
    assert s["flags"]["planner_failures"] == 1 and s["plan"] == {}
    assert s["routes"]["access_incidents"] == ["p1", "p2"] and s["verdicts"]


@pytest.mark.parametrize("role", ["planner", "reader", "local"])
def test_budget_exceeded_propagates(role):
    exc = BudgetExceeded("budget")
    if role == "planner":
        ag = make_ag(planner=PlannerLLM(raise_exc=exc))
    elif role == "reader":
        ag = make_ag(reader=ReaderLLM(raise_exc=exc))
    else:
        ag = replace(make_ag(), local_llm=LocalLLM(raise_on={"router"}, exc=exc))
    with pytest.raises(BudgetExceeded):
        run_plain(new_state(CASE, PAGES), ag)


# ---------------------------------------------------------------- reading ids
def test_reading_ids_are_unique_across_both_iterations(with_baseline):
    s = run_plain(new_state(CASE, PAGES), make_ag(grade="unsure"))
    ids = [r["id"] for r in s["readings"]]
    assert len(ids) == len(set(ids)) and any(r["iteration"] == 2 for r in s["readings"])


def _r(n):
    return {"id": reading_id(1, "data", "p1", "data_regions", n), "iteration": 1, "family": "data", "page_id": "p1",
            "field": "data_regions", "value": None, "quote": "q"}


def test_a_duplicate_reading_id_fails_loudly():
    nodes.assert_unique_ids([_r(0), _r(1)], {_r(2)["id"]})                       # distinct and new: fine
    with pytest.raises(nodes.DuplicateReadingId):
        nodes.assert_unique_ids([_r(0), _r(0)], set())                            # twice within one call
    with pytest.raises(nodes.DuplicateReadingId):
        nodes.assert_unique_ids([_r(1)], {_r(1)["id"]})                           # already taken by an earlier call


def test_the_specialists_node_checks_ids(monkeypatch):
    seen = []
    real = nodes.assert_unique_ids
    monkeypatch.setattr(nodes, "assert_unique_ids", lambda new, taken: (seen.append(len(new)), real(new, taken)))
    reader = ReaderLLM()
    run_plain(new_state(CASE, PAGES), make_ag(grade="unsure", reader=reader))
    assert len(seen) == len(reader.calls) > 0                                     # once per specialist call, both iterations


# ---------------------------------------------------------------- privacy and side effects
def test_no_page_text_or_model_output_reaches_trace_or_notes(with_baseline):
    pages = [dict(p, text=p["text"] + f" {CANARY} appears here.", title=f"{CANARY} title") for p in PAGES]
    canary_read = {("ai", "p1"): [{"field": "subprocessors", "value": {"name": f"{CANARY} Corp", "category": "hosting", "country": "US"},
                                   "quote": f"{CANARY} appears here.", "display": f"{CANARY} display"}]}
    base = baseline_of(changes={"sso_saml": "none"})
    base["p1"].append({"id": reading_id(0, "ai", "p1", "subprocessors", 0), "iteration": 0, "family": "ai", "page_id": "p1",
                       "field": "subprocessors", "value": [{"name": f"{CANARY} Other", "category": "email", "country": "US"}],
                       "quote": f"{CANARY} baseline quote", "display": CANARY})
    base_dir, _ = with_baseline(base)
    for ag in (make_ag(baseline_dir=base_dir, reader=ReaderLLM(extra=canary_read, fail_pages={"p2"}), families=[CANARY, "ai"]),
               make_ag(baseline_dir=base_dir, reader=ReaderLLM(extra=canary_read), raise_on={"router", "grader", "decision"}),
               make_ag(baseline_dir=base_dir, reader=ReaderLLM(extra=canary_read), grade="unsure", choice=7)):
        s = run_plain(new_state(CASE, pages), ag)
        assert s["conflicts"]
        assert CANARY not in json.dumps(s["trace"].to_json()) and CANARY not in s["trace"].to_markdown()
        assert not any(CANARY in n for n in s["notes"])


def test_no_node_writes_a_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_plain(new_state(CASE, PAGES), make_ag(grade="unsure"))
    build_langgraph(make_ag()).invoke(new_state(CASE, PAGES))
    mermaid(make_ag())
    assert list(tmp_path.iterdir()) == []


def test_every_handover_has_seconds_and_only_counts_and_ids():
    s = run_plain(new_state(CASE, PAGES), make_ag(grade="unsure"))
    for h in s["trace"].to_json()["handovers"]:
        assert isinstance(h["seconds"], float) and h["seconds"] >= 0
        assert h["to"] in NODE_ORDER + ["end"]
    assert s["trace"].to_json()["handovers"][-1]["agent"] == "audit"


# ---------------------------------------------------------------- Task 8A follow-ups (ruling R32: M1, M2, M4, M5, M6)
def test_an_iteration_2_case_read_failure_is_its_own_count_and_iteration_1_stands():
    reader = ReaderLLM(fail_iter2=True)
    s = run_plain(new_state(CASE, PAGES), make_ag(grade="unsure", reader=reader))
    second = [c for c in reader.calls if c["iteration"] == 2]
    assert s["iteration"] == 2 and len(second) == len(FAMILIES)
    assert s["flags"]["case_read_failures"] == len(second) and s["flags"]["specialist_failures"] == 0
    assert not any(n.startswith(nodes.FAILED_PAGE_NOTE) for n in s["notes"])         # no page counts as failed: iteration 1 read them
    assert [n for n in s["notes"] if n.startswith("specialists: case read failed: ")] == \
        [f"specialists: case read failed: {f}" for f in FAMILIES]
    spec = [h for h in s["trace"].to_json()["handovers"] if h["agent"] == "specialists"]
    assert [h["summary"]["failed_pages"] for h in spec] == [[], []] and spec[1]["summary"]["case_read_failures"] == len(second)
    assert {r["iteration"] for r in s["readings"]} == {1}
    assert set(s["facts"]) == set(READ_FIELDS) and all(s["verdicts"][f"R-{f}"] == "PASS" for f in READ_FIELDS)


def test_case_read_notes_are_labelled_by_iteration_not_by_page_count():
    # the planner sends data to p2 alone and the router tags every page with certifications only: data has ONE routed page in
    # iteration 2, so a page-count proxy would label its case-read notes with the page id
    spoof = [{"page_id": "p9", "field": "dpa_available", "value": "no", "quote": "x"}]
    reader = ReaderLLM(iter2_extra={"data": spoof})
    planner = PlannerLLM()
    planner.structured = lambda system, user, schema, stage="", **kw: PlanOut(routes=[
        PageRoute(page_id="p1", families=["access_incidents", "ai"]), PageRoute(page_id="p2", families=["data"])])
    s = run_plain(new_state(CASE, PAGES), make_ag(grade="unsure", reader=reader, planner=planner, families=["certifications"]))
    data_read = next(c for c in reader.calls if c["iteration"] == 2 and c["family"] == "data")
    assert len(data_read["pages"]) == 1
    dropped = [n for n in s["notes"] if "is not one of the pages read" in n]
    assert dropped and all(n.startswith("specialists: data: case read: ") for n in dropped)


def test_decide_conflicts_node_abstains_on_an_out_of_range_choice(with_baseline):
    base_dir, _ = with_baseline(baseline_of(changes={"sso_saml": "none"}))
    s = run_plain(new_state(CASE, PAGES), make_ag(baseline_dir=base_dir))
    ag7 = make_ag(baseline_dir=base_dir, choice=7)
    s.update(flags={k: 0 for k in nodes.FLAG_KEYS}, notes=[], decisions={}, conflicts=[])
    out = nodes.decide_conflicts(s, ag7)
    assert out["decisions"] == {"p1:sso_saml": None}
    assert out["flags"]["decision_invalid"] == 1 and out["flags"]["decision_failures"] == 0
    s.update(out)
    assert nodes.merge(s, ag7)["facts"]["sso_saml"]["value"] == "all_plans"            # abstain -> the default (iteration-1) reading


def test_decision_failures_and_invalid_answers_are_counted_apart(with_baseline):
    base_dir, _ = with_baseline(baseline_of(changes={"sso_saml": "none"}))
    failed = run_plain(new_state(CASE, PAGES), make_ag(baseline_dir=base_dir, raise_on={"decision"}))
    assert failed["flags"]["decision_failures"] == 1 and failed["flags"]["decision_invalid"] == 0
    invalid = run_plain(new_state(CASE, PAGES), make_ag(baseline_dir=base_dir, choice=7))
    assert invalid["flags"]["decision_failures"] == 0 and invalid["flags"]["decision_invalid"] == 1


def test_both_engines_share_the_step_bound(monkeypatch):
    from vendor_verdict import graph as graph_mod
    assert build_langgraph(AG).config["recursion_limit"] == graph_mod._MAX_STEPS
    monkeypatch.setattr(graph_mod, "next_after_evaluate", lambda state: "specialists")         # an evaluator that never stops
    a, b = new_state(CASE, PAGES), new_state(CASE, PAGES)
    with pytest.raises(RuntimeError):                   # "unsure": iteration 2 is reached, after which a pass reads nothing new
        run_plain(a, make_ag(grade="unsure"))
    from langgraph.errors import GraphRecursionError
    with pytest.raises(GraphRecursionError):
        build_langgraph(make_ag(grade="unsure")).invoke(b)
    assert len(a["trace"].to_json()["handovers"]) == len(b["trace"].to_json()["handovers"]) == graph_mod._MAX_STEPS


# ---------------------------------------------------------------- GATE-P: display is page-derived text (item 5d)
def test_a_short_display_canary_never_reaches_trace_notes_or_output(capsys):
    short = f"{CANARY} is not mentioned"                         # under the trace's 80-character heuristic, and absence-worded
    assert len(short) < 80
    extra = {("ai", "p1"): [{"field": "trains_on_customer_data", "value": "no", "quote": "We do not train models on customer data.",
                             "display": short}],
             ("access_incidents", "p1"): [{"field": "security_incidents", "value": [], "quote": "Invented trust page.", "display": short},
                                          {"field": "sso_saml", "value": "all_plans", "quote": "Single sign-on (SAML) is offered on every plan.",
                                           "display": CANARY}]}
    for engine in ("plain", "langgraph"):
        ag = make_ag(grade="unsure", reader=ReaderLLM(extra=extra))
        s = run_plain(new_state(CASE, PAGES), ag) if engine == "plain" else build_langgraph(ag).invoke(new_state(CASE, PAGES))
        assert any("absence_statement" in n for n in s["notes"])               # the canary display went down the new note path
        assert CANARY not in json.dumps(s["trace"].to_json()) and CANARY not in s["trace"].to_markdown()
        assert not any(CANARY in n for n in s["notes"])
    out = capsys.readouterr()
    assert CANARY not in out.out and CANARY not in out.err
