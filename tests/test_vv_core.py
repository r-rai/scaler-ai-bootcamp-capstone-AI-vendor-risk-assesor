"""vendor_verdict core: config constants, case state helpers, call cache, handover trace, data loading.

Invented data only: no competition file is read and no LLM is called."""
import copy
import json
import pickle
import re

import pytest

from vendor_verdict import data
from vendor_verdict.cache import CallCache
from vendor_verdict.config import (BASELINE_PROMPT, EXPOSED_HOLDOUT_ROWS, FAMILIES, FIELD_FAMILY, GRADER_FIELDS, LOCAL_SCOPE,
                                   LOCAL_SLOTS, MAX_ITERATIONS)
from vendor_verdict.state import CaseState, reading_id
from vendor_verdict.trace import Trace

ALL_FIELDS = ["security_certifications", "data_regions", "data_retention", "dpa_available",
              "trains_on_customer_data", "subprocessors", "sso_saml", "security_incidents"]


# ------------------------------------------------------------------------------------------------ config
def test_families_cover_every_rules_lite_field_once():
    fields = [f for fs in FAMILIES.values() for f in fs]
    assert sorted(fields) == sorted(["security_certifications", "data_regions", "data_retention", "dpa_available",
                                     "trains_on_customer_data", "subprocessors", "sso_saml", "security_incidents"])
    assert len(fields) == len(set(fields))


def test_field_family_is_the_exact_inverse_of_families():
    assert FIELD_FAMILY == {f: fam for fam, fs in FAMILIES.items() for f in fs}
    assert FIELD_FAMILY["dpa_available"] == "data" and FIELD_FAMILY["sso_saml"] == "access_incidents"


def test_grader_fields_default_to_all_eight_fields():
    assert isinstance(GRADER_FIELDS, frozenset)
    assert sorted(GRADER_FIELDS) == sorted(ALL_FIELDS)


def test_local_scope_is_router_grader_decision_maker_without_injection():
    assert LOCAL_SCOPE == frozenset({"router", "grader", "decision_maker"})
    assert "injection" not in LOCAL_SCOPE
    assert LOCAL_SLOTS == 2


def test_loop_and_baseline_constants():
    assert MAX_ITERATIONS == 2
    assert BASELINE_PROMPT == ("e42203d", "9566ad4b6b95")


def test_exposed_holdout_rows_are_exactly_the_seven_label_exposed_ids():
    assert isinstance(EXPOSED_HOLDOUT_ROWS, frozenset)
    assert len(EXPOSED_HOLDOUT_ROWS) == 7
    assert EXPOSED_HOLDOUT_ROWS == {"c0060__SEC-4.2", "c0312__SEC-4.2-T2", "c0678__SEC-4.2-T2", "c0062__DP-3.1",
                                    "c0288__SEC-7.1", "c0426__SEC-4.2", "c0512__SEC-9.3"}


# ------------------------------------------------------------------------------------------------ state
def test_reading_id_format_is_iteration_family_page_field_n():
    assert reading_id(1, "data", "c9999_p2", "data_regions", 0) == "1:data:c9999_p2:data_regions:0"
    assert reading_id(2, "ai", "p7", "subprocessors", 3) == "2:ai:p7:subprocessors:3"


def test_case_state_is_a_partial_typed_dict_that_carries_raw_pages_and_a_trace():
    assert CaseState.__total__ is False
    keys = set(CaseState.__annotations__)
    assert {"case", "pages", "recs", "plan", "router_tags", "routes", "review_fixes", "readings", "grades", "baseline",
            "conflicts", "decisions", "facts", "verdicts", "overall", "iteration", "targets", "notes", "flags", "trace"} <= keys
    state: CaseState = {"case": {"case_id": "c9999"}, "pages": [], "iteration": 1, "trace": Trace("c9999")}
    assert state["trace"].case_id == "c9999"


# ------------------------------------------------------------------------------------------------ cache
def test_cache_key_changes_with_every_part_and_empty_salt_adds_nothing(tmp_path):
    c = CallCache(tmp_path)
    k = c.key("specialist", "v1", "m", "payload")
    assert k == c.key("specialist", "v1", "m", "payload", "")
    assert len({k, c.key("router", "v1", "m", "payload"), c.key("specialist", "v2", "m", "payload"),
                c.key("specialist", "v1", "m2", "payload"), c.key("specialist", "v1", "m", "other"),
                c.key("specialist", "v1", "m", "payload", "s1")}) == 6


def test_cache_key_is_a_sha256_hex_digest_and_part_boundaries_matter(tmp_path):
    c = CallCache(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{64}", c.key("r", "v", "m", "p"))
    assert c.key("ab", "c", "m", "p") != c.key("a", "bc", "m", "p")     # NUL-joined: moving a character across parts changes the key


def test_cache_roundtrip_counts_hits(tmp_path):
    c = CallCache(tmp_path); k = c.key("r", "v", "m", "p")
    assert c.get(k) is None; c.put(k, {"out": 1}); assert c.get(k) == {"out": 1}; assert (c.hits, c.misses) == (1, 1)


def test_cache_put_is_atomic_and_creates_the_root_lazily(tmp_path):
    root = tmp_path / "agents" / "deep"
    c = CallCache(root)
    assert not root.exists()                                            # a read-only look (get) must not create folders
    assert c.get(c.key("r", "v", "m", "p")) is None and not root.exists()
    k = c.key("r", "v", "m", "p")
    c.put(k, {"out": [1, 2], "note": "naïve"})
    assert [p.name for p in root.iterdir()] == [f"{k}.json"]            # no .tmp left behind
    c.put(k, {"out": 3})                                                # overwrite goes through replace
    assert CallCache(root).get(k) == {"out": 3}


def test_cache_treats_a_corrupt_entry_as_a_miss(tmp_path):
    c = CallCache(tmp_path); k = c.key("r", "v", "m", "p")
    (tmp_path / f"{k}.json").write_text("{not json", encoding="utf-8")
    assert c.get(k) is None and (c.hits, c.misses) == (0, 1)


def test_cache_default_root_is_data_kaggle_cache_agents():
    assert CallCache().root.parts[-3:] == ("data_kaggle", "cache", "agents")


# ------------------------------------------------------------------------------------------------ trace
def test_trace_records_handover_in_order():
    t = Trace("c9999"); t.handover("planner", "plan_reviewer", {"pages": 3}); t.handover("plan_reviewer", "router", {"fixes": 1})
    assert [h["agent"] for h in t.to_json()["handovers"]] == ["planner", "plan_reviewer"]
    assert "planner → plan_reviewer" in t.to_markdown()


def test_trace_json_carries_fields_totals_and_is_serialisable():
    t = Trace("c9999")
    t.handover("router", "specialist", {"pages": 4, "ids": ["p1", "p2"]}, tokens_in=100, tokens_out=20, seconds=1.5)
    t.handover("specialist", "grader", {"readings": 7}, tokens_in=50, tokens_out=10, seconds=0.5)
    out = json.loads(json.dumps(t.to_json()))
    assert out["case_id"] == "c9999"
    assert out["handovers"][0] == {"agent": "router", "to": "specialist", "summary": {"pages": 4, "ids": ["p1", "p2"]},
                                   "tokens_in": 100, "tokens_out": 20, "seconds": 1.5}
    assert out["totals"] == {"handovers": 2, "tokens_in": 150, "tokens_out": 30, "seconds": 2.0}


def test_trace_markdown_lists_each_handover_with_its_counts():
    t = Trace("c9999"); t.handover("grader", "decision_maker", {"conflicts": 2}, tokens_in=7, tokens_out=3, seconds=0.25)
    md = t.to_markdown()
    assert "c9999" in md and "grader → decision_maker" in md and "conflicts=2" in md


def test_trace_survives_copy_and_pickle_so_a_graph_engine_can_checkpoint_the_state():
    t = Trace("c9999"); t.handover("router", "specialist", {"pages": 4}, tokens_in=5)
    for clone in (copy.deepcopy(t), pickle.loads(pickle.dumps(t))):
        assert clone.to_json() == t.to_json()
        clone.handover("specialist", "grader", {"readings": 1})
    assert len(t.to_json()["handovers"]) == 1                           # the clones are independent


def test_trace_summaries_refuse_free_text():
    t = Trace("c9999")
    with pytest.raises(ValueError):
        t.handover("specialist", "grader", {"quote": "This vendor stores customer tickets in the EU only, and " * 3})
    with pytest.raises(ValueError):
        t.handover("specialist", "grader", {"nested": {"deep": ["x" * 200]}})
    assert t.to_json()["handovers"] == []                               # a refused handover leaves no record


# ------------------------------------------------------------------------------------------------ data
def _write_competition_folder(root):
    header = "case_id,split,vendor,legal_entity,official_domain,hq_country,hq_city,product,question,tier,users,as_of"
    rows = [f"c000{i},{'test' if i == 5 else 'train'},V{i},V{i} Ltd,v{i}.example,Ireland,Dublin,a tool,Can we use V{i}?,T3,10,2026-11-01"
            for i in range(1, 6)]
    (root / "cases.csv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    pages = [{"case_id": "c0001", "page_id": "c0001_p1", "url": "https://v1.example/a", "title": "t", "text": "invented"},
             {"case_id": "c0001", "page_id": "c0001_p2", "url": "https://v1.example/b", "title": "t", "text": "invented"},
             {"case_id": "c0003", "page_id": "c0003_p1", "url": "https://v3.example/a", "title": "t", "text": "invented"}]
    (root / "pages.jsonl").write_text("\n".join(json.dumps(p) for p in pages) + "\n", encoding="utf-8")


def test_data_load_reuses_the_runner_loaders_and_selection(tmp_path):
    _write_competition_folder(tmp_path)
    cases, pages, ids = data.load(tmp_path, "train", "dev")
    assert set(cases) == {"c0001", "c0002", "c0003", "c0004", "c0005"}
    assert ids == ["c0001", "c0003"]                                    # label-blind halves: even positions are dev
    assert data.load(tmp_path, "train", "holdout")[2] == ["c0002", "c0004"]
    assert [p["page_id"] for p in pages["c0001"]] == ["c0001_p1", "c0001_p2"]
    assert pages["c0002"] == []                                         # a case without pages reads as empty, not KeyError


def test_data_load_passes_limit_and_only_through_to_select_cases(tmp_path):
    _write_competition_folder(tmp_path)
    assert data.load(tmp_path, "train", "all", limit=3)[2] == ["c0001", "c0002", "c0003"]
    assert data.load(tmp_path, "train", "all", only=["c0004", "c0002"])[2] == ["c0002", "c0004"]
    assert data.load(tmp_path, "test", "all")[2] == ["c0005"]
