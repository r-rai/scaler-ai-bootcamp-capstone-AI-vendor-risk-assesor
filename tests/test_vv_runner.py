"""vendor_verdict runner (runner.py, __main__.py): outputs, run.json, the partial-run guard, the submission rule (R4), estimates and the
console leak guard.

Invented data only: every data dir is written into `tmp_path` (nothing under `data_kaggle/` is read) and every model is a stub injected
through the `runner.run(..., agents=...)` seam or a patched `runner.build_agents` (no real LLM is built, nothing reaches :8080)."""
import copy
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from assessor.llm import LLMError
from assessor.tracing import BudgetExceeded

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))            # as tests/test_kaggle_runner.py does: run_kaggle sets VRA_CACHE and sys.path on import
import run_kaggle  # noqa: E402,F401

from vendor_verdict import baseline as baseline_mod  # noqa: E402
from vendor_verdict import runner  # noqa: E402
from vendor_verdict.cache import CallCache  # noqa: E402
from vendor_verdict.config import FAMILIES, FIELD_FAMILY, LOCAL_SCOPE  # noqa: E402
from vendor_verdict.nodes import Agents  # noqa: E402
from vendor_verdict.prompts import SPECIALIST_SYSTEM  # noqa: E402
from vendor_verdict.schemas import DecisionOut, GradeOut, PageRoute, PlanOut, RouterOut, SpecialistRaw  # noqa: E402
from vendor_verdict.state import reading_id  # noqa: E402

CANARY = "CANARY-PAGE-TEXT"
CASE_IDS = ["c0001", "c0002", "c0003"]
REQS = [{"id": f"R-{f}", "field": f, "severity": "major"} for f in FIELD_FAMILY]
TEXT = {
    "p1": "Invented trust page. Single sign-on (SAML) is offered on every plan. We do not train models on customer data.",
    "p2": "Invented privacy page. Customer data is deleted after 30 days. A data processing agreement is available on request.",
    "p3": "Invented registry entry. Acme Widgets Inc. holds SOC 2 Type II, issued 2026-02-01 by Example Audit LLP.",
}
META = {
    "p1": ("https://acme.example/trust", "official", "Trust", "2026-03-01"),
    "p2": ("https://acme.example/privacy", "official", "Privacy", "2026-05-01"),
    "p3": ("https://registry.example/acme", "regulator", "Registry", "2026-02-01"),
}
CERT = {"name": "SOC 2 Type II", "status": "certified", "date": "2026-02-01", "valid_until": None, "issuer": "Example Audit LLP",
        "holder": "Acme Widgets Inc."}
READ = {   # (family, page suffix) -> readings; every quote is a literal substring of its page
    ("access_incidents", "p1"): [{"field": "sso_saml", "value": "all_plans", "quote": "Single sign-on (SAML) is offered on every plan."}],
    ("ai", "p1"): [{"field": "trains_on_customer_data", "value": "no", "quote": "We do not train models on customer data."}],
    ("data", "p2"): [{"field": "data_retention", "value": {"default_days": 30, "deletion_on_request_days": None},
                      "quote": "Customer data is deleted after 30 days."},
                     {"field": "dpa_available", "value": "on_request", "quote": "A data processing agreement is available on request."}],
    ("certifications", "p3"): [{"field": "security_certifications", "value": CERT, "quote": "Acme Widgets Inc. holds SOC 2 Type II"}],
}
FAMILY_OF_SYSTEM = {v: k for k, v in SPECIALIST_SYSTEM.items()}


def _suffix(pid: str) -> str:
    return pid.rsplit("_", 1)[-1]


def _case(cid):
    return {"case_id": cid, "split": "test", "vendor": "Acme", "legal_entity": "Acme Widgets Inc.", "official_domain": "acme.example",
            "tier": "T3", "users": "35", "as_of": "2026-11-01", "question": "Can support use Acme with tickets?"}


def write_data(d: Path, page_text=None, split="test") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    cols = list(_case("x"))
    with (d / "cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for cid in CASE_IDS:
            w.writerow({**_case(cid), "split": split})
    with (d / "pages.jsonl").open("w", encoding="utf-8") as f:
        for cid in CASE_IDS:
            for s, (url, st, title, upd) in META.items():
                text = TEXT[s] + f" Case {cid} page." + (f" {page_text} here." if page_text else "")
                f.write(json.dumps({"case_id": cid, "page_id": f"{cid}_{s}", "url": url, "source_type": st, "title": title,
                                    "publisher": "Acme", "published_date": "2026-01-01", "last_updated": upd, "text": text}) + "\n")
    with (d / "test.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "case_id"])
        for cid in CASE_IDS:
            for r in REQS:
                w.writerow([f"{cid}__{r['id']}", cid])
            w.writerow([f"{cid}__OVERALL", cid])
    return d


class FakeRL:
    @staticmethod
    def evaluate(facts, requirements, tier, users, as_of):
        return {r["id"]: ("PASS" if (facts.get(r["field"]) or {}).get("value") not in (None, "", [], {}) else "UNKNOWN")
                for r in requirements}

    @staticmethod
    def overall(verdicts, requirements):
        return "READY_FOR_APPROVAL" if all(v == "PASS" for v in verdicts.values()) else "CONDITIONAL"


class StubMetric:
    def __init__(self):
        self.calls = 0

    def components(self, sol, sub):
        self.calls += 1
        return {"score": 1.0}


class PlannerLLM:
    model = "stub-main"

    def __init__(self, raise_exc=None):
        self.calls, self.raise_exc = [], raise_exc

    def structured(self, system, user, schema, stage="", **kw):
        self.calls.append(stage)
        if self.raise_exc is not None:
            raise self.raise_exc
        pids = [json.loads(line)["page_id"] for line in user.split("\n") if line.startswith("{")]
        fams = {"p1": ["access_incidents", "ai"], "p2": ["data"], "p3": ["certifications"]}
        return PlanOut(routes=[PageRoute(page_id=p, families=fams[_suffix(p)]) for p in pids])


class ReaderLLM:
    """read_page -> READ[(family, suffix)]; read_case -> the same readings of the requested fields. `fail` holds page suffixes whose
    iteration-1 read fails; `fail_iter2` fails every case read; `raise_on` maps a case id to an exception raised on its first read;
    `delay` maps a case id to seconds slept before answering; `value` replaces every reading's value and display (leak tests)."""
    model = "stub-reader"

    def __init__(self, fail=(), fail_iter2=False, raise_on=None, delay=None, value=None):
        self.calls, self.fail, self.fail_iter2 = [], set(fail), fail_iter2
        self.raise_on, self.delay, self.value = raise_on or {}, delay or {}, value

    def structured(self, system, user, schema, stage="", **kw):
        assert schema is SpecialistRaw
        family = FAMILY_OF_SYSTEM[system]
        cid = next(c for c in CASE_IDS if f" Case {c} page." in user)
        if user.startswith("Report only these fields: "):
            fields = user.split("\n", 1)[0][len("Report only these fields: "):].split(", ")
            pids = [line[len("page_id: "):] for line in user.split("\n") if line.startswith("page_id: ")]
            self.calls.append(("case", family, tuple(pids)))
            if self.fail_iter2:
                raise LLMError("provider error 500 " + CANARY)
            out = [{**r, "page_id": p} for p in pids for r in READ.get((family, _suffix(p)), []) if r["field"] in fields]
            return SpecialistRaw(readings=copy.deepcopy(out))
        s = next(s for s in META if TEXT[s] in user)
        self.calls.append(("page", family, s))
        if cid in self.delay:
            time.sleep(self.delay[cid])
        if cid in self.raise_on:
            raise self.raise_on[cid]
        if s in self.fail:
            raise LLMError("provider error 500 " + CANARY)
        out = copy.deepcopy(READ.get((family, s), []))
        if self.value is not None:
            out = [{**r, "value": self.value, "display": self.value} for r in out]
        return SpecialistRaw(readings=out)


class LocalLLM:
    model = "stub-local"

    def __init__(self, grade="supported", choice=0, raise_on=()):
        self.grade, self.choice, self.raise_on, self.calls = grade, choice, set(raise_on), []

    def structured(self, system, user, schema, stage="", max_repairs=2, **kw):
        self.calls.append(stage)
        if stage in self.raise_on:
            raise LLMError("local server error " + CANARY)
        if schema is RouterOut:
            return RouterOut(families=list(FAMILIES))
        if schema is GradeOut:
            return GradeOut(grade=self.grade)
        if schema is DecisionOut:
            return DecisionOut(choice=self.choice)
        raise AssertionError(schema)


def make_agents(cache_root: Path, *, reader=None, planner=None, local=None, salt="", ablate=(), baseline_dir=None) -> Agents:
    return Agents(main_llm=planner or PlannerLLM(), reader_llm=reader or ReaderLLM(), local_llm=local or LocalLLM(),
                  cache=CallCache(cache_root), rl=FakeRL, reqs=REQS, baseline_dir=baseline_dir, baseline_model="stub-baseline",
                  salt=salt, ablate=frozenset(ablate))


def run_fixture(base: Path, fail_pages=(), salt="", ablate=(), page_text=None, cache_only=False, cache_from=None, engine="plain",
                workers=2, reader=None, local=None, split="test", only=None, metric=None, **kw) -> Path:
    """Run the 3 invented test cases into `base / "out"` and return that dir. `fail_pages` holds page suffixes (p1, p2, p3)."""
    data = write_data(base / "data", page_text=page_text, split=split)
    cache_root = (cache_from.parent if cache_from is not None else base) / "cache"
    ag = make_agents(cache_root, reader=reader or ReaderLLM(fail=fail_pages), local=local, salt=salt, ablate=ablate)
    out = base / "out"
    runner.run(data, split, "all", agents=ag, out=out, cache_only=cache_only, engine=engine, workers=workers, only=only,
               metric=metric or StubMetric(), **kw)
    return out


def run_json(out: Path) -> dict:
    return json.loads((out / "run.json").read_text(encoding="utf-8"))


def preds(out: Path) -> dict[str, str]:
    with (out / "preds.csv").open(encoding="utf-8") as f:
        return {r["id"]: r["verdict"] for r in csv.DictReader(f)}


# ---------------------------------------------------------------- the brief's tests
def test_partial_run_writes_no_submission(tmp_path):
    out = run_fixture(tmp_path, fail_pages={"p2"})
    assert json.loads((out / "run.json").read_text())["partial"] is True and not (out / "submission.csv").exists()


def test_salted_or_ablated_test_run_writes_no_submission(tmp_path):
    assert not (run_fixture(tmp_path / "s", salt="s1") / "submission.csv").exists()
    assert not (run_fixture(tmp_path / "g", ablate={"grader"}) / "submission.csv").exists()


def test_decision_maker_ablation_and_a_clean_run_write_the_submission(tmp_path):      # ruling R4: the candidate must be producible
    for name, ablate in (("plain", ()), ("dm", {"decision_maker"})):
        metric = StubMetric()
        out = run_fixture(tmp_path / name, ablate=ablate, metric=metric)
        assert (out / "submission.csv").exists() and metric.calls == 1
        rj = run_json(out)
        assert rj["partial"] is False and rj["submission"] == {"written": True, "reason": "ok", "problems": []}
        with (out / "submission.csv").open(encoding="utf-8") as f:
            assert [r["id"] for r in csv.DictReader(f)] == [r["id"] for r in csv.DictReader((tmp_path / name / "data" / "test.csv").open())]


@pytest.mark.parametrize("ablate", [{"router", "grader"}, {"grader"}, {"iteration2"}, {"decision_maker", "grader"},      # R54: {"router"} alone now writes
                                    {"decision_maker", "iteration2"}, {"router", "decision_maker", "grader"}])
def test_any_other_ablation_writes_no_submission(tmp_path, ablate):
    out = run_fixture(tmp_path, ablate=ablate)
    assert not (out / "submission.csv").exists() and run_json(out)["submission"]["written"] is False


@pytest.mark.parametrize("ablate", [{"router"}, {"router", "decision_maker"}])
def test_router_off_candidate_configurations_write_the_submission(tmp_path, ablate):      # ruling R54: gates.candidate() can select them
    metric = StubMetric()
    out = run_fixture(tmp_path, ablate=ablate, metric=metric)
    assert (out / "submission.csv").exists() and metric.calls == 1
    rj = run_json(out)
    assert rj["ablate"] == sorted(ablate) and rj["partial"] is False
    assert rj["submission"] == {"written": True, "reason": "ok", "problems": []}
    with (out / "submission.csv").open(encoding="utf-8") as f:
        assert [r["id"] for r in csv.DictReader(f)] == [r["id"] for r in csv.DictReader((tmp_path / "data" / "test.csv").open())]


def test_the_allowed_submission_ablations_are_exactly_the_four_candidate_sets():      # R54 (R4 amended)
    assert len(runner.SUBMISSION_ABLATIONS) == 4
    assert set(runner.SUBMISSION_ABLATIONS) == {frozenset(), frozenset({"decision_maker"}), frozenset({"router"}),
                                                frozenset({"router", "decision_maker"})}


@pytest.mark.parametrize("ablate,ok", [
    (frozenset(), True), (frozenset({"decision_maker"}), True), (frozenset({"router"}), True),
    (frozenset({"router", "decision_maker"}), True),
    (frozenset({"grader"}), False), (frozenset({"iteration2"}), False), (frozenset({"router", "grader"}), False),
    (frozenset({"decision_maker", "iteration2"}), False), (frozenset({"router", "decision_maker", "grader"}), False),
])
def test_submission_rule_ablation_sets(ablate, ok):      # R54: the other conditions all hold, only the ablation set varies
    ids = ["c1", "c2"]
    allowed, reason = runner.submission_rule("test", "", ablate, False, ids, set(ids))
    assert allowed is ok
    assert (reason == "ok") if ok else ("candidate configurations" in reason)


def test_console_prints_no_page_text(tmp_path, capsys):
    run_fixture(tmp_path, page_text="CANARY-PAGE-TEXT"); assert "CANARY-PAGE-TEXT" not in capsys.readouterr().out


def test_cache_only_rerun_is_byte_identical(tmp_path):
    a = run_fixture(tmp_path / "a"); b = run_fixture(tmp_path / "b", cache_only=True, cache_from=a)
    assert (a / "preds.csv").read_bytes() == (b / "preds.csv").read_bytes()


# ---------------------------------------------------------------- the submission rule (R4) on its own
@pytest.mark.parametrize("split,salt,ablate,partial,ok", [
    ("test", "", frozenset(), False, True),
    ("test", "", frozenset({"decision_maker"}), False, True),
    ("test", "", frozenset({"grader"}), False, False),
    ("test", "", frozenset({"router"}), False, True),                       # R54 (was refused under R4 alone)
    ("test", "", frozenset({"iteration2"}), False, False),
    ("test", "", frozenset({"decision_maker", "router"}), False, True),     # R54 (was refused under R4 alone)
    ("test", "s1", frozenset(), False, False),
    ("test", "", frozenset(), True, False),
    ("train", "", frozenset(), False, False),
])
def test_submission_rule(split, salt, ablate, partial, ok):
    allowed, reason = runner.submission_rule(split, salt, ablate, partial, ["c1", "c2"], {"c1", "c2"})
    assert allowed is ok and isinstance(reason, str) and reason


def test_submission_rule_needs_every_test_case():
    assert runner.submission_rule("test", "", frozenset(), False, ["c1"], {"c1", "c2"})[0] is False
    assert runner.submission_rule("test", "", frozenset(), False, ["c1", "c2", "c3"], {"c1", "c2"})[0] is False


def test_a_subset_of_the_test_cases_writes_no_submission(tmp_path):
    out = run_fixture(tmp_path, only=["c0001", "c0002"])
    assert not (out / "submission.csv").exists() and run_json(out)["partial"] is False
    assert {i.split("__")[0] for i in preds(out)} == {"c0001", "c0002"}


def test_a_train_run_writes_no_submission(tmp_path):
    out = run_fixture(tmp_path, split="train")
    assert not (out / "submission.csv").exists() and len(preds(out)) == 3 * (len(REQS) + 1)


# ---------------------------------------------------------------- partial semantics
def test_a_stale_submission_is_removed_by_a_later_partial_run(tmp_path):
    out = tmp_path / "out"
    (out / "traces").mkdir(parents=True)
    for name in ("submission.csv", "run.json", "preds.csv"):
        (out / name).write_text("stale", encoding="utf-8")
    (out / "traces" / "c9999.json").write_text("{}", encoding="utf-8")
    run_fixture(tmp_path, fail_pages={"p2"})
    assert not (out / "submission.csv").exists() and not (out / "traces" / "c9999.json").exists()
    assert run_json(out)["partial"] is True and "stale" not in (out / "preds.csv").read_text()


def test_failed_pages_are_counted_and_listed(tmp_path):
    out = run_fixture(tmp_path, fail_pages={"p2"})
    rj = run_json(out)
    assert rj["partial_reasons"] == ["failed_pages"]
    assert rj["pages_readable"] == 9 and rj["pages_failed"] == 3 and rj["pages_read"] == 6 and rj["pages_unrouted"] == 0
    assert rj["failed_page_ids"] == ["c0001_p2", "c0002_p2", "c0003_p2"]
    assert rj["flags"]["specialist_failures"] == 3 * 4 == rj["failures"]          # p2 is routed to all 4 families in each case
    assert rj["cases"] == {"selected": 3, "finished": 3, "errors": 0, "unfinished": 0}


def test_an_iteration_2_case_read_failure_is_counted_but_not_partial(tmp_path):
    reader = ReaderLLM(fail_iter2=True)
    out = run_fixture(tmp_path, reader=reader, local=LocalLLM(grade="unsure"))
    rj = run_json(out)
    assert any(c[0] == "case" for c in reader.calls)
    assert rj["iteration2_cases"] == 3 and rj["flags"]["case_read_failures"] > 0 and rj["failures"] >= rj["flags"]["case_read_failures"]
    assert rj["partial"] is False and rj["pages_failed"] == 0 and (out / "submission.csv").exists()


@pytest.mark.parametrize("engine", ["plain", "langgraph"])
def test_budget_stop_makes_the_run_partial(tmp_path, engine):
    reader = ReaderLLM(raise_on={"c0002": BudgetExceeded("budget " + CANARY)})
    out = run_fixture(tmp_path, reader=reader, engine=engine, workers=1)
    rj = run_json(out)
    assert rj["partial"] is True and rj["stopped"] == "budget" and "budget" in rj["partial_reasons"]
    assert not (out / "submission.csv").exists()
    assert "c0001__OVERALL" in preds(out) and not any(i.startswith("c0002") for i in preds(out))


@pytest.mark.parametrize("engine", ["plain", "langgraph"])
def test_interrupt_writes_what_finished_and_no_submission(tmp_path, engine):
    reader = ReaderLLM(raise_on={"c0002": KeyboardInterrupt()})
    out = run_fixture(tmp_path, reader=reader, engine=engine, workers=1)
    rj = run_json(out)
    assert rj["partial"] is True and rj["stopped"] == "interrupted" and "interrupted" in rj["partial_reasons"]
    assert not (out / "submission.csv").exists() and (out / "traces" / "c0001.json").exists()
    assert "c0001__OVERALL" in preds(out) and not any(i.startswith("c0002") for i in preds(out))


def test_a_case_that_raises_is_counted_and_the_run_is_partial(tmp_path):
    reader = ReaderLLM(raise_on={"c0002": ValueError("bug " + CANARY)})
    out = run_fixture(tmp_path, reader=reader)
    rj = run_json(out)
    assert rj["partial"] is True and rj["case_errors"] == {"ValueError": 1} and rj["error_case_ids"] == ["c0002"]
    assert rj["cases"]["errors"] == 1 and not (out / "submission.csv").exists() and CANARY not in json.dumps(rj)


# ---------------------------------------------------------------- determinism
def test_output_order_does_not_depend_on_completion_order(tmp_path):
    slow = run_fixture(tmp_path / "slow", reader=ReaderLLM(delay={"c0001": 0.05}), workers=3)
    serial = run_fixture(tmp_path / "serial", workers=1)
    assert (slow / "preds.csv").read_bytes() == (serial / "preds.csv").read_bytes()
    ids = list(preds(serial))
    assert ids[0].startswith("c0001") and ids[-1] == "c0003__OVERALL"


def test_the_cache_only_rerun_makes_no_model_call_and_is_not_partial(tmp_path):
    a = run_fixture(tmp_path / "a", local=LocalLLM(grade="unsure"))
    reader, local = ReaderLLM(), LocalLLM(grade="unsure")
    b = run_fixture(tmp_path / "b", cache_only=True, cache_from=a, reader=reader, local=local)
    assert reader.calls == [] and local.calls == []
    rj = run_json(b)
    assert rj["partial"] is False and rj["cache_only"] is True and all(r["calls"] == 0 for r in rj["roles"].values())
    assert rj["roles"]["specialist"]["cache_hits"] > 0 and rj["roles"]["specialist"]["not_cached"] == 0
    assert (a / "preds.csv").read_bytes() == (b / "preds.csv").read_bytes()


def test_a_cache_only_miss_is_a_failure_not_a_call(tmp_path):
    reader = ReaderLLM()
    out = run_fixture(tmp_path, cache_only=True, reader=reader)
    rj = run_json(out)
    assert reader.calls == [] and rj["partial"] is True and rj["pages_failed"] == 9
    assert rj["roles"]["specialist"]["not_cached"] > 0 and rj["roles"]["specialist"]["calls"] == 0


def test_both_engines_give_the_same_preds(tmp_path):
    a = run_fixture(tmp_path / "p", engine="plain", local=LocalLLM(grade="unsure"))
    b = run_fixture(tmp_path / "l", engine="langgraph", local=LocalLLM(grade="unsure"))
    assert (a / "preds.csv").read_bytes() == (b / "preds.csv").read_bytes()
    assert run_json(b)["engine"] == "langgraph"


# ---------------------------------------------------------------- run.json
def test_run_json_fields(tmp_path):
    out = run_fixture(tmp_path, local=LocalLLM(grade="unsure"))
    rj = run_json(out)
    for k in ("stopped_reason", "cases_total", "cases_done", "cases_pending", "split", "subset", "engine", "workers", "cache_only", "salt", "ablate", "partial", "partial_reasons", "stopped", "cases",
              "pages_readable", "pages_read", "pages_failed", "pages_unrouted", "failed_page_ids", "failures", "flags", "local_fallbacks",
              "iteration2_cases", "roles", "cache", "prompt_versions", "LOCAL_SCOPE", "route_recall", "route_recall_note", "submission",
              "preds_rows", "case_errors", "error_case_ids", "unfinished_case_ids", "spend", "wall_seconds"):
        assert k in rj, k
    assert rj["LOCAL_SCOPE"] == sorted(LOCAL_SCOPE) and rj["ablate"] == [] and rj["salt"] == ""
    assert set(rj["roles"]) == {"planner", "specialist", "router", "grader", "decision_maker"} | {f"specialist:{f}" for f in FAMILIES}
    for r in rj["roles"].values():
        assert set(r) == {"calls", "not_cached", "cache_hits", "cache_misses", "put_failures", "failures", "tokens_in", "tokens_out",
                          "seconds"}
    assert set(rj["prompt_versions"]) == {"planner", "router", "grader", "decision_maker"} | {f"specialist:{f}" for f in FAMILIES}
    assert rj["roles"]["planner"]["calls"] == 3 and rj["roles"]["router"]["calls"] == 9
    assert rj["roles"]["specialist"]["calls"] == sum(rj["roles"][f"specialist:{f}"]["calls"] for f in FAMILIES) > 0
    assert rj["roles"]["specialist"]["tokens_in"] > 0 and rj["roles"]["grader"]["calls"] > 0
    assert rj["route_recall"] is None and rj["route_recall_note"]                 # no baseline cache in this run
    assert rj["iteration2_cases"] == 3 and rj["preds_rows"] == 3 * (len(REQS) + 1)
    assert rj["local_fallbacks"] == {"router_all_families": 0, "grader_unsure": 0, "decision_abstain": 0}
    assert rj["spend"] is None and rj["stopped_reason"] == "complete" and rj["partial"] is False
    assert (rj["cases_total"], rj["cases_done"], rj["cases_pending"]) == (3, 3, 0)


def test_local_fallbacks_and_role_failures_are_counted_apart(tmp_path):
    out = run_fixture(tmp_path, local=LocalLLM(raise_on={"router", "grader"}))
    rj = run_json(out)
    assert rj["local_fallbacks"]["router_all_families"] == 9 == rj["roles"]["router"]["failures"]
    assert rj["local_fallbacks"]["grader_unsure"] == rj["roles"]["grader"]["failures"] > 0
    assert rj["failures"] == sum(rj["flags"].values())
    assert rj["partial"] is False                                                  # local fallbacks never make a run partial


def test_route_recall_with_a_baseline(tmp_path, monkeypatch):
    def fake(model, recs, cache_dir, system=None):
        out = {}
        for r in recs:
            pid = str(r["page"]["page_id"])
            out[pid] = [{"id": reading_id(0, FIELD_FAMILY[x["field"]], pid, x["field"], 0), "iteration": 0,
                         "family": FIELD_FAMILY[x["field"]], "page_id": pid, "field": x["field"], "value": x["value"],
                         "quote": x["quote"], "display": ""} for (fam, s), xs in READ.items() if s == _suffix(pid) for x in xs]
        return out
    monkeypatch.setattr(baseline_mod, "baseline_readings", fake)
    data = write_data(tmp_path / "data")
    (tmp_path / "base").mkdir()
    ag = make_agents(tmp_path / "cache", baseline_dir=tmp_path / "base")
    rj = runner.run(data, "test", "all", agents=ag, out=tmp_path / "out", metric=StubMetric())
    assert rj["route_recall"] == {"covered": 15, "total": 15, "missed": []} and rj["route_recall_note"] is None


def test_run_json_is_strict_json_with_plain_numbers(tmp_path):
    out = run_fixture(tmp_path)
    text = (out / "run.json").read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite number {c}"))
    assert text.isascii()


def test_jsonable_coerces_numpy_and_non_finite():
    np = pytest.importorskip("numpy")
    got = runner.jsonable({"a": np.int64(3), "b": np.float32(1.5), "c": float("nan"), "d": float("inf"), "e": (1, 2), "f": {3}})
    assert got == {"a": 3, "b": 1.5, "c": None, "d": None, "e": [1, 2], "f": [3]}
    assert type(got["a"]) is int and type(got["b"]) is float


def test_traces_and_graph_are_written(tmp_path):
    out = run_fixture(tmp_path)
    for cid in CASE_IDS:
        t = json.loads((out / "traces" / f"{cid}.json").read_text(encoding="utf-8"))
        assert t["case_id"] == cid and t["handovers"][-1]["agent"] == "audit"
        assert (out / "traces" / f"{cid}.md").read_text(encoding="utf-8").startswith(f"# Handover trace: {cid}")
    assert "validate_sources" in (out / "graph.mmd").read_text(encoding="utf-8")


def test_trace_numbers_are_coerced():
    from vendor_verdict.trace import Trace
    np = pytest.importorskip("numpy")
    t = Trace("c1")
    t.handover("a", "b", {"n": 1}, tokens_in=np.int64(5), tokens_out=None, seconds=float("nan"))
    clean = runner.clean_trace(t)
    data = clean.to_json()
    assert data["handovers"][0]["tokens_in"] == 5 and type(data["handovers"][0]["tokens_in"]) is int
    assert data["handovers"][0]["tokens_out"] == 0 and data["handovers"][0]["seconds"] == 0.0
    json.dumps(data, allow_nan=False)
    assert "c1" in clean.to_markdown()


# ---------------------------------------------------------------- leak guard
def test_no_canary_reaches_console_or_run_json(tmp_path, capsys):
    reader = ReaderLLM(fail={"p2"}, value=CANARY, raise_on={"c0003": ValueError(CANARY)})
    out = run_fixture(tmp_path, page_text=CANARY, reader=reader, local=LocalLLM(raise_on={"decision", "grader"}))
    cap = capsys.readouterr()
    assert CANARY not in cap.out and CANARY not in cap.err
    assert CANARY not in (out / "run.json").read_text(encoding="utf-8")
    for p in (out / "traces").iterdir():
        assert CANARY not in p.read_text(encoding="utf-8")


# ---------------------------------------------------------------- estimate
def test_estimate_makes_no_call_writes_nothing_and_counts_uncached(tmp_path):
    data = write_data(tmp_path / "data")
    cases, pages, ids = runner.load_split(data, "test", "all")
    planner, reader, local = PlannerLLM(), ReaderLLM(), LocalLLM()
    ag = make_agents(tmp_path / "cache", planner=planner, reader=reader, local=local)
    est = runner.estimate(cases, pages, ids, ag)
    assert planner.calls == reader.calls == local.calls == [] and not (tmp_path / "cache").exists()
    assert ag.cache.hits == ag.cache.misses == 0
    assert est["cases"] == 3 and est["pages_readable"] == 9
    assert est["planner"]["calls"] == est["planner"]["uncached"] == 3 and est["planner"]["tokens_in"] > 0
    assert est["router"]["pages"] == 9 and est["router"]["uncached"] == 9
    worst = est["specialist"]["worst"]
    assert worst["per_page"] == 4 and worst["calls"] == 36 and worst["uncached"] == 36 and worst["tokens_in"] > 0
    assert est["specialist"]["expected"] is None and est["specialist"]["expected_note"]
    assert est["iteration2_worst"]["calls"] == 12 and est["iteration2_worst"]["tokens_in"] > 0
    assert est["iteration2_worst"]["tokens_out"] == round(est["iteration2_worst"]["tokens_in"] * run_kaggle.OUT_PER_IN)
    lines = runner.format_estimate(est)
    assert any("upper bound" in s for s in lines) and any("measured" in s for s in lines)

    # after a run the caches hold the planner, router and routed specialist calls
    runner.run(data, "test", "all", agents=ag, out=tmp_path / "out", metric=StubMetric())
    est2 = runner.estimate(cases, pages, ids, replace(ag, main_llm=PlannerLLM(), reader_llm=ReaderLLM(), local_llm=LocalLLM()))
    assert est2["planner"]["uncached"] == 0 and est2["router"]["uncached"] == 0
    routed = run_json(tmp_path / "out")["roles"]["specialist"]["calls"]
    assert routed == 27 and est2["specialist"]["worst"]["uncached"] == 36 - routed      # p1, p2: 4 families each, p3: certifications


def test_estimate_with_a_baseline_gives_the_expected_specialists_per_page(tmp_path, monkeypatch):
    def fake(model, recs, cache_dir, system=None):
        out = {}
        for r in recs:
            pid = str(r["page"]["page_id"])
            out[pid] = [{"id": reading_id(0, fam, pid, x["field"], 0), "iteration": 0, "family": fam, "page_id": pid,
                         "field": x["field"], "value": x["value"], "quote": x["quote"], "display": ""}
                        for (fam, s), xs in READ.items() if s == _suffix(pid) for x in xs]
        return out
    monkeypatch.setattr(baseline_mod, "baseline_readings", fake)
    data = write_data(tmp_path / "data")
    (tmp_path / "base").mkdir()
    cases, pages, ids = runner.load_split(data, "test", "all")
    est = runner.estimate(cases, pages, ids, make_agents(tmp_path / "cache", baseline_dir=tmp_path / "base"))
    exp = est["specialist"]["expected"]
    assert exp["calls"] == 3 * 4 and exp["per_page"] == pytest.approx(4 / 3) and exp["uncached"] == 12      # p1: 2, p2: 1, p3: 1
    assert est["grader"]["readings_expected"] == 3 * 5


def test_estimate_respects_ablations(tmp_path):
    data = write_data(tmp_path / "data")
    cases, pages, ids = runner.load_split(data, "test", "all")
    est = runner.estimate(cases, pages, ids, make_agents(tmp_path / "cache", ablate={"router", "iteration2"}))
    assert est["router"]["calls"] == 0 and est["iteration2_worst"]["calls"] == 0


# ---------------------------------------------------------------- CLI
def test_default_out_dir():
    assert runner.default_out("test", "all", frozenset(), "") == ROOT / "reports" / "vendor_verdict" / "test_all"
    assert runner.default_out("train", "dev", frozenset({"router", "grader"}), "") == \
        ROOT / "reports" / "vendor_verdict" / "train_dev_ablate-grader+router"
    assert runner.default_out("train", "dev", frozenset(), "s/1").name == "train_dev_salt-s_1"


def test_cli_estimate_exits_before_any_call_and_writes_nothing(tmp_path, monkeypatch, capsys):
    data = write_data(tmp_path / "data", page_text=CANARY)
    planner, reader, local = PlannerLLM(), ReaderLLM(), LocalLLM()
    ag = make_agents(tmp_path / "cache", planner=planner, reader=reader, local=local)
    monkeypatch.setattr(runner, "build_agents", lambda a, data_dir: (ag, None, StubMetric()))
    rc = runner.main(["run", "--data", str(data), "--split", "test", "--subset", "all", "--estimate", "--out", str(tmp_path / "out")])
    assert rc == 0 and planner.calls == reader.calls == local.calls == []
    assert not (tmp_path / "out").exists() and not (tmp_path / "cache").exists()
    out = capsys.readouterr().out
    assert "estimate" in out and CANARY not in out


def test_cli_run_passes_flags_through(tmp_path, monkeypatch, capsys):
    data = write_data(tmp_path / "data")
    seen = {}

    def build(a, data_dir):
        seen.update(salt=a.salt, ablate=a.ablate)
        return make_agents(tmp_path / "cache", salt=a.salt, ablate=runner.parse_ablate(a.ablate)), None, StubMetric()
    monkeypatch.setattr(runner, "build_agents", build)
    rc = runner.main(["run", "--data", str(data), "--split", "test", "--subset", "all", "--ablate", "decision_maker",
                      "--engine", "langgraph", "--workers", "2", "--limit", "2", "--out", str(tmp_path / "out")])
    rj = run_json(tmp_path / "out")
    assert rc == 0 and rj["engine"] == "langgraph" and rj["ablate"] == ["decision_maker"] and rj["cases"]["selected"] == 2
    assert rj["limit"] == 2 and not (tmp_path / "out" / "submission.csv").exists()          # 2 of 3 test cases: no submission


def test_cli_refuses_an_unknown_ablation():
    with pytest.raises(SystemExit):
        runner.parse_ablate("planner")


def test_put_failures_are_counted_per_role(tmp_path):
    blocker = tmp_path / "cache"
    blocker.write_text("a file where the cache dir should be", encoding="utf-8")          # every put fails
    out = run_fixture(tmp_path)
    rj = run_json(out)
    assert rj["cache"]["put_failures"] > 0
    assert rj["roles"]["planner"]["put_failures"] == 3 and rj["roles"]["router"]["put_failures"] == 9
    assert sum(rj["roles"][r]["put_failures"] for r in ("planner", "specialist", "router", "grader", "decision_maker")) == \
        rj["cache"]["put_failures"]


def test_cases_cancelled_by_a_stop_are_unfinished_not_errors(tmp_path):
    reader = ReaderLLM(raise_on={"c0001": BudgetExceeded("budget")}, delay={"c0001": 0.05})
    out = run_fixture(tmp_path, reader=reader, workers=1)
    rj = run_json(out)
    assert rj["case_errors"] == {} and rj["cases"]["errors"] == 0
    assert rj["cases"]["finished"] + rj["cases"]["unfinished"] == 3 and "c0001" in rj["unfinished_case_ids"]


def test_a_stop_ends_the_cases_in_flight_at_their_next_model_call(tmp_path):
    # c0001 hits the budget after 0.1 s; c0002 runs alongside, 0.05 s per read over 9 reads, so it is still reading at the stop
    reader = ReaderLLM(raise_on={"c0001": BudgetExceeded("budget")}, delay={"c0001": 0.1, "c0002": 0.05})
    out = run_fixture(tmp_path, reader=reader, workers=2)
    rj = run_json(out)
    assert rj["stopped"] == "budget" and "c0002" in rj["unfinished_case_ids"] and rj["cases"]["finished"] == 0


@pytest.mark.parametrize("kind,reason", [("pages", "failed_pages"), ("budget", "budget"), ("interrupt", "interrupted")])
def test_stopped_reason(tmp_path, kind, reason):
    reader = {"pages": ReaderLLM(fail={"p2"}), "budget": ReaderLLM(raise_on={"c0002": BudgetExceeded("b")}),
              "interrupt": ReaderLLM(raise_on={"c0002": KeyboardInterrupt()})}[kind]
    rj = run_json(run_fixture(tmp_path, reader=reader, workers=1))
    assert rj["stopped_reason"] == reason and rj["partial"] is True
    if kind != "pages":
        assert rj["cases_total"] == 3 and rj["cases_done"] >= 1 and rj["cases_done"] + rj["cases_pending"] == 3
        assert not (tmp_path / "out" / "traces" / "c0002.json").exists()


def test_stop_exceptions_is_the_seam(tmp_path, monkeypatch):
    class Quota(Exception):
        pass
    monkeypatch.setattr(runner, "STOP_EXCEPTIONS", (BudgetExceeded, Quota))
    rj = run_json(run_fixture(tmp_path, reader=ReaderLLM(raise_on={"c0001": Quota()}), workers=1))
    assert rj["partial"] is True and rj["case_errors"] == {} and rj["stopped_reason"] == "quota" and rj["cases_done"] == 0


def test_resume_into_the_same_out_overwrites(tmp_path):
    first = run_fixture(tmp_path, reader=ReaderLLM(raise_on={"c0002": BudgetExceeded("b")}), workers=1)
    assert run_json(first)["partial"] is True
    again = run_fixture(tmp_path)                                    # the same command again, warm cache, same --out
    rj = run_json(again)
    assert again == first and rj["partial"] is False and rj["stopped_reason"] == "complete"
    assert rj["roles"]["planner"]["cache_hits"] >= 1 and len(preds(again)) == 3 * (len(REQS) + 1)
    assert (again / "submission.csv").exists() and sorted(p.name for p in (again / "traces").iterdir()) == \
        sorted(f"{c}.{e}" for c in CASE_IDS for e in ("json", "md"))


# ---------------------------------------------------------------- fix round 1
def test_a_cache_only_run_with_local_misses_is_partial(tmp_path, monkeypatch):
    # I1: a grader prompt change after the filling run: every grade is a cache-only miss ("unsure"), no page fails
    a = run_fixture(tmp_path / "a")
    from vendor_verdict.agents import grader as grader_agent
    monkeypatch.setattr(grader_agent, "GRADER_SYSTEM", grader_agent.GRADER_SYSTEM + " (edited)")
    b = run_fixture(tmp_path / "b", cache_only=True, cache_from=a)
    rj = run_json(b)
    assert rj["roles"]["grader"]["not_cached"] > 0 and rj["pages_failed"] == 0
    assert rj["partial"] is True and "cache_misses" in rj["partial_reasons"] and rj["stopped_reason"] == "cache_misses"
    assert rj["not_cached"] == sum(rj["roles"][r]["not_cached"] for r in ("planner", "specialist", "router", "grader", "decision_maker"))
    assert not (b / "submission.csv").exists() and rj["submission"]["written"] is False


def test_a_complete_cache_only_run_has_no_cache_misses(tmp_path):
    a = run_fixture(tmp_path / "a")
    rj = run_json(run_fixture(tmp_path / "b", cache_only=True, cache_from=a))
    assert rj["not_cached"] == 0 and rj["stopped_reason"] == "complete" and rj["submission"]["written"] is True


class _EnabledCache:
    """Stands in for the assessor's on-disk LLM cache (never touched: the test only checks it is replaced)."""
    enabled = True


def test_run_turns_the_assessor_llm_cache_off(tmp_path, monkeypatch):
    import assessor.cache as acache
    monkeypatch.setattr(acache, "_CACHE", _EnabledCache())
    run_fixture(tmp_path)
    assert isinstance(acache.get_cache(), acache.NullCache)


def test_build_agents_turns_the_assessor_llm_cache_off_before_any_model_exists(tmp_path, monkeypatch):
    import assessor.cache as acache
    import assessor.llm as allm
    monkeypatch.setattr(acache, "_CACHE", _EnabledCache())
    seen = []

    class FakeLLM:                                   # never a real LLM: records the process cache, then reports mock mode
        mock_mode, model = True, "fake"

        def __init__(self, *a, **k):
            seen.append(acache.get_cache())

        @classmethod
        def extractor(cls, *a, **k):
            return cls()
    monkeypatch.setattr(allm, "LLM", FakeLLM)
    args = runner.build_parser().parse_args(["run", "--split", "test", "--subset", "all"])
    with pytest.raises(SystemExit):
        runner.build_agents(args, tmp_path)
    assert seen and all(isinstance(c, acache.NullCache) for c in seen)


class ChargingReader(ReaderLLM):
    """Charges a real Budget for every call, at the configured prices, as LLM._account does."""

    def __init__(self, budget, **kw):
        super().__init__(**kw)
        self.budget = budget

    def structured(self, system, user, schema, stage="", **kw):
        out = super().structured(system, user, schema, stage=stage, **kw)
        self.budget.charge(200_000_000, 0, model=self.model, role="main", stage=stage)    # tens of USD per call at any real price
        return out


def test_no_money_cap_by_default(tmp_path, monkeypatch):
    import assessor.tracing as tracing
    monkeypatch.setattr(tracing, "price_for", lambda *a, **kw: (0.15, 0.60))     # pinned: a PRICE_IN_PER_MTOK=0 env cannot void it
    budget = runner.make_budget(None)
    out = run_fixture(tmp_path, reader=ChargingReader(budget), budget=budget)
    rj = run_json(out)
    assert rj["spend"]["usd"] > 5 and rj["partial"] is False and rj["stopped_reason"] == "complete"


def test_an_explicit_money_cap_still_stops_the_run(tmp_path):
    budget = runner.make_budget(1.0)
    rj = run_json(run_fixture(tmp_path, reader=ChargingReader(budget), budget=budget, workers=1))
    assert rj["stopped"] == "budget" and rj["stopped_reason"] == "budget" and rj["partial"] is True


def test_max_usd_defaults_to_no_cap():
    a = runner.build_parser().parse_args(["run", "--split", "test", "--subset", "all"])
    assert a.max_usd is None and runner.make_budget(None).max_usd == float("inf")
    assert runner.build_parser().parse_args(["run", "--split", "test", "--subset", "all", "--max-usd", "2.5"]).max_usd == 2.5


# ---------------------------------------------------------------- Task 8B2: runner hygiene
def test_the_flag_table_covers_every_node_flag():
    from vendor_verdict.nodes import FLAG_KEYS
    assert set(runner._FLAG_ROLE) == set(FLAG_KEYS) and set(runner._FLAG_ROLE.values()) <= set(runner.ROLES)


def _refused(tmp_path, monkeypatch, capsys, *argv, split="test"):
    """main() with arguments it must refuse: nothing is built, run, deleted or written."""
    data = write_data(tmp_path / "data", split=split)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    (out / "submission.csv").write_text("earlier", encoding="utf-8")
    monkeypatch.setattr(runner, "build_agents", lambda a, d: pytest.fail("built agents for refused arguments"))
    monkeypatch.setattr(runner, "run", lambda *a, **kw: pytest.fail("started a run for refused arguments"))
    rc = runner.main(["run", "--data", str(data), "--split", split, "--subset", "all", "--out", str(out), *argv])
    assert rc == runner.EXIT_USAGE and (out / "submission.csv").read_text() == "earlier" and sorted(out.iterdir()) == [out / "submission.csv"]
    cap = capsys.readouterr()
    return cap.out + cap.err


def test_the_usage_exit_code_is_its_own():
    assert runner.EXIT_USAGE not in (0, 1, 3, 69, 75, 130)
    assert f"{runner.EXIT_USAGE} " in runner.__doc__ and "refused" in runner.__doc__


@pytest.mark.parametrize("only,named", [("c9999", ["c9999"]), ("c0001,c9999,c8888", ["c8888", "c9999"])])
def test_only_with_unknown_case_ids_is_refused_and_names_them(tmp_path, monkeypatch, capsys, only, named):
    text = _refused(tmp_path, monkeypatch, capsys, "--only", only)
    assert all(c in text for c in named) and "c0001" not in text


@pytest.mark.parametrize("only", [",", " , ", ""])
def test_only_with_no_case_id_is_refused(tmp_path, monkeypatch, capsys, only):
    assert "--only" in _refused(tmp_path, monkeypatch, capsys, "--only", only)


def test_only_with_an_id_of_another_subset_is_refused(tmp_path, monkeypatch, capsys):
    data = write_data(tmp_path / "data", split="train")            # dev = c0001, c0003 (even positions); holdout = c0002
    monkeypatch.setattr(runner, "build_agents", lambda a, d: pytest.fail("built agents for refused arguments"))
    rc = runner.main(["run", "--data", str(data), "--split", "train", "--subset", "dev", "--only", "c0002", "--out",
                      str(tmp_path / "out")])
    assert rc == runner.EXIT_USAGE and "c0002" in capsys.readouterr().err and not (tmp_path / "out").exists()


@pytest.mark.parametrize("argv", [("--limit", "-1"), ("--max-usd", "nan"), ("--max-usd", "inf"), ("--max-usd=-inf",),
                                  ("--max-usd", "-0.5"), ("--ablate", "planner")])
def test_bad_numbers_and_ablations_are_refused(tmp_path, monkeypatch, capsys, argv):
    assert argv[0].split("=")[0] in _refused(tmp_path, monkeypatch, capsys, *argv)


@pytest.mark.parametrize("extra", [("--cache-only",), ("--estimate",), ("--allow-local-down",), ("--wait-quota",)])
def test_prewarm_with_an_option_that_makes_no_sense_for_it_is_refused(tmp_path, monkeypatch, capsys, extra):
    assert extra[0] in _refused(tmp_path, monkeypatch, capsys, "--prewarm-local", *extra)


def test_no_money_cap_stays_the_default_and_a_zero_cap_is_allowed(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "build_agents", lambda a, d: (seen.append(a.max_usd), (make_agents(tmp_path / "cache"), None,
                                                                                       StubMetric()))[1])
    data = write_data(tmp_path / "data")
    for extra in ((), ("--max-usd", "0")):
        runner.main(["run", "--data", str(data), "--split", "test", "--subset", "all", "--out", str(tmp_path / "out"), *extra])
    assert seen == [None, 0.0]


def test_default_out_dir_names_for_limited_and_only_runs():
    import hashlib
    h = hashlib.sha1(b"c0001,c0002").hexdigest()[:8]
    base = ROOT / "reports" / "vendor_verdict"
    assert runner.default_out("test", "all", frozenset(), "") == base / "test_all"                 # a full run keeps its name
    assert runner.default_out("test", "all", frozenset(), "", limit=None, only=None) == base / "test_all"
    assert runner.default_out("test", "all", frozenset(), "", limit=0) == base / "test_all"
    assert runner.default_out("test", "all", frozenset(), "", limit=5) == base / "test_all_limit-5"
    assert runner.default_out("test", "all", frozenset(), "", only=["c0002", "c0001"]) == base / f"test_all_only-{h}"
    assert runner.default_out("train", "dev", frozenset({"router", "grader"}), "s/1", limit=2, only=["c0001", "c0002"]) == \
        base / f"train_dev_limit-2_only-{h}_ablate-grader+router_salt-s_1"
    assert runner.default_out("test", "all", frozenset({"decision_maker"}), "") == base / "test_all_ablate-decision_maker"
    name = runner.default_out("test", "all", frozenset(), "", only=["c0001", "c0002"]).name
    assert "c0001" not in name and "c0002" not in name and len(name) == len("test_all_only-") + 8


def test_a_limited_run_never_touches_a_full_runs_default_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)                  # the default location, under tmp_path
    data = write_data(tmp_path / "data")
    monkeypatch.setattr(runner, "build_agents", lambda a, d: (make_agents(tmp_path / "cache"), None, StubMetric()))
    base = tmp_path / "reports" / "vendor_verdict"
    assert runner.main(["run", "--data", str(data), "--split", "test", "--subset", "all"]) == 0
    full = base / "test_all"
    assert (full / "submission.csv").exists()
    before = {p.name: p.read_bytes() for p in full.iterdir() if p.is_file()}
    assert runner.main(["run", "--data", str(data), "--split", "test", "--subset", "all", "--limit", "1"]) == 0
    assert runner.main(["run", "--data", str(data), "--split", "test", "--subset", "all", "--only", "c0002,c0001"]) == 0
    assert {p.name: p.read_bytes() for p in full.iterdir() if p.is_file()} == before
    dirs = sorted(p.name for p in base.iterdir())
    assert len(dirs) == 3 and "test_all_limit-1" in dirs and not any("c000" in d for d in dirs)
    for d in dirs[1:]:
        assert not (base / d / "submission.csv").exists() and run_json(base / d)["cases"]["selected"] in (1, 2)


def _no_orphan_submission(out: Path) -> None:
    assert not (out / "submission.csv.tmp").exists() and not (out / "run.json.tmp").exists()
    if (out / "submission.csv").exists():
        rj = run_json(out)                                         # a complete run.json that matches it
        assert rj["submission"]["written"] is True and rj["submission"]["problems"] == []


def test_a_failed_submission_write_leaves_no_submission(tmp_path, monkeypatch):
    real = runner.rk.write_submission

    def half(path, *a, **kw):
        real(path, *a, **kw)
        raise OSError("disk full")
    monkeypatch.setattr(runner.rk, "write_submission", half)
    with pytest.raises(OSError):
        run_fixture(tmp_path)
    out = tmp_path / "out"
    assert not (out / "submission.csv").exists()
    _no_orphan_submission(out)


def test_a_failed_run_json_write_leaves_no_submission(tmp_path, monkeypatch):
    real = runner._write_atomic

    def fail_run_json(path, text):
        if Path(path).name == "run.json":
            raise OSError("disk full")
        return real(path, text)
    monkeypatch.setattr(runner, "_write_atomic", fail_run_json)
    with pytest.raises(OSError):
        run_fixture(tmp_path)
    out = tmp_path / "out"
    assert not (out / "submission.csv").exists() and not (out / "run.json").exists()
    _no_orphan_submission(out)


def test_a_complete_run_writes_run_json_and_then_the_submission(tmp_path):
    out = run_fixture(tmp_path)
    assert (out / "submission.csv").exists() and run_json(out)["submission"]["written"] is True
    _no_orphan_submission(out)


def test_stop_exceptions_hold_quota_exhausted():
    from vendor_verdict.quota import QuotaExhausted
    assert QuotaExhausted in runner.STOP_EXCEPTIONS                 # so the outcome loop needs no second QuotaExhausted


def test_previous_resume_count_of_an_unreadable_run_json_is_one(tmp_path):
    (tmp_path / "run.json").write_text("not json", encoding="utf-8")
    assert runner.previous_resume_count(tmp_path) == 1 and "counts as 1" in runner.previous_resume_count.__doc__
