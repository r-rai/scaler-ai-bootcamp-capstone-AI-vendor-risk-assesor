"""vendor_verdict gates (gates.py): the pre-registered rules, the input acceptance, the R45/R46 glue and the counts-only output.

Invented data only: every run directory, gold map, s0 map and submission is written into `tmp_path`; nothing under `data_kaggle/` is
read, no model is built (the conftest forces the mock provider) and nothing opens a socket. The one runner-made run.json comes from the
runner tests' stub fixture (tests/test_vv_runner.py)."""
import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run_kaggle  # noqa: E402,F401

from vendor_verdict import gates  # noqa: E402
from vendor_verdict import runner  # noqa: E402
from vendor_verdict.config import EXPOSED_HOLDOUT_ROWS  # noqa: E402
from vendor_verdict.gates import (GateRefused, candidate, check_run, r_dm, r_hold, r_route, r_smoke, r_test,  # noqa: E402
                                  replay_identical, wrong_counts)

DM = frozenset({"decision_maker"})
RT = frozenset({"router"})
RTDM = RT | DM
PV = {"planner": "aaaaaaaaaaaa", "router": "bbbbbbbbbbbb", "decision_maker": "cccccccccccc"}
CANARY = "ZZCANARYREQ"                      # an invented requirement id: must never reach stdout or gates.json
FRESH = {"prompt_versions": PV, "inputs": []}       # a hand-written section that summarize() accepts as fresh (R55a)
LABEL_WORDS = ("PASS", "FAIL", "UNKNOWN", "NEEDS_REVIEW", "BLOCKED", "CONDITIONAL", "READY_FOR_APPROVAL")


# ---------------------------------------------------------------- the brief's tests (verbatim)
def test_wrong_counts_excludes_rows_and_counts_false_confidence():
    gold = {"c1__A": "UNKNOWN", "c1__B": "PASS", "c2__A": "FAIL"}
    assert wrong_counts({"c1__A": "PASS", "c1__B": "PASS", "c2__A": "PASS"}, gold, frozenset({"c2__A"})) == (1, 1)
def test_r_hold_needs_both_counts_not_worse():
    assert r_hold((2, 0), (3, 1)) and not r_hold((2, 2), (3, 1))
def test_r_dm_counts_only_changed_rows():
    assert r_dm({"x": "PASS", "y": "FAIL"}, {"x": "FAIL", "y": "FAIL"}, {"x": "PASS", "y": "FAIL"}, frozenset()) == (True, 1, 0)
def test_candidate_drops_dm_when_r_dm_fails():
    ok = {"R-smoke": True, "R-route": True, "R-hold": True, "R-test": True}
    assert candidate({**ok, "R-DM": True}) == ("B", frozenset())
    assert candidate({**ok, "R-DM": False}) == ("B", frozenset({"decision_maker"}))
    assert candidate({**ok, "R-hold": False, "R-DM": True}) == ("none", frozenset())
def test_partial_run_is_refused():
    with pytest.raises(GateRefused): check_run({"partial": True})


# ---------------------------------------------------------------- pure rules: boundaries (mutation checks)
def test_r_hold_boundaries():
    assert r_hold((3, 1), (3, 1))                         # equal on both counts passes
    assert not r_hold((4, 1), (3, 1)) and not r_hold((3, 2), (3, 1))
    assert r_hold((0, 0), (0, 0))


def test_r_route_boundaries():
    assert r_route(7, 7) and not r_route(6, 7) and not r_route(8, 7)


def test_r_test_boundary_at_30_changed_rows():
    sub2 = {f"t{i}__R": "PASS" for i in range(40)}
    def changed(n):
        return {k: ("FAIL" if i < n else v) for i, (k, v) in enumerate(sub2.items())}
    assert r_test(changed(30), sub2) == (True, 30)
    assert r_test(changed(31), sub2) == (False, 31)
    assert r_test(dict(sub2), sub2) == (True, 0)


def test_r_test_counts_ids_missing_on_either_side():
    assert r_test({"a": "PASS", "b": "PASS"}, {"a": "PASS", "c": "PASS"}) == (True, 2)


def test_r_dm_tie_passes_and_a_loss_fails():
    gold = {"x": "PASS", "y": "FAIL"}
    assert r_dm({"x": "PASS", "y": "PASS"}, {"x": "FAIL", "y": "FAIL"}, gold, frozenset()) == (True, 1, 1)
    assert r_dm({"x": "FAIL", "y": "FAIL"}, {"x": "PASS", "y": "FAIL"}, gold, frozenset()) == (False, 0, 1)
    assert r_dm({"x": "PASS"}, {"x": "PASS"}, gold, frozenset()) == (True, 0, 0)       # no changed row


def test_r_dm_excludes_rows_and_ignores_rows_without_gold():
    gold = {"x": "PASS", "y": "FAIL"}
    assert r_dm({"x": "FAIL", "z": "PASS"}, {"x": "PASS", "z": "FAIL"}, gold, frozenset({"x"})) == (True, 0, 0)
    # a changed row where neither side is right counts on neither side
    assert r_dm({"y": "PASS"}, {"y": "UNKNOWN"}, gold, frozenset()) == (True, 0, 0)


def test_wrong_counts_false_confidence_only_on_requirement_rows():
    gold = {"c1__R": "NEEDS_REVIEW", "c1__S": "UNKNOWN", "c1__OVERALL": "CONDITIONAL"}
    preds = {"c1__R": "FAIL", "c1__S": "NEEDS_REVIEW", "c1__OVERALL": "BLOCKED"}
    assert wrong_counts(preds, gold, frozenset()) == (3, 1)


def test_wrong_counts_missing_prediction_is_wrong_and_extra_predictions_are_ignored():
    assert wrong_counts({"other__R": "PASS"}, {"c1__R": "PASS"}, frozenset()) == (1, 0)
    assert wrong_counts({"c1__R": "PASS"}, {"c1__R": "PASS"}, frozenset()) == (0, 0)


def test_an_overall_row_is_never_false_confident():
    assert wrong_counts({"c__OVERALL": "PASS"}, {"c__OVERALL": "UNKNOWN"}, frozenset()) == (1, 0)


def test_wrong_counts_gold_pass_or_fail_is_never_false_confident():
    assert wrong_counts({"c__R": "FAIL"}, {"c__R": "PASS"}, frozenset()) == (1, 0)


@pytest.mark.parametrize("off", ["R-smoke", "R-route", "R-hold", "R-test"])
def test_candidate_is_none_when_any_required_rule_fails(off):
    rules = {"R-smoke": True, "R-route": True, "R-hold": True, "R-test": True, "R-DM": True}
    assert candidate({**rules, off: False}) == ("none", frozenset())
    assert candidate({**rules, off: False, "R-DM": False}) == ("none", frozenset())
    missing = {k: v for k, v in rules.items() if k != off}
    assert candidate(missing) == ("none", frozenset())          # R45: not computable counts as a fail


def test_candidate_edge_cases():
    assert candidate({}) == ("none", frozenset())
    ok = {"R-smoke": True, "R-route": True, "R-hold": True, "R-test": True}
    assert candidate(ok) == ("B", DM)                           # R-DM not computable: the pre-registered switch


BRANCHES = [  # (holdout R-route, R-DM) -> ablations of the candidate
    (True, True, frozenset()), (True, False, DM), (False, True, RT), (False, False, RTDM)]


@pytest.mark.parametrize("route,dm,ablate", BRANCHES)
def test_configuration_order(route, dm, ablate):
    assert gates.configuration(route, dm) == ablate
    assert gates.router_config(route) == (frozenset() if route else RT)


@pytest.mark.parametrize("route,dm,ablate", BRANCHES)
def test_candidate_branches_with_the_router(route, dm, ablate):
    ok = {"R-smoke": True, "R-hold": True, "R-test": True, "R-DM": dm, "R-route": route}
    assert candidate(ok, router_off=not route) == ("B", ablate)
    for off in ("R-smoke", "R-hold", "R-test"):
        assert candidate({**ok, off: False}, router_off=not route) == ("none", frozenset())
        assert candidate({k: v for k, v in ok.items() if k != off}, router_off=not route) == ("none", frozenset())


def test_router_off_makes_r_route_true_by_construction_and_on_keeps_it_a_gate():
    ok = {"R-smoke": True, "R-hold": True, "R-test": True, "R-DM": True}
    assert candidate({**ok, "R-route": False}, router_off=True) == ("B", RT)
    assert candidate(ok, router_off=True) == ("B", RT)
    assert candidate({**ok, "R-route": False}) == ("none", frozenset())
    assert candidate({**ok, "R-DM": None}, router_off=True) == ("B", RTDM)           # not computable: DM off


def test_r_smoke():
    good = {"partial": False, "failures": 0, "pages_readable": 9, "pages_read": 9}
    assert r_smoke(good)
    assert not r_smoke({**good, "partial": True})
    assert not r_smoke({**good, "failures": 1})
    assert not r_smoke({**good, "pages_read": 8})
    assert not r_smoke({k: v for k, v in good.items() if k != "partial"})


def test_replay_identical():
    run_b = {"partial": False, "stopped_reason": "complete", "cache_only": True, "not_cached": 0}
    assert replay_identical(b"id,verdict\nx,PASS\n", b"id,verdict\nx,PASS\n", run_b)
    assert not replay_identical(b"id,verdict\nx,PASS\n", b"id,verdict\nx,FAIL\n", run_b)
    assert not replay_identical(b"a", b"a", {**run_b, "partial": True})
    assert not replay_identical(b"a", b"a", {**run_b, "stopped_reason": "quota"})
    assert not replay_identical(b"a", b"a", {**run_b, "cache_only": False})
    assert not replay_identical(b"a", b"a", {**run_b, "not_cached": 1})


# ---------------------------------------------------------------- input acceptance
def rj(target="holdout", ablate=(), n=4, **over) -> dict:
    split, subset = gates.TARGETS[target]
    dev = target == "dev"
    run = {"split": split, "subset": subset, "partial": False, "stopped_reason": "complete", "salt": "",
           "limit": 20 if dev else None, "only": None, "cases_total": 20 if dev else n, "cache_only": False, "not_cached": 0,
           "ablate": sorted(ablate), "pages_readable": 10, "pages_read": 10, "failures": 0, "prompt_versions": dict(PV),
           "route_recall": {"covered": 5, "total": 5, "missed": []},
           "submission": {"written": target == "test" and set(ablate) <= DM, "reason": "ok", "problems": []}}
    run.update(over)
    return run


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(gates, "FROZEN_PROMPT_VERSIONS", dict(PV))


@pytest.mark.parametrize("target", ["holdout", "test"])
def test_a_complete_run_is_accepted(frozen, target):
    check_run(rj(target), target, cases_expected=4)
    check_run(rj(target, ablate=DM), target, ablate=DM, cases_expected=4)
    check_run(rj(target, cache_only=True, not_cached=0), target, cases_expected=4)


@pytest.mark.parametrize("over", [
    {"partial": True}, {"partial": None}, {"stopped_reason": "quota"}, {"stopped_reason": "failed_pages"}, {"salt": "s1"},
    {"limit": 2}, {"only": ["c1"]}, {"cases_total": 3}, {"cases_total": 5}, {"cache_only": True, "not_cached": 2},
    {"ablate": ["decision_maker"]}, {"ablate": ["grader"]}, {"split": "test"}, {"subset": "dev"},
])
def test_holdout_input_acceptance_refusals(frozen, over):
    with pytest.raises(GateRefused):
        check_run(rj("holdout", **over), "holdout", cases_expected=4)


def test_an_ablation_mismatch_is_refused_both_ways(frozen):
    with pytest.raises(GateRefused):
        check_run(rj("test"), "test", ablate=DM, cases_expected=4)
    with pytest.raises(GateRefused):
        check_run(rj("test", ablate={"decision_maker", "grader"}), "test", ablate=DM, cases_expected=4)


@pytest.mark.parametrize("over", [{"limit": None}, {"limit": 19}, {"limit": 21}, {"cases_total": 19}, {"only": ["c1"]},
                                  {"subset": "holdout"}, {"salt": "x"}, {"ablate": ["router"]}])
def test_dev_input_acceptance_refusals(over):
    with pytest.raises(GateRefused):
        check_run(rj("dev", **over), "dev")


def test_a_run_missing_keys_is_refused_not_a_key_error(frozen):
    for target in ("dev", "holdout", "test"):
        with pytest.raises(GateRefused):
            check_run({"partial": False}, target, cases_expected=4)
    with pytest.raises(GateRefused):
        check_run("not a dict", "dev")


def test_holdout_and_test_need_the_full_subset_size(frozen):
    with pytest.raises(GateRefused):
        check_run(rj("holdout"), "holdout")                     # cases_expected unknown


@pytest.mark.parametrize("target", ["holdout", "test"])
def test_unfrozen_prompts_are_refused_for_holdout_and_test(monkeypatch, target):
    monkeypatch.setattr(gates, "FROZEN_PROMPT_VERSIONS", None)
    with pytest.raises(GateRefused, match="NOT YET RECORDED|frozen"):
        check_run(rj(target), target, cases_expected=4)


@pytest.mark.parametrize("target", ["holdout", "test"])
def test_a_prompt_version_mismatch_is_refused(monkeypatch, target):
    monkeypatch.setattr(gates, "FROZEN_PROMPT_VERSIONS", {**PV, "router": "dddddddddddd"})
    with pytest.raises(GateRefused):
        check_run(rj(target), target, cases_expected=4)
    monkeypatch.setattr(gates, "FROZEN_PROMPT_VERSIONS", {**PV, "grader": "eeeeeeeeeeee"})       # an extra frozen role
    with pytest.raises(GateRefused):
        check_run(rj(target), target, cases_expected=4)


def test_dev_runs_without_frozen_prompts(monkeypatch):
    monkeypatch.setattr(gates, "FROZEN_PROMPT_VERSIONS", None)
    check_run(rj("dev"), "dev")


def test_the_module_docstring_is_the_pre_registration():
    doc = gates.__doc__
    assert "Frozen prompt versions (recorded at Task 10 step 4" in doc
    assert "NOT YET RECORDED" not in doc
    for word in ("R-smoke", "R-route", "R-hold", "R-DM", "R-test", "30", "refines", "exposed", "not computable"):
        assert word in doc
    # Task 10 step 4 (the freeze): the docstring, the constant and the prompts the code really uses must all agree. Editing any prompt
    # after the freeze fails this test on purpose.
    assert gates.FROZEN_PROMPT_VERSIONS == runner.prompt_versions()
    assert all(v in doc for v in gates.FROZEN_PROMPT_VERSIONS.values())


def test_a_runner_made_run_json_passes_the_acceptance(tmp_path, monkeypatch):
    """Contract: the gates read the runner's real key names (runner stub fixture, invented data)."""
    from tests.test_vv_runner import run_fixture, run_json
    out = run_fixture(tmp_path)
    run = run_json(out)
    monkeypatch.setattr(gates, "FROZEN_PROMPT_VERSIONS", runner.prompt_versions())
    check_run(run, "test", cases_expected=run["cases_total"])
    assert r_smoke(run)
    assert run["route_recall"] is None                          # no baseline in the fixture: R-route is not computable


# ---------------------------------------------------------------- the glue: invented run directories
def write_run(root: Path, target: str, ablate=(), run=None, preds=None, submission=None) -> Path:
    split, subset = gates.TARGETS[target]
    d = root / runner.default_out(split, subset, frozenset(ablate), "", limit=20 if target == "dev" else None).name
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(run if run is not None else rj(target, ablate)), encoding="utf-8")
    write_csv(d / "preds.csv", preds or {})
    if submission is not None:
        write_csv(d / "submission.csv", submission)
    return d


def write_csv(path: Path, rows: dict) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "verdict"])
        w.writerows(rows.items())
    return path


# holdout fixture: rows r0..r9 plus one exposed row; gold PASS everywhere except the false-confidence row r9 (UNKNOWN)
EXPOSED = sorted(EXPOSED_HOLDOUT_ROWS)[0]
GOLD = {**{f"h{i}__{CANARY}": "PASS" for i in range(9)}, f"h9__{CANARY}": "UNKNOWN", EXPOSED: "PASS"}


def preds_with(wrong: int, fc: bool = False, exposed_wrong: bool = False) -> dict:
    p = {k: ("FAIL" if i < wrong else "PASS") for i, k in enumerate(list(GOLD)[:9])}
    p[f"h9__{CANARY}"] = "PASS" if fc else "UNKNOWN"
    p[EXPOSED] = "FAIL" if exposed_wrong else "PASS"
    return p


def holdout(root, *, full=None, abl=None, s0=None, gates_json=None, say=None):
    gj = gates_json or root / "gates.json"
    return gates.gate("holdout", runs_root=root, gates_json=gj, cases_expected=4, gold=lambda: dict(GOLD),
                      s0=lambda: s0 if s0 is not None else preds_with(2), say=say or (lambda *_: None))


def test_r46_dm_passes_so_r_hold_reads_the_full_run(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), preds=preds_with(1))             # the DM fixes a row: full run has 1 wrong
    write_run(tmp_path, "holdout", DM, preds=preds_with(3))             # abstaining leaves 3 wrong (worse than s0's 2)
    sec = holdout(tmp_path)["holdout"]
    assert sec["rules"]["R-DM"]["pass"] and sec["config"] == []
    assert sec["rules"]["R-hold"]["pass"] and sec["rules"]["R-hold"]["run"] == "train_holdout"


def test_r46_dm_fails_so_r_hold_reads_the_ablated_run(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), preds=preds_with(3))             # the DM breaks a row: 3 wrong
    write_run(tmp_path, "holdout", DM, preds=preds_with(1))
    sec = holdout(tmp_path)["holdout"]
    assert not sec["rules"]["R-DM"]["pass"] and sec["rules"]["R-DM"]["computable"]
    assert sec["config"] == ["decision_maker"]
    assert sec["rules"]["R-hold"]["pass"] and sec["rules"]["R-hold"]["run"] == "train_holdout_ablate-decision_maker"


def test_r53_router_off_chain_with_r_dm_not_computable(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), run=rj("holdout", partial=True), preds=preds_with(0))     # refused: R-route not computable
    write_run(tmp_path, "holdout", DM, preds=preds_with(0))            # router-on runs are never read once the router is off
    write_run(tmp_path, "holdout", RT, run=rj("holdout", RT, partial=True), preds=preds_with(0))  # refused: R-DM not computable
    write_run(tmp_path, "holdout", RTDM, preds=preds_with(1))
    sec = holdout(tmp_path)["holdout"]
    assert sec["router"] == "off" and not sec["rules"]["R-route"]["computable"]
    assert sec["rules"]["R-DM"] == {**sec["rules"]["R-DM"], "pass": False, "computable": False}
    assert sec["config"] == ["decision_maker", "router"]
    assert sec["rules"]["R-hold"]["pass"] and sec["rules"]["R-hold"]["run"] == "train_holdout_ablate-decision_maker+router"
    assert sec["rules"]["R-route (final configuration)"]["pass"]


def test_r53_router_off_r_dm_compares_the_router_ablated_pair(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), run=rj("holdout", route_recall={"covered": 158, "total": 163, "missed": []}),
              preds=preds_with(9))
    write_run(tmp_path, "holdout", DM, preds=preds_with(9))
    write_run(tmp_path, "holdout", RT, preds=preds_with(1))            # the DM helps in the router-off pair
    write_run(tmp_path, "holdout", RTDM, preds=preds_with(3))
    sec = holdout(tmp_path)["holdout"]
    assert sec["router"] == "off" and sec["rules"]["R-route"]["computable"] and not sec["rules"]["R-route"]["pass"]
    assert sec["rules"]["R-DM"]["pass"] and sec["rules"]["R-DM"]["runs"] == ["train_holdout_ablate-router",
                                                                              "train_holdout_ablate-decision_maker+router"]
    assert sec["config"] == ["router"] and sec["rules"]["R-hold"]["pass"]
    assert sec["rules"]["R-hold"]["run"] == "train_holdout_ablate-router"


def test_r53_router_off_dm_fails_reads_the_doubly_ablated_run(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), run=rj("holdout", route_recall={"covered": 4, "total": 5, "missed": []}), preds=preds_with(0))
    write_run(tmp_path, "holdout", RT, preds=preds_with(3))
    write_run(tmp_path, "holdout", RTDM, preds=preds_with(1))
    sec = holdout(tmp_path)["holdout"]
    assert not sec["rules"]["R-DM"]["pass"] and sec["rules"]["R-DM"]["computable"]
    assert sec["config"] == ["decision_maker", "router"] and sec["rules"]["R-hold"]["pass"]


def test_r53_router_off_but_its_runs_missing_gives_no_candidate(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), run=rj("holdout", route_recall={"covered": 4, "total": 5, "missed": []}), preds=preds_with(0))
    write_run(tmp_path, "holdout", DM, preds=preds_with(0))
    gj = holdout(tmp_path)
    assert gj["holdout"]["config"] == ["decision_maker", "router"] and not gj["holdout"]["rules"]["R-hold"]["computable"]
    assert gj["candidate"]["name"] == "none"


def test_r53_a_router_run_with_the_wrong_ablation_set_is_refused(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), run=rj("holdout", route_recall=None), preds=preds_with(0))
    d = write_run(tmp_path, "holdout", RT, run=rj("holdout", ablate=["router", "grader"]), preds=preds_with(0))
    assert d.name == "train_holdout_ablate-router"
    write_run(tmp_path, "holdout", RTDM, preds=preds_with(0))
    sec = holdout(tmp_path)["holdout"]
    assert not sec["rules"]["R-DM"]["computable"] and "ablation set" in sec["rules"]["R-DM"]["reason"]


def test_r46_both_runs_missing_gives_no_candidate(tmp_path, frozen):
    gj = holdout(tmp_path)
    sec = gj["holdout"]
    assert not sec["rules"]["R-DM"]["computable"] and not sec["rules"]["R-hold"]["computable"]
    assert not sec["rules"]["R-hold"]["pass"] and gj["candidate"]["name"] == "none"


def test_r46_full_run_passes_hold_but_dm_is_not_computable_still_reads_the_ablated_run(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), preds=preds_with(0))             # would pass R-hold, but the ablated run is missing
    sec = holdout(tmp_path)["holdout"]
    assert sec["config"] == ["decision_maker"]
    assert not sec["rules"]["R-hold"]["pass"] and not sec["rules"]["R-hold"]["computable"]


def test_holdout_excludes_the_exposed_rows(tmp_path, frozen):
    # full run 2 wrong + the exposed row wrong; s0 2 wrong: equal once the exposed row is excluded
    write_run(tmp_path, "holdout", (), preds=preds_with(2, exposed_wrong=True))
    write_run(tmp_path, "holdout", DM, preds=preds_with(2, exposed_wrong=True))
    sec = holdout(tmp_path, s0=preds_with(2))["holdout"]
    assert sec["rules"]["R-hold"]["pass"] and sec["rules"]["R-hold"]["b"] == [2, 0]


def test_holdout_false_confidence_fails_r_hold(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), preds=preds_with(1, fc=True))
    write_run(tmp_path, "holdout", DM, preds=preds_with(1, fc=True))
    sec = holdout(tmp_path)["holdout"]
    # wrong rows tie with s0 (2 = 2: the false-confident row is also wrong), false-confident 1 > 0
    assert not sec["rules"]["R-hold"]["pass"] and sec["rules"]["R-hold"]["b"] == [2, 1] and sec["rules"]["R-hold"]["s0"] == [2, 0]


def test_holdout_unfrozen_prompts_make_every_rule_not_computable(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "FROZEN_PROMPT_VERSIONS", None)
    write_run(tmp_path, "holdout", (), preds=preds_with(0))
    write_run(tmp_path, "holdout", DM, preds=preds_with(0))
    sec = holdout(tmp_path)["holdout"]
    assert not any(r["computable"] for r in sec["rules"].values())


def test_holdout_s0_refusal_is_not_computable(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), preds=preds_with(0))
    write_run(tmp_path, "holdout", DM, preds=preds_with(0))

    def s0():
        raise GateRefused("s0 baseline: 1 holdout case not in the extraction cache")
    sec = gates.gate("holdout", runs_root=tmp_path, gates_json=tmp_path / "g.json", cases_expected=4, gold=lambda: dict(GOLD),
                     s0=s0, say=lambda *_: None)["holdout"]
    assert not sec["rules"]["R-hold"]["computable"] and not sec["rules"]["R-hold"]["pass"]


def test_holdout_r_route_reads_only_the_full_run(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), preds=preds_with(3))
    write_run(tmp_path, "holdout", DM, run=rj("holdout", DM, route_recall={"covered": 4, "total": 5, "missed": []}),
              preds=preds_with(1))                                  # DM fails: the ablated run is chosen, its recall is not read
    sec = holdout(tmp_path)["holdout"]
    assert sec["rules"]["R-route"]["pass"] and sec["rules"]["R-route"]["run"] == "train_holdout" and sec["router"] == "on"
    assert sec["config"] == ["decision_maker"] and "R-route (final configuration)" not in sec["rules"]


def test_holdout_without_route_recall_is_not_computable(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), run=rj("holdout", route_recall=None), preds=preds_with(1))
    write_run(tmp_path, "holdout", DM, run=rj("holdout", DM, route_recall=None), preds=preds_with(3))
    sec = holdout(tmp_path)["holdout"]
    assert not sec["rules"]["R-route"]["pass"] and not sec["rules"]["R-route"]["computable"] and sec["router"] == "off"


def test_dev_gate_reads_the_limit_20_run(tmp_path):
    write_run(tmp_path, "dev")
    gj = gates.gate("dev", runs_root=tmp_path, gates_json=tmp_path / "g.json", say=lambda *_: None)
    assert gj["dev"]["rules"]["R-smoke"]["pass"] and gj["dev"]["rules"]["R-smoke"]["run"] == "train_dev_limit-20"
    write_run(tmp_path, "dev", run=rj("dev", failures=1))
    gj = gates.gate("dev", runs_root=tmp_path, gates_json=tmp_path / "g.json", say=lambda *_: None)
    assert not gj["dev"]["rules"]["R-smoke"]["pass"] and gj["dev"]["rules"]["R-smoke"]["computable"]


def test_dev_gate_never_reads_gold(tmp_path):
    write_run(tmp_path, "dev")

    def boom():
        raise AssertionError("dev read gold")
    gates.gate("dev", runs_root=tmp_path, gates_json=tmp_path / "g.json", gold=boom, s0=boom, say=lambda *_: None)


def test_a_missing_dev_run_is_not_computable(tmp_path):
    r = gates.gate("dev", runs_root=tmp_path, gates_json=tmp_path / "g.json", say=lambda *_: None)["dev"]["rules"]["R-smoke"]
    assert not r["pass"] and not r["computable"]


# ---------------------------------------------------------------- test target
TEST_SUB = {f"t{i}__R": "PASS" for i in range(40)}


def test_gate_reads_the_run_in_the_r_dm_configuration(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    route = {"R-route": {"pass": True, "computable": True}}
    gj.write_text(json.dumps({"holdout": {"rules": {**route, "R-DM": {"pass": True, "computable": True}}, "config": [], **FRESH}}))
    write_run(tmp_path, "test", (), run=rj("test", n=40), submission=TEST_SUB)
    write_run(tmp_path, "test", DM, run=rj("test", DM, n=40), submission={k: "FAIL" for k in TEST_SUB})
    sub2 = write_csv(tmp_path / "sub2.csv", TEST_SUB)
    out = gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, say=lambda *_: None)
    r = out["test"]["rules"]["R-test"]
    assert r["pass"] and r["changed"] == 0 and r["run"] == "test_all"
    gj.write_text(json.dumps({"holdout": {"rules": {**route, "R-DM": {"pass": False, "computable": True}}, "config": ["decision_maker"],
                                          **FRESH}}))
    out = gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, say=lambda *_: None)
    r = out["test"]["rules"]["R-test"]
    assert not r["pass"] and r["changed"] == 40 and r["run"] == "test_all_ablate-decision_maker"


def test_without_a_holdout_entry_the_test_gate_uses_the_router_and_dm_ablated_run(tmp_path, frozen):
    write_run(tmp_path, "test", RTDM, run=rj("test", RTDM, n=40), submission=TEST_SUB)
    out = gates.gate("test", runs_root=tmp_path, gates_json=tmp_path / "g.json", cases_expected=40,
                     sub2=write_csv(tmp_path / "sub2.csv", TEST_SUB), say=lambda *_: None)
    assert out["test"]["config"] == ["decision_maker", "router"] and out["test"]["rules"]["R-test"]["pass"]
    assert out["test"]["rules"]["R-test"]["run"] == "test_all_ablate-decision_maker+router"


def test_written_true_without_the_file_is_not_computable(tmp_path, frozen):
    run = rj("test", RTDM, n=40, submission={"written": True, "reason": "ok", "problems": []})
    d = write_run(tmp_path, "test", RTDM, run=run)                  # run.json says written: true; no submission.csv
    assert json.loads((d / "run.json").read_text())["submission"]["written"] is True
    out = gates.gate("test", runs_root=tmp_path, gates_json=tmp_path / "g.json", cases_expected=40,
                     sub2=write_csv(tmp_path / "sub2.csv", TEST_SUB), say=lambda *_: None)
    r = out["test"]["rules"]["R-test"]
    assert not r["pass"] and not r["computable"]


def test_a_missing_submission_2_is_not_computable(tmp_path, frozen):
    write_run(tmp_path, "test", RTDM, run=rj("test", RTDM, n=40), submission=TEST_SUB)
    out = gates.gate("test", runs_root=tmp_path, gates_json=tmp_path / "g.json", cases_expected=40,
                     sub2=tmp_path / "absent.csv", say=lambda *_: None)
    assert not out["test"]["rules"]["R-test"]["computable"]


def _full_green(tmp_path, gj: Path, *, dm_pass=True, test_recall=None, router_on=True):
    ab = (frozenset() if router_on else RT) | (frozenset() if dm_pass else DM)
    gj.write_text(json.dumps({
        "dev": {"rules": {"R-smoke": {"pass": True, "computable": True}}, **FRESH},
        "holdout": {"rules": {"R-DM": {"pass": dm_pass, "computable": True}, "R-hold": {"pass": True, "computable": True},
                              "R-route": {"pass": router_on, "computable": True}}, "config": sorted(ab), **FRESH}}))
    over = {} if test_recall is None else {"route_recall": test_recall}
    d = write_run(tmp_path, "test", ab, run=rj("test", ab, n=40, **over), preds={"t0__R": "PASS"}, submission=TEST_SUB)
    return d, write_csv(tmp_path / "sub2.csv", TEST_SUB)


def test_candidate_combines_all_targets(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    _, sub2 = _full_green(tmp_path, gj)
    out = gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, say=lambda *_: None)
    assert out["candidate"]["name"] == "B" and out["candidate"]["ablate"] == []
    assert out["candidate"]["replay_identical"] is None and out["candidate"]["submittable"] is False


def test_candidate_with_the_dm_switched_off(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    _, sub2 = _full_green(tmp_path, gj, dm_pass=False)
    out = gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, say=lambda *_: None)
    assert out["candidate"]["name"] == "B" and out["candidate"]["ablate"] == ["decision_maker"]


def test_r_route_needs_holdout_and_test(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    _, sub2 = _full_green(tmp_path, gj, test_recall={"covered": 4, "total": 5, "missed": ["p:x"]})
    out = gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, say=lambda *_: None)
    assert out["rules"]["R-route"] is False and out["candidate"]["name"] == "none"


@pytest.mark.parametrize("router_on,dm_pass,ablate", [(True, True, []), (True, False, ["decision_maker"]), (False, True, ["router"]),
                                                     (False, False, ["decision_maker", "router"])])
def test_the_candidate_configuration_end_to_end(tmp_path, frozen, router_on, dm_pass, ablate):
    gj = tmp_path / "gates.json"
    # with the router off a test-run R-route miss does not matter (true by construction)
    _, sub2 = _full_green(tmp_path, gj, dm_pass=dm_pass, router_on=router_on,
                          test_recall=None if router_on else {"covered": 4, "total": 5, "missed": []})
    out = gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, say=lambda *_: None)
    assert out["test"]["config"] == ablate and out["test"]["rules"]["R-test"]["pass"]
    assert out["candidate"]["name"] == "B" and out["candidate"]["ablate"] == ablate
    assert out["candidate"]["router"] == ("on" if router_on else "off") and out["rules"]["R-route"] is True


def test_a_holdout_router_flip_makes_the_test_entry_stale(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    _, sub2 = _full_green(tmp_path, gj)
    gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, say=lambda *_: None)
    data = json.loads(gj.read_text())
    data["holdout"]["rules"]["R-route"]["pass"] = False
    gj.write_text(json.dumps(data))
    out = gates.gate("dev", runs_root=tmp_path / "nodev", gates_json=gj, say=lambda *_: None)
    assert out["rules"]["R-route"] is True and out["rules"]["R-test"] is False and out["candidate"]["name"] == "none"


def test_a_stale_test_entry_does_not_count(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    _, sub2 = _full_green(tmp_path, gj)
    gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, say=lambda *_: None)
    data = json.loads(gj.read_text())
    data["holdout"]["rules"]["R-DM"]["pass"] = False                  # holdout re-gated later: the test entry no longer matches
    data["holdout"]["config"] = ["decision_maker"]
    gj.write_text(json.dumps(data))
    out = gates.gate("dev", runs_root=tmp_path / "nodev", gates_json=gj, say=lambda *_: None)
    assert out["rules"]["R-test"] is False and out["candidate"]["name"] == "none"
    assert out["rules"]["R-route"] is False                       # router on: the stale test half of R-route does not count


def test_gates_json_merges_targets(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    write_run(tmp_path, "dev")
    gates.gate("dev", runs_root=tmp_path, gates_json=gj, say=lambda *_: None)
    holdout(tmp_path, gates_json=gj)
    data = json.loads(gj.read_text())
    assert set(data) >= {"dev", "holdout", "rules", "candidate"}


def test_replay_check(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    cand, sub2 = _full_green(tmp_path, gj)
    rep = tmp_path / "replay"
    shutil.copytree(cand, rep)
    rjr = json.loads((rep / "run.json").read_text())
    (rep / "run.json").write_text(json.dumps({**rjr, "cache_only": True, "not_cached": 0}))
    out = gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, replay=rep, say=lambda *_: None)
    assert out["candidate"]["replay_identical"] is True and out["candidate"]["submittable"] is True
    write_csv(rep / "preds.csv", {"t0__R": "FAIL"})
    out = gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, replay=rep, say=lambda *_: None)
    assert out["candidate"]["replay_identical"] is False and out["candidate"]["submittable"] is False
    write_csv(rep / "preds.csv", {"t0__R": "PASS"})
    (rep / "run.json").write_text(json.dumps({**rjr, "cache_only": True, "not_cached": 0, "partial": True}))
    out = gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, replay=rep, say=lambda *_: None)
    assert out["candidate"]["replay_identical"] is False


def test_the_replay_may_not_be_the_candidate_dir(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    cand, sub2 = _full_green(tmp_path, gj)
    run = json.loads((cand / "run.json").read_text())
    (cand / "run.json").write_text(json.dumps({**run, "cache_only": True, "not_cached": 0}))     # would otherwise qualify as a replay
    out = gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, replay=cand, say=lambda *_: None)
    assert out["candidate"]["replay_identical"] is False and out["candidate"]["submittable"] is False


# ---------------------------------------------------------------- output rule
def test_no_label_or_row_id_reaches_stdout_or_gates_json(tmp_path, frozen, capsys):
    gj = tmp_path / "gates.json"
    write_run(tmp_path, "holdout", (), preds=preds_with(1))
    write_run(tmp_path, "holdout", DM, preds=preds_with(3))
    gates.gate("holdout", runs_root=tmp_path, gates_json=gj, cases_expected=4, gold=lambda: dict(GOLD), s0=lambda: preds_with(2))
    gates.gate("dev", runs_root=tmp_path, gates_json=gj)
    text = capsys.readouterr().out + gj.read_text()
    assert "R-DM" in text and "R-hold" in text
    assert CANARY not in text and EXPOSED not in text
    for word in LABEL_WORDS:
        assert word not in text


def test_cli_dispatch_and_refusals(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_gate(target, **kw):
        seen.update(target=target, **kw)
        return {}
    monkeypatch.setattr(gates, "gate", fake_gate)
    assert gates.main(["--target", "dev", "--runs-root", str(tmp_path), "--gates-json", str(tmp_path / "g.json")]) == 0
    assert seen["target"] == "dev" and seen["runs_root"] == tmp_path
    assert gates.main(["--target", "dev", "--runs-root", str(tmp_path), "--replay", str(tmp_path)]) == 64      # --replay: test only
    with pytest.raises(SystemExit):
        gates.main(["--target", "nope"])


def test_main_module_routes_gates(monkeypatch):
    called = {}
    monkeypatch.setattr(gates, "main", lambda argv: (called.update(argv=argv), 0)[1])
    monkeypatch.setattr(runner, "main", lambda argv=None: (called.update(run=argv), 5)[1])
    from vendor_verdict import __main__ as entry            # importing it runs nothing: the entry point is guarded
    assert entry.dispatch(["gates", "--target", "dev"]) == 0 and called["argv"] == ["--target", "dev"]
    assert entry.dispatch(["run", "--split", "test"]) == 5 and called["run"] == ["run", "--split", "test"]


def test_s0_baseline_refuses_on_the_mock_provider(tmp_path):
    with pytest.raises(GateRefused):
        gates.s0_holdout_preds(tmp_path)


# ---------------------------------------------------------------- fix round 1: fresh inputs (R55a), pinned s0 model (R55b), history
def test_a_dev_entry_on_other_prompt_versions_is_not_fresh(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    write_run(tmp_path, "dev", run=rj("dev", prompt_versions={**PV, "router": "ffffffffffff"}))
    out = gates.gate("dev", runs_root=tmp_path, gates_json=gj, say=lambda *_: None)
    assert out["dev"]["rules"]["R-smoke"]["pass"]                    # the smoke itself passed ...
    assert out["fresh"]["dev"] is False and out["rules"]["R-smoke"] is False      # ... but not on the frozen prompts
    write_run(tmp_path, "dev")
    out = gates.gate("dev", runs_root=tmp_path, gates_json=gj, say=lambda *_: None)
    assert out["fresh"]["dev"] is True and out["rules"]["R-smoke"] is True


def test_with_frozen_none_no_section_is_fresh(tmp_path, monkeypatch):
    gj = tmp_path / "gates.json"
    monkeypatch.setattr(gates, "FROZEN_PROMPT_VERSIONS", dict(PV))
    _, sub2 = _full_green(tmp_path, gj)
    assert gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2,
                      say=lambda *_: None)["candidate"]["name"] == "B"
    monkeypatch.setattr(gates, "FROZEN_PROMPT_VERSIONS", None)
    out = gates.gate("dev", runs_root=tmp_path / "nodev", gates_json=gj, say=lambda *_: None)
    assert out["fresh"] == {"dev": False, "holdout": False, "test": False} and out["candidate"]["name"] == "none"


def test_sections_record_input_hashes(tmp_path, frozen):
    import hashlib
    write_run(tmp_path, "holdout", (), preds=preds_with(1))
    d = write_run(tmp_path, "holdout", DM, preds=preds_with(3))
    sec = holdout(tmp_path)["holdout"]
    rec = {i["dir"]: i for i in sec["inputs"]}
    assert set(rec) == {"train_holdout", "train_holdout_ablate-decision_maker"}
    assert rec[d.name]["run.json"] == hashlib.sha256((d / "run.json").read_bytes()).hexdigest()
    assert rec[d.name]["preds.csv"] == hashlib.sha256((d / "preds.csv").read_bytes()).hexdigest()


def test_a_holdout_dir_rewritten_after_gating_is_not_fresh(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    write_run(tmp_path, "holdout", (), preds=preds_with(1))
    d = write_run(tmp_path, "holdout", DM, preds=preds_with(3))
    out = holdout(tmp_path, gates_json=gj)
    assert out["fresh"]["holdout"] is True and out["rules"]["R-hold"] is True
    write_csv(d / "preds.csv", preds_with(2))                          # re-run after gating: a re-gate is needed
    out = gates.gate("dev", runs_root=tmp_path / "nodev", gates_json=gj, say=lambda *_: None)
    assert out["fresh"]["holdout"] is False and out["rules"]["R-hold"] is False and out["rules"]["R-DM"] is False


def test_a_refused_input_that_appears_later_makes_the_section_stale(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    write_run(tmp_path, "holdout", (), preds=preds_with(1))
    holdout(tmp_path, gates_json=gj)                                  # the DM-ablated run was missing
    write_run(tmp_path, "holdout", DM, preds=preds_with(3))
    out = gates.gate("dev", runs_root=tmp_path / "nodev", gates_json=gj, say=lambda *_: None)
    assert out["fresh"]["holdout"] is False


def test_a_test_dir_rewritten_after_gating_is_not_fresh(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    cand, sub2 = _full_green(tmp_path, gj)
    gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=sub2, say=lambda *_: None)
    write_csv(cand / "submission.csv", {k: "FAIL" for k in TEST_SUB})
    out = gates.gate("dev", runs_root=tmp_path / "nodev", gates_json=gj, say=lambda *_: None)
    assert out["fresh"]["test"] is False and out["candidate"]["name"] == "none"


def test_the_test_gate_ignores_a_stale_holdout_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "FROZEN_PROMPT_VERSIONS", dict(PV))
    gj = tmp_path / "gates.json"
    _full_green(tmp_path, gj)
    data = json.loads(gj.read_text())
    data["holdout"]["prompt_versions"] = {**PV, "router": "ffffffffffff"}
    gj.write_text(json.dumps(data))
    out = gates.gate("test", runs_root=tmp_path, gates_json=gj, cases_expected=40, sub2=tmp_path / "x.csv", say=lambda *_: None)
    assert out["test"]["config"] == ["decision_maker", "router"]


def test_r_hold_records_the_pinned_s0_model(tmp_path, frozen):
    write_run(tmp_path, "holdout", (), preds=preds_with(1))
    write_run(tmp_path, "holdout", DM, preds=preds_with(3))
    r = holdout(tmp_path)["holdout"]["rules"]["R-hold"]
    assert r["s0_model"] == gates.BASELINE_MODEL == "MiniMax-M3.1-Flash"


def test_an_unreadable_gates_json_is_refused_and_left_alone(tmp_path, capsys):
    gj = tmp_path / "gates.json"
    gj.write_text("{not json")
    write_run(tmp_path, "dev")
    with pytest.raises(GateRefused, match="JSONDecodeError"):
        gates.gate("dev", runs_root=tmp_path, gates_json=gj, say=lambda *_: None)
    assert gj.read_text() == "{not json"
    assert gates.main(["--target", "dev", "--runs-root", str(tmp_path), "--gates-json", str(gj)]) != 0
    assert "refused" in capsys.readouterr().out and gj.read_text() == "{not json"


def test_an_unknown_target_is_refused(tmp_path):
    with pytest.raises(GateRefused):
        gates.gate("nope", runs_root=tmp_path, gates_json=tmp_path / "g.json", say=lambda *_: None)
    assert not (tmp_path / "g.json").exists()


def test_every_gate_run_appends_one_history_record(tmp_path, frozen):
    gj = tmp_path / "gates.json"
    write_run(tmp_path, "dev")
    gates.gate("dev", runs_root=tmp_path, gates_json=gj, say=lambda *_: None)
    first = json.loads(gj.read_text())["history"]
    write_run(tmp_path, "dev", run=rj("dev", failures=2))
    gates.gate("dev", runs_root=tmp_path, gates_json=gj, say=lambda *_: None)
    hist = json.loads(gj.read_text())["history"]
    assert len(first) == 1 and len(hist) == 2 and hist[0] == first[0]
    assert hist[0]["target"] == "dev" and hist[0]["rules"]["R-smoke"] == {"pass": True, "computable": True}
    assert hist[1]["rules"]["R-smoke"] == {"pass": False, "computable": True}
    assert hist[1]["inputs"] == ["train_dev_limit-20"] and hist[1]["utc"].endswith("Z")


def test_history_that_is_not_a_list_is_refused(tmp_path):
    gj = tmp_path / "gates.json"
    gj.write_text(json.dumps({"history": {"x": 1}}))
    with pytest.raises(GateRefused):
        gates.gate("dev", runs_root=tmp_path, gates_json=gj, say=lambda *_: None)


class _StubExtractorLLM:
    mock_mode = False

    def __init__(self, model):
        self.model = model


TINY_SYSTEM = "Invented extraction prompt for the gates test."
TINY_RL = """
def evaluate(facts, requirements, tier, users, as_of):
    return {}


def overall(verdicts, requirements):
    return "CONDITIONAL"
"""


def tiny_data(tmp_path: Path, *, cache_all=True) -> Path:
    from eval import kaggle_policy as kp
    from tests.test_vv_runner import write_data
    d = write_data(tmp_path / "data", split="train")                 # 3 invented train cases; holdout = the odd position, c0002
    (d / "requirements.json").write_text("[]", encoding="utf-8")
    (d / "rules_lite.py").write_text(TINY_RL, encoding="utf-8")
    with (d / "train_labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "verdict"])
        w.writerows([["c0001__OVERALL", "CONDITIONAL"], ["c0002__OVERALL", "BLOCKED"], ["c0002__R-x", "PASS"]])
    cache = d / "cache" / "extract"
    cache.mkdir(parents=True)
    pages = [json.loads(line) for line in (d / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
    for i, pg in enumerate(p for p in pages if p["case_id"] == "c0002"):
        if cache_all or i:
            key = kp.cache_key(gates.BASELINE_MODEL, pg, "", system=TINY_SYSTEM)
            (cache / f"{key}.json").write_text(json.dumps({"claims": [], "in": 1, "out": 1}), encoding="utf-8")
    return d


def test_real_input_loaders_on_an_invented_data_dir(tmp_path):
    d = tiny_data(tmp_path)
    assert gates.subset_size(d, "train", "holdout") == 1 and gates.subset_size(d, "train", "dev") == 2
    assert gates.subset_size(tmp_path / "absent", "train", "holdout") is None
    gold = gates.load_gold(d)
    assert set(gold) == {"c0002__OVERALL", "c0002__R-x"}             # holdout cases only
    preds = gates.s0_holdout_preds(d, llm=_StubExtractorLLM(gates.BASELINE_MODEL), system=TINY_SYSTEM)
    assert set(preds) == {"c0002__OVERALL"}


def test_s0_refuses_another_extractor_model(tmp_path):
    d = tiny_data(tmp_path)
    with pytest.raises(GateRefused, match="MiniMax-M3.1-Flash"):
        gates.s0_holdout_preds(d, llm=_StubExtractorLLM("Some-Other-Model"), system=TINY_SYSTEM)


def test_s0_refuses_a_page_missing_from_the_cache(tmp_path):
    d = tiny_data(tmp_path, cache_all=False)
    with pytest.raises(GateRefused, match="1 holdout cases"):
        gates.s0_holdout_preds(d, llm=_StubExtractorLLM(gates.BASELINE_MODEL), system=TINY_SYSTEM)
