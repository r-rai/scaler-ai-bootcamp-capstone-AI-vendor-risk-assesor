"""The live-accuracy harness scores misses; its classifier and gate maths must be right BEFORE we pay for a live run."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("run_live_real", Path(__file__).resolve().parent.parent / "eval" / "run_live_real.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


@pytest.mark.parametrize("expected,got,deadline,want", [
    ("PASS", "PASS", False, "correct"),
    ("NEEDS_REVIEW", "NEEDS_REVIEW", False, "correct"),
    ("PASS", "UNKNOWN", False, "unknown_not_found"),
    ("PASS", "UNKNOWN", True, "unknown_deadline"),
    ("PASS", "NEEDS_REVIEW", False, "needs_review"),
    ("PASS", "FAIL", False, "wrong_value"),
    ("FAIL", "PASS", False, "wrong_value"),
    ("NEEDS_REVIEW", "PASS", False, "guess"),
    ("UNKNOWN", "FAIL", False, "guess"),
    ("FAIL", "NOT_APPLICABLE", False, "other"),
    ("PASS", None, False, "other"),
])
def test_classify(expected, got, deadline, want):
    assert h.classify(expected, got, deadline_hit=deadline) == want


def _row(cat, web=True):
    return {"category": cat, "web_driven": web}


def test_gate_excludes_internal_avl_and_reports_deadline_separately():
    rows = [{"id": "a", "error": None, "wall_s": 10, "usd": 0.1, "deadline_hit": True, "search_calls": {"tavily": 3},
             "got_overall": "BLOCKED", "expected_overall": "BLOCKED",
             "requirements": {"R1": _row("correct"), "R2": _row("unknown_deadline"), "R3": _row("wrong_value"),
                              "PROC-1.1": _row("wrong_value", web=False)}}]
    s = h.summarise(rows)
    assert (s["web_driven"]["n"], s["web_driven"]["correct"]) == (3, 1)          # PROC-1.1 not in the gate number
    assert s["web_driven_excluding_deadline_unknowns"]["n"] == 2                  # deadline misses reported separately
    assert s["proc_1_1_internal"]["n"] == 1
    assert s["deadline_runs"] == ["a"] and s["guesses"] == 0


def test_requirements_map_has_the_internal_field():
    f = h.requirement_fields()
    assert f["PROC-1.1"] in h.NON_WEB_FIELDS and f["DP-3.1"] == "data_regions"
