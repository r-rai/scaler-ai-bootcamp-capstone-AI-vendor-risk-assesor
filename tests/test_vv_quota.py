"""Quota-window resilience (vendor_verdict.quota and its wiring into the planner, the specialists and the runner): detection, a clean
stop, an optional wait, an exact resume, exit statuses, and the carried-over runner minors (3b, 3c, 3d).

Invented data only (every data dir is in `tmp_path`), stub models, an injected clock and sleep: no real time sleep, no real LLM, no
call to a local server, nothing under `data_kaggle/` is read."""
import json
import re
import sys
import threading
import time
from pathlib import Path

import pytest

from assessor.llm import LLMError
from assessor.tracing import BudgetExceeded

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run_kaggle  # noqa: E402,F401

from tests.test_vv_runner import (CASE_IDS, PlannerLLM, ReaderLLM, StubMetric, make_agents, preds, run_json,  # noqa: E402
                                  write_data)
from vendor_verdict import nodes, quota, runner  # noqa: E402
from vendor_verdict.agents import planner as planner_mod  # noqa: E402
from vendor_verdict.agents import specialists as specialists_mod  # noqa: E402
from vendor_verdict.agents.planner import plan_routes  # noqa: E402
from vendor_verdict.agents.specialists import read_case, read_page  # noqa: E402
from vendor_verdict.cache import CallCache  # noqa: E402
from vendor_verdict.graph import new_state  # noqa: E402
from vendor_verdict.quota import QuotaExhausted, QuotaGate, is_quota_error  # noqa: E402
from vendor_verdict.schemas import SpecialistRaw  # noqa: E402

PROVIDER_CANARY = "CANARY-PROVIDER-MESSAGE"
PAGE_CANARY = "CANARY-PAGE-TEXT-Q"
PAGE = {"page_id": "p1", "url": "https://acme.example/trust", "title": "Trust", "published_date": "2026-01-01",
        "last_updated": "2026-02-01", "text": "Acme invented trust page. " * 5}
CASE = {"case_id": "c9999", "vendor": "Acme", "legal_entity": "Acme Inc.", "official_domain": "acme.example", "as_of": "2026-11-01",
        "tier": "T2", "users": "50", "question": "Can we use Acme?"}
REC = {"page": {**PAGE, "source_type": "official", "publisher": "Acme"}, "role": "official", "why": "", "quarantine": [], "extract": None}


@pytest.fixture(autouse=True)
def no_ladder_sleep(monkeypatch):
    """The agents' retry ladder sleeps (4, 12, 36, 60 s) are recorded, never slept."""
    slept = []
    monkeypatch.setattr(specialists_mod, "_sleep", slept.append)
    monkeypatch.setattr(planner_mod, "_sleep", slept.append)
    return slept


class StatusError(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


def llm_error(msg: str, status: int | None = None) -> LLMError:
    """An LLMError raised `from` an SDK-like error, as assessor/llm.py raises it."""
    try:
        try:
            raise StatusError(msg, status)
        except StatusError as cause:
            raise LLMError(f"LLM call failed at stage 'x': {msg}") from cause
    except LLMError as exc:
        return exc


class Scripted:
    """Raises the given exceptions in order, then answers with `out` (a model object) forever."""
    model = "stub-model"

    def __init__(self, errors, out):
        self.errors, self.out, self.calls = list(errors), out, 0

    def structured(self, system, user, schema, stage="", **kw):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.out


def persistent(exc):
    return Scripted([exc] * 100, None)


# ---------------------------------------------------------------- detection
def test_rate_limit_kind_by_message_and_by_status():
    assert is_quota_error(LLMError("HTTP 429 too many requests")) == "rate_limit"
    assert is_quota_error(LLMError("Rate Limit reached for requests")) == "rate_limit"
    assert is_quota_error(llm_error("too many requests", status=429)) == "rate_limit"


def test_keyword_kind_is_one_unverified_constant():
    assert quota.QUOTA_KEYWORDS == ("quota", "usage limit", "limit exceeded", "insufficient balance")
    assert "UNVERIFIED" in quota.__doc__
    for word in quota.QUOTA_KEYWORDS:
        assert is_quota_error(LLMError(f"provider says: {word.upper()} for this key")) == "keyword"


def test_not_a_quota_error():
    assert is_quota_error(LLMError("provider error 500")) is None
    assert is_quota_error(llm_error("bad gateway", status=502)) is None
    assert is_quota_error(runner.NotCached("not in the cache (cache-only run) quota")) is None
    # the repair-exhaustion error quotes model (page-derived) output: never a quota error, its text never reaches run.json
    assert is_quota_error(LLMError("Could not get valid X after 2 repairs: page says 429 quota")) is None


def test_quota_exhausted_carries_status_kind_and_a_clipped_detail_but_prints_none_of_it():
    exc = quota.exhausted(llm_error(PROVIDER_CANARY + " x" * 300, status=429), "rate_limit")
    assert exc.status == 429 and exc.kind == "rate_limit" and len(exc.detail) == 200 and PROVIDER_CANARY in exc.detail
    assert PROVIDER_CANARY not in str(exc) and PROVIDER_CANARY not in repr(exc)


def test_quota_exhausted_cannot_be_swallowed_by_except_exception(tmp_path):
    assert issubclass(QuotaExhausted, BaseException) and not issubclass(QuotaExhausted, Exception)
    swallowed = []
    with pytest.raises(QuotaExhausted):
        try:
            read_page(persistent(LLMError("HTTP 429")), CallCache(tmp_path), "ai", PAGE)
        except Exception as exc:  # noqa: BLE001  (what any agent fallback would do)
            swallowed.append(exc)
    assert swallowed == []


# ---------------------------------------------------------------- the agents
def test_a_429_that_recovers_within_the_ladder_is_absorbed(tmp_path, no_ladder_sleep):
    out = SpecialistRaw(readings=[{"field": "trains_on_customer_data", "value": "no", "quote": "Acme invented trust page."}])
    llm = Scripted([LLMError("HTTP 429")] * 3, out)
    r, notes = read_page(llm, CallCache(tmp_path / "a"), "ai", PAGE)
    assert llm.calls == 4 and no_ladder_sleep == [4, 12, 36] and len(r) == 1 and notes == []
    from vendor_verdict.schemas import PageRoute, PlanOut
    llm = Scripted([LLMError("HTTP 429")] * 4, PlanOut(routes=[PageRoute(page_id="p1", families=["ai"])]))
    assert plan_routes(llm, CallCache(tmp_path / "b"), CASE, [REC]) == ({"ai": ["p1"]}, [])
    assert llm.calls == 5


@pytest.mark.parametrize("err,kind,status", [(LLMError("HTTP 429 " + PROVIDER_CANARY), "rate_limit", None),
                                             (llm_error("slow down", status=429), "rate_limit", 429),
                                             (LLMError("monthly Quota exhausted " + PROVIDER_CANARY), "keyword", None)])
def test_a_persistent_quota_error_raises_from_every_paid_agent(tmp_path, err, kind, status, no_ladder_sleep):
    for call in (lambda llm: read_page(llm, CallCache(tmp_path / "p"), "ai", PAGE),
                 lambda llm: read_case(llm, CallCache(tmp_path / "c"), "ai", [PAGE], ["trains_on_customer_data"]),
                 lambda llm: plan_routes(llm, CallCache(tmp_path / "l"), CASE, [REC])):
        llm = persistent(err)
        no_ladder_sleep.clear()
        with pytest.raises(QuotaExhausted) as got:
            call(llm)
        assert got.value.kind == kind and got.value.status == status and llm.calls == 5 and no_ladder_sleep == [4, 12, 36, 60]
        assert not (tmp_path / "p").exists() or not any((tmp_path / "p").iterdir())       # nothing cached


def test_a_non_quota_error_keeps_todays_failed_page_and_planner_failed(tmp_path):
    llm = persistent(LLMError("provider error 500"))
    assert read_page(llm, CallCache(tmp_path), "ai", PAGE) == ([], ["failed: LLMError"]) and llm.calls == 1
    assert read_case(persistent(LLMError("provider error 500")), CallCache(tmp_path), "ai", [PAGE],
                     ["trains_on_customer_data"]) == ([], ["failed: LLMError"])
    assert plan_routes(persistent(LLMError("provider error 500")), CallCache(tmp_path), CASE, [REC]) == ({}, ["planner failed"])


@pytest.mark.parametrize("node", ["plan", "specialists"])
def test_the_nodes_let_quota_through(tmp_path, node):
    ag = make_agents(tmp_path / "cache", planner=persistent(LLMError("HTTP 429")), reader=persistent(LLMError("HTTP 429")))
    data = write_data(tmp_path / "data")
    cases, pages, _ = runner.load_split(data, "test", "all")
    state = new_state(cases["c0001"], pages["c0001"])
    state.update(nodes.validate_sources(state, ag))
    if node == "specialists":
        state["routes"] = {"ai": [str(r["page"]["page_id"]) for r in state["recs"]]}
    with pytest.raises(QuotaExhausted):
        try:
            getattr(nodes, node)(state, ag)
        except Exception:  # noqa: BLE001
            pytest.fail("QuotaExhausted was caught by except Exception")


# ---------------------------------------------------------------- the runner: stub provider with a quota window
class Clock:
    """The injected clock and sleep: time moves only when the gate sleeps."""

    def __init__(self):
        self.t, self.sleeps, self.lock = 0.0, [], threading.Lock()

    def now(self):
        return self.t

    def sleep(self, s):
        with self.lock:
            self.sleeps.append(s)
            self.t += s


class Window:
    """The hosted quota: `ok` successful paid calls (None = unlimited), then a 429 until the clock reaches `heal_at` (None = never).
    `paid` counts the successful calls (the ones the provider charges for)."""

    def __init__(self, ok=None, heal_at=None, clock=None):
        self.ok, self.heal_at, self.clock = ok, heal_at, clock
        self.paid = self.failed = 0
        self.log: list[tuple[float, int, bool, bool]] = []         # (clock, thread, success, from the probe)
        self.lock = threading.Lock()

    def check(self):
        with self.lock:
            now = self.clock.now() if self.clock else 0.0
            probe = quota._PROBE.get() is not None
            closed = self.ok is not None and self.ok <= 0 and (self.heal_at is None or now < self.heal_at)
            self.log.append((now, threading.get_ident(), not closed, probe))
            if closed:
                self.failed += 1
                raise LLMError("HTTP 429 " + PROVIDER_CANARY + " " + "y" * 300)
            if self.ok is not None:
                self.ok -= 1
            self.paid += 1


class WReader(ReaderLLM):
    def __init__(self, window, **kw):
        super().__init__(**kw)
        self.window = window

    def structured(self, system, user, schema, stage="", **kw):
        self.window.check()
        return super().structured(system, user, schema, stage=stage, **kw)


class WPlanner(PlannerLLM):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def structured(self, system, user, schema, stage="", **kw):
        self.window.check()
        return super().structured(system, user, schema, stage=stage, **kw)


def go(base: Path, *, reader_window=None, planner_window=None, out=None, cache=None, workers=1, engine="plain", page_text=None, **kw):
    data = write_data(base / "data", page_text=page_text)
    reader = WReader(reader_window) if reader_window is not None else ReaderLLM()
    planner = WPlanner(planner_window) if planner_window is not None else PlannerLLM()
    ag = make_agents(cache or base / "cache", reader=reader, planner=planner)
    out = out or base / "out"
    rj = runner.run(data, "test", "all", agents=ag, out=out, workers=workers, engine=engine, metric=StubMetric(), **kw)
    return out, rj


def _no_seconds(obj):
    if isinstance(obj, dict):
        return {k: _no_seconds(v) for k, v in obj.items() if k != "seconds"}
    if isinstance(obj, list):
        return [_no_seconds(v) for v in obj]
    return obj


def traces(out: Path) -> dict:
    return {p.name: _no_seconds(json.loads(p.read_text(encoding="utf-8"))) for p in sorted((out / "traces").glob("*.json"))}


# run.json fields that count THIS run's calls or time (a resume makes fewer calls), plus the two the brief names
RUN_ONLY = {"quota", "resume_count", "wall_seconds", "roles", "cache", "spend", "local"}


def _comparable(rj: dict) -> dict:
    return {k: v for k, v in rj.items() if k not in RUN_ONLY}


def _lookups(rj: dict) -> dict:
    return {r: v["cache_hits"] + v["cache_misses"] for r, v in rj["roles"].items()}


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    base = tmp_path_factory.mktemp("uninterrupted")
    rw, pw = Window(), Window()
    out, rj = go(base, reader_window=rw, planner_window=pw)
    assert rj["stopped_reason"] == "complete"
    return {"out": out, "rj": rj, "reader_paid": rw.paid, "planner_paid": pw.paid}


@pytest.mark.parametrize("where", ["first", "middle", "last"])
@pytest.mark.parametrize("engine", ["plain", "langgraph"])
def test_resume_equivalence(tmp_path, uninterrupted, where, engine):
    n = uninterrupted["reader_paid"]
    k = {"first": 0, "middle": n // 2, "last": n - 1}[where]
    rw, pw = Window(ok=k), Window()
    out1, rj1 = go(tmp_path, reader_window=rw, planner_window=pw, engine=engine)
    # run 1: a clean quota stop
    assert rj1["partial"] is True and rj1["stopped_reason"] == "quota" and rj1["stopped"] == "quota"
    assert not (out1 / "submission.csv").exists() and rj1["submission"]["written"] is False
    assert rj1["cases_done"] + rj1["cases_pending"] + rj1["cases"]["errors"] == rj1["cases_total"] == 3
    assert rj1["cases"]["errors"] == 0 and rj1["pages_failed"] == 0 and rj1["flags"]["planner_failures"] == 0
    assert rj1["quota"]["tripped"] is True and rj1["quota"]["kind"] == "rate_limit" and rj1["resume_count"] == 0
    done = {i.split("__")[0] for i in preds(out1)}
    assert done == {c for c in CASE_IDS if c not in rj1["unfinished_case_ids"]} and len(done) == rj1["cases_done"]
    assert sorted(p.stem for p in (out1 / "traces").glob("*.json")) == sorted(done)        # no trace of an abandoned case
    assert rw.paid == k
    # run 2: the same command, the same --out and cache, the window healthy again
    rw.ok = None
    out2, rj2 = go(tmp_path, reader_window=rw, planner_window=pw, engine=engine)
    assert out2 == out1 and rj2["stopped_reason"] == "complete" and rj2["partial"] is False and rj2["resume_count"] == 1
    u = uninterrupted
    assert (out2 / "preds.csv").read_bytes() == (u["out"] / "preds.csv").read_bytes()      # both engines give the same preds
    if engine == "plain":
        assert traces(out2) == traces(u["out"])
        assert _comparable(rj2) == _comparable(u["rj"]) and _lookups(rj2) == _lookups(u["rj"])
    assert (out2 / "submission.csv").exists() and rj2["quota"] is None
    # no successful call is paid for twice
    assert rw.paid == u["reader_paid"] and pw.paid == u["planner_paid"]


def test_a_quota_stop_with_parallel_workers_is_clean_and_resumes(tmp_path, uninterrupted):
    rw = Window(ok=uninterrupted["reader_paid"] // 3)
    out, rj = go(tmp_path, reader_window=rw, workers=3)
    assert rj["stopped_reason"] == "quota" and rj["cases_done"] + rj["cases_pending"] == 3 and rj["cases"]["errors"] == 0
    assert rj["pages_failed"] == 0 and rj["quota"]["wait_mode"] is False
    rw.ok = None
    out, rj = go(tmp_path, reader_window=rw, workers=3)
    assert (out / "preds.csv").read_bytes() == (uninterrupted["out"] / "preds.csv").read_bytes()
    assert rw.paid == uninterrupted["reader_paid"]


def test_a_429_that_recovers_leaves_no_quota_block(tmp_path, uninterrupted):
    class Bursty(Window):
        def check(self):
            with self.lock:
                self.paid += 1
                burst = self.paid in (2, 3)
            if burst:
                raise LLMError("HTTP 429 burst")
    out, rj = go(tmp_path, reader_window=Bursty())
    assert rj["stopped_reason"] == "complete" and rj["quota"] is None
    assert (out / "preds.csv").read_bytes() == (uninterrupted["out"] / "preds.csv").read_bytes()


# ---------------------------------------------------------------- WAIT mode
@pytest.mark.parametrize("engine,n_failed_probes", [("plain", 0), ("plain", 2), ("langgraph", 2)])
def test_wait_mode_probes_one_at_a_time_then_completes(tmp_path, uninterrupted, engine, n_failed_probes):
    clock = Clock()
    poll = 600.0
    w = Window(ok=5, heal_at=n_failed_probes * poll + 1, clock=clock)       # probes at 600 .. N*600 fail, probe N+1 passes
    out, rj = go(tmp_path, reader_window=w, planner_window=w, workers=3, wait_quota=True, quota_poll_minutes=10,
                 quota_max_hours=6, quota_clock=clock.now, quota_sleep=clock.sleep, engine=engine)
    q = rj["quota"]
    assert rj["stopped_reason"] == "complete" and rj["partial"] is False and (out / "submission.csv").exists()
    assert q["tripped"] is True and q["wait_mode"] is True and q["probes"] == n_failed_probes + 1
    assert q["waited_seconds"] == (n_failed_probes + 1) * poll and clock.sleeps == [poll] * (n_failed_probes + 1)
    assert (out / "preds.csv").read_bytes() == (uninterrupted["out"] / "preds.csv").read_bytes()
    # one probe at a time: every call made after a wait and before the window heals comes from ONE probe thread and is its ladder
    for k in range(1, n_failed_probes + 1):
        at = [e for e in w.log if e[0] == k * poll]
        assert len(at) == 5 and len({e[1] for e in at}) == 1 and all(e[3] and not e[2] for e in at)
    first_after_heal = next(e for e in w.log if e[0] == (n_failed_probes + 1) * poll)
    assert first_after_heal[3] is True                      # the first call after the last wait is the probe's
    assert runner.exit_code(rj) == 0
    # paid exactly as an uninterrupted run (the planner and the reader share the window here)
    assert w.paid == uninterrupted["reader_paid"] + uninterrupted["planner_paid"]


def test_wait_mode_gives_up_after_the_max_hours(tmp_path):
    clock = Clock()
    w = Window(ok=0, heal_at=None, clock=clock)
    out, rj = go(tmp_path, reader_window=w, planner_window=w, workers=3, wait_quota=True, quota_poll_minutes=10,
                 quota_max_hours=0.5, quota_clock=clock.now, quota_sleep=clock.sleep)
    q = rj["quota"]
    assert rj["stopped_reason"] == "quota" and rj["partial"] is True and not (out / "submission.csv").exists()
    assert q["probes"] == 3 and q["waited_seconds"] == 1800 and q["gave_up"] is True
    assert rj["cases_done"] == 0 and rj["cases_pending"] == 3 and runner.exit_code(rj) == 75


def test_wait_mode_is_interruptible(tmp_path):
    clock = Clock()

    def interrupted_sleep(s):
        raise KeyboardInterrupt
    w = Window(ok=0, clock=clock)
    out, rj = go(tmp_path, reader_window=w, planner_window=w, workers=3, wait_quota=True, quota_clock=clock.now,
                 quota_sleep=interrupted_sleep)
    assert rj["stopped"] == "interrupted" and rj["partial"] is True and runner.exit_code(rj) == 130
    assert rj["cases_done"] + rj["cases_pending"] == 3 and rj["quota"]["tripped"] is True


def test_the_gate_in_stop_mode_refuses_paid_calls_after_the_trip():
    gate = QuotaGate()
    with gate.admit():
        pass
    gate.trip(QuotaExhausted("rate_limit", 429, "x"))
    with pytest.raises(QuotaExhausted) as got:
        with gate.admit():
            pytest.fail("admitted")
    assert got.value.from_gate and gate.refused == 1 and gate.trips == 1
    with pytest.raises(QuotaExhausted):
        gate.run_case(lambda: pytest.fail("a new case started"))


def test_the_gate_lets_one_probe_out_per_wait():
    clock = Clock()
    gate = QuotaGate(wait=True, poll_seconds=60, max_seconds=600, clock=clock.now, sleep=clock.sleep)
    gate.trip(QuotaExhausted("keyword", None, "q"))
    inside, lock, attempts, seen = [0], threading.Lock(), [0], []

    def case():
        with gate.admit():
            with lock:
                inside[0] += 1
                attempts[0] += 1
                fail = attempts[0] <= 3                     # three failed probes, then the window is open
                if attempts[0] <= 4:                        # the probes: alone inside, and in the probe's context
                    seen.append((inside[0], quota._PROBE.get() is not None, gate.state))
            try:
                if fail:
                    raise QuotaExhausted("keyword", None, "q")
            finally:
                with lock:
                    inside[0] -= 1
        return "ok"
    res = []
    threads = [threading.Thread(target=lambda: res.append(gate.run_case(case))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert res == ["ok"] * 4 and gate.probes == 4 and gate.waited_seconds == 240 and gate.state == quota.OPEN
    assert gate.trips == 4                                  # the first trip plus three failed probes
    assert seen == [(1, True, quota.PROBING)] * 4


# ---------------------------------------------------------------- reporting
def test_console_shows_no_provider_message_or_page_text(tmp_path, capsys):
    out, rj = go(tmp_path, reader_window=Window(ok=2), page_text=PAGE_CANARY, workers=2)
    cap = capsys.readouterr()
    for stream in (cap.out, cap.err):
        assert PROVIDER_CANARY not in stream and PAGE_CANARY not in stream
    assert "quota window exhausted: " in cap.out and "run the same command again to resume (finished calls are cached)" in cap.out
    assert re.search(r"quota window exhausted: \d+ of 3 cases finished", cap.out)
    assert "kind rate_limit" in cap.out and "status " in cap.out
    text = (out / "run.json").read_text(encoding="utf-8")
    assert PAGE_CANARY not in text and PROVIDER_CANARY in rj["quota"]["first_error"] and len(rj["quota"]["first_error"]) <= 200
    for p in (out / "traces").iterdir():
        body = p.read_text(encoding="utf-8")
        assert PROVIDER_CANARY not in body and PAGE_CANARY not in body


def test_the_quota_block_fields(tmp_path):
    _, rj = go(tmp_path, reader_window=Window(ok=0))
    q = rj["quota"]
    assert set(q) >= {"tripped", "kind", "status", "first_error", "first_hit_utc", "wait_mode", "probes", "waited_seconds"}
    assert q["status"] is None and q["probes"] == 0 and q["waited_seconds"] == 0 and q["wait_mode"] is False
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", q["first_hit_utc"])
    assert rj["partial_reasons"] == ["quota", "unfinished_cases"]


def test_resume_count(tmp_path):
    out, rj = go(tmp_path)
    assert rj["resume_count"] == 0
    assert go(tmp_path)[1]["resume_count"] == 1 and go(tmp_path)[1]["resume_count"] == 2
    (out / "run.json").write_text("not json", encoding="utf-8")
    assert go(tmp_path)[1]["resume_count"] == 1


# ---------------------------------------------------------------- exit statuses (3c a)
@pytest.mark.parametrize("rj,code", [
    ({"partial": False, "stopped": None, "submission": {"problems": []}}, 0),
    ({"partial": True, "stopped": "quota", "submission": {"problems": []}}, 75),
    ({"partial": True, "stopped": "interrupted", "submission": {"problems": []}}, 130),
    ({"partial": True, "stopped": "budget", "submission": {"problems": []}}, 3),
    ({"partial": True, "stopped": None, "submission": {"problems": []}}, 3),
    ({"partial": False, "stopped": None, "submission": {"problems": ["x"]}}, 1),
])
def test_exit_code_table(rj, code):
    assert runner.exit_code(rj) == code


def _cli(tmp_path, monkeypatch, reader, *extra):
    data = write_data(tmp_path / "data")
    monkeypatch.setattr(runner, "build_agents", lambda a, d: (make_agents(tmp_path / "cache", reader=reader), None, StubMetric()))
    return runner.main(["run", "--data", str(data), "--split", "test", "--subset", "all", "--out", str(tmp_path / "out"), *extra])


def test_main_exits_75_on_a_quota_stop_then_0_on_the_resume(tmp_path, monkeypatch, capsys):
    w = Window(ok=4)
    assert _cli(tmp_path, monkeypatch, WReader(w), "--workers", "1") == 75
    w.ok = None
    assert _cli(tmp_path, monkeypatch, WReader(w), "--workers", "1") == 0
    assert PROVIDER_CANARY not in capsys.readouterr().out


@pytest.mark.parametrize("reader,code", [(lambda: ReaderLLM(fail={"p2"}), 3),
                                         (lambda: ReaderLLM(raise_on={"c0002": ValueError("bug")}), 3),
                                         (lambda: ReaderLLM(raise_on={"c0002": BudgetExceeded("b")}), 3)])
def test_main_exits_non_zero_on_other_partial_runs(tmp_path, monkeypatch, reader, code):
    assert _cli(tmp_path, monkeypatch, reader(), "--workers", "1") == code


def test_cli_quota_flags(tmp_path, monkeypatch):
    a = runner.build_parser().parse_args(["run", "--split", "test", "--subset", "all"])
    assert a.wait_quota is False and a.quota_poll_minutes == 10 and a.quota_max_hours == 6
    seen = {}
    monkeypatch.setattr(runner, "build_agents", lambda a, d: (make_agents(tmp_path / "cache"), None, StubMetric()))
    monkeypatch.setattr(runner, "run", lambda *a, **kw: seen.update(kw) or {"partial": False, "stopped": None,
                                                                             "submission": {"problems": []}})
    runner.main(["run", "--data", str(tmp_path), "--split", "test", "--subset", "all", "--wait-quota", "--quota-poll-minutes", "3",
                 "--quota-max-hours", "1.5", "--out", str(tmp_path / "o")])
    assert seen["wait_quota"] is True and seen["quota_poll_minutes"] == 3 and seen["quota_max_hours"] == 1.5


def test_estimate_states_the_quota_window(tmp_path):
    data = write_data(tmp_path / "data")
    cases, pages, ids = runner.load_split(data, "test", "all")
    lines = runner.format_estimate(runner.estimate(cases, pages, ids, make_agents(tmp_path / "cache")))
    line = next(s for s in lines if "quota window" in s)
    assert "unknown" in line and "under half" in line and "unmeasured" in line and "resum" in line


# ---------------------------------------------------------------- 3c b, c: case arithmetic and error frames
def test_cases_total_is_done_plus_pending_plus_errors(tmp_path):
    reader = ReaderLLM(raise_on={"c0001": ValueError("bug"), "c0003": BudgetExceeded("b")})
    data = write_data(tmp_path / "data")
    rj = runner.run(data, "test", "all", agents=make_agents(tmp_path / "cache", reader=reader), out=tmp_path / "out", workers=1,
                    metric=StubMetric())
    assert rj["cases"]["errors"] == 1 and rj["cases_done"] == 1 and rj["cases_pending"] == 1
    assert rj["cases_total"] == rj["cases_done"] + rj["cases_pending"] + rj["cases"]["errors"] == 3


def test_a_case_error_records_repo_frames_without_the_message(tmp_path, capsys):
    reader = ReaderLLM(raise_on={"c0002": ValueError("bug " + PAGE_CANARY)})
    data = write_data(tmp_path / "data")
    out = tmp_path / "out"
    rj = runner.run(data, "test", "all", agents=make_agents(tmp_path / "cache", reader=reader), out=out, workers=1,
                    metric=StubMetric())
    frames = rj["error_frames"]["c0002"]
    assert rj["case_errors"] == {"ValueError": 1} and frames
    assert all(re.fullmatch(r"[\w./-]+\.py:\d+", f) for f in frames)
    assert any(f.startswith("vendor_verdict/agents/specialists.py:") for f in frames)
    assert any(f.startswith("tests/test_vv_runner.py:") for f in frames)          # the raising stub: the bug's location
    assert not any(".venv" in f or "site-packages" in f for f in frames)
    assert PAGE_CANARY not in (out / "run.json").read_text(encoding="utf-8") and PAGE_CANARY not in capsys.readouterr().out


# ---------------------------------------------------------------- 3d: the money cap wiring of build_agents
@pytest.mark.parametrize("argv,cap", [((), float("inf")), (("--max-usd", "2.5"), 2.5)])
def test_build_agents_wires_the_money_cap(tmp_path, monkeypatch, argv, cap):
    import assessor.llm as allm
    seen = []

    class FakeLLM:                                   # never a real LLM: records the budget it is given
        mock_mode, model = False, "fake"

        def __init__(self, *a, **kw):
            seen.append(kw["budget"].max_usd)
            self.targets = [{"local": True}]

        @classmethod
        def extractor(cls, *a, **kw):
            return cls(*a, **kw)
    monkeypatch.setattr(allm, "LLM", FakeLLM)
    (tmp_path / "requirements.json").write_text("[]", encoding="utf-8")
    (tmp_path / "rules_lite.py").write_text("def evaluate(*a):\n    return {}\n\ndef overall(*a):\n    return ''\n", encoding="utf-8")
    args = runner.build_parser().parse_args(["run", "--split", "train", "--subset", "all", *argv])
    ag, budget, metric = runner.build_agents(args, tmp_path)
    assert seen == [cap, cap, cap] and budget.max_usd == cap and metric is None


def test_budget_exceeded_stays_a_stop_exception():
    assert BudgetExceeded in runner.STOP_EXCEPTIONS and QuotaExhausted in runner.STOP_EXCEPTIONS


# ---------------------------------------------------------------- mutation-driven extras
def test_error_frames_leave_out_the_virtualenv_and_outside_files():
    ns: dict = {}
    for name, fn in (("venv", ROOT / ".venv" / "lib" / "site-packages" / "fakepkg.py"), ("outside", Path("/elsewhere/x.py"))):
        exec(compile(f"def {name}(g):\n    return g()\n", str(fn), "exec"), ns)

    def boom():
        raise ValueError("message " + PAGE_CANARY)
    try:
        ns["venv"](lambda: ns["outside"](boom))
    except ValueError as exc:
        frames = runner.error_frames(exc)
    assert frames and all(f.startswith("tests/test_vv_quota.py:") for f in frames)
    assert not any(PAGE_CANARY in f for f in frames)


def test_a_halt_aborts_the_quota_gate(tmp_path, monkeypatch):
    calls = []
    real = QuotaGate.abort
    monkeypatch.setattr(QuotaGate, "abort", lambda self: (calls.append(1), real(self)))
    reader = ReaderLLM(raise_on={"c0002": BudgetExceeded("b")})
    data = write_data(tmp_path / "data")
    runner.run(data, "test", "all", agents=make_agents(tmp_path / "cache", reader=reader), out=tmp_path / "out", workers=1,
               metric=StubMetric())
    assert calls


def test_the_gate_aborts_its_waiters():
    gate = QuotaGate(wait=True, poll_seconds=60, max_seconds=600, clock=Clock().now, sleep=lambda s: None, abort_exc=KeyboardInterrupt)
    gate.trip(QuotaExhausted("rate_limit", None, ""))
    gate._sleeper = True                                    # someone else holds the wait: these workers block on the condition
    got = []

    def worker():
        try:
            gate.run_case(lambda: "ran")
        except BaseException as exc:  # noqa: BLE001
            got.append(type(exc))
    ts = [threading.Thread(target=worker) for _ in range(3)]
    for t in ts:
        t.start()
    gate.abort()
    for t in ts:
        t.join(timeout=10)
    assert got == [KeyboardInterrupt] * 3


def test_the_wait_starts_only_after_a_call_in_flight_returns():
    clock = Clock()
    gate = None
    inflight_at_sleep = []

    def sleep(s):
        inflight_at_sleep.append(gate._inflight)
        clock.sleep(s)
    gate = QuotaGate(wait=True, poll_seconds=60, max_seconds=600, clock=clock.now, sleep=sleep)
    entered, release = threading.Event(), threading.Event()

    def in_flight():                                        # a paid call admitted while the gate was open
        with gate.admit():
            entered.set()
            release.wait(timeout=10)
    a = threading.Thread(target=in_flight)
    a.start()
    entered.wait(timeout=10)
    gate.trip(QuotaExhausted("rate_limit", None, ""))
    b = threading.Thread(target=lambda: gate.run_case(lambda: "ok"))
    b.start()
    for _ in range(200_000):                                # yield (no timed sleep) until b holds the wait
        if gate._sleeper:
            break
        time.sleep(0)                                       # a zero-length thread yield, not a timed wait
    for _ in range(2_000):                                  # then give b the chance to sleep (it must not: a call is in flight)
        if clock.sleeps:
            break
        time.sleep(0)
    release.set()
    a.join(timeout=10)
    b.join(timeout=10)
    assert inflight_at_sleep == [0] and gate.probes == 1 and gate.state == quota.OPEN


# ---------------------------------------------------------------- fix round 1, I1 (ruling R39): a known status wins
@pytest.mark.parametrize("status", [400, 401, 404, 405, 413, 415, 422])
def test_a_known_non_quota_status_is_never_a_quota_error(status):
    assert quota.NON_QUOTA_STATUSES == {400, 401, 404, 405, 413, 415, 422}
    assert is_quota_error(llm_error("context length limit exceeded", status=status)) is None
    assert is_quota_error(llm_error("HTTP 429 rate limit quota", status=status)) is None


def test_a_429_inside_a_number_is_not_a_rate_limit():
    assert is_quota_error(llm_error("maximum context length is 8192 tokens; you requested 14293 tokens", status=400)) is None
    assert is_quota_error(LLMError("you requested 14293 tokens")) is None            # no status: word boundary only


def test_status_less_rate_limit_wording_and_a_429_status_stay_rate_limit():
    assert is_quota_error(LLMError("rate limit reached")) == "rate_limit"
    assert is_quota_error(LLMError("429 Too Many Requests")) == "rate_limit"
    assert is_quota_error(llm_error("slow down", status=429)) == "rate_limit"


@pytest.mark.parametrize("status", [None, 402, 403, 429, 500, 503])
def test_the_keyword_kind_needs_no_status_or_a_quota_like_one(status):
    kind = is_quota_error(llm_error("insufficient balance on this key", status=status))
    assert kind == ("rate_limit" if status == 429 else "keyword")


def test_a_deterministic_400_is_one_failed_call_again(tmp_path, no_ladder_sleep):
    llm = persistent(llm_error("context length limit exceeded; you requested 14293 tokens", status=400))
    assert read_page(llm, CallCache(tmp_path), "ai", PAGE) == ([], ["failed: LLMError"])
    assert llm.calls == 1 and no_ladder_sleep == []


def test_an_oversized_page_400_is_a_failed_page_not_a_quota_stop(tmp_path, monkeypatch):
    class Oversized(ReaderLLM):
        def structured(self, system, user, schema, stage="", **kw):
            if "Invented privacy page." in user:                      # p2 of every case
                raise llm_error("context length limit exceeded; you requested 14293 tokens", status=400)
            return super().structured(system, user, schema, stage=stage, **kw)
    data = write_data(tmp_path / "data")
    monkeypatch.setattr(runner, "build_agents", lambda a, d: (make_agents(tmp_path / "cache2", reader=Oversized()), None,
                                                              StubMetric()))
    rc = runner.main(["run", "--data", str(data), "--split", "test", "--subset", "all", "--out", str(tmp_path / "o2")])
    rj = run_json(tmp_path / "o2")
    assert rj["stopped_reason"] == "failed_pages" and rj["partial"] is True and rj["quota"] is None and rc == 3
    assert rj["pages_failed"] == 3 and rj["cases_done"] == 3


# ---------------------------------------------------------------- fix round 1, I2: the sleeper hands over to the probe atomically
def test_no_second_sleeper_between_the_wait_and_the_probe():
    clock = Clock()
    ids = {}
    parked, reacted = threading.Event(), threading.Event()
    armed = [False]

    def sleep(s):
        ids["sleeper"] = threading.get_ident()
        if len(clock.sleeps) == 0:
            parked.wait(timeout=10)                         # the other worker is waiting at the gate before the wait ends
            armed[0] = True
        else:
            reacted.set()                                   # a second sleeper (the race) reacted
        clock.sleep(s)

    gate = QuotaGate(wait=True, poll_seconds=60, max_seconds=600, clock=clock.now, sleep=sleep)

    class HookCond(threading.Condition):
        """After the sleeper's first release once its wait is over, let the parked worker run before the sleeper goes on."""

        def wait(self, timeout=None):
            if "sleeper" in ids and threading.get_ident() != ids["sleeper"]:
                (reacted if armed[0] else parked).set()
            return super().wait(timeout)

        def __exit__(self, *exc):
            out = super().__exit__(*exc)
            if armed[0] and threading.get_ident() == ids.get("sleeper"):
                armed[0] = False
                reacted.wait(timeout=10)                    # the woken worker either waits again (correct) or sleeps (the race)
            return out
    gate._cond = HookCond()
    gate.trip(QuotaExhausted("rate_limit", 429, ""))
    res = []
    s = threading.Thread(target=lambda: res.append(gate.run_case(lambda: "s")))
    s.start()
    for _ in range(200_000):                                # a zero-length thread yield, not a timed wait
        if "sleeper" in ids:
            break
        time.sleep(0)
    w = threading.Thread(target=lambda: res.append(gate.run_case(lambda: "w")))
    w.start()
    s.join(timeout=20)
    w.join(timeout=20)
    assert sorted(res) == ["s", "w"] and clock.sleeps == [60] and gate.waited_seconds == 60 and gate.probes == 1
