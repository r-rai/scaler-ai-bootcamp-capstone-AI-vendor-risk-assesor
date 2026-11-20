"""Local-server plumbing (vendor_verdict.local: the shared limiter, the health check, `--prewarm-local`, the run.json `local` block)
and the `selfcheck` subcommand (vendor_verdict.selfcheck: the real graph through a rules_lite with replay readers).

Invented data only (every data dir is in `tmp_path`; nothing under `data_kaggle/` is read), stub models, an injected HTTP getter and
clock: no real sleep, no real LLM, no socket to a local server (the getter is always injected or patched to fail the test)."""
import json
import socket
import sys
import threading
from pathlib import Path

import pytest

from assessor.llm import LLMError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run_kaggle  # noqa: E402,F401
from eval import kaggle_policy as kp  # noqa: E402

from tests.test_vv_runner import (CANARY, CASE_IDS, REQS, FakeRL, LocalLLM, PlannerLLM, ReaderLLM, StubMetric, make_agents,  # noqa: E402
                                  run_json, write_data)
from vendor_verdict import baseline as baseline_mod  # noqa: E402
from vendor_verdict import local, runner, selfcheck  # noqa: E402
from vendor_verdict.config import FAMILIES, FIELD_FAMILY, LOCAL_SLOTS  # noqa: E402
from vendor_verdict.quota import QuotaExhausted  # noqa: E402
from vendor_verdict.schemas import GradeOut, RouterOut  # noqa: E402

URL = "http://local.invalid:1/v1"
PROVIDER_CANARY = "CANARY-PROVIDER-BODY"


@pytest.fixture(autouse=True)
def no_real_http(monkeypatch):
    """The default getter fails the test: every health check here goes through an injected getter, or makes none."""
    def refuse(url, timeout):
        raise AssertionError("a test reached the real HTTP getter")
    monkeypatch.setattr(local, "http_get", refuse)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.5
        return self.t


class TargetLocal(LocalLLM):
    """The stub local model with the llama.cpp target shape `assessor.llm.LLM` has (the health URL comes from it)."""
    targets = [{"local": True, "base_url": URL, "provider": "llamacpp"}]


def getter(status=200, body='{"status":"ok"}', exc=None):
    seen = []

    def get(url, timeout):
        seen.append((url, timeout))
        if exc is not None:
            raise exc
        return status, body
    get.seen = seen
    return get


# ---------------------------------------------------------------- the limiter
class Blocking:
    """A local model whose calls wait on `go`; it records how many are in flight at once."""
    model = "stub-local"

    def __init__(self):
        self.go, self.lock, self.cond = threading.Event(), threading.Lock(), threading.Condition()
        self.in_flight = self.max_seen = self.calls = 0

    def structured(self, system, user, schema, stage="", **kw):
        with self.cond:
            self.in_flight += 1
            self.calls += 1
            self.max_seen = max(self.max_seen, self.in_flight)
            self.cond.notify_all()
        try:
            assert self.go.wait(timeout=10)
            return RouterOut(families=["ai"])
        finally:
            with self.cond:
                self.in_flight -= 1


def test_six_threads_share_two_slots():
    inner = Blocking()
    lim = local.LocalLimiter(inner, slots=2)
    threads = [threading.Thread(target=lim.structured, args=("s", "u", RouterOut), kwargs={"stage": "router"}) for _ in range(6)]
    for t in threads:
        t.start()
    with inner.cond:
        assert inner.cond.wait_for(lambda: inner.in_flight == 2, timeout=10)
    # two calls run and four are queued at the semaphore; none of the four may get in while the two are held
    assert lim.in_flight == 2 and inner.calls == 2
    inner.go.set()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()
    st = lim.stats()
    assert inner.max_seen == 2 and st["max_in_flight"] == 2 and st["calls"] == 6 and lim.in_flight == 0


@pytest.mark.parametrize("exc", [LLMError("local " + CANARY), runner.RunStopped(), QuotaExhausted("rate_limit", 429, "x"),
                                 KeyboardInterrupt()])
def test_every_exit_path_releases_the_permit(exc):
    class Raises:
        model = "stub-local"

        def structured(self, *a, **kw):
            raise exc
    lim = local.LocalLimiter(Raises(), slots=1)
    with pytest.raises(type(exc)):
        lim.structured("s", "u", RouterOut, stage="router")
    lim._llm = LocalLLM()                              # the one permit must be free again: a following call must not hang
    done = []
    t = threading.Thread(target=lambda: done.append(lim.structured("s", "u", RouterOut, stage="router")))
    t.start()
    t.join(timeout=10)
    assert not t.is_alive() and isinstance(done[0], RouterOut)
    assert lim.stats()["calls"] == 2 and lim.in_flight == 0


def test_counters_are_exact_and_wait_uses_the_clock():
    lim = local.LocalLimiter(LocalLLM(), slots=2, clock=Clock())
    for _ in range(3):
        lim.structured("s", "u", GradeOut, stage="grader", max_repairs=0)
    # each call reads the clock twice (before and after taking the permit): 0.5 s queued per call with this clock
    assert lim.stats() == {"calls": 3, "max_in_flight": 1, "wait_seconds": 1.5}
    lim.reset()
    assert lim.stats() == {"calls": 0, "max_in_flight": 0, "wait_seconds": 0.0}


def test_the_wrapper_keeps_what_the_agents_read():
    inner = TargetLocal()
    inner.budget, inner.tracer, inner.mock_mode = "B", "T", False
    lim = local.limited(inner)
    assert lim.model == "stub-local" and lim.targets == inner.targets and lim.budget == "B" and lim.tracer == "T"
    assert lim.mock_mode is False and local.limited(lim) is lim
    lim.structured("s", "u", RouterOut, stage="router", max_repairs=0)
    assert inner.calls == ["router"]                   # stage and keywords reach the model unchanged


def test_build_agents_wraps_the_real_local_model(tmp_path, monkeypatch):
    import assessor.llm as allm

    class FakeLLM:                                     # never a real LLM
        mock_mode, model = False, "fake"

        def __init__(self, *a, **kw):
            self.targets = [{"local": True}]

        @classmethod
        def extractor(cls, *a, **kw):
            return cls()
    monkeypatch.setattr(allm, "LLM", FakeLLM)
    (tmp_path / "requirements.json").write_text("[]", encoding="utf-8")
    (tmp_path / "rules_lite.py").write_text("def evaluate(*a):\n    return {}\n\ndef overall(*a):\n    return ''\n", encoding="utf-8")
    args = runner.build_parser().parse_args(["run", "--split", "train", "--subset", "all"])
    ag, _, _ = runner.build_agents(args, tmp_path)
    assert isinstance(ag.local_llm, local.LocalLimiter) and ag.local_llm.slots == LOCAL_SLOTS
    assert not isinstance(ag.main_llm, local.LocalLimiter) and not isinstance(ag.reader_llm, local.LocalLimiter)


def test_a_run_with_six_workers_never_exceeds_the_slots(tmp_path):
    class Slow(LocalLLM):                              # each local call waits for a third in flight (or 0.03 s)
        def __init__(self):
            super().__init__()
            self.cond, self.n, self.peak = threading.Condition(), 0, 0

        def structured(self, *a, **kw):
            with self.cond:
                self.n += 1
                self.peak = max(self.peak, self.n)
                self.cond.notify_all()
                self.cond.wait_for(lambda: self.n > LOCAL_SLOTS, timeout=0.03)
            try:
                return super().structured(*a, **kw)
            finally:
                with self.cond:
                    self.n -= 1
    slow = Slow()
    data = write_data(tmp_path / "data")
    out = tmp_path / "out"
    runner.run(data, "test", "all", agents=make_agents(tmp_path / "cache", local=slow), out=out, workers=6, metric=StubMetric(),
               say=lambda s: None)
    blk = run_json(out)["local"]
    assert 1 <= slow.peak <= LOCAL_SLOTS and 1 <= blk["max_in_flight"] <= LOCAL_SLOTS     # 3 cases at once, at most 2 in flight
    assert blk["calls"] == len(slow.calls) > 0 and blk["slots"] == LOCAL_SLOTS


# ---------------------------------------------------------------- the health check
def test_health_url_and_verdicts():
    assert local.health_url(TargetLocal()) == "http://local.invalid:1/health"
    assert local.health_url(LocalLLM()) is None                                  # a stub without a target: nothing to check
    ok = getter()
    assert local.local_health("http://h/health", get=ok) == {"ok": True, "detail": "ok"} and ok.seen == [("http://h/health", 2.0)]
    assert local.local_health("u", get=getter(status=503, body=PROVIDER_CANARY)) == {"ok": False, "detail": "http 503"}
    assert local.local_health("u", get=getter(body='{"status":"loading ' + PROVIDER_CANARY + '"}'))["ok"] is False
    assert local.local_health("u", get=getter(body="not json " + PROVIDER_CANARY))["detail"] == "unexpected reply"
    down = local.local_health("u", get=getter(exc=ConnectionRefusedError(PROVIDER_CANARY)))
    assert down == {"ok": False, "detail": "unreachable: ConnectionRefusedError"}


def _main(tmp_path, monkeypatch, *extra, local_llm=None, get=None, reader=None, page_text=None):
    data = write_data(tmp_path / "data", page_text=page_text)
    holder = {}

    def build(a, d):
        ag = make_agents(tmp_path / "cache", local=local_llm or TargetLocal(), reader=reader)
        holder["ag"] = ag
        return ag, None, StubMetric()
    monkeypatch.setattr(runner, "build_agents", build)
    rc = runner.main(["run", "--data", str(data), "--split", "test", "--subset", "all", "--out", str(tmp_path / "out"), *extra],
                     http_get=get)
    return rc, holder["ag"]


def test_a_down_server_refuses_a_paid_run(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "preds.csv").write_text("earlier", encoding="utf-8")
    called = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: called.append(1))
    reader = ReaderLLM()
    get = getter(exc=ConnectionRefusedError(PROVIDER_CANARY))
    rc, ag = _main(tmp_path, monkeypatch, get=get, reader=reader)
    text = capsys.readouterr().out
    assert rc == local.EXIT_LOCAL_DOWN == runner.EXIT_LOCAL_DOWN == 69 and called == [] and reader.calls == []
    assert get.seen == [("http://local.invalid:1/health", 2.0)]
    assert (out / "preds.csv").read_text() == "earlier"                         # refused before run() removes anything
    assert "DOWN" in text and "--allow-local-down" in text and PROVIDER_CANARY not in text


def test_allow_local_down_proceeds_and_records_it(tmp_path, monkeypatch, capsys):
    rc, _ = _main(tmp_path, monkeypatch, "--allow-local-down", get=getter(status=503))
    rj = run_json(tmp_path / "out")
    assert rc == 0 and rj["local"]["health"] == "down_allowed" and rj["partial"] is False


def test_the_console_holds_no_page_text_and_no_reply_body(tmp_path, monkeypatch, capsys):
    rc, _ = _main(tmp_path, monkeypatch, "--allow-local-down", get=getter(status=500, body=PROVIDER_CANARY), page_text=CANARY)
    text = capsys.readouterr().out
    assert rc == 0 and "health: DOWN (http 500)" in text and "local server: health down_allowed" in text
    assert CANARY not in text and PROVIDER_CANARY not in text


def test_an_up_server_is_ok(tmp_path, monkeypatch):
    get = getter()
    rc, _ = _main(tmp_path, monkeypatch, get=get)
    assert rc == 0 and run_json(tmp_path / "out")["local"]["health"] == "ok" and len(get.seen) == 1


@pytest.mark.parametrize("extra", [("--cache-only",), ("--estimate",), ("--ablate", "router,grader,decision_maker")])
def test_cache_only_estimate_and_no_local_role_skip_the_check(tmp_path, monkeypatch, extra):
    get = getter(exc=ConnectionRefusedError("down"))
    rc, _ = _main(tmp_path, monkeypatch, *extra, get=get)
    assert get.seen == []
    assert rc == (runner.EXIT_PARTIAL if extra == ("--cache-only",) else 0)     # cache-only on an empty cache: every call misses
    if extra != ("--estimate",):
        assert run_json(tmp_path / "out")["local"]["health"] == "skipped"


def test_one_local_role_left_still_needs_the_check(tmp_path, monkeypatch):
    get = getter(exc=ConnectionRefusedError("down"))
    rc, _ = _main(tmp_path, monkeypatch, "--ablate", "router,grader", get=get)
    assert rc == local.EXIT_LOCAL_DOWN and len(get.seen) == 1


def test_an_old_style_stub_run_makes_no_http_call(tmp_path, monkeypatch):
    # the autouse fixture makes the default getter fail the test: a stub without a llama.cpp target is never checked
    rc, _ = _main(tmp_path, monkeypatch, local_llm=LocalLLM())
    assert rc == 0 and run_json(tmp_path / "out")["local"]["health"] == "skipped"


def test_the_default_getter_is_what_main_uses(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(local, "http_get", lambda url, timeout: (seen.append(url), (503, ""))[1])
    rc, _ = _main(tmp_path, monkeypatch)
    assert rc == local.EXIT_LOCAL_DOWN and seen == ["http://local.invalid:1/health"]


def test_the_local_block(tmp_path):
    data = write_data(tmp_path / "data")
    out = tmp_path / "out"
    runner.run(data, "test", "all", agents=make_agents(tmp_path / "cache"), out=out, workers=2, metric=StubMetric(), say=lambda s: None)
    blk = run_json(out)["local"]
    assert set(blk) == {"slots", "health", "calls", "max_in_flight", "wait_seconds", "prewarm"}
    assert blk["health"] == "not_checked" and blk["prewarm"] is None and blk["slots"] == LOCAL_SLOTS
    rj = run_json(out)
    assert blk["calls"] == sum(rj["roles"][r]["calls"] for r in ("router", "grader", "decision_maker"))
    assert "local_fallbacks" in rj                     # the node fallback counts stay where they were


# ---------------------------------------------------------------- prewarm
def _prewarm(tmp_path, monkeypatch, *extra, get=None, local_llm=None, planner=None, reader=None):
    data = write_data(tmp_path / "data")
    ag = make_agents(tmp_path / "cache", local=local_llm or TargetLocal(), planner=planner, reader=reader)
    monkeypatch.setattr(runner, "build_agents", lambda a, d: (ag, None, StubMetric()))
    monkeypatch.setattr(runner, "run", lambda *a, **kw: pytest.fail("prewarm must not start a run"))
    rc = runner.main(["run", "--data", str(data), "--split", "test", "--subset", "all", "--out", str(tmp_path / "out"),
                      "--prewarm-local", *extra], http_get=get or getter())
    return rc, data


def test_prewarm_fills_the_router_cache_and_a_later_run_hits_it(tmp_path, monkeypatch, capsys):
    first, planner, reader = TargetLocal(), PlannerLLM(), ReaderLLM()
    rc, data = _prewarm(tmp_path, monkeypatch, local_llm=first, planner=planner, reader=reader)
    text = capsys.readouterr().out
    assert rc == 0 and planner.calls == [] and reader.calls == [] and first.calls == ["router"] * 9
    assert "pages 9" in text and "new calls 9" in text and "cache hits 0" in text and CANARY not in text
    assert not (tmp_path / "out").exists()                                       # writes nothing but the cache

    second = TargetLocal()
    rc, _ = _prewarm(tmp_path, monkeypatch, local_llm=second)
    assert rc == 0 and second.calls == [] and "cache hits 9" in capsys.readouterr().out

    third = TargetLocal()
    out = tmp_path / "run"
    monkeypatch.undo()
    monkeypatch.setattr(local, "http_get", lambda url, timeout: pytest.fail("no http"))
    runner.run(data, "test", "all", agents=make_agents(tmp_path / "cache", local=third), out=out, metric=StubMetric(),
               say=lambda s: None)
    rj = run_json(out)
    assert rj["roles"]["router"]["cache_hits"] == 9 and rj["roles"]["router"]["calls"] == 0 and "router" not in third.calls


def test_prewarm_honours_limit_and_only(tmp_path, monkeypatch, capsys):
    lim_local = TargetLocal()
    rc, _ = _prewarm(tmp_path / "l", monkeypatch, "--limit", "1", local_llm=lim_local)
    assert rc == 0 and len(lim_local.calls) == 3 and "cases 1" in capsys.readouterr().out
    only_local = TargetLocal()
    rc, _ = _prewarm(tmp_path / "o", monkeypatch, "--only", "c0002,c0003", local_llm=only_local)
    assert rc == 0 and len(only_local.calls) == 6 and "cases 2" in capsys.readouterr().out


def test_prewarm_refuses_when_the_server_is_down(tmp_path, monkeypatch):
    stub = TargetLocal()
    rc, _ = _prewarm(tmp_path, monkeypatch, get=getter(exc=ConnectionRefusedError("down")), local_llm=stub)
    assert rc == local.EXIT_LOCAL_DOWN and stub.calls == []


def test_prewarm_counts_failures_and_goes_through_the_limiter(tmp_path):
    data = write_data(tmp_path / "data")
    cases, pages, ids = runner.load_split(data, "test", "all")
    lim = local.LocalLimiter(LocalLLM(raise_on={"router"}), slots=2)
    got = local.prewarm(cases, pages, sorted(ids), lim, tmp_path / "cache", clock=Clock())
    assert got == {"cases": 3, "pages": 9, "hits": 0, "calls": 9, "failures": 9, "seconds": 0.5}
    assert lim.stats()["calls"] == 9 and not (tmp_path / "cache").exists()      # failures are never cached


def test_prewarm_with_the_router_ablated_does_nothing(tmp_path, monkeypatch, capsys):
    stub = TargetLocal()
    rc, _ = _prewarm(tmp_path, monkeypatch, "--ablate", "router", local_llm=stub)
    assert rc == 0 and stub.calls == [] and "nothing to prewarm" in capsys.readouterr().out


# ---------------------------------------------------------------- selfcheck (invented data dir; the real code path)
SC_CASES = ["c0001", "c0002", "c0003"]          # train split: dev = even positions (c0001, c0003)


def _claims(cid):
    return {
        "p1": [{"field": "sso_saml", "value": "all_plans", "display": "SSO on all plans", "quote": "Single sign-on (SAML) is offered on every plan.",
                "evidence_level": "claimed"},
               {"field": "trains_on_customer_data", "value": "no", "display": "no training", "quote": "We do not train models on customer data.",
                "evidence_level": "claimed"},
               {"field": "subprocessors", "value": [{"name": "Example Cloud", "purpose": "hosting", "category": "hosting", "country": "US"},
                          {"name": "Example Mail", "purpose": "email", "category": "email", "country": None}],
                "display": "two sub-processors", "quote": "Sub-processors: Example Cloud (hosting), Example Mail (email).",
                "evidence_level": "claimed"}],
        "p2": [{"field": "data_retention", "value": {"default_days": 30, "deletion_on_request_days": None}, "display": "30 days",
                "quote": "Customer data is deleted after 30 days.", "evidence_level": "claimed"},
               {"field": "dpa_available", "value": "on_request", "display": "DPA on request",
                "quote": "A data processing agreement is available on request.", "evidence_level": "claimed"}],
        "p3": [],
    }


SC_BASELINE_PROMPT = "Invented baseline extraction prompt for the selfcheck tests (not the submitted one)."


@pytest.fixture
def no_git_baseline(monkeypatch):
    """R58: the selfcheck tests never touch git history (a depth-1 CI clone has none). The real `selfcheck` still recovers the prompt
    from git through `baseline_mod.baseline_prompt`; here that returns an invented text and any git recovery raises, so a test that
    still reached git history would fail on a full clone too."""
    def no_history(*a, **k):
        raise AssertionError("a selfcheck test tried to recover a prompt from git history")
    monkeypatch.setattr(kp, "prompt_text_from_git", no_history)
    monkeypatch.setattr(baseline_mod, "baseline_prompt", lambda: SC_BASELINE_PROMPT)


def write_selfcheck_data(d: Path, model: str, missing_case=None, page_text="") -> Path:
    write_data(d, split="train", page_text=page_text or None)
    # the sub-processor sentence the claims quote
    lines = []
    for line in (d / "pages.jsonl").read_text(encoding="utf-8").splitlines():
        p = json.loads(line)
        if p["page_id"].endswith("_p1"):
            p["text"] += " Sub-processors: Example Cloud (hosting), Example Mail (email)."
        lines.append(json.dumps(p))
    (d / "pages.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cache = d / "cache" / "extract"
    cache.mkdir(parents=True)
    system = SC_BASELINE_PROMPT                                     # the text `no_git_baseline` injects
    for line in lines:
        p = json.loads(line)
        cid, s = p["case_id"], p["page_id"].rsplit("_", 1)[1]
        if cid == missing_case:
            continue
        key = kp.cache_key(model, p, "", system=system)
        (cache / f"{key}.json").write_text(json.dumps({"model": model, "salt": "", "claims": _claims(cid)[s], "in": 1, "out": 1}),
                                           encoding="utf-8")
    return d


def sc_reqs():
    """One requirement per family field in the matcher's shape (`run_kaggle.assemble_case` also runs our matcher), from the repo's own
    data/requirements.yaml (not competition data), renamed."""
    import yaml
    rows = yaml.safe_load((ROOT / "data" / "requirements.yaml").read_text(encoding="utf-8"))["requirements"]
    first = {}
    for r in rows:
        first.setdefault(r["field"], r)
    return [{**first[f], "id": f"INV-{f}"} for f in sorted(FIELD_FAMILY) if f in first]


def _check(d, model="stub-baseline", **kw):
    lines = []
    res = selfcheck.check(d, split="train", subset="dev", limit=3, rl=FakeRL, reqs=sc_reqs(), baseline_model=model,
                          baseline_dir=d / "cache" / "extract", say=lines.append, **kw)
    return res, lines


@pytest.mark.usefixtures("no_git_baseline")
def test_selfcheck_replays_the_baseline_rows(tmp_path):
    d = write_selfcheck_data(tmp_path / "data", "stub-baseline")
    files = sorted(p.relative_to(d) for p in d.rglob("*"))
    res, lines = _check(d)
    assert res["exit"] == 0 and res["cases_run"] == 2 and res["skipped"] == [] and res["errors"] == 0
    assert res["rows"] == 2 * (len(sc_reqs()) + 1) and len(sc_reqs()) == 8 and res["differing"] == [] and res["equal"] == res["rows"]
    assert res["unexpected"] == 0 and res["network"] == 0 and res["answers"]["specialist"] > 0
    assert sorted(p.relative_to(d) for p in d.rglob("*")) == files                # nothing written into the data dir
    assert any(line.startswith("selfcheck OK") for line in lines)


@pytest.mark.usefixtures("no_git_baseline")
def test_selfcheck_skips_a_case_without_a_baseline_cache(tmp_path):
    d = write_selfcheck_data(tmp_path / "data", "stub-baseline", missing_case="c0003")
    res, lines = _check(d)
    assert res["cases_run"] == 1 and res["skipped"] == ["c0003"] and res["exit"] == 0
    assert any("no baseline cache" in line and "c0003" in line for line in lines)


@pytest.mark.usefixtures("no_git_baseline")
def test_selfcheck_with_no_covered_case_fails(tmp_path):
    d = write_selfcheck_data(tmp_path / "data", "stub-baseline")
    res, _ = _check(d, model="another-model")                                    # a wrong model name: no case has a cache
    assert res["cases_run"] == 0 and res["exit"] != 0


@pytest.mark.usefixtures("no_git_baseline")
def test_an_unexpected_stub_request_fails_the_selfcheck(tmp_path, monkeypatch):
    d = write_selfcheck_data(tmp_path / "data", "stub-baseline")
    real = selfcheck.LocalStub.structured

    def odd(self, system, user, schema, *a, stage="", **kw):              # the router asks for something the stub does not know
        return real(self, system, user, GradeOut if stage == "router" else schema, *a, stage=stage, **kw)
    monkeypatch.setattr(selfcheck.LocalStub, "structured", odd)
    res, lines = _check(d)
    assert res["unexpected"] > 0 and res["exit"] != 0 and any(line.startswith("selfcheck FAILED") for line in lines)


@pytest.mark.usefixtures("no_git_baseline")
def test_a_network_attempt_fails_the_selfcheck(tmp_path, monkeypatch):
    d = write_selfcheck_data(tmp_path / "data", "stub-baseline")
    real = selfcheck.PlannerStub.structured

    def dial(self, *a, **kw):
        s = socket.socket()
        try:
            s.connect(("127.0.0.1", 9))                                    # the guard refuses before any connection is made
        except OSError:
            pass
        finally:
            s.close()
        return real(self, *a, **kw)
    monkeypatch.setattr(selfcheck.PlannerStub, "structured", dial)
    res, _ = _check(d)
    assert res["network"] == 2 and res["exit"] != 0                            # one planner call per dev case


@pytest.mark.usefixtures("no_git_baseline")
def test_selfcheck_prints_no_page_text(tmp_path, capsys):
    d = write_selfcheck_data(tmp_path / "data", "stub-baseline", page_text=CANARY)
    res, lines = _check(d)
    assert res["exit"] == 0
    assert CANARY not in "\n".join(lines) and CANARY not in capsys.readouterr().out
    assert "Example Cloud" not in "\n".join(lines)


def test_list_values_are_replayed_one_item_per_reading():
    r = {"page_id": "p1", "field": "subprocessors", "value": [{"name": "A"}, {"name": "B"}], "quote": "q", "display": "d"}
    assert [x["value"] for x in selfcheck.specialist_items([r])] == [{"name": "A"}, {"name": "B"}]
    empty = {**r, "field": "security_incidents", "value": []}
    assert [x["value"] for x in selfcheck.specialist_items([empty])] == [[]]
    plain = {**r, "field": "sso_saml", "value": "all_plans"}
    assert selfcheck.specialist_items([plain]) == [{"page_id": "p1", "field": "sso_saml", "value": "all_plans", "quote": "q", "display": "d"}]


def test_the_selfcheck_subcommand_parses_dev_only():
    a = runner.build_parser().parse_args(["selfcheck"])
    assert (a.cmd, a.split, a.subset, a.limit) == ("selfcheck", "train", "dev", 3)
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["selfcheck", "--subset", "holdout"])
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["selfcheck", "--split", "test"])


def test_families_cover_the_router_stub():
    assert selfcheck.LocalStub().structured("s", "u", RouterOut, stage="router").families == list(FAMILIES)
    assert set(CASE_IDS) == set(SC_CASES)


# ---------------------------------------------------------------- the `local` block across a quota stop and its resume
@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_local_counts_across_a_quota_stop_and_its_resume(tmp_path, monkeypatch, where):
    """`local` counts THIS run's calls (like `roles` and `cache`), so it differs between a stop+resume and an uninterrupted run; what
    must hold instead: slots, health and prewarm are equal, and the two runs' local calls add up to the uninterrupted run's (every
    successful local call is cached, so none is made twice). max_in_flight and wait_seconds are not comparable across runs."""
    from tests.test_vv_quota import Window, go
    from vendor_verdict.agents import planner as planner_mod
    from vendor_verdict.agents import specialists as specialists_mod
    monkeypatch.setattr(planner_mod, "_sleep", lambda s: None)           # the retry ladder is recorded nowhere and never slept
    monkeypatch.setattr(specialists_mod, "_sleep", lambda s: None)
    say = {"say": lambda s: None}
    full_w = Window()
    _, full = go(tmp_path / "u", reader_window=full_w, planner_window=Window(), **say)
    n = full_w.paid
    rw = Window(ok={"first": 0, "middle": n // 2, "last": n - 1}[where])
    _, stopped = go(tmp_path / "r", reader_window=rw, planner_window=Window(), **say)
    rw.ok = None
    _, resumed = go(tmp_path / "r", reader_window=rw, planner_window=Window(), **say)
    assert stopped["stopped_reason"] == "quota" and resumed["stopped_reason"] == "complete"
    for k in ("slots", "health", "prewarm"):
        assert stopped["local"][k] == resumed["local"][k] == full["local"][k]
    assert full["local"]["calls"] > 0
    assert stopped["local"]["calls"] + resumed["local"]["calls"] == full["local"]["calls"]


# ---------------------------------------------------------------- Task 8B2: Part B leftovers
def test_the_slots_default_is_read_when_the_limiter_is_built(monkeypatch):
    from vendor_verdict import config
    monkeypatch.setattr(config, "LOCAL_SLOTS", 5)
    assert local.limited(LocalLLM()).slots == 5 and local.LocalLimiter(LocalLLM()).slots == 5


class StopAfter(LocalLLM):
    """A local stub that sets `stop` (the event a Ctrl-C sets) once it has answered `n` calls; it records each call's case id."""

    def __init__(self, stop, n):
        super().__init__()
        self.stop, self.n, self.cases = stop, n, []

    def structured(self, system, user, schema, stage="", **kw):
        self.cases.append(next(c for c in CASE_IDS if f" Case {c} page." in user))
        out = super().structured(system, user, schema, stage=stage, **kw)
        if len(self.cases) >= self.n:
            self.stop.set()
        return out


def test_a_stop_during_prewarm_leaves_the_rest_unrouted(tmp_path, monkeypatch):
    data = write_data(tmp_path / "data")
    cases, pages, ids = runner.load_split(data, "test", "all")
    triaged = []
    real = local.rk.triage
    monkeypatch.setattr(local.rk, "triage", lambda case, pgs: (triaged.append(case["case_id"]), real(case, pgs))[1])
    stop = threading.Event()
    stub = StopAfter(stop, 4)                          # case 1: 3 pages; case 2: the stop comes after its first page
    got = local.prewarm(cases, pages, sorted(ids), local.LocalLimiter(stub, slots=1), tmp_path / "cache", workers=1, stop=stop)
    assert stub.cases == ["c0001"] * 3 + ["c0002"] and triaged == ["c0001", "c0002"]       # c0003 never started
    assert got["cases"] == 1 and got["pages"] == 3 and got["calls"] == 4 and got["failures"] == 0


def test_a_stopped_prewarm_prints_the_partial_counts_and_exits_130(tmp_path, monkeypatch, capsys):
    from vendor_verdict import config
    monkeypatch.setattr(config, "LOCAL_SLOTS", 1)          # one worker: the stop lands after case 1, deterministically
    data = write_data(tmp_path / "data", page_text=CANARY)
    cases, pages, ids = runner.load_split(data, "test", "all")
    stop = threading.Event()
    stub = type("T", (StopAfter,), {"targets": TargetLocal.targets})(stop, 4)
    ag = make_agents(tmp_path / "cache", local=stub)
    rc = local.prewarm_cli(cases, pages, sorted(ids), ag, get=getter(), stop=stop)
    text = capsys.readouterr().out
    assert rc == local.EXIT_INTERRUPTED == runner.EXIT_INTERRUPTED == 130 and "c0003" not in stub.cases
    assert "INTERRUPTED" in text and "cases 1 of 3" in text and "new calls 4" in text and CANARY not in text


def test_prewarm_and_a_run_word_a_down_server_the_same_way(tmp_path, monkeypatch, capsys):
    get = getter(exc=ConnectionRefusedError("down"))
    rc, _ = _prewarm(tmp_path / "p", monkeypatch, get=get)
    pre = [line for line in capsys.readouterr().out.splitlines() if "health" in line]
    monkeypatch.setattr(runner, "run", lambda *a, **kw: pytest.fail("refused"))
    rc2, _ = _main(tmp_path / "r", monkeypatch, get=get)
    run_lines = [line for line in capsys.readouterr().out.splitlines() if "health" in line]
    assert rc == rc2 == local.EXIT_LOCAL_DOWN and pre == run_lines and len(pre) == 1


def test_the_prewarm_slot_of_the_local_block_is_documented_as_reserved():
    assert "reserved, always null in run.json" in local.block.__doc__
