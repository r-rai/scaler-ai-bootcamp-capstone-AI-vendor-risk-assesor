"""The local llama.cpp server inside the parallel runner (ledger rulings R11, R31(a), R34): one shared limiter, a health check at the
start of a run that can pay, and the free Router-only pre-warm pass.

- `LocalLimiter` wraps the local model with ONE `threading.BoundedSemaphore(config.LOCAL_SLOTS)` shared by every case worker: the
  server has 2 slots and is prefill-bound, so more concurrent calls would only queue on the server. The permit is held only while the
  call runs and is released on every exit path (an exception, `RunStopped`, `QuotaExhausted`, KeyboardInterrupt). It counts calls,
  the peak number in flight (never above the slots) and the seconds spent queued for a permit. `build_agents` wraps the real local
  model; `runner.run` wraps whatever it is given unless it is already wrapped, so every run reports the counts.
- `local_health` is one cheap `GET <server>/health` (llama-server answers it without the API key; 200 with {"status": "ok"} = up, a
  503 "loading model" = down). It never makes a model call, and its `detail` is a fixed phrase plus a status code or an exception class
  name, never the reply body.
- `start_check`: a run that can make an uncached hosted call (not --cache-only; --estimate never gets here) and uses a local role
  (router, grader, decision_maker in `config.LOCAL_SCOPE`, not ablated) checks the server first and refuses to start when it is down
  (exit `EXIT_LOCAL_DOWN`), unless --allow-local-down: a degraded run is not reproducible and its extra iteration-2 reads burn quota
  window time (R34). A model without a llama.cpp target (a test stub) has nothing to check.
- `prewarm` runs ONLY the Router over every readable page of the selected cases, through the limiter and the Router's own cache keys
  (`agents.router.tag_pages`, exactly as the graph's `route_pages` calls it), so a later run finds every routing in the cache. It makes
  no hosted call and writes nothing but cache entries. Cases are submitted one by one; a Ctrl-C sets a stop event: the cases in flight
  end at their next local call (`PrewarmStopped`), queued ones never start, the partial counts are printed and the exit is 130.

Console lines are counts and fixed words only (never page text, a reply body or an exception message).
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval import run_kaggle as rk

from . import config
from .agents import router
from .cache import CallCache

EXIT_LOCAL_DOWN = 69                                   # EX_UNAVAILABLE: the local server is down, nothing was started
EXIT_INTERRUPTED = 130                                 # runner.EXIT_INTERRUPTED (this module cannot import the runner)
LOCAL_ROLES = ("router", "grader", "decision_maker")
HEALTH_TIMEOUT_S = 2.0
# run.json `local.health`: ok | down_allowed (down, --allow-local-down) | skipped (no check needed: cache-only, no local role in use,
# or no llama.cpp target to check) | not_checked (runner.run called directly, without the CLI's check)
HEALTH_VALUES = ("ok", "down_allowed", "skipped", "not_checked")


class LocalLimiter:
    """The local model behind one shared semaphore. Forwards every attribute (`model`, `targets`, `budget`, `tracer`, `mock_mode`)."""

    def __init__(self, llm, slots: int | None = None, clock=time.perf_counter):
        # `config.LOCAL_SLOTS` is read here, not bound when the module is imported
        self._llm, self.slots, self._clock = llm, int(config.LOCAL_SLOTS if slots is None else slots), clock
        self._sem = threading.BoundedSemaphore(self.slots)
        self._lock = threading.Lock()
        self.in_flight = 0
        self.reset()

    @property
    def model(self):
        return self._llm.model

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._llm, name)

    def reset(self) -> None:
        with self._lock:
            self.calls, self.max_in_flight, self.wait_seconds = 0, self.in_flight, 0.0

    def stats(self) -> dict:
        with self._lock:
            return {"calls": self.calls, "max_in_flight": self.max_in_flight, "wait_seconds": round(self.wait_seconds, 3)}

    def _call(self, name: str, *args, **kw):
        t0 = self._clock()
        with self._sem:                                # released on every exit path, BaseException included
            waited = self._clock() - t0
            with self._lock:
                self.in_flight += 1
                self.calls += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
                self.wait_seconds += waited
            try:
                return getattr(self._llm, name)(*args, **kw)
            finally:
                with self._lock:
                    self.in_flight -= 1

    def structured(self, *args, **kw):
        return self._call("structured", *args, **kw)

    def chat(self, *args, **kw):
        return self._call("chat", *args, **kw)

    def chat_tools(self, *args, **kw):
        return self._call("chat_tools", *args, **kw)


def limited(llm) -> LocalLimiter:
    return llm if isinstance(llm, LocalLimiter) else LocalLimiter(llm)


def block(lim: LocalLimiter, health: str, prewarm: dict | None = None) -> dict:
    """run.json `local`: {slots, health, calls, max_in_flight, wait_seconds, prewarm}. Exact names (Task 9 reads them). `prewarm` is
    reserved, always null in run.json (ruling 12e): the prewarm pass is a separate command that writes only the cache, and
    `roles.router.cache_hits` carries the warm-cache signal."""
    return {"slots": lim.slots, "health": health, **lim.stats(), "prewarm": prewarm}


def format_block(b: dict) -> str:
    return (f"local server: health {b['health']}; calls {b['calls']}, max in flight {b['max_in_flight']} of {b['slots']} slots, "
            f"queued {b['wait_seconds']:.1f} s")


# ------------------------------------------------------------------------------------------------ health
def health_url(llm) -> str | None:
    """`<server>/health` of the model's first llama.cpp target (its base URL without the OpenAI `/v1`), or None when it has none."""
    for t in getattr(llm, "targets", None) or []:
        base = t.get("base_url") if isinstance(t, dict) and t.get("local") else None
        if isinstance(base, str) and base:
            base = base.rstrip("/")
            return (base[:-3] if base.endswith("/v1") else base) + "/health"
    return None


def http_get(url: str, timeout: float) -> tuple[int, str]:
    """(HTTP status, body) of one GET; the default getter (tests inject their own or patch this one)."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:         # noqa: S310  a configured loopback URL
            return resp.status, resp.read(4096).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""


def local_health(url: str, *, get=None, timeout: float = HEALTH_TIMEOUT_S) -> dict:
    """{"ok": bool, "detail": str}. One GET, no model call; `detail` never holds the reply body."""
    get = get or http_get
    try:
        status, body = get(url, timeout)
    except Exception as exc:                            # noqa: BLE001  unreachable, refused, timed out: down
        return {"ok": False, "detail": f"unreachable: {type(exc).__name__}"}
    if status != 200:
        return {"ok": False, "detail": f"http {status}"}
    try:
        ok = json.loads(body).get("status") == "ok"
    except (ValueError, AttributeError):
        return {"ok": False, "detail": "unexpected reply"}
    return {"ok": True, "detail": "ok"} if ok else {"ok": False, "detail": "not ready"}


def needs_check(cache_only: bool, ablate) -> bool:
    """R31(a): a run that can pay (not cache-only) and uses at least one local role."""
    return not cache_only and any(r in config.LOCAL_SCOPE and r not in ablate for r in LOCAL_ROLES)


def start_check(llm, *, cache_only: bool, ablate, allow_down: bool, say=print, get=None) -> str | None:
    """The run's `local.health` value, or None = refuse to start (the caller exits EXIT_LOCAL_DOWN before anything runs)."""
    if not needs_check(cache_only, ablate):
        say("local server health: skipped (" + ("cache-only run" if cache_only else "no local role in use") + ")")
        return "skipped"
    url = health_url(llm)
    if url is None:
        say("local server health: skipped (no llama.cpp target to check)")
        return "skipped"
    res = local_health(url, get=get)
    if res["ok"]:
        say("local server health: ok")
        return "ok"
    if allow_down:
        say(f"local server health: DOWN ({res['detail']}); --allow-local-down: the local roles fall back (router all families, "
            "grader unsure, decision abstain) and the run is not reproducible")
        return "down_allowed"
    say(f"local server health: DOWN ({res['detail']}); refusing to start (exit {EXIT_LOCAL_DOWN}): start it with "
        "scripts/serve_local_llm.sh, or pass --allow-local-down to run with the local fallbacks")
    return None


# ------------------------------------------------------------------------------------------------ prewarm
class PrewarmStopped(BaseException):
    """The prewarm pass is stopping: the case in flight ends at its next local call. A BaseException, so the Router's
    `except Exception` cannot count it as a failed page."""


class _Stoppable:
    """The limiter, refusing every call once `stop` is set."""

    def __init__(self, lim: LocalLimiter, stop: threading.Event):
        self._lim, self._stop = lim, stop

    @property
    def model(self):
        return self._lim.model

    def structured(self, *args, **kw):
        if self._stop.is_set():
            raise PrewarmStopped()
        return self._lim.structured(*args, **kw)


def prewarm(cases: dict, pages: dict, ids: list[str], lim: LocalLimiter, cache_root: Path, *, workers: int | None = None,
            clock=time.perf_counter, stop: threading.Event | None = None) -> dict:
    """Route every readable page of `ids` with the local model only: {cases, pages, hits, calls, failures, seconds}. `cases` and
    `pages` count the fully routed cases and their Router page lookups (as the graph makes them), `hits` the cached routings, `calls`
    the new local calls, `failures` the failed ones (never cached; the graph would tag those pages with every family). `workers`
    defaults to `config.LOCAL_SLOTS`. Once `stop` is set (a Ctrl-C here sets it) no case starts and no further local call is made."""
    t0 = clock()
    stop = stop if stop is not None else threading.Event()
    cache = CallCache(Path(cache_root))
    before = lim.stats()["calls"]
    guarded = _Stoppable(lim, stop)

    def one(cid: str) -> tuple[int, int] | None:
        if stop.is_set():
            return None                                 # never started
        recs = rk.triage(cases[cid], pages[cid])
        _, failures = router.tag_pages(guarded, cache, recs)
        return sum(1 for r in recs if rk.readable(r)), failures

    pool = ThreadPoolExecutor(max_workers=max(1, config.LOCAL_SLOTS if workers is None else workers))
    futs = [pool.submit(one, cid) for cid in ids]     # one by one, so a stop can leave the queued cases unstarted
    try:
        for f in futs:
            f.exception()                               # waits; a Ctrl-C lands here, on the main thread
    except KeyboardInterrupt:
        stop.set()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    got = []
    for f in futs:
        if f.cancelled():
            continue
        exc = f.exception()
        if exc is not None and not isinstance(exc, PrewarmStopped):
            raise exc                                   # a bug, as pool.map would have raised it
        if exc is None and f.result() is not None:
            got.append(f.result())
    return {"cases": len(got), "pages": sum(p for p, _ in got), "hits": cache.hits, "calls": lim.stats()["calls"] - before,
            "failures": sum(f for _, f in got), "seconds": round(clock() - t0, 3)}


def prewarm_cli(cases: dict, pages: dict, ids: list[str], ag, *, ablate=frozenset(), say=print, get=None,
                stop: threading.Event | None = None) -> int:
    """`run --prewarm-local`: the run's health check (`start_check`, one wording; a down server is exit EXIT_LOCAL_DOWN), then
    `prewarm`; 0 when done, EXIT_INTERRUPTED after a stop (a Ctrl-C, or `stop` set by the caller) with the partial counts."""
    if "router" in ablate or "router" not in config.LOCAL_SCOPE:
        say("prewarm: the router is ablated or not a local role: nothing to prewarm")
        return 0
    lim = limited(ag.local_llm)
    if start_check(lim, cache_only=False, ablate=ablate, allow_down=False, say=say, get=get) is None:
        return EXIT_LOCAL_DOWN
    stop = stop if stop is not None else threading.Event()
    p = prewarm(cases, pages, ids, lim, Path(ag.cache.root), stop=stop)
    counts = (f"pages {p['pages']}, cache hits {p['hits']}, new calls {p['calls']}, failures {p['failures']}, "
              f"{p['seconds']:.1f} s")
    if stop.is_set():
        say(f"prewarm INTERRUPTED (router only, local; no hosted call): cases {p['cases']} of {len(ids)} routed, {counts}; "
            f"exit {EXIT_INTERRUPTED} (the routed pages are cached: the same command again resumes)")
        return EXIT_INTERRUPTED
    say(f"prewarm (router only, local; no hosted call): cases {p['cases']}, {counts}")
    return 0
