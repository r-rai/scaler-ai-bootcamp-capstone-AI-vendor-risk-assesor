"""Quota-window resilience for the paid (hosted MiniMax) calls: detection, a clean stop, an optional wait, an exact resume.

UNVERIFIED: the real wording of MiniMax's 5-hour quota-window error is not known. `is_quota_error` recognises the repo's existing
rate-limit rule (a 429) and a best-guess keyword list (`QUOTA_KEYWORDS`, ONE constant); neither has been checked against a real
MiniMax quota error. The first quota error's status, kind and clipped message are written to run.json (never to the console) so the
real text can be read after a run and the list corrected.

The agents keep their short retry ladder (4, 12, 36, 60 s), so a transient burst is absorbed exactly as before. Only when the ladder
is exhausted does a quota-kind error become `QuotaExhausted` (instead of a failed page or "planner failed"). Every successful call is
in the CallCache and a failed call never is, so a resume is the same command run again: the finished calls are cache hits.

`QuotaGate` is shared by the runner's case workers:
- STOP mode (default): the first `QuotaExhausted` trips the gate for good; no new case starts, every later paid call is refused at
  once (no provider call), and the runner stops the run cleanly (stopped_reason "quota", exit status 75).
- WAIT mode: a case that hits the quota is abandoned (nothing recorded) and its worker waits at the gate, as does every other worker.
  After each poll interval ONE worker is released as the probe and restarts its case from the top (all finished calls are cache
  hits). A probe case that ends without a quota error opens the gate for everyone; a probe that hits the quota re-arms the wait.
  Before each wait the gate checks that the total wait stays within the maximum; if not, it stops as in STOP mode.
  States: open -> closed (a trip) -> [one sleeper waits poll seconds] -> probing (one worker) -> open | closed (re-armed) | stopped.
"""
from __future__ import annotations

import contextvars
import threading
import time
from datetime import datetime, timezone

from assessor.llm import LLMError

# UNVERIFIED best guess at the words of a quota-window error (lowercase substrings). Not checked against a real MiniMax error.
QUOTA_KEYWORDS: tuple[str, ...] = ("quota", "usage limit", "limit exceeded", "insufficient balance")
DETAIL_CHARS = 200
# assessor/llm.py's repair-exhaustion error quotes the model's own (page-derived) output: it is never a quota error, and its text
# must never reach run.json.
_REPAIR_PREFIX = "Could not get valid "

OPEN, CLOSED, PROBING, STOPPED = "open", "closed", "probing", "stopped"


class QuotaExhausted(BaseException):
    """The hosted model's quota window is exhausted (after the retry ladder). A BaseException, like the runner's `RunStopped`, so no
    agent's `except Exception` / `except LLMError` fallback can turn it into a failed page or a "planner failed": it must reach the
    runner. Its `str()` is a fixed phrase; the provider's message is only in `detail` (clipped), which is written to run.json only."""

    def __init__(self, kind: str, status: int | None = None, detail: str = "", from_gate: bool = False):
        super().__init__("quota window exhausted")
        self.kind, self.status, self.detail, self.from_gate = kind, status, str(detail)[:DETAIL_CHARS], from_gate


def _status(exc: BaseException) -> int | None:
    seen = 0
    e: BaseException | None = exc
    while e is not None and seen < 5:                   # LLMError is raised `from` the SDK error, which carries the status
        code = getattr(e, "status_code", None)
        if isinstance(code, int):
            return code
        e, seen = e.__cause__, seen + 1
    return None


def is_quota_error(exc: BaseException) -> str | None:
    """"rate_limit" (the repo's rule: "429" in the message, "rate limit" in any case, or status 429), "keyword" (a QUOTA_KEYWORDS
    word in the lowercased message; UNVERIFIED) or None. Never for the runner's cache-only refusal or the repair-exhaustion error."""
    if type(exc).__name__ == "NotCached":
        return None
    msg = str(exc)
    if msg.startswith(_REPAIR_PREFIX):
        return None
    low = msg.lower()
    if "429" in msg or "rate limit" in low or _status(exc) == 429:
        return "rate_limit"
    if any(k in low for k in QUOTA_KEYWORDS):
        return "keyword"
    return None


def exhausted(exc: BaseException, kind: str) -> QuotaExhausted:
    return QuotaExhausted(kind, _status(exc), str(exc))


def call_with_ladder(fn, attempts: int, sleep):
    """`fn()` with the agents' retry ladder: a quota-kind LLMError is retried after 4, 12, 36, 60 s; after the last attempt it raises
    `QuotaExhausted`. Any other LLMError ends the attempts at once. Returns (result or None, the last LLMError or None)."""
    err, kind = None, None
    for attempt in range(attempts):
        try:
            return fn(), None
        except LLMError as exc:
            err, kind = exc, is_quota_error(exc)
            if kind is None:
                return None, err                        # not worth waiting for: the caller's ordinary failure
            if attempt < attempts - 1:
                sleep(min(60, 4 * 3 ** attempt))        # 4, 12, 36, 60 s
    raise exhausted(err, kind) from None                # from None: the provider text stays out of any printed traceback


_PROBE: contextvars.ContextVar = contextvars.ContextVar("vendor_verdict_quota_probe", default=None)


class QuotaGate:
    """Thread-safe gate shared by the case workers (see the module docstring). `clock` and `sleep` are injectable; the default sleep
    waits on `stop` (the run's stop event), so a Ctrl-C (which sets it through the runner's halt) ends a wait at once."""

    def __init__(self, *, wait: bool = False, poll_seconds: float = 600.0, max_seconds: float = 6 * 3600.0,
                 stop: threading.Event | None = None, clock=None, sleep=None, abort_exc: type[BaseException] = RuntimeError):
        self.wait_mode, self.poll_seconds, self.max_seconds = bool(wait), float(poll_seconds), float(max_seconds)
        self._stop = stop or threading.Event()
        self._clock = clock or time.monotonic
        self._sleep = sleep or self._stop.wait
        self._abort_exc = abort_exc
        self._cond = threading.Condition()
        self.state = OPEN
        self.first: dict | None = None
        self.trips = self.probes = self.refused = 0
        self.waited_seconds = 0.0
        self._sleeper = False
        self._aborted = False
        self._probe_id: object | None = None
        self._inflight = 0                                  # admitted paid calls not yet returned

    # ---- observation
    @property
    def tripped(self) -> bool:
        return self.first is not None

    @property
    def stopped(self) -> bool:
        return self.state == STOPPED

    def _is_probe(self) -> bool:
        return self._probe_id is not None and _PROBE.get() is self._probe_id

    # ---- the paid-call check (GuardedLLM)
    def admit(self):
        """Context manager around one paid call: refused (QuotaExhausted from the gate, no provider call) unless the gate is open or
        this is the probe's call."""
        gate = self

        class _Admit:
            def __enter__(self_inner):
                with gate._cond:
                    if not (gate.state == OPEN or (gate.state == PROBING and gate._is_probe())):
                        gate.refused += 1
                        raise gate._replay()
                    gate._inflight += 1
                    return self_inner

            def __exit__(self_inner, *exc):
                with gate._cond:
                    gate._inflight -= 1
                    gate._cond.notify_all()
                return False
        return _Admit()

    def _replay(self) -> QuotaExhausted:
        f = self.first or {"kind": "unknown", "status": None}
        return QuotaExhausted(f["kind"], f["status"], "", from_gate=True)

    # ---- transitions
    def trip(self, exc: QuotaExhausted, probe: bool = False) -> None:
        with self._cond:
            if not exc.from_gate:
                self.trips += 1
                if self.first is None:
                    self.first = {"kind": exc.kind, "status": exc.status, "detail": exc.detail,
                                  "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if self.state == STOPPED:
                return
            if not self.wait_mode:
                self.state = STOPPED
            elif self.state == OPEN or (self.state == PROBING and probe):
                self.state, self._probe_id = CLOSED, None       # a failed probe re-arms the wait
            self._cond.notify_all()

    def abort(self) -> None:
        """The run is stopping (interrupt, budget, give-up): wake every waiter."""
        with self._cond:
            self._aborted = True
            self._cond.notify_all()

    def _enter(self) -> str:
        """"run", "probe" or "stop" (the gate stopped or the run is aborting)."""
        with self._cond:
            while True:
                if self._aborted or self.state == STOPPED:
                    return "stop"
                if self.state == OPEN:
                    return "run"
                if self.state == CLOSED and not self._sleeper:
                    if self.waited_seconds + self.poll_seconds > self.max_seconds:
                        self.state = STOPPED                    # the total wait would pass --quota-max-hours
                        self._cond.notify_all()
                        return "stop"
                    self._sleeper = True
                    # a call admitted before the trip may still be at the provider: let it return first, so the wait (and
                    # the probe after it) never overlaps another paid call
                    while self._inflight and not self._aborted:
                        self._cond.wait()
                    if self._aborted:
                        self._sleeper = False
                        self._cond.notify_all()
                        return "stop"
                    break
                self._cond.wait()
        t0 = self._clock()
        try:
            self._sleep(self.poll_seconds)
        finally:
            with self._cond:
                self.waited_seconds += max(0.0, self._clock() - t0)
                self._sleeper = False
                self._cond.notify_all()
        with self._cond:
            if self._aborted:
                return "stop"
            if self.state != CLOSED:
                return "again"                          # cannot happen while one sleeper holds the wait; look again
            self.state, self._probe_id = PROBING, object()
            self.probes += 1
            return "probe"

    def run_case(self, fn):
        """Run one case under the gate: wait for it to open (WAIT mode), restart the case from the top after a quota error in WAIT
        mode, re-raise it in STOP mode. A case never started because the gate stopped raises the replayed QuotaExhausted (the
        runner counts the case as unfinished); one stopped by an abort raises `abort_exc`."""
        while True:
            turn = self._enter()
            if turn == "again":
                continue
            if turn == "stop":
                if self._aborted and self.state != STOPPED:
                    raise self._abort_exc()
                raise self._replay()
            probe = turn == "probe"
            token = _PROBE.set(self._probe_id if probe else None)
            opens = False                       # a probe opens the gate when its case ends without a quota error and without a stop
            try:
                out = fn()
                opens = True
            except QuotaExhausted as exc:
                self.trip(exc, probe=probe)     # a failed probe re-arms the wait
                if not self.wait_mode or self.state == STOPPED:
                    raise
                continue
            except Exception:
                opens = True                    # the case failed for another reason (a bug): the provider was not the problem
                raise
            finally:
                _PROBE.reset(token)
                if probe and opens:
                    self._open_after_probe()
            return out

    def _open_after_probe(self) -> None:
        """The probe's case ended without a quota error (finished, or failed for another reason): the window is open again."""
        with self._cond:
            if self.state == PROBING:
                self.state, self._probe_id = OPEN, None
                self._cond.notify_all()

    # ---- run.json
    def block(self) -> dict | None:
        if self.first is None:
            return None
        f = self.first
        return {"tripped": True, "kind": f["kind"], "status": f["status"], "first_error": f["detail"], "first_hit_utc": f["utc"],
                "wait_mode": self.wait_mode, "probes": self.probes, "waited_seconds": round(self.waited_seconds, 3),
                "trips": self.trips, "refused_calls": self.refused, "poll_seconds": self.poll_seconds,
                "max_wait_seconds": self.max_seconds, "gave_up": self.wait_mode and self.state == STOPPED}
