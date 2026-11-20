"""The handover trace of one case: which agent passed what to which agent, with seconds (tokens: see the last paragraph).

Summaries are meant to hold counts and ids only, never page text. `handover` enforces that on a best-effort basis and never
raises: a string longer than MAX_SUMMARY_STR is replaced by a `<text:N chars>` placeholder and an odd type is coerced or replaced
by a `<TypeName>` placeholder, so a stray model-derived value can neither leak competition text into a trace nor crash a run.
This is a heuristic, not a guarantee: a short quote still passes.

Tokens per handover are NOT recorded by the runtime (ruling R56): parallel cases share the global per-role counters, so a per-node
delta would need new plumbing. Every handover therefore carries tokens_in = tokens_out = 0 in the JSON (the fields stay), the
Markdown prints seconds only for a handover (and a total) with no recorded tokens, and the per-role token totals are in run.json
`roles` (estimates for cached calls)."""
from __future__ import annotations

import enum
import numbers
import os
import threading
from decimal import Decimal

MAX_SUMMARY_STR = 80        # longest string a summary keeps as is: ids and field names fit, a quote or a sentence does not
_MAX_DEPTH = 6              # deeper nesting (or a self-referencing container) is cut off


def _clean_str(text: str) -> str:
    return text if len(text) <= MAX_SUMMARY_STR else f"<text:{len(text)} chars>"


def _clean(value, depth: int = 0):
    """A JSON-safe copy of `value` that holds no long string. Never raises for any value a caller can pass."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _clean_str(value)
    if isinstance(value, numbers.Integral):                      # int, numpy integers
        return int(value)
    if isinstance(value, (numbers.Real, Decimal)):               # float, numpy floats, Decimal
        return float(value)
    if isinstance(value, os.PathLike):
        return _clean_str(os.fspath(value))
    if isinstance(value, enum.Enum):
        return _clean(value.value, depth + 1)
    if depth >= _MAX_DEPTH:
        return "<nested>"
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [_clean(v, depth + 1) for v in items]
    if isinstance(value, dict):
        return {_clean_str(str(k)): _clean(v, depth + 1) for k, v in value.items()}
    return f"<{type(value).__name__}>"


class Trace:
    def __init__(self, case_id: str):
        self.case_id = case_id
        self._handovers: list[dict] = []
        self._lock = threading.Lock()                    # parallel specialists record into the same trace

    def __getstate__(self) -> dict:                      # a Lock can be neither deep-copied nor pickled: leave it out and rebuild it
        return {"case_id": self.case_id, "_handovers": self._handovers}

    def __setstate__(self, state: dict) -> None:
        self.case_id, self._handovers, self._lock = state["case_id"], state["_handovers"], threading.Lock()

    def handover(self, agent: str, to: str, summary: dict, tokens_in: int = 0, tokens_out: int = 0, seconds: float = 0.0) -> None:
        try:
            clean = _clean(summary)
        except Exception:                                # a summary that cannot even be read must not fail the node that made it
            clean = {"summary": "<unrecordable>"}
        record = {"agent": agent, "to": to, "summary": clean,
                  "tokens_in": tokens_in, "tokens_out": tokens_out, "seconds": seconds}
        with self._lock:
            self._handovers.append(record)

    def to_json(self) -> dict:
        with self._lock:
            handovers = [dict(h) for h in self._handovers]
        return {"case_id": self.case_id, "handovers": handovers,
                "totals": {"handovers": len(handovers), "tokens_in": sum(h["tokens_in"] for h in handovers),
                           "tokens_out": sum(h["tokens_out"] for h in handovers), "seconds": sum(h["seconds"] for h in handovers)}}

    def to_markdown(self) -> str:
        data = self.to_json()
        lines = [f"# Handover trace: {self.case_id}", ""]
        for i, h in enumerate(data["handovers"], 1):
            counts = ", ".join(f"{k}={v}" for k, v in h["summary"].items()) or "-"
            lines.append(f"{i}. {h['agent']} → {h['to']}: {counts} ({_cost(h)})")
        t = data["totals"]
        lines += ["", f"Total: {t['handovers']} handovers, {_cost(t)}"]
        return "\n".join(lines) + "\n"


def _cost(rec: dict) -> str:
    """"N in / M out tokens, S s", or "S s" alone when no token was recorded (R56: the runtime records none per handover)."""
    seconds = f"{rec['seconds']:.2f} s"
    if not (rec["tokens_in"] or rec["tokens_out"]):
        return seconds
    return f"{rec['tokens_in']} in / {rec['tokens_out']} out tokens, {seconds}"
