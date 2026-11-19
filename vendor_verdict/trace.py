"""The handover trace of one case: which agent passed what to which agent, with tokens and seconds.

Summaries hold counts and ids only, never page text: `handover` refuses a free-text value, so a trace can be printed,
committed and shown in a presentation without leaking competition text."""
from __future__ import annotations

import threading

MAX_SUMMARY_STR = 80        # longest string a summary may hold: ids and field names fit, a quote or a sentence does not


def _check(value, path: str):
    """A JSON-safe copy of `value`, or ValueError if it holds a string too long to be an id."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > MAX_SUMMARY_STR:
            raise ValueError(f"trace summary {path}: a string of {len(value)} characters looks like free text (counts and ids only)")
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [_check(v, f"{path}[{i}]") for i, v in enumerate(items)]
    if isinstance(value, dict):
        return {str(k): _check(v, f"{path}.{k}") for k, v in value.items()}
    raise ValueError(f"trace summary {path}: unsupported type {type(value).__name__}")


class Trace:
    def __init__(self, case_id: str):
        self.case_id = case_id
        self._handovers: list[dict] = []
        self._lock = threading.Lock()                    # parallel specialists record into the same trace

    def __getstate__(self) -> dict:                      # the state dict is copied and checkpointed by the LangGraph engine; a Lock cannot be
        return {"case_id": self.case_id, "_handovers": self._handovers}

    def __setstate__(self, state: dict) -> None:
        self.case_id, self._handovers, self._lock = state["case_id"], state["_handovers"], threading.Lock()

    def handover(self, agent: str, to: str, summary: dict, tokens_in: int = 0, tokens_out: int = 0, seconds: float = 0.0) -> None:
        record = {"agent": agent, "to": to, "summary": _check(summary, "summary"),
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
            lines.append(f"{i}. {h['agent']} → {h['to']}: {counts} "
                         f"({h['tokens_in']} in / {h['tokens_out']} out tokens, {h['seconds']:.2f} s)")
        t = data["totals"]
        lines += ["", f"Total: {t['handovers']} handovers, {t['tokens_in']} in / {t['tokens_out']} out tokens, {t['seconds']:.2f} s"]
        return "\n".join(lines) + "\n"
