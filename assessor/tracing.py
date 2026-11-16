"""
tracing.py — a trace of every step, and a budget that can stop the run.

From the Deep Search course: Module 9 (Cost, latency and budgets) and the `tracing.py` / `budget.py`
pair in the code pack. Production RAG course Module 12 ("Observability") covers the same idea for RAG.

Why both exist:
* A trace (one JSON line per event) is how you debug a wrong verdict: you read backwards from the verdict
  to the evidence to the page. The report alone never tells you WHY.
* A budget (tokens, USD, wall-clock) is how you stop an agent from spending without limit when the web
  is slow or a model loops.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import observability as obs
from .config import RUNS_DIR, SETTINGS
from .errors import mask_secrets


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token). Replace with tiktoken for exact counts (Production RAG Lab 13)."""
    return max(1, len(text) // 4)


class BudgetExceeded(RuntimeError):
    pass


def price_for(model: str = "", role: str = "main", local: bool = False) -> tuple[float, float]:
    """(input, output) USD per million tokens. Local models are free; the small role may have its own price."""
    if local:
        return 0.0, 0.0
    if role == "small" or (model and model == SETTINGS.llm_small_model):
        return SETTINGS.price_small_in_per_mtok, SETTINGS.price_small_out_per_mtok
    return SETTINGS.price_in_per_mtok, SETTINGS.price_out_per_mtok


@dataclass
class Budget:
    """Tokens, money, time and tool calls for ONE run (Deep Search course, budget.py).

    Every LLM call and tool call is charged here. `exhausted()` is the cooperative check the research
    loop and the LLM wrapper make before each step; `charge()` also raises once the USD cap is passed,
    so no code path can keep spending by forgetting to check. All methods are thread-safe: research
    workers charge the same Budget in parallel.
    """

    max_usd: float = field(default_factory=lambda: SETTINGS.max_usd)
    max_tokens: int = field(default_factory=lambda: SETTINGS.max_tokens)
    max_seconds: float = field(default_factory=lambda: SETTINGS.max_seconds)
    max_tool_calls: int = field(default_factory=lambda: SETTINGS.max_tool_calls)
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    started: float = field(default_factory=time.time)
    by_stage: dict[str, dict[str, float]] = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)      # one record per LLM call, persisted to llm_usage
    saved_usd: float = 0.0                               # cloud cost avoided by caches and local models
    tools: dict[str, int] = field(default_factory=dict)  # "search:tavily" -> calls (free-tier quota tracking)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # -- charging -----------------------------------------------------------
    def _stage(self, stage: str) -> dict[str, float]:
        return self.by_stage.setdefault(stage, {"llm_calls": 0, "tokens": 0, "usd": 0.0, "tool_calls": 0})

    def charge(self, input_tokens: int, output_tokens: int, *, model: str = "", role: str = "main",
               stage: str = "other", provider: str = "", local: bool = False, cached: bool = False) -> None:
        p_in, p_out = price_for(model, role, local or cached)
        cost = (input_tokens * p_in + output_tokens * p_out) / 1_000_000
        if local or cached:   # what the same call would have cost on the cloud model
            c_in, c_out = price_for(model, role, False)
            avoided = (input_tokens * c_in + output_tokens * c_out) / 1_000_000
        else:
            avoided = 0.0
        with self._lock:
            self.llm_calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.spent_usd += cost
            self.saved_usd += avoided
            st = self._stage(stage.split("#")[0])
            st["llm_calls"] += 1
            st["tokens"] += input_tokens + output_tokens
            st["usd"] += cost
            self.calls.append({"stage": stage.split("#")[0], "model": model, "role": role, "provider": provider,
                               "input_tokens": input_tokens, "output_tokens": output_tokens, "usd": round(cost, 6),
                               "avoided_usd": round(avoided, 6), "local": local, "cached": cached})
            over = self.spent_usd > self.max_usd
        if over:
            raise BudgetExceeded(f"Budget of ${self.max_usd:.2f} exceeded (spent ${self.spent_usd:.3f}). Raise VRA_MAX_USD or reduce pages.")

    def charge_tool(self, stage: str = "tools", provider: str = "") -> None:
        with self._lock:
            self.tool_calls += 1
            self._stage(stage)["tool_calls"] += 1
            key = f"{stage}:{provider}"
            self.tools[key] = self.tools.get(key, 0) + 1

    # -- checking -----------------------------------------------------------
    def restart_clock(self) -> None:
        """A Budget can outlive one question (a shared Pipeline); the wall clock is per run."""
        with self._lock:
            self.started = time.time()

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def exhausted(self) -> str | None:
        """A human-readable reason if any limit is hit, else None."""
        if self.spent_usd >= self.max_usd:
            return f"cost limit reached (${self.spent_usd:.4f} of ${self.max_usd:.2f})"
        if self.max_tokens and self.total_tokens >= self.max_tokens:
            return f"token limit reached ({self.total_tokens:,} of {self.max_tokens:,})"
        if self.max_seconds and self.elapsed_s >= self.max_seconds:
            return f"time limit reached ({self.elapsed_s:.0f}s of {self.max_seconds:.0f}s)"
        if self.max_tool_calls and self.tool_calls >= self.max_tool_calls:
            return f"tool-call limit reached ({self.tool_calls} of {self.max_tool_calls})"
        return None

    def check(self) -> None:
        reason = self.exhausted()
        if reason:
            raise BudgetExceeded(f"Budget stop: {reason}.")

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "tool_calls": self.tool_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": round(self.spent_usd, 5), "saved_usd": round(self.saved_usd, 5),
                "seconds": round(self.elapsed_s, 2),
                "by_stage": {k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in self.by_stage.items()}}


class Tracer:
    """Writes runs/<timestamp>-<name>/trace.jsonl. Use `t.event(stage, **data)` liberally."""

    def __init__(self, name: str = "assess", root: Path | None = None, enabled: bool = True):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        # Random suffix: two API requests in the same second must not share a run folder.
        self.run_id = f"{stamp}-{name}-{secrets.token_hex(3)}"
        self.dir = (root or RUNS_DIR) / self.run_id
        self.enabled = enabled
        self.events: list[dict] = []
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.dir / "trace.jsonl", "a", encoding="utf-8")
        else:
            self._fh = None
        self._stage_start: dict[str, float] = {}
        self._lock = threading.Lock()

    def event(self, stage: str, **data: Any) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "stage": stage, **data}
        # A trace is read by people and shipped to log storage: mask anything key-shaped before it is written.
        line = mask_secrets(json.dumps(rec, default=str, ensure_ascii=False))
        with self._lock:
            self.events.append(rec)
            if self._fh:
                self._fh.write(line + "\n")
                self._fh.flush()
        obs.on_trace_event(self.run_id, stage, data)
        if SETTINGS.verbose:
            short = {k: v for k, v in data.items() if k not in ("text",)}
            print(f"  [trace] {stage}: {json.dumps(short, default=str, ensure_ascii=False)[:160]}")

    def start(self, stage: str) -> None:
        self._stage_start[stage] = time.time()

    def end(self, stage: str, **data: Any) -> None:
        t0 = self._stage_start.pop(stage, None)
        self.event(stage, duration_ms=round((time.time() - t0) * 1000) if t0 else None, **data)

    def write_json(self, name: str, obj: Any) -> Path | None:
        if not self.enabled:
            return None
        p = self.dir / name
        p.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return p

    def write_text(self, name: str, text: str) -> Path | None:
        if not self.enabled:
            return None
        p = self.dir / name
        p.write_text(text, encoding="utf-8")
        return p

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
