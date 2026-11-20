"""
llm.py — one small wrapper around "an LLM", copied in spirit from the Deep Search course (deepsearch/llm.py).

Two modes:
* mock (default): no network. Every call site passes a `mock=` function that produces the answer with
  simple rules, so the whole project runs offline and the tests are repeatable.
* live: any OpenAI-compatible provider (including MiniMax at https://api.minimax.io/v1). Set LLM_PROVIDER in .env.

Roles and routing (live mode):
* main    — planning, query generation, grounded policy answers      (LLM_PROVIDER)
* small   — routing, grading, identity gate, injection classifier    (LLM_SMALL_PROVIDER, e.g. llamacpp)
* extract — claim extraction from web pages                          (LLM_EXTRACT_PROVIDER)
A role served by a local provider falls back to the main provider's small model when the local server
is down or slow; a circuit breaker stops every call from waiting on a dead server. Every call is charged
to the shared Budget with its model, role and stage, which is what the cost dashboard reads.

The method that matters is `structured()`: ask for JSON, validate it with a Pydantic schema, and if
validation fails send the error back and ask the model to repair its output. This validate → repair loop
is what turns fragile LLM output into a contract (Deep Search Module 4, Lab 2).
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from typing import Callable, Type, TypeVar

from pydantic import ValidationError

from .config import SETTINGS
from .schemas import BaseModel
from .tracing import Budget, Tracer, estimate_tokens

class LLMError(RuntimeError):
    pass


REASONING_EFFORT_PROVIDERS = frozenset({"minimax", "openai", "openrouter"})   # accept reasoning_effort
BREAKER_THRESHOLD = 3      # consecutive failures before a provider is skipped
BREAKER_COOLDOWN_S = 60.0
HEDGE_MIN_LEFT_S = 6.0     # a duplicate that cannot finish before the run's deadline is pure waste
_BREAKERS: dict[str, dict[str, float]] = {}
_CLIENTS: dict[tuple[str, str], object] = {}
_STATE_LOCK = threading.Lock()
_HEDGE_TLS = threading.local()   # `_create` leaves the hedge details of the call it just made for `_complete` (same thread)


def breaker_open(provider: str) -> bool:
    with _STATE_LOCK:
        b = _BREAKERS.get(provider)
        return bool(b and b["open_until"] > time.time())


def _breaker_record(provider: str, ok: bool) -> None:
    with _STATE_LOCK:
        b = _BREAKERS.setdefault(provider, {"fails": 0, "open_until": 0.0})
        if ok:
            b["fails"], b["open_until"] = 0, 0.0
        else:
            b["fails"] += 1
            if b["fails"] >= BREAKER_THRESHOLD:
                b["open_until"] = time.time() + BREAKER_COOLDOWN_S


def reset_breakers() -> None:
    with _STATE_LOCK:
        _BREAKERS.clear()


def _client(cfg: dict):
    """One OpenAI-compatible client per endpoint, shared across threads and requests (connection reuse)."""
    key = (cfg["base_url"], cfg["api_key"])
    with _STATE_LOCK:
        if key not in _CLIENTS:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError("Live mode needs the openai package: pip install openai") from exc
            # The SDK retries connection errors, 408/429 and 5xx with exponential backoff and jitter.
            _CLIENTS[key] = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=SETTINGS.llm_timeout_s,
                                   max_retries=0 if cfg.get("local") else 2)
        return _CLIENTS[key]

T = TypeVar("T", bound=BaseModel)


def clean_reasoning(text: str) -> str:
    """Strip <think>...</think> blocks emitted by reasoning models (DeepSeek-R1, MiniMax-M3, Qwen3).

    Note: MiniMax-M3 with thinking enabled can occasionally emit opening JSON/text tokens before
    closing `</think>`. If stripping `<think>...</think>` removes `{` and leaves a dangling JSON body,
    we inspect inside `<think>` as a fallback during JSON extraction.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return cleaned or text.strip()


def _extract_json(text: str) -> str:
    # First try after stripping <think> blocks
    for candidate in (clean_reasoning(text), text.strip()):
        fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", candidate, re.S)
        if fence:
            return fence.group(1).strip()
        if (candidate.startswith("{") and candidate.endswith("}")) or (
            candidate.startswith("[") and candidate.endswith("]")
        ):
            return candidate
        # Find balanced or outermost JSON object
        first, last = candidate.find("{"), candidate.rfind("}")
        if first != -1 and last > first:
            snippet = candidate[first : last + 1]
            # Remove any accidental </think> tag inside a split reasoning output
            snippet_clean = re.sub(r"</?think>", "", snippet).strip()
            try:
                json.loads(snippet_clean)
                return snippet_clean
            except Exception:  # noqa: BLE001
                return snippet
    return text.strip()


class LLM:
    def __init__(self, budget: Budget | None = None, tracer: Tracer | None = None, model: str | None = None, role: str = "main",
                 provider: str | None = None):
        self.budget = budget or Budget()
        self.tracer = tracer
        self.role = role
        self.provider = (provider or SETTINGS.provider_for(role)).lower()
        if SETTINGS.llm_provider == "mock" and provider is None:
            self.provider = "mock"
        self.mock_mode = self.provider == "mock"
        self.targets: list[dict] = []          # ordered: preferred endpoint first, then fallbacks
        if self.mock_mode:
            self.model = "mock-llm"
            self.local = False
        else:
            cfg = SETTINGS.resolved_llm(role=role, provider=self.provider)
            if model:
                cfg["model"] = model
            self.targets.append(cfg)
            # A non-main role on another provider (typically a local server) can fall back to the main provider.
            if SETTINGS.llm_fallback and provider is None and self.provider != SETTINGS.llm_provider and SETTINGS.llm_provider != "mock":
                try:
                    self.targets.append(SETTINGS.resolved_llm(role="small", provider=SETTINGS.llm_provider))
                except ValueError:
                    pass
            self.model = cfg["model"]
            self.local = bool(cfg.get("local"))

    @classmethod
    def small(cls, budget: Budget | None = None, tracer: Tracer | None = None) -> "LLM":
        """A cheaper, faster model for high-volume decisions: routing, grading, identity, injection classification."""
        return cls(budget=budget, tracer=tracer, role="small")

    @classmethod
    def local_only(cls, budget: Budget | None = None, tracer: Tracer | None = None) -> "LLM":
        """For text that must not leave this machine (restricted policy clauses): small role, local endpoints only,
        and no cloud fallback target. With no local model configured it is a mock-mode LLM, so callers use their
        deterministic rule fallbacks instead."""
        llm = cls(budget=budget, tracer=tracer, role="small")
        llm.targets = [t for t in llm.targets if t.get("local")]
        if not llm.targets:
            llm.provider, llm.mock_mode, llm.model, llm.local = "mock", True, "mock-llm", False
        return llm

    @classmethod
    def extractor(cls, budget: Budget | None = None, tracer: Tracer | None = None) -> "LLM":
        """The model that reads web pages and returns typed claims (LLM_EXTRACT_PROVIDER)."""
        return cls(budget=budget, tracer=tracer, role="extract")

    @classmethod
    def for_bugfix(cls, budget: Budget | None = None, tracer: Tracer | None = None) -> "LLM":
        """The bug-fix agent's model. BUGFIX_LLM_PROVIDER / BUGFIX_LLM_MODEL override the main settings."""
        return cls(budget=budget or Budget(max_usd=SETTINGS.bugfix_max_usd, max_seconds=0, max_tool_calls=0), tracer=tracer,
                   model=SETTINGS.bugfix_model or None, role="main", provider=SETTINGS.bugfix_provider or None)

    # ------------------------------------------------------------------ transport
    def _trace(self, stage: str, **data) -> None:
        if self.tracer:
            self.tracer.event(stage, **data)

    def _create_once(self, cfg: dict, messages: list[dict], *, json_mode: bool = False, schema: dict | None = None,
                     schema_name: str = "output", tools: list[dict] | None = None, retries: int | None = None):
        """ONE request to one endpoint (the SDK may retry it inside its own limits). `retries` can only lower those limits."""
        kwargs: dict = {"model": cfg["model"], "messages": messages}
        if tools:
            kwargs["tools"], kwargs["tool_choice"] = tools, "auto"
        if SETTINGS.llm_temperature is not None:
            temp = SETTINGS.llm_temperature
            # Older MiniMax gateways reject temperature 0.0.
            if cfg["provider"] == "minimax" and temp <= 0.0:
                temp = 0.01
            kwargs["temperature"] = temp
        effort = SETTINGS.llm_small_reasoning_effort if self.role in ("small", "extract") else SETTINGS.llm_reasoning_effort
        if effort and cfg["provider"] in REASONING_EFFORT_PROVIDERS:
            kwargs["reasoning_effort"] = effort
        if schema is not None and cfg["provider"] == "llamacpp":
            # llama.cpp compiles the JSON Schema to a grammar: the small model cannot emit invalid JSON.
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}}
        elif json_mode and cfg["provider"] != "ollama":
            kwargs["response_format"] = {"type": "json_object"}
        client = _client(cfg)
        timeout, sdk_retries = self._call_limits(cfg)
        if retries is not None:
            sdk_retries = min(sdk_retries, retries)
        if (timeout, sdk_retries) != (SETTINGS.llm_timeout_s, 0 if cfg.get("local") else 2):
            client = client.with_options(timeout=timeout, max_retries=sdk_retries)
        try:
            return client.chat.completions.create(**kwargs)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            # Only a provider that REJECTS an optional parameter (HTTP 400/422) gets the bare retry. A timeout, a
            # dropped connection, a 429 or a 5xx must not: the SDK has already retried those, and a second call
            # without `reasoning_effort` lets a thinking model reason without limit (measured: 6 s -> 48 s).
            if getattr(exc, "status_code", None) not in (400, 422) or (
                    "response_format" not in kwargs and "reasoning_effort" not in kwargs):
                raise
            self._trace("llm_param_fallback", provider=cfg["provider"], error=str(exc)[:160],
                        dropped=[k for k in ("response_format", "reasoning_effort") if k in kwargs])
            kwargs.pop("response_format", None)
            kwargs.pop("reasoning_effort", None)
            return client.chat.completions.create(**kwargs)  # type: ignore[attr-defined]

    def _create(self, cfg: dict, messages: list[dict], **kw):
        """The response of one logical request (see `_send`). This is the one seam every provider call goes through (the
        egress tests replace it to prove restricted text never reaches a cloud call), so `_complete` calls it and nothing
        else; the details of a hedge travel to `_complete` through a thread-local, because the response is all it returns."""
        resp, hedge = self._send(cfg, messages, **kw)
        _HEDGE_TLS.info = hedge
        return resp

    def _hedge_after(self, cfg: dict) -> float:
        """Seconds after which this call gets a duplicate request; 0 = this call is never hedged. Only the page reader on
        a cloud endpoint is: the local server has one GPU, and main-role calls are legitimately long."""
        after = SETTINGS.llm_hedge_after_s
        if after <= 0 or self.role != "extract" or cfg.get("local"):
            return 0.0
        b = self.budget
        if getattr(b, "max_seconds", 0) and b.max_seconds - b.elapsed_s - after < HEDGE_MIN_LEFT_S:
            return 0.0
        return after

    def _send(self, cfg: dict, messages: list[dict], **kw) -> tuple[object, dict | None]:
        """One logical request to one endpoint: (response, hedge info or None).

        A call still running after `_hedge_after` seconds gets ONE duplicate request and the first good answer wins. Measured
        on 283 live extract calls, the slow ones are provider stalls (4-50 tokens/s against about 109 for a normal call), so
        a second request usually answers in a few seconds. Rules: a failure BEFORE the hedge time is raised as it always was
        (the hedge is for slow calls, not a hidden retry); the duplicate has no SDK retries of its own, so one call can never
        become four requests; both attempts run in daemon threads so an abandoned one cannot keep the process alive; and the
        whole call is bounded even if the transport never answers."""
        after = self._hedge_after(cfg)
        if not after:
            return self._create_once(cfg, messages, **kw), None
        timeout, retries = self._call_limits(cfg)
        cap = (retries + 1) * timeout + after + 1.0
        b = self.budget
        if getattr(b, "max_seconds", 0):
            cap = min(cap, max(3.0, b.max_seconds - b.elapsed_s) + 1.0)
        box: queue.Queue = queue.Queue()

        def attempt(name: str, sdk_retries: int | None) -> None:
            try:
                box.put((name, True, self._create_once(cfg, messages, retries=sdk_retries, **kw)))
            except Exception as exc:  # noqa: BLE001 — handed to the waiting caller
                box.put((name, False, exc))

        def launch(name: str, sdk_retries: int | None) -> None:
            threading.Thread(target=attempt, args=(name, sdk_retries), daemon=True, name=f"llm-attempt-{name}").start()

        t0 = time.monotonic()
        launch("first", None)
        hedged = False
        failed: dict[str, Exception] = {}
        while True:
            wait = (cap if hedged else after) - (time.monotonic() - t0)
            try:
                name, ok, value = box.get(timeout=max(wait, 0.0))
            except queue.Empty:
                if hedged:
                    raise TimeoutError(f"LLM call to {cfg['model']} got no answer within {cap:.0f}s") from None
                hedged = True
                launch("hedge", 0)
                continue
            if ok:
                return value, ({"winner": name, "after_ms": round(after * 1000), "wall_ms": round((time.monotonic() - t0) * 1000)}
                               if hedged else None)
            failed[name] = value
            if not hedged or len(failed) == 2:
                raise failed["first"]

    def _account_hedge(self, resp, cfg: dict, messages: list[dict], stage: str, info: dict) -> None:
        """The abandoned request is billed by the provider although nobody reads it: charge it as an ESTIMATE (same prompt,
        output assumed equal to the winner's) so the Budget can never under-report, and trace it as such."""
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) or estimate_tokens(json.dumps(messages, default=str))
        out_tok = getattr(usage, "completion_tokens", None) or 0
        self._trace("llm_hedge", llm_stage=stage, winner=info["winner"], after_ms=info["after_ms"], wall_ms=info["wall_ms"],
                    abandoned_input_tokens=in_tok, abandoned_output_tokens=out_tok, estimated=True)
        self.budget.charge(in_tok, out_tok, model=cfg["model"], role=self.role, stage=f"{stage}#hedge",
                           provider=cfg["provider"], local=False)

    def _call_limits(self, cfg: dict) -> tuple[float, int]:
        """HTTP timeout and SDK retries for ONE call: bounded by the role's cap and by the time the run has left,
        so a stalled provider can never hold a worker thread past the deadline."""
        timeout = SETTINGS.llm_timeout_s
        if self.role in ("small", "extract"):
            timeout = min(timeout, SETTINGS.llm_small_timeout_s)
        retries = 0 if cfg.get("local") else 2
        b = self.budget
        if getattr(b, "max_seconds", 0):
            left = b.max_seconds - b.elapsed_s
            timeout = min(timeout, max(3.0, left - 1.0))
            if self.role in ("small", "extract"):
                # These run inside the research deadline, where a stalled call is expensive. Main-role calls keep their
                # retries: a 429, 5xx or dropped connection costs about a second to retry, not a full timeout.
                while retries and (retries + 1) * timeout > max(left, 0):
                    retries -= 1
        return timeout, retries

    def _complete(self, messages: list[dict], *, stage: str, **kw):
        """Try each target in order, skipping providers whose circuit breaker is open."""
        self.budget.check()   # cooperative stop: never start a call the budget cannot pay for
        last: Exception | None = None
        for i, cfg in enumerate(self.targets):
            if breaker_open(cfg["provider"]) and i < len(self.targets) - 1:
                self._trace("llm_fallback", llm_stage=stage, skipped=cfg["provider"], reason="circuit open")
                continue
            t0 = time.time()
            _HEDGE_TLS.info = None
            try:
                resp = self._create(cfg, messages, **kw)
                _breaker_record(cfg["provider"], True)
                ms = round((time.time() - t0) * 1000)
            except Exception as exc:  # noqa: BLE001
                _breaker_record(cfg["provider"], False)
                last = exc
                if i < len(self.targets) - 1:
                    self._trace("llm_fallback", llm_stage=stage, failed=cfg["provider"], error=str(exc)[:160],
                                next=self.targets[i + 1]["provider"])
            else:
                hedge, _HEDGE_TLS.info = getattr(_HEDGE_TLS, "info", None), None
                if hedge:       # outside the try: a spent budget must stop the run, not be mistaken for a provider failure
                    self._account_hedge(resp, cfg, messages, stage, hedge)
                return resp, cfg, ms
        raise LLMError(f"LLM call failed at stage {stage!r}: {last}") from last

    def _account(self, resp, cfg: dict, ms: int, *, stage: str, in_est: int, raw: str, **extra) -> None:
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) or in_est
        out_tok = getattr(usage, "completion_tokens", None) or estimate_tokens(raw)
        self._trace("llm", llm_stage=stage, model=cfg["model"], provider=cfg["provider"], role=self.role,
                    input_tokens=in_tok, output_tokens=out_tok, duration_ms=ms, **extra)
        self.budget.charge(in_tok, out_tok, model=cfg["model"], role=self.role, stage=stage, provider=cfg["provider"],
                           local=bool(cfg.get("local")))

    # ------------------------------------------------------------------
    def chat_tools(self, messages: list[dict], tools: list[dict], *, stage: str = "tools",
                   mock: Callable[[], dict] | None = None) -> dict:
        """One tool-calling turn. Returns {"content", "tool_calls": [{id,name,arguments(str)}], "message"}.

        `message` is the assistant message to append to history verbatim (tool-call continuity).
        In mock mode `mock()` returns {"content": str, "tool_calls": [{"name","arguments": dict|str}]}.
        """
        if self.mock_mode:
            if mock is None:
                raise LLMError(f"Stage {stage!r} has no mock implementation.")
            out = mock()
            calls = []
            for i, c in enumerate(out.get("tool_calls") or []):
                args = c.get("arguments", {})
                calls.append({"id": c.get("id") or f"mock-{i}", "name": c["name"],
                              "arguments": args if isinstance(args, str) else json.dumps(args)})
            content = out.get("content") or ""
            self.budget.charge(estimate_tokens(json.dumps(messages, default=str)[-4000:]), estimate_tokens(content),
                               model=self.model, role=self.role, stage=stage, provider="mock")
            msg: dict = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
            return {"content": content, "tool_calls": calls, "message": msg}

        resp, cfg, ms = self._complete(messages, stage=stage, tools=tools)
        m = resp.choices[0].message
        raw = m.content or ""
        calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
                 for tc in (getattr(m, "tool_calls", None) or [])]
        self._account(resp, cfg, ms, stage=stage, in_est=estimate_tokens(json.dumps(messages, default=str)[-4000:]),
                      raw=raw, tool_calls=[c["name"] for c in calls])
        # Keep the assistant message exactly as returned (including any reasoning text) for continuity.
        msg = {"role": "assistant", "content": raw}
        if calls:
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
        return {"content": clean_reasoning(raw) if raw else "", "tool_calls": calls, "message": msg}

    def chat(self, system: str, user: str, *, stage: str = "llm", json_mode: bool = False,
             mock: Callable[[], str] | None = None, schema: dict | None = None, schema_name: str = "output") -> str:
        if self.mock_mode:
            if mock is None:
                raise LLMError(f"Stage {stage!r} has no mock implementation. Set LLM_PROVIDER to a real provider or add a mock.")
            out = mock()
            if self.tracer:
                self.tracer.event("llm", llm_stage=stage, model=self.model, role=self.role, mock=True)
            self.budget.charge(estimate_tokens(system + user), estimate_tokens(out), model=self.model, role=self.role,
                               stage=stage, provider="mock")
            return out

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        resp, cfg, ms = self._complete(messages, stage=stage, json_mode=json_mode, schema=schema, schema_name=schema_name)
        raw_text = resp.choices[0].message.content or ""
        self._account(resp, cfg, ms, stage=stage, in_est=estimate_tokens(system + user), raw=raw_text)
        return clean_reasoning(raw_text) if not json_mode else raw_text

    def structured(self, system: str, user: str, schema: Type[T], *, stage: str = "llm",
                   mock: Callable[[], dict] | None = None, max_repairs: int = 2) -> T:
        """Ask for JSON matching `schema`; validate; repair on failure."""
        if self.mock_mode:
            if mock is None:
                raise LLMError(f"Stage {stage!r} has no mock implementation.")
            data = mock()
            if self.tracer:
                self.tracer.event("llm", llm_stage=stage, model=self.model, role=self.role, mock=True, schema=schema.__name__)
            self.budget.charge(estimate_tokens(system + user), estimate_tokens(json.dumps(data, default=str)),
                               model=self.model, role=self.role, stage=stage, provider="mock")
            return schema.model_validate(data)

        json_schema = schema.model_json_schema()
        # Content-addressed cache of VALIDATED outputs: the key is the model plus the full prompt, so an answer
        # is only ever reused for exactly the same question about exactly the same text (see cache.py).
        from .cache import LLM_TTL_S, get_cache, sha

        cache = get_cache()
        cache_key = sha(self.model, system, user, schema.__name__) if cache.enabled and (SETTINGS.llm_temperature or 0) <= 0.01 else ""
        if cache_key:
            hit = cache.get("llm", cache_key, run_id=self.tracer.run_id if self.tracer else "")
            if hit:
                try:
                    out = schema.model_validate(hit["data"])
                    self._trace("llm", llm_stage=stage, model=hit.get("model", self.model), provider=hit.get("provider", self.provider),
                                role=self.role, cached=True)
                    self.budget.charge(hit.get("in", 0), hit.get("out", 0), model=self.model, role=self.role, stage=stage,
                                       provider=self.provider, cached=True)
                    return out
                except ValidationError:
                    pass   # schema changed since it was stored: ask again
        tok_before = (self.budget.input_tokens, self.budget.output_tokens)
        schema_hint = json.dumps(json_schema, indent=0)
        sys_full = f"{system}\n\nReturn ONLY a valid JSON object that matches this JSON Schema (no markdown fences, no commentary):\n{schema_hint}"
        prompt = user
        last_err = ""
        for attempt in range(max_repairs + 1):
            raw = self.chat(sys_full, prompt, stage=f"{stage}#{attempt}", json_mode=True, schema=json_schema,
                            schema_name=schema.__name__)
            try:
                out = schema.model_validate_json(_extract_json(raw))
                if cache_key:
                    cache.put("llm", cache_key, {"data": out.model_dump(mode="json"), "model": self.model, "provider": self.provider,
                                                 "in": self.budget.input_tokens - tok_before[0],
                                                 "out": self.budget.output_tokens - tok_before[1]}, ttl=LLM_TTL_S)
                return out
            except (ValidationError, ValueError) as exc:
                last_err = str(exc)
                prompt = (f"{user}\n\nYour previous answer was not valid:\n{raw[:2000]}\n\n"
                          f"Validation error:\n{last_err[:1500]}\n\nReturn corrected JSON only.")
        raise LLMError(f"Could not get valid {schema.__name__} after {max_repairs} repairs: {last_err[:300]}")
