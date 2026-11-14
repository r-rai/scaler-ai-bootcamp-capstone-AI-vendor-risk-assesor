"""
llm.py — one small wrapper around "an LLM", copied in spirit from the Deep Search course (deepsearch/llm.py).

Two modes:
* mock (default): no network. Every call site passes a `mock=` function that produces the answer with
  simple rules, so the whole project runs offline and the tests are repeatable.
* live: any OpenAI-compatible provider (including MiniMax at https://api.minimax.io/v1). Set LLM_PROVIDER in .env.

The method that matters is `structured()`: ask for JSON, validate it with a Pydantic schema, and if
validation fails send the error back and ask the model to repair its output. This validate → repair loop
is what turns fragile LLM output into a contract (Deep Search Module 4, Lab 2).
"""

from __future__ import annotations

import json
import re
from typing import Callable, Type, TypeVar

from pydantic import ValidationError

from .config import SETTINGS
from .schemas import BaseModel
from .tracing import Budget, Tracer, estimate_tokens

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


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
    def __init__(self, budget: Budget | None = None, tracer: Tracer | None = None, model: str | None = None, role: str = "main"):
        self.budget = budget or Budget()
        self.tracer = tracer
        self.provider = SETTINGS.llm_provider
        self.mock_mode = self.provider == "mock"
        self.role = role
        self._client = None
        if self.mock_mode:
            self.model = "mock-llm"
        else:
            cfg = SETTINGS.resolved_llm(role=role)
            self.model = model or cfg["model"]
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError("Live mode needs the openai package: pip install openai") from exc
            self._client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=90, max_retries=2)

    @classmethod
    def small(cls, budget: Budget | None = None, tracer: Tracer | None = None) -> "LLM":
        """A cheaper, faster model for high-volume jobs such as claim extraction, routing, and grading (e.g. MiniMax-M3.1-Flash)."""
        return cls(budget=budget, tracer=tracer, model=SETTINGS.llm_small_model or None, role="small")

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, *, stage: str = "llm", json_mode: bool = False,
             mock: Callable[[], str] | None = None) -> str:
        if self.mock_mode:
            if mock is None:
                raise LLMError(f"Stage {stage!r} has no mock implementation. Set LLM_PROVIDER to a real provider or add a mock.")
            out = mock()
            self.budget.charge(estimate_tokens(system + user), estimate_tokens(out))
            if self.tracer:
                self.tracer.event("llm", llm_stage=stage, model=self.model, role=self.role, mock=True)
            return out

        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if SETTINGS.llm_temperature is not None:
            # MiniMax temperature must be in (0.0, 1.0] on some legacy endpoints or [0.0, 2.0] on M3;
            # if provider is minimax and temperature == 0.0, 0.01 avoids zero-temp rejection on older gateways.
            temp = SETTINGS.llm_temperature
            if self.provider == "minimax" and temp <= 0.0:
                temp = 0.01
            kwargs["temperature"] = temp
        if json_mode and self.provider not in ("ollama",):
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — fallback without response_format if provider rejects json_object
            if json_mode and "response_format" in kwargs:
                kwargs.pop("response_format", None)
                try:
                    resp = self._client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
                except Exception as exc2:  # noqa: BLE001
                    raise LLMError(f"LLM call failed at stage {stage!r}: {exc2}") from exc2
            else:
                raise LLMError(f"LLM call failed at stage {stage!r}: {exc}") from exc
        raw_text = resp.choices[0].message.content or ""
        text = clean_reasoning(raw_text) if not json_mode else raw_text
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) or estimate_tokens(system + user)
        out_tok = getattr(usage, "completion_tokens", None) or estimate_tokens(raw_text)
        self.budget.charge(in_tok, out_tok)
        if self.tracer:
            self.tracer.event("llm", llm_stage=stage, model=self.model, role=self.role, input_tokens=in_tok, output_tokens=out_tok)
        return text

    def structured(self, system: str, user: str, schema: Type[T], *, stage: str = "llm",
                   mock: Callable[[], dict] | None = None, max_repairs: int = 2) -> T:
        """Ask for JSON matching `schema`; validate; repair on failure."""
        if self.mock_mode:
            if mock is None:
                raise LLMError(f"Stage {stage!r} has no mock implementation.")
            data = mock()
            self.budget.charge(estimate_tokens(system + user), estimate_tokens(json.dumps(data, default=str)))
            if self.tracer:
                self.tracer.event("llm", llm_stage=stage, model=self.model, role=self.role, mock=True, schema=schema.__name__)
            return schema.model_validate(data)

        schema_hint = json.dumps(schema.model_json_schema(), indent=0)
        sys_full = f"{system}\n\nReturn ONLY a valid JSON object that matches this JSON Schema (no markdown fences, no commentary):\n{schema_hint}"
        prompt = user
        last_err = ""
        for attempt in range(max_repairs + 1):
            raw = self.chat(sys_full, prompt, stage=f"{stage}#{attempt}", json_mode=True)
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as exc:
                last_err = str(exc)
                prompt = (f"{user}\n\nYour previous answer was not valid:\n{raw[:2000]}\n\n"
                          f"Validation error:\n{last_err[:1500]}\n\nReturn corrected JSON only.")
        raise LLMError(f"Could not get valid {schema.__name__} after {max_repairs} repairs: {last_err[:300]}")
