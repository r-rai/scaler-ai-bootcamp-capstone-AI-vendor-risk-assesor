"""
config.py — every setting in one place, read from .env.

Same pattern as the Deep Search course (deepsearch/config.py) and the Agentic RAG kit (config.py):
copy .env.example to .env, edit, and nothing else in the code needs to change.

Defaults are OFFLINE: LLM_PROVIDER=mock and SEARCH_PROVIDER=offline. The whole project runs with no
API key and no internet. Switch providers only after the offline run is green.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is in requirements.txt; this keeps `import assessor` working without it
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
POLICY_DIR = DATA_DIR / "policies"
VENDOR_CORPUS = DATA_DIR / "vendor_web" / "corpus.json"
REAL_VENDOR_CORPUS = DATA_DIR / "vendor_web" / "real_vendors_corpus.json"
REQUIREMENTS_FILE = DATA_DIR / "requirements.yaml"
GOLD_SET = DATA_DIR / "gold_set.json"
GOLD_SET_REAL = DATA_DIR / "gold_set_real.json"
RUNS_DIR = Path(os.environ.get("VRA_RUNS_DIR", "").strip() or ROOT / "runs")


LOCAL_PROVIDERS = frozenset({"llamacpp", "ollama"})   # run on this machine: free, and data never leaves it


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # --- LLM -------------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "mock").lower())
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_small_model: str = field(default_factory=lambda: _env("LLM_SMALL_MODEL", ""))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    # Per-role providers. Blank = same as LLM_PROVIDER. Typical production split on this laptop:
    #   LLM_PROVIDER=minimax (planning, synthesis)   LLM_SMALL_PROVIDER=llamacpp (routing, grading, identity, injection)
    llm_small_provider: str = field(default_factory=lambda: _env("LLM_SMALL_PROVIDER", "").lower())
    llm_extract_provider: str = field(default_factory=lambda: _env("LLM_EXTRACT_PROVIDER", "").lower())
    llm_fallback: bool = field(default_factory=lambda: _env("LLM_FALLBACK", "1") == "1")
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 60.0))
    # The high-volume roles (small, extract) normally answer in 2-15 s; a call past this is a stall, not work.
    llm_small_timeout_s: float = field(default_factory=lambda: _env_float("LLM_SMALL_TIMEOUT_S", 30.0))
    # Page reading (extract role, cloud only): a call still running after this many seconds gets ONE duplicate request and
    # the first good answer wins. 0 = off. Measured on 283 live calls: p50 5 s, p90 16 s, p99 49 s, and the slow ones are
    # provider stalls, not long outputs; lowering the timeout instead would lose pages. See llm.py `_send`.
    llm_hedge_after_s: float = field(default_factory=lambda: _env_float("LLM_HEDGE_AFTER_S", 10.0))
    # Reasoning effort for providers with adaptive thinking (verified on MiniMax). The high-volume roles
    # (small, extract) classify and copy facts; "low" cut their latency and output tokens by more than half
    # with the same answers. Blank = provider default.
    llm_reasoning_effort: str = field(default_factory=lambda: _env("LLM_REASONING_EFFORT", "").lower())
    llm_small_reasoning_effort: str = field(default_factory=lambda: _env("LLM_SMALL_REASONING_EFFORT", "low").lower())

    # --- Search --------------------------------------------------------------
    search_provider: str = field(default_factory=lambda: _env("SEARCH_PROVIDER", "offline").lower())
    # repr=False: printing SETTINGS must never reveal a secret (tests/test_config_secrets.py)
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY", ""), repr=False)
    results_per_query: int = field(default_factory=lambda: _env_int("VRA_RESULTS_PER_QUERY", 5))

    # --- Firecrawl: reads a page of the assessed vendor's own site when our honest fetcher is refused (HTTP 401/403/
    # 429/503). Off without a key. The monthly figure is OUR ceiling, not the plan size. See fetching.py, GOVERNANCE.md §3.
    firecrawl_api_key: str = field(default_factory=lambda: _env("FIRECRAWL_API_KEY", ""), repr=False)
    firecrawl_enabled: bool = field(default_factory=lambda: _env("VRA_FIRECRAWL", "1") == "1")
    firecrawl_max_per_run: int = field(default_factory=lambda: _env_int("VRA_FIRECRAWL_MAX_PER_RUN", 8))
    firecrawl_monthly_credits: int = field(default_factory=lambda: _env_int("VRA_FIRECRAWL_MONTHLY_CREDITS", 300))
    firecrawl_timeout_s: float = field(default_factory=lambda: _env_float("VRA_FIRECRAWL_TIMEOUT_S", 15.0))

    # --- Embeddings & Reranker (free local models or built-in fallback) -------
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "auto").lower())
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    reranker: str = field(default_factory=lambda: _env("RERANKER", "auto").lower())
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))
    # A stalled Hugging Face download must never hang startup, /ready or the tests: models are loaded from the
    # local cache first; a missing model is downloaded in a thread with this deadline, then we fall back to builtin.
    model_download: bool = field(default_factory=lambda: _env("VRA_MODEL_DOWNLOAD", "1").lower() in ("1", "true", "yes"))
    model_load_timeout_s: float = field(default_factory=lambda: _env_float("VRA_MODEL_LOAD_TIMEOUT_S", 90.0))
    # fastembed otherwise caches in the system temp dir, which a reboot or `wsl --shutdown` wipes.
    fastembed_cache: str = field(default_factory=lambda: _env(
        "FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed")))

    # --- Agent behaviour -----------------------------------------------------
    as_of: date = field(default_factory=lambda: date.fromisoformat(_env("VRA_AS_OF", "2026-11-01")))
    max_usd: float = field(default_factory=lambda: _env_float("VRA_MAX_USD", 0.50))
    max_seconds: float = field(default_factory=lambda: _env_float("VRA_MAX_SECONDS", 90.0))     # wall clock per run
    max_tokens: int = field(default_factory=lambda: _env_int("VRA_MAX_TOKENS", 400_000))
    max_tool_calls: int = field(default_factory=lambda: _env_int("VRA_MAX_TOOL_CALLS", 120))    # searches + fetches
    max_iterations: int = field(default_factory=lambda: _env_int("VRA_MAX_ITERATIONS", 2))      # research passes
    # template = the fixed 9-field checklist (instant, free, auditable); llm = the main model writes the plan.
    planner: str = field(default_factory=lambda: _env("VRA_PLANNER", "template").lower())
    max_parallel: int = field(default_factory=lambda: _env_int("VRA_MAX_PARALLEL", 4))                # search + download pools
    llm_parallel: int = field(default_factory=lambda: _env_int("VRA_LLM_PARALLEL", 6))                # pages read at once
    verbose: bool = field(default_factory=lambda: _env("VRA_VERBOSE", "1") == "1")

    # --- Service: identity, limits (app/api.py, app/ui.py) -------------------
    env: str = field(default_factory=lambda: _env("VRA_ENV", "dev").lower())                    # dev | prod
    tokens_file: str = field(default_factory=lambda: _env("VRA_TOKENS_FILE", ""))
    max_concurrent_jobs: int = field(default_factory=lambda: _env_int("VRA_MAX_CONCURRENT_JOBS", 2))
    rate_per_min: int = field(default_factory=lambda: _env_int("VRA_RATE_PER_MIN", 60))
    max_question_chars: int = field(default_factory=lambda: _env_int("VRA_MAX_QUESTION_CHARS", 2000))

    # --- Spend caps across runs (0 = no cap). VRA_MAX_USD above is the cap for ONE assessment. ---
    daily_usd: float = field(default_factory=lambda: _env_float("VRA_DAILY_USD", 0.0))
    monthly_usd: float = field(default_factory=lambda: _env_float("VRA_MONTHLY_USD", 0.0))
    tavily_monthly_credits: int = field(default_factory=lambda: _env_int("TAVILY_MONTHLY_CREDITS", 1000))

    # --- Caching (assessor/cache.py) -----------------------------------------
    cache_enabled: bool = field(default_factory=lambda: _env("VRA_CACHE", "1") == "1")
    semantic_cache_threshold: float = field(default_factory=lambda: _env_float("VRA_SEMANTIC_CACHE_THRESHOLD", 0.92))
    semantic_cache_ttl_s: int = field(default_factory=lambda: _env_int("VRA_SEMANTIC_CACHE_TTL_S", 7 * 86400))
    search_cache_ttl_s: int = field(default_factory=lambda: _env_int("VRA_SEARCH_CACHE_TTL_S", 86400))
    page_cache_ttl_s: int = field(default_factory=lambda: _env_int("VRA_PAGE_CACHE_TTL_S", 7 * 86400))

    # --- Logging and observability (assessor/observability.py) ----------------
    log_level: str = field(default_factory=lambda: _env("VRA_LOG_LEVEL", "INFO").upper())
    log_json: bool = field(default_factory=lambda: _env("VRA_LOG_JSON", "1") == "1")
    otlp_endpoint: str = field(default_factory=lambda: _env("OTEL_EXPORTER_OTLP_ENDPOINT", ""))

    # --- Web fetching (assessor/fetching.py) ----------------------------------
    fetcher_url: str = field(default_factory=lambda: _env("VRA_FETCHER_URL", ""))      # sandboxed fetch service, if any
    fetch_max_bytes: int = field(default_factory=lambda: _env_int("VRA_FETCH_MAX_BYTES", 5_000_000))
    fetch_timeout_s: float = field(default_factory=lambda: _env_float("VRA_FETCH_TIMEOUT_S", 12.0))
    pages_per_question: int = field(default_factory=lambda: _env_int("VRA_PAGES_PER_QUESTION", 2))

    # --- Bug-fix agent (defaults to the main LLM settings) --------------------
    bugfix_provider: str = field(default_factory=lambda: _env("BUGFIX_LLM_PROVIDER", "").lower())
    bugfix_model: str = field(default_factory=lambda: _env("BUGFIX_LLM_MODEL", ""))
    bugfix_max_usd: float = field(default_factory=lambda: _env_float("BUGFIX_MAX_USD", 1.00))
    bugfix_max_steps: int = field(default_factory=lambda: _env_int("BUGFIX_MAX_STEPS", 25))

    # --- Cost estimates (only used for the budget maths) --------------------
    price_in_per_mtok: float = field(default_factory=lambda: _env_float("PRICE_IN_PER_MTOK", 0.15))
    price_out_per_mtok: float = field(default_factory=lambda: _env_float("PRICE_OUT_PER_MTOK", 0.60))
    # Small-model prices default to the main prices unless set (MiniMax Flash is cheaper than M3).
    price_small_in_per_mtok: float = field(default_factory=lambda: _env_float("PRICE_SMALL_IN_PER_MTOK", _env_float("PRICE_IN_PER_MTOK", 0.15)))
    price_small_out_per_mtok: float = field(default_factory=lambda: _env_float("PRICE_SMALL_OUT_PER_MTOK", _env_float("PRICE_OUT_PER_MTOK", 0.60)))

    @property
    def prod(self) -> bool:
        return self.env == "prod"

    @property
    def offline(self) -> bool:
        return self.llm_provider == "mock"

    def resolved_llm(self, role: str = "main", provider: str | None = None) -> dict:
        """Translate LLM_PROVIDER into (base_url, api_key, model) for the OpenAI-compatible client.

        Supports MiniMax (`https://api.minimax.io/v1` with `MiniMax-M3` and `MiniMax-M3.1-Flash`),
        Gemini, Groq, Ollama, OpenRouter, OpenAI, Anthropic, and custom OpenAI-compatible endpoints.
        """
        p = (provider or self.llm_provider).lower()
        table = {
            "minimax":    (
                _env("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
                _env("MINIMAX_API_KEY"),
                "MiniMax-M3",
                "MiniMax-M3.1-Flash",
            ),
            "openai":     ("https://api.openai.com/v1", _env("OPENAI_API_KEY"), "gpt-6-luna", "gpt-6-luna"),
            "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/", _env("GEMINI_API_KEY"), "gemini-3.8-flash", "gemini-3.8-flash"),
            "groq":       ("https://api.groq.com/openai/v1", _env("GROQ_API_KEY"), "openai/gpt-oss-20b", "llama-3.3-70b-versatile"),
            "ollama":     (_env("OLLAMA_BASE_URL", "http://localhost:11434/v1"), "ollama", "qwen3:8b", "qwen3:4b"),
            "llamacpp":   (_env("LLAMACPP_BASE_URL", "http://localhost:8080/v1"), _env("LLAMACPP_API_KEY", "local"),
                           _env("LLAMACPP_MODEL", "local-small"), _env("LLAMACPP_MODEL", "local-small")),
            "anthropic":  ("https://api.anthropic.com/v1/", _env("ANTHROPIC_API_KEY"), "claude-haiku-5-5", "claude-haiku-5-5"),
            "openrouter": ("https://openrouter.ai/api/v1", _env("OPENROUTER_API_KEY"), "deepseek/deepseek-chat", "deepseek/deepseek-v4-flash"),
            "custom":     (_env("LLM_BASE_URL", "http://localhost:8000/v1"), _env("LLM_API_KEY", "none"), "", ""),
        }
        if p not in table:
            raise ValueError(
                f"Unknown LLM_PROVIDER={p!r}. Use mock, minimax, openai, gemini, groq, ollama, llamacpp, anthropic, openrouter or custom."
            )
        base_url, api_key, default_main, default_small = table[p]
        if not api_key:
            raise ValueError(f"LLM_PROVIDER={p} needs an API key in .env (see .env.example).")
        # LLM_MODEL / LLM_SMALL_MODEL name models of the MAIN provider only; any other provider uses its own default.
        # A llama.cpp server serves one model under its own alias, whatever LLM_MODEL says.
        is_main = p == self.llm_provider and p != "llamacpp"
        if role in ("small", "extract"):
            chosen_model = ((self.llm_small_model or self.llm_model) if is_main else "") or default_small
        else:
            chosen_model = (self.llm_model if is_main else "") or default_main
        return {"base_url": base_url, "api_key": api_key, "model": chosen_model, "provider": p,
                "local": p in LOCAL_PROVIDERS}

    def provider_for(self, role: str = "main") -> str:
        """Which provider serves a role. Mock mode is all-or-nothing: no role escapes it."""
        if self.llm_provider == "mock":
            return "mock"
        if role == "small":
            return self.llm_small_provider or self.llm_provider
        if role == "extract":
            # Extraction reads whole pages (long prompts) and its accuracy drives the report. It defaults to the
            # MAIN provider's small model, not the local one; move it with LLM_EXTRACT_PROVIDER after measuring.
            return self.llm_extract_provider or self.llm_provider
        return self.llm_provider


SETTINGS = Settings()
