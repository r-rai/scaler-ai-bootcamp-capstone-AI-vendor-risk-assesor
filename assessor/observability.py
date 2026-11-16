"""
observability.py — logs, metrics and traces (Production RAG Module 12, "Observability").

Three signals, each answering a different question:
  logs     what happened, in order           JSON lines on stderr and in runs/logs/app.jsonl
  metrics  how much / how fast / how often   Prometheus text at GET /metrics
  traces   where the time went in one run    OpenTelemetry spans, exported over OTLP when configured

Design rules:
  * No new hard dependency. `prometheus_client` and the OpenTelemetry SDK are used when installed
    (requirements-prod.txt); without them a small built-in registry still serves /metrics and spans are no-ops.
  * Labels are low-cardinality and carry no content: never a question, a vendor name, a URL or a user.
  * Secrets are masked before a log line is written.
  * The per-run `trace.jsonl` (assessor/tracing.py) stays the evidence trail; `on_trace_event()` turns the same
    events into metrics so the two can never disagree.

This module imports nothing from the pipeline, so every other module may import it.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import RUNS_DIR, SETTINGS
from .errors import mask_secrets

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")

# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

_STD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rec: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if _request_id.get():
            rec["request_id"] = _request_id.get()
        if _run_id.get():
            rec["run_id"] = _run_id.get()
        for k, v in record.__dict__.items():
            if k not in _STD_ATTRS and not k.startswith("_"):
                rec[k] = v
        if record.exc_info:
            rec["exc"] = self.formatException(record.exc_info)[-2000:]
        return mask_secrets(json.dumps(rec, default=str, ensure_ascii=False))


_LOG_LOCK = threading.Lock()
_LOG_READY = False


def setup_logging(to_file: bool = True) -> None:
    """Idempotent. Call once from an entry point (API, UI, CLI)."""
    global _LOG_READY
    with _LOG_LOCK:
        if _LOG_READY:
            return
        root = logging.getLogger("vra")
        root.setLevel(getattr(logging, SETTINGS.log_level, logging.INFO))
        root.propagate = False
        fmt: logging.Formatter = JsonFormatter() if SETTINGS.log_json else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        root.addHandler(stream)
        if to_file:
            try:
                log_dir = RUNS_DIR / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                fh = logging.handlers.RotatingFileHandler(log_dir / "app.jsonl", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
                fh.setFormatter(JsonFormatter())
                root.addHandler(fh)
            except OSError:
                pass   # read-only filesystem (container): stderr is still collected
        _LOG_READY = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"vra.{name}")


def bind_run(run_id: str) -> None:
    _run_id.set(run_id)


# ---------------------------------------------------------------------------
# Metrics: prometheus_client when installed, otherwise a small compatible registry
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised only when the optional package is installed
    import prometheus_client as _prom
except ImportError:  # CI and the offline starter kit
    _prom = None

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 90, 120, 300)
COST_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


class _Metric:
    """Fallback counter / gauge / histogram with the same call shape as prometheus_client."""

    def __init__(self, kind: str, name: str, doc: str, labels: tuple[str, ...], buckets: tuple[float, ...] = ()):
        self.kind, self.name, self.doc, self.labels_, self.buckets = kind, name, doc, labels, buckets
        self.values: dict[tuple, Any] = {}
        self.lock = threading.Lock()

    def labels(self, *vals: Any, **kw: Any) -> "_Child":
        key = tuple(str(v) for v in vals) if vals else tuple(str(kw.get(l, "")) for l in self.labels_)
        return _Child(self, key)

    def inc(self, n: float = 1.0) -> None:
        _Child(self, ()).inc(n)

    def set(self, v: float) -> None:
        _Child(self, ()).set(v)

    def observe(self, v: float) -> None:
        _Child(self, ()).observe(v)

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.doc}", f"# TYPE {self.name} {self.kind}"]
        with self.lock:
            for key, val in sorted(self.values.items()):
                lab = ",".join(f'{l}="{_esc(v)}"' for l, v in zip(self.labels_, key))
                if self.kind == "histogram":
                    cum = 0
                    for b, n in zip(self.buckets, val["buckets"]):
                        cum += n
                        out.append(f'{self.name}_bucket{{{lab + "," if lab else ""}le="{b}"}} {cum}')
                    out.append(f'{self.name}_bucket{{{lab + "," if lab else ""}le="+Inf"}} {val["count"]}')
                    out.append(f"{self.name}_sum{{{lab}}} {val['sum']}" if lab else f"{self.name}_sum {val['sum']}")
                    out.append(f"{self.name}_count{{{lab}}} {val['count']}" if lab else f"{self.name}_count {val['count']}")
                else:
                    suffix = "_total" if self.kind == "counter" and not self.name.endswith("_total") else ""
                    out.append(f"{self.name}{suffix}{{{lab}}} {val}" if lab else f"{self.name}{suffix} {val}")
        return out


class _Child:
    def __init__(self, m: _Metric, key: tuple):
        self.m, self.key = m, key

    def inc(self, n: float = 1.0) -> None:
        with self.m.lock:
            self.m.values[self.key] = self.m.values.get(self.key, 0.0) + n

    def set(self, v: float) -> None:
        with self.m.lock:
            self.m.values[self.key] = float(v)

    def observe(self, v: float) -> None:
        with self.m.lock:
            h = self.m.values.setdefault(self.key, {"buckets": [0] * len(self.m.buckets), "sum": 0.0, "count": 0})
            for i, b in enumerate(self.m.buckets):
                if v <= b:
                    h["buckets"][i] += 1
                    break
            h["sum"] += v
            h["count"] += 1


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


_FALLBACK: list[_Metric] = []


def _counter(name: str, doc: str, labels: tuple[str, ...] = ()):
    if _prom:
        return _prom.Counter(name, doc, labels)
    m = _Metric("counter", name, doc, labels)
    _FALLBACK.append(m)
    return m


def _gauge(name: str, doc: str, labels: tuple[str, ...] = ()):
    if _prom:
        return _prom.Gauge(name, doc, labels)
    m = _Metric("gauge", name, doc, labels)
    _FALLBACK.append(m)
    return m


def _histogram(name: str, doc: str, labels: tuple[str, ...] = (), buckets: tuple[float, ...] = LATENCY_BUCKETS):
    if _prom:
        return _prom.Histogram(name, doc, labels, buckets=buckets)
    m = _Metric("histogram", name, doc, labels, buckets)
    _FALLBACK.append(m)
    return m


HTTP_REQUESTS = _counter("vra_http_requests", "HTTP requests", ("method", "route", "status"))
HTTP_SECONDS = _histogram("vra_http_request_seconds", "HTTP request latency", ("route",))
RUNS = _counter("vra_runs", "Pipeline runs by route and outcome", ("route", "outcome"))
RUN_SECONDS = _histogram("vra_run_seconds", "End-to-end pipeline latency", ("route",))
RUN_COST = _histogram("vra_run_cost_usd", "Cloud LLM cost per run", ("route",), COST_BUCKETS)
STAGE_SECONDS = _histogram("vra_stage_seconds", "Latency per pipeline node", ("stage",))
LLM_CALLS = _counter("vra_llm_calls", "LLM calls", ("provider", "model", "role", "stage"))
LLM_TOKENS = _counter("vra_llm_tokens", "LLM tokens", ("provider", "model", "direction"))
LLM_COST = _counter("vra_llm_cost_usd", "Cloud LLM spend in USD", ("provider", "model"))
LLM_AVOIDED = _counter("vra_llm_avoided_usd", "Cloud spend avoided by local models and caches", ("reason",))
LLM_SECONDS = _histogram("vra_llm_seconds", "LLM call latency", ("provider", "role"))
LLM_FALLBACKS = _counter("vra_llm_fallbacks", "Calls that fell back to another provider", ("from_provider",))
CACHE_EVENTS = _counter("vra_cache_events", "Cache lookups", ("cache", "result"))
GUARDRAIL_EVENTS = _counter("vra_guardrail_events", "Guardrail actions", ("layer", "action"))
TOOL_CALLS = _counter("vra_tool_calls", "Search and fetch calls", ("tool", "result"))
VERDICTS = _counter("vra_verdicts", "Requirement verdicts produced", ("verdict",))
OVERALL = _counter("vra_overall_outcomes", "Assessment outcomes", ("overall",))
BUDGET_STOPS = _counter("vra_budget_stops", "Runs stopped by a budget limit", ("limit",))
DECISIONS = _counter("vra_approval_decisions", "Human decisions recorded", ("decision", "status_after"))
JOBS = _gauge("vra_jobs", "Async jobs by status", ("status",))
SPEND_TODAY = _gauge("vra_spend_usd_today", "Cloud LLM spend today (UTC)")
SPEND_MONTH = _gauge("vra_spend_usd_month", "Cloud LLM spend this month (UTC)")
SPEND_CAP = _gauge("vra_spend_cap_usd", "Configured spend caps", ("period",))
TAVILY_CREDITS = _gauge("vra_tavily_credits_used_month", "Tavily searches this month against the free quota")
EVAL_METRIC = _gauge("vra_eval_metric", "Latest evaluation scorecard values", ("gold_set", "metric"))
AUDIT_CHAIN_OK = _gauge("vra_audit_chain_ok", "1 if the audit log hash chain verifies")
LOCAL_LLM_UP = _gauge("vra_local_llm_up", "1 if the local model server answered its last call")


def _budget_limit(reason: str) -> str:
    for k in ("cost", "token", "time", "tool-call"):
        if k in reason:
            return k
    return "cost" if "Budget of" in reason else "other"


def on_trace_event(run_id: str, stage: str, data: dict[str, Any]) -> None:
    """Called for every Tracer event: the trace and the metrics are fed from the same facts."""
    try:
        if stage == "llm":
            provider, model = str(data.get("provider") or "mock"), str(data.get("model", ""))
            LLM_CALLS.labels(provider, model, str(data.get("role", "")), str(data.get("llm_stage", "")).split("#")[0]).inc()
            if data.get("input_tokens"):
                LLM_TOKENS.labels(provider, model, "input").inc(data["input_tokens"])
                LLM_TOKENS.labels(provider, model, "output").inc(data.get("output_tokens", 0))
            if data.get("duration_ms") is not None:
                LLM_SECONDS.labels(provider, str(data.get("role", ""))).observe(data["duration_ms"] / 1000)
            if provider in ("llamacpp", "ollama"):
                LOCAL_LLM_UP.set(1)
        elif stage == "llm_fallback":
            src = str(data.get("failed") or data.get("skipped") or "")
            LLM_FALLBACKS.labels(src).inc()
            if src in ("llamacpp", "ollama"):
                LOCAL_LLM_UP.set(0)
        elif stage == "cache":
            CACHE_EVENTS.labels(str(data.get("cache", "")), "hit" if data.get("hit") else "miss").inc()
        elif stage in ("quarantined", "identity_rejected", "empty_page", "skip_from_memory", "ssrf_blocked", "person_data_removed"):
            GUARDRAIL_EVENTS.labels(stage, "blocked").inc()
        elif stage == "guard_input":
            if data.get("refused"):
                GUARDRAIL_EVENTS.labels("input", "refused").inc()
            if data.get("redactions"):
                GUARDRAIL_EVENTS.labels("input", "redacted").inc(len(data["redactions"]))
        elif stage == "output_guard":
            GUARDRAIL_EVENTS.labels("output", str(data.get("action", "edited"))).inc()
        elif stage == "budget_stop":
            BUDGET_STOPS.labels(_budget_limit(str(data.get("reason", "")))).inc()
        elif stage == "search":
            TOOL_CALLS.labels("search", "ok").inc()
        elif stage == "search_failed":
            TOOL_CALLS.labels("search", "error").inc()
        elif stage == "fetch":
            TOOL_CALLS.labels("fetch", "ok").inc()
        elif stage == "fetch_failed":
            TOOL_CALLS.labels("fetch", "error").inc()
        elif stage == "node":
            STAGE_SECONDS.labels(str(data.get("name", ""))).observe(float(data.get("duration_ms", 0)) / 1000)
    except Exception:  # noqa: BLE001 — telemetry must never break a run
        pass


def observe_run(state: dict, budget: Any) -> None:
    """Called once per pipeline run by the entry point (API, UI, CLI)."""
    try:
        route = str(state.get("route") or "unknown")
        rep = state.get("report")
        outcome = rep.overall if rep is not None else ("refused" if route == "refuse" else "answered")
        RUNS.labels(route, outcome).inc()
        RUN_SECONDS.labels(route).observe(max(0.0, time.time() - float(state.get("started", time.time()))))
        RUN_COST.labels(route).observe(float(budget.spent_usd))
        for call in budget.calls:
            if call["usd"]:
                LLM_COST.labels(call["provider"], call["model"]).inc(call["usd"])
            if call.get("avoided_usd"):
                LLM_AVOIDED.labels("cache" if call.get("cached") else "local_model").inc(call["avoided_usd"])
        if rep is not None:
            OVERALL.labels(rep.overall).inc()
            for r in rep.results:
                VERDICTS.labels(r.verdict).inc()
            for note in rep.guardrail_notes:
                if note.startswith("output:"):
                    GUARDRAIL_EVENTS.labels("output", "edited").inc()
    except Exception:  # noqa: BLE001
        pass


def count_guardrail(layer: str, action: str) -> None:
    GUARDRAIL_EVENTS.labels(layer, action).inc()


def count_decision(decision: str, status_after: str) -> None:
    DECISIONS.labels(decision, status_after).inc()


def set_queue_depth(by_status: dict[str, int]) -> None:
    for status in ("queued", "processing", "completed", "failed"):
        JOBS.labels(status).set(by_status.get(status, 0))


def _refresh_gauges() -> None:
    """Values that live in the database are read at scrape time, so they survive restarts."""
    try:
        from . import costs, db

        set_queue_depth(db.count_jobs())
        ov = costs.spend_summary()
        SPEND_TODAY.set(ov["today_usd"])
        SPEND_MONTH.set(ov["month_usd"])
        SPEND_CAP.labels("run").set(SETTINGS.max_usd)
        SPEND_CAP.labels("day").set(SETTINGS.daily_usd)
        SPEND_CAP.labels("month").set(SETTINGS.monthly_usd)
        TAVILY_CREDITS.set(ov["tavily_credits_used"])
        AUDIT_CHAIN_OK.set(1 if db.verify_audit_chain()["ok"] else 0)
        seen: set[str] = set()
        for run in db.get_eval_history(limit=6):
            if run["gold_set"] in seen:
                continue
            seen.add(run["gold_set"])
            for k, v in run["metrics"].items():
                if isinstance(v, (int, float)):
                    EVAL_METRIC.labels(run["gold_set"], k).set(float(v))
    except Exception:  # noqa: BLE001
        pass


def prometheus_payload() -> tuple[str, str]:
    _refresh_gauges()
    if _prom:
        return _prom.generate_latest().decode("utf-8"), _prom.CONTENT_TYPE_LATEST
    lines: list[str] = []
    for m in _FALLBACK:
        lines.extend(m.render())
    return "\n".join(lines) + "\n", "text/plain; version=0.0.4; charset=utf-8"


@contextmanager
def request_scope(request_id: str, method: str, path: str) -> Iterator[dict]:
    token = _request_id.set(request_id)
    t0 = time.perf_counter()
    scope: dict[str, Any] = {"status": 500, "route": path}
    try:
        yield scope
    finally:
        dt = time.perf_counter() - t0
        route = scope.get("route") or path
        if route not in ("/metrics", "/health"):
            HTTP_REQUESTS.labels(method, route, str(scope["status"])).inc()
            HTTP_SECONDS.labels(route).observe(dt)
            get_logger("http").info("request", extra={"method": method, "route": route, "status": scope["status"],
                                                      "duration_ms": round(dt * 1000, 1)})
        _request_id.reset(token)


# ---------------------------------------------------------------------------
# Traces (OpenTelemetry). The API package may be absent in CI: everything degrades to a no-op.
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace as _otel_trace
except ImportError:
    _otel_trace = None

_TRACING_READY = False


def setup_tracing(service_name: str) -> bool:
    """Export spans over OTLP/HTTP when OTEL_EXPORTER_OTLP_ENDPOINT is set and the SDK is installed."""
    global _TRACING_READY
    if _TRACING_READY or not SETTINGS.otlp_endpoint or _otel_trace is None:
        return _TRACING_READY
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        endpoint = SETTINGS.otlp_endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint += "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        _otel_trace.set_tracer_provider(provider)
        _TRACING_READY = True
    except Exception as exc:  # noqa: BLE001 — SDK not installed or endpoint unreachable at start
        get_logger("otel").warning("tracing disabled", extra={"error": str(exc)[:200]})
    return _TRACING_READY


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Any]:
    """A span when OpenTelemetry is available, otherwise nothing. Attributes must not contain content or secrets."""
    if _otel_trace is None:
        yield None
        return
    tracer = _otel_trace.get_tracer("vra")
    with tracer.start_as_current_span(name) as sp:
        for k, v in attrs.items():
            if v is not None:
                sp.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else str(v))
        yield sp


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

def _http_ok(url: str, api_key: str = "", timeout: float = 2.0) -> tuple[bool, str]:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 — operator-configured model endpoint
            return 200 <= r.status < 500, f"HTTP {r.status}"
    except urllib.error.HTTPError as exc:
        return exc.code < 500, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return False, type(exc).__name__


def readiness_checks() -> dict[str, dict[str, Any]]:
    from . import db
    from .config import LOCAL_PROVIDERS

    checks: dict[str, dict[str, Any]] = {}
    try:
        db.count_jobs()
        checks["database"] = {"ok": True, "required": True, "detail": "reachable"}
    except Exception as exc:  # noqa: BLE001
        checks["database"] = {"ok": False, "required": True, "detail": type(exc).__name__}
    try:
        from .graph import shared_components

        lib = shared_components()["library"]
        checks["policy_index"] = {"ok": bool(lib.chunks), "required": True,
                                  "detail": f"{len(lib.chunks)} chunks, embedder={lib.embedder.backend}, reranker={lib.reranker.backend}"}
    except Exception as exc:  # noqa: BLE001
        checks["policy_index"] = {"ok": False, "required": True, "detail": type(exc).__name__}
    seen: set[str] = set()
    for role in ("main", "small", "extract"):
        provider = SETTINGS.provider_for(role)
        if provider == "mock" or provider in seen:
            continue
        seen.add(provider)
        try:
            cfg = SETTINGS.resolved_llm(role=role, provider=provider)
            ok, detail = _http_ok(cfg["base_url"].rstrip("/") + "/models", cfg["api_key"])
        except ValueError as exc:
            ok, detail = False, str(exc)[:120]
        # A local model with a cloud fallback is optional: the service still works, at a cost.
        optional = provider in LOCAL_PROVIDERS and SETTINGS.llm_fallback and SETTINGS.llm_provider not in ("mock", provider)
        checks[f"llm:{provider}"] = {"ok": ok, "required": not optional, "detail": detail}
        if provider in LOCAL_PROVIDERS:
            LOCAL_LLM_UP.set(1 if ok else 0)
    return checks
