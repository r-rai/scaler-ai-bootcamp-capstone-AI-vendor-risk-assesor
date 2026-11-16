# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AI Vendor Risk Assessor (capstone). Given a question like "Can support use LedgerLens AI with customer tickets?", it retrieves internal policies (hybrid BM25 + vector RRF, cross-encoder rerank, ACL), researches the vendor on the web, deterministically checks each requirement, drafts vendor questionnaire items, and stops at a human approval gate. It never approves anything itself.

## Commands

Use the `.venv` (`source .venv/bin/activate`). Defaults are offline: `LLM_PROVIDER=mock`, `SEARCH_PROVIDER=offline`, so tests and demos need no API key or network. Copy `.env.example` to `.env` for live providers.

```bash
python scripts/check_setup.py                 # environment sanity check
python run_assessor.py "<question>" --tier T3 --users 35 [--groups public,employees]
python -m pytest -q                           # full suite (also: python scripts/run_all_tests.py)
python -m pytest tests/test_matcher.py::test_name -q   # single test
python eval/run_eval.py --strict --real --ablations    # scorecard; writes reports/eval_report.md (CI runs this)
python scripts/cache_admin.py stats|purge ...  # caches (audited purges); scripts/retention.py [--apply]; scripts/verify_audit.py
make help                                     # Makefile shortcuts (test, test-builtin, eval, lint, models, docker targets - docker unverified)
uvicorn app.api:app --reload --port 8000      # FastAPI backend
streamlit run app/ui.py                       # reviewer UI
python scripts/bugfix_agent.py scan|fix [<issue-id>|--manual "desc"]   # human-approved bug-fix agent
```

**Gotchas:** `tests/conftest.py` now forces mock/offline and VRA_CACHE=0, but `eval/run_eval.py` and `run_assessor.py` still read `.env` (which may select `LLM_PROVIDER=minimax`): export `LLM_PROVIDER=mock SEARCH_PROVIDER=offline` for the eval. Run the suite on both backends: default, and `EMBEDDER=builtin RERANKER=builtin` (the CI backend). `eval/run_eval.py` overwrites `reports/eval_report.md`. Embedder/reranker models live in `~/.cache/fastembed` and load cache-first with a download deadline (`assessor/rag_core.py::load_fastembed`); if the real backend is slow or missing the suite still runs on builtin. Never run `pkill -f` / `pgrep -f` with a pattern that appears in your own command line.

CI (`.github/workflows/ci.yml`) has five jobs: tests + strict eval (builtin backend, `requirements.txt` only), ruff (`ruff.toml`, correctness rules only), pip-audit on `requirements-prod.lock` + gitleaks, the same tests on the real fastembed backend, and a Docker build (never run: Docker is not installed here). Governance and operations docs: `docs/GOVERNANCE.md`, `docs/RUNBOOK.md`.

## Architecture

- **`assessor/graph.py`** is the orchestrator: `guard_input → route → {policy_qa | vendor_research → facts_only | vendor_research → match → questionnaire → human_gate → report | refuse}`. The same node functions run either as plain Python (`Pipeline.run_plain`) or as a LangGraph `StateGraph` (`Pipeline.build_langgraph`, with `MemorySaver` and `interrupt_before=["human_gate"]`). Keep node logic engine-agnostic; shared state is the `State` TypedDict. Completed assessments persist via `assessor/db.py` (SQLite in `runs/assessments.db`).
- **Policy side:** `rag_core.py` (BM25/vectors/RRF/reranker primitives, with NumPy fallbacks when fastembed is absent) → `policy_rag.py` (`PolicyLibrary`; ACL, active-status, trust and injection filters applied *before* ranking) over `data/policies/*`.
- **Vendor side:** `vendor_research.py` (plan creator → plan reviewer → parallel workers sharing a `Budget`, identity gate, injection quarantine, freshness decay), `memory.py` (SEC-12.1 change tracking in `runs/vendor_memory.json`). Offline mode reads `data/vendor_web/*corpus.json`.
- **Decision side:** `matcher.py` checks facts against `data/requirements.yaml` deterministically (no LLM); `questionnaire.py`, `report.py` build outputs; `schemas.py` holds the Pydantic contracts between stages.
- **Safety/quality:** `router.py` (4-way semantic router: policy/vendor/assessment/refuse), `guardrails.py` (PII redaction, injection regex/base64/LLM classifier, output policy), `graders.py` (answer + hallucination graders with retry).
- **`llm.py`** wraps any OpenAI-compatible provider plus the `mock` provider; model routing uses `LLM.small()` for high-volume tasks and the main model for planning/synthesis. Settings all come from `config.py` (env-driven).
- **`app/api.py`** (FastAPI; `X-API-Token` ACL, async jobs via `POST /assess?async_job=true`, `/vendors/{v}/history`, `/metrics`) and **`app/ui.py`** (Streamlit) sit on top of `Pipeline`.
- **`bugfix/`** + `scripts/bugfix_agent.py`: separate agentic bug-fixing tool (detect → read-only diagnose → human approve → branch `bugfix/<id>` → scoped fix → verify vs. baseline → human commit/discard). Runtime errors are logged by `assessor/errors.py` to `runs/issues/errors.jsonl`. It requires a clean git tree and an interactive terminal; it blocks reads of `.env`/`.git`/`runs/` and protects tests and gold sets.

## Live research

`vendor_research.py::live_iteration` streams: searches, page downloads and extraction run concurrently, and each page is read as soon as it arrives. At `VRA_MAX_SECONDS` the run is closed (`run.closed`): finished pages count, in-flight work is discarded, missing fields become UNKNOWN. `fetching.search` caps every provider call (`SEARCH_HARD_CAP_S`) and falls through Tavily/ddgs. Live runs are non-deterministic (the web changes); verdicts for a given set of facts are not. Settings: see `.env.example` (sections 6-9). Docker, compose and `ops/` are written but unverified until someone has run them.

## Conventions

- `tests/conftest.py` pins `AS_OF = 2026-11-01` and provides `fact()`/`profile()` helpers; freshness-dependent tests must use that date, not today.
- Gold sets (`data/gold_set*.json`) feed the eval thresholds; don't edit them to make an eval pass.
- `.env` holds secrets and is gitignored — edit `.env.example` for documenting new settings.
