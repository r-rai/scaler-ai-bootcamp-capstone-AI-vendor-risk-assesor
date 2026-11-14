# AI Vendor Risk Assessor — Complete Capstone Solution (M0–M6)

**Capstone Project by Shivank Agrawal · Future with Shivank**

> *"Can our support team use LedgerLens AI with customer tickets?"*
> The assessor reads internal policies (with hybrid BM25 + vector search, RRF, cross-encoder reranking, and strict ACL), researches the vendor on the web (with a Plan Creator/Reviewer, parallel workers, identity gate, prompt-injection quarantine, and freshness decay), checks every requirement deterministically, drafts the exact questionnaire items to send the vendor, and hands a review file to a named human approver. It never approves anything itself.

---

## 1. Quickstart on Windows WSL2 (with `uv` and MiniMax API)

```bash
cd project-vendor-risk-assessor

# Create and activate virtual environment using uv (or python3 -m venv .venv)
uv venv .venv
source .venv/bin/activate

# Install core + live mode + product dependencies
uv pip install -r requirements.txt
uv pip install openai ddgs trafilatura httpx fastapi "uvicorn[standard]" streamlit fastembed

# Configure environment
cp .env.example .env
```

### Configuring Your MiniMax API Key & Free Tools in `.env`
MiniMax provides an OpenAI-compatible Chat Completions endpoint (`https://api.minimax.io/v1`). To minimize cost and latency, this solution uses **model routing**:
- **Main LLM (`MiniMax-M3`)**: Used for research planning and grounded policy synthesis.
- **Small LLM (`MiniMax-M3.1-Flash`)**: Used for high-volume claim extraction, 4-way question routing, Answer/Hallucination grading, and second-layer prompt-injection classification.
- **Free Search & Local RAG**: Uses `ddgs` (DuckDuckGo metasearch, 100% free, no API key required) + `trafilatura` for page extraction, and `fastembed` (`BAAI/bge-small-en-v1.5` + `Xenova/ms-marco-MiniLM-L-6-v2`) running 100% locally on WSL2 CPU/GPU (with a zero-download NumPy semantic bi-encoder/cross-encoder fallback built in).

```ini
LLM_PROVIDER=minimax
MINIMAX_API_KEY=your-minimax-api-key
MINIMAX_BASE_URL=https://api.minimax.io/v1
LLM_MODEL=MiniMax-M3
LLM_SMALL_MODEL=MiniMax-M3.1-Flash

SEARCH_PROVIDER=offline   # use 'offline' for deterministic tests/demos, or 'ddgs' for live web search
EMBEDDER=auto             # uses fastembed/sentence-transformers if installed, else built-in vector engine
RERANKER=auto
```

---

## 2. Running the CLI, Tests, Evaluation & Product Services

### Run the CLI Demo (Offline & Live)
```bash
python scripts/check_setup.py
python run_assessor.py                                                                          # LedgerLens AI, T3, 35 users -> BLOCKED
python run_assessor.py "Can we use Parakeet Translate for support chats?" --tier T3 --users 60  # -> CONDITIONAL
python run_assessor.py "Can the product team use Parakeet Translate for roadmap docs?" --tier T2 --users 8 # -> READY FOR APPROVAL
python run_assessor.py "Can marketing use QuillDesk for customer replies?" --tier T3 --users 12 # -> BLOCKED (2 blockers)
python run_assessor.py "Who approves a tool that handles customer data?"                        # -> AI-5.1 (Head of InfoSec and DPO)
python run_assessor.py --groups public,employees                                                # -> Approved Vendor List hidden by ACL
```

### Run the Full Test Suite (116 Tests) & Scorecard Evaluation
```bash
python -m pytest -q                                       # or: python scripts/run_all_tests.py
python eval/run_eval.py --strict --real --ablations       # writes reports/eval_report.md
```

### Run the FastAPI Service & Streamlit Reviewer UI (Milestone 5)
```bash
# Terminal 1: FastAPI backend (with server-side token ACL, async job queue, SQLite persistence, and SEC-12.1 history)
uvicorn app.api:app --reload --port 8000

# Terminal 2: Streamlit Reviewer Workstation
streamlit run app/ui.py
```

---

## 3. Milestone Implementation Summary (M0–M6)

| Milestone | Files Implemented | Key Engineering Highlights |
|---|---|---|
| **M1: Production Policy Library** | `assessor/rag_core.py`, `assessor/policy_rag.py`, `data/policies/*` | Hybrid search (BM25 + dense vectors fused via RRF `k=60`) + cross-encoder reranker. Fixes `pq-who-approves` (`AI-5.1` ranked #1). Pre-ranking ACL, active status (`draft` and `archived` excluded), trust, and injection filters. Added 3 new policies (`ai_incident_response_runbook.md`, `vendor_offboarding_policy.md`, `ai_policy_2027_draft.md`) and 3 new gold questions. |
| **M2: Vendor Research (Deep Search)** | `assessor/vendor_research.py`, `assessor/memory.py`, `data/gold_set_real.json`, `data/vendor_web/real_vendors_corpus.json` | Plan Creator → Plan Reviewer (`create_research_plan`, `review_research_plan`), parallel workers (`ThreadPoolExecutor`) with shared `Budget`, small-model routing (`LLM.small()`), spotlighted extraction, identity gate, and SEC-12.1 change-tracking memory (`runs/vendor_memory.json`). Hand-checked 10 real AI/SaaS vendors × 9 fields. |
| **M3: Agentic Router & Graders** | `assessor/router.py`, `assessor/graders.py`, `tests/test_routing_30.py` | Semantic 4-way LLM router (`policy`, `vendor`, `assessment`, `refuse`) achieving 100% accuracy on the 30-question test suite, plus the "Break the Router on purpose" ablation test. Answer Grader + Hallucination Grader with self-correction retry loop. |
| **M4: Layered Security & Red Team** | `assessor/guardrails.py`, `tests/test_red_team.py` | Indian PAN, Aadhaar, mobile, email, card, and API key redaction (`AI-4.1`); regex + base64 + second-layer semantic/LLM injection classifier (`classify_injection`); 15 red-team attack tests asserting exact catch points. |
| **M5: Product, Graph, API & UI** | `assessor/db.py`, `assessor/graph.py`, `app/api.py`, `app/ui.py` | SQLite persistence (`runs/assessments.db`), LangGraph checkpointer (`MemorySaver`) + `interrupt_before=["human_gate"]`, server-side API token ACL (`X-API-Token`), async job queue (`POST /assess?async_job=true`), `/vendors/{vendor}/history`, `/metrics` (p50/p95), and full Streamlit UI. |
| **M6: Evaluation, Ablations & CI** | `eval/run_eval.py`, `.github/workflows/ci.yml` | Calibrated `reason_faithfulness` judge, 10 real-vendor gold set evaluation, 5-way component ablation table, and GitHub Actions CI pipeline. |
