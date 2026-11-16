# AI Vendor Risk Assessor — capstone starter kit

**Capstone project by Shivank Agrawal · Future with Shivank**

> *"Can our support team use LedgerLens AI with customer tickets?"*
> The assessor reads our policies, researches the vendor on the web, checks every requirement
> deterministically, drafts the questions we still need to ask the vendor, and hands a review file to a
> human approver. It never approves anything itself.

This repo is the **starter kit**. It runs end to end **offline** (no API key, no internet) against a
fictional company (Nimbu Mart) and three fictional vendors, so you can see the finished shape on day one.
Then you replace the simplified parts with the real components you built in the three courses:

| Course | What you bring into this project |
|---|---|
| **Production RAG from Scratch** | the policy library: ingestion, chunking by clause, hybrid search + reranking, grounded answers with citations, access control, evaluation, the FastAPI service |
| **Agentic RAG** | the Router (policy / vendor / assessment), the Answer Grader and the Hallucination Grader |
| **Deep Search Agents** | vendor research: planner, parallel executor, identity gate, freshness + authority scoring, merging with provenance, prompt-injection quarantine, budgets, LangGraph |

Open **`Capstone-AI-Vendor-Risk-Assessor-Shivank-Agrawal.html`** in your browser for the full brief:
milestones, the module-by-module course map, the rubric and the demo checklist.

---

## 1. Setup (10 minutes)

Python 3.10 or newer.

```bash
cd vendor-risk-assessor
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env
python scripts/check_setup.py
```

## 2. Run it

```bash
python run_assessor.py                                                          # LedgerLens AI, T3, 35 users → BLOCKED
python run_assessor.py "Can we use Parakeet Translate for support chats?" --tier T3 --users 60   # → CONDITIONAL
python run_assessor.py "Can the product team use Parakeet Translate for roadmap docs?" --tier T2 --users 8   # → READY FOR APPROVAL
python run_assessor.py "Can marketing use QuillDesk for customer replies?" --tier T3 --users 12   # → BLOCKED (2 blockers)
python run_assessor.py "Where must customer personal data be stored?"            # policy question
python run_assessor.py "Does QuillDesk have SOC 2?"                               # vendor facts only
python run_assessor.py --groups public,employees                                  # outside Security: the Approved Vendor List is hidden
python run_assessor.py --interactive --approver "A. Sharma"                       # record a human decision
```

Each assessment writes `runs/<timestamp>-assess/`:

| File | What it is |
|---|---|
| `report.md` | the review file (PROC-2.1): verdict table with source URL + date, facts, open questions, approval block |
| `report.json` | the same, structured |
| `profile.json` | everything established about the vendor, with scores and conflicts |
| `trace.jsonl` | every step: searches, rejections, quarantines, LLM calls, verdicts. **Read this to debug.** |
| `vendor_questions.txt` | the email to send the vendor |

Tests and evaluation:

```bash
python -m pytest -q              # 70 tests
python eval/run_eval.py          # scorecard against data/gold_set.json → reports/eval_report.md
python eval/run_eval.py --strict # exit 1 on a failed threshold (use in CI)
```

## 3. Switch on a real LLM and real search (after the offline run is green)

Edit `.env`: pick one `LLM_PROVIDER` block (Gemini and Groq have free tiers; Ollama is local) and set
`SEARCH_PROVIDER=ddgs` (no key) or `tavily`. Install the extras: `pip install openai ddgs trafilatura httpx`.
Then assess a **real** vendor:

```bash
python run_assessor.py "Can our support team use <Tool> for ticket summaries?" --vendor "<Tool>" --domain <vendor.com> --tier T3 --users 35 --max-usd 0.30
```

Live research is an iterative loop (plan → parallel workers → identity/injection gates → merge → evaluate) that streams pages
through extraction and stops at `VRA_MAX_SECONDS` (default 90), returning whatever it has finished; facts it did not
establish are reported UNKNOWN. See section 6 for the service, models, caches and measured numbers.

## 4. The code, in reading order

```
assessor/
  schemas.py          every contract between stages (Pydantic)                 ← read first
  matcher.py          requirement × fact → verdict. Deterministic. Tested.      ← the heart
  policy_rag.py       policy library: clause chunks, BM25, ACL, versions, extractive answers
  vendor_research.py  deep search retarget: plan → gather → identity → quarantine → extract → score → merge
  router.py           policy / vendor / assessment / refuse
  graders.py          relevance + groundedness checks on policy answers
  guardrails.py       input redaction, intent check, output policy
  questionnaire.py    UNKNOWN / NEEDS_REVIEW → questions from template NM-SEC-Q
  report.py           Markdown + JSON review file
  graph.py            the orchestrator: plain Python AND LangGraph, same nodes
  llm.py / tracing.py / config.py   provider wrapper, trace + budget, settings
data/
  policies/           8 Nimbu Mart documents + _metadata.json (access, status, trust)
  requirements.yaml   11 machine-checkable rules, each tied to a clause
  vendor_web/corpus.json   the fictional web: 23 pages, deliberately messy
  gold_set.json       hand-checked expected verdicts (5 cases) + policy questions
  gold_set_template.json   fill this for REAL vendors
eval/run_eval.py      the scorecard
tests/                run `python -m pytest -q`
app/api.py, app/ui.py FastAPI + Streamlit skeletons (Milestone 5)
```

Every file starts with a **COURSE MAP** comment and a **LEARNER TODO** block that names the milestone and
the course module/lab to use.

## 5. Rules of the project (read before you write any code)

1. **The LLM never decides a verdict.** It reads pages and drafts text. `matcher.py` decides.
2. **UNKNOWN beats a guess.** A PASS without a URL is a bug; the output guardrail downgrades it.
3. **A claim is not evidence.** "We are SOC 2 compliant" is `claimed`; a report, certificate or registry entry is `documented`.
4. **Identity is a gate.** A page about a same-name company is rejected before anything is read from it.
5. **Untrusted content never reaches the verdict.** Pages that talk to the AI are quarantined; forum rumours are dropped and logged.
6. **Access control lives in retrieval.** The Approved Vendor List is invisible to people outside Security/Procurement, and the assistant says so instead of leaking it.
7. **Humans approve.** The report ends in an approval block that only a named person can fill.

All companies, people, domains and documents in `data/` are fictional and were written for this course.

## 6. Running it as a service (production hardening)

Branch `prod-hardening`. Every setting is documented in `.env.example`; mock/offline stays the default, so CI and the
demo need no key, network or GPU.

**What was added.** Hashed-token auth and role-checked, multi-signature approvals (T3 needs Head of InfoSec and DPO;
the requester cannot approve; BLOCKED cannot be approved); a hash-chained audit log (`scripts/verify_audit.py`);
provenance stamp on every report; per-role model routing with a circuit breaker; SQLite caches (embeddings, search,
pages, validated LLM output, and a semantic cache for policy questions only, scoped by ACL and corpus hash); a real
cross-encoder reranker; SSRF-safe fetching; a shared prompt-injection scanner plus an optional local ONNX classifier;
JSON logs, Prometheus metrics, OpenTelemetry spans and a cost table.

**Local models** (GTX 1650, 4 GB): `scripts/setup_models.sh` downloads them, `scripts/serve_local_llm.sh` serves the
small-role model with llama.cpp. Bake-off winner Qwen3-4B-Instruct-2507 Q4_K_M (about 3.1 GB VRAM) scored 30/30 routing,
18/18 injection, 33/33 identity and 8/8 grader cases; Phi-4-mini scored 28/30 on routing. The injection classifier
(protectai deberta-v3 ONNX) catches blatant attacks but misses paraphrases and once flagged a clean page, so it is an
extra layer at a 0.99 threshold, never the only one. Verdicts never come from a model: `matcher.py` is rules only.

**Model files.** The embedder and reranker are read from `~/.cache/fastembed` (override `FASTEMBED_CACHE_PATH`), not
`/tmp`. A model that is not cached is downloaded with a deadline (`VRA_MODEL_LOAD_TIMEOUT_S`), and the builtin backend is
used if that stalls, so a bad network cannot hang startup or `/ready`. If large downloads stall at 0 bytes, set
`HF_HUB_DISABLE_XET=1` (what `setup_models.sh` does by default).

**Measured live run** (Zoho Desk Zia, T3, 35 users; MiniMax main and extract, local Qwen for small roles, ddgs search,
caches off, builtin embedder and reranker, 2026-11-17, residential network): two clean runs took 78 s and 73 s wall-clock
and read 8 and 9 pages for 5 and 6 facts. Both stopped on the time limit rather than finishing every sub-question. A
run that shared the network with a model download took 100 s and 3 pages, which is why the numbers above are the clean
ones. The overall outcome differed between the two runs (BLOCKED, then CONDITIONAL) because the live web returned
different pages; for the same facts the matcher always gives the same verdicts. DuckDuckGo from this network times out
on roughly half of the queries; Tavily (`SEARCH_PROVIDER=tavily`, free credits) is the better primary where a key is
available. No live accuracy figure against `data/gold_set_real.json` has been measured yet.

**Docker.** `Dockerfile`, `docker-compose.yml` (`api`, `ui`, `llm`, `fetcher` sandbox, `obs` profile) and `ops/`
(Prometheus alerts, Grafana dashboards, Loki, promtail) are written but **have not been built or run**: Docker is not
installed on the development machine. Treat them as untested until `docker compose --profile obs up` has been run.

**Checks.** `LLM_PROVIDER=mock SEARCH_PROVIDER=offline python -m pytest -q`, also with `EMBEDDER=builtin RERANKER=builtin`,
then `python eval/run_eval.py --strict --real --ablations`.

