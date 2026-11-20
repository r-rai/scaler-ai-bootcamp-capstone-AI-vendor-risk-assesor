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
python -m pytest -q              # 1478 tests (1477 pass, 2 skip without fastapi)
python eval/run_eval.py          # scorecard against data/gold_set.json → reports/eval_report.md
python eval/run_eval.py --strict # exit 1 on a failed threshold (use in CI)
```

## 3. Switch on a real LLM and real search (after the offline run is green)

Edit `.env`: pick one `LLM_PROVIDER` block (Gemini and Groq have free tiers; Ollama is local) and set
`SEARCH_PROVIDER=ddgs` (no key) or `tavily`. Install the extras: `pip install -r requirements-prod.txt`
(or the individual ones: `pip install openai ddgs trafilatura httpx`).
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
  policies/           11 Nimbu Mart documents + _metadata.json (access, status, trust)
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

Everything in `data/policies/`, `data/vendor_web/corpus.json` and `data/gold_set.json` is fictional and was written for this
course. `data/gold_set_real.json` and `data/vendor_web/real_vendors_corpus.json` are the exception: they hold cached
snapshots of **real** vendor trust pages (Zoho, Freshworks, LiveKit, Notion, Anthropic, OpenAI and four others) so the
real-vendor evaluation runs offline and deterministically in CI.

## 6. Running it as a service (production hardening)

Production hardening (hash-chained audit log, hashed-token auth, SSRF-safe fetching, caches, metrics). Every setting is
documented in `.env.example`; mock/offline stays the default, so CI and the demo need no key, network or GPU.

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
used if that stalls, so a bad network cannot hang startup or `/ready` (the fastembed backend; the optional
sentence-transformers fallback, not installed here, has no such deadline). If large downloads stall at 0 bytes, set
`HF_HUB_DISABLE_XET=1` (what `setup_models.sh` does by default).

**Measured live run** (Zoho Desk Zia, T3, 35 users; MiniMax for every role in the rows from `8dc4304` on (no local model running), ddgs search,
caches off, 2026-11-17, nothing else using the network). The 90 s limit applies to the run itself; the CLI adds a cold start
(about 9 s with caches off because the policy corpus is re-embedded, about 1 s with the default caches).

| Code at | Backends | Wall | In-run | Pages extracted / seen | Fields (of 9) | Searches failed |
|---|---|---|---|---|---|---|
| `1161a20` | builtin | 78 s | 76 s | 6 / 8 | 5 | 5 of 9 |
| `1161a20` | builtin | 73 s | 73 s | 8 / 9 | 6 | 4 of 9 |
| `a2a4682` | fastembed | 68 s | 56 s | 3 / 4 | 2 | 16 of 17 |
| `a2a4682` | fastembed | 97 s | 87 s | 5 / 6 | 8 | 13 of 16 |
| `8dc4304` | fastembed | 75 s | 65 s | 19 read | 7 | 7 of 18 |
| `8dc4304` | fastembed | 97 s | 88 s | 16 read | 7 | 2 of 9 |
| `69df346` | fastembed | 52 s | 42 s | 23 read | 8 | 2 of 16 |
| `69df346` | fastembed | 75 s | 66 s | 24 read | 8 | 3 of 18 |
| `69df346` | fastembed | 56 s | 47 s | 24 read | 7 | 1 of 16 |

(Rows from `8dc4304` on count pages read and fields filled from `profile.json`; the earlier rows counted extracted / seen
pages, so the columns are not exactly comparable.)

**What the two `8dc4304` rows showed.** Not search: in both runs one extraction call (the 20,000-character
`zoho.com/security` page) took 45 s and 82 s while every other call took 2 to 16 s, so a single call held the first pass open
until the deadline. Those two calls reported only 1,001 and 760 output tokens, so the likeliest cause is plain provider latency
plus the old 60 s timeout with two SDK retries (82 s fits a timeout and one retry); it was not proven. At `69df346` small and
extract calls are capped at `LLM_SMALL_TIMEOUT_S` (30 s) and every call's timeout is bounded by the time the run has left (unit
tested). The same commit fixes a separate latent bug found while reproducing: `LLM._create` retried any failure, timeouts
included, without `reasoning_effort`, and without it the same 3.3k-token page took 48 s and 4,743 output tokens instead of 6 s
and 519. Now only an HTTP 400/422 gets that retry.

So, for this vendor on this network: **three of three runs at `69df346` finished under 90 s wall-clock (52, 75 and 56 s) with
7 or 8 of 9 fields**, against one of the two before it. Three runs is a small sample, and none of them reached the 30 s cap
(slowest extract call 26.6 s) or used the parameter fallback, so they show MiniMax was not stalling that time, not that the cap
worked; the unit tests are what show that. All five runs ended BLOCKED on DP-3.1
(customer data stored in India). **That verdict was a matcher bug, not the vendor:** the extractor returned the region as `IN`
and the matcher compared exact strings against `India` (fixed in `f39df3e`; the five runs above are the pre-fix behaviour).
DuckDuckGo from this network fails some queries (1 to 7 per run). The earlier statement that Tavily was "untested" understated
it: it could not run at all, because `tavily-python` was not installed in the venv and the ddgs fallback hid that. It is
declared in `requirements-prod.txt`; install that file to use `SEARCH_PROVIDER=tavily`.

**Live accuracy against `data/gold_set_real.json` (2026-11-17; MiniMax main and Flash extract, Tavily search, caches off, gold
checked on 2026-11-01).** Harness: `SEARCH_PROVIDER=tavily VRA_CACHE=0 python eval/run_live_real.py` (one vendor at a time,
per-run and batch cost caps, search provider asserted from the trace, profiles saved so `--rescore DIR` re-runs the matcher
for free; `--dry` checks the scoring offline). The gold set stores expected *verdicts*, so "field accuracy" here is
per-requirement verdict accuracy over the 8 web-driven requirements (PROC-1.1 comes from the internal Approved Vendor List,
10/10, and is excluded).

| | Correct / 80 | Guesses | Overall verdict | Cost | Wall time |
|---|---|---|---|---|---|
| Baseline, 10 vendors, one run each | 32 (40%) | 4 | 4 of 10 | $0.10 | 46 to 80 s (deadline never hit) |
| Same saved profiles, matcher after `f39df3e` | 34 (42.5%) | 4 | 6 of 10 | $0 (rescored) | n/a |

**This is below the 80% gate, so extraction stays on MiniMax Flash and must not move to the local model.** Most misses are
conservative (22 UNKNOWN because nothing was found, 17 NEEDS_REVIEW), not wrong values. By requirement (baseline): SEC-4.2
1/10, SEC-9.3 2/10, SEC-7.1 4/10, DP-3.1 4/10 (6/10 after the fix), AI-3.4 4/10, DP-2.2 5/10, DP-5.4 5/10, AI-2.1 7/10.
Known causes: official pages such as `zoho.com/trust` are extracted as `claimed` with confidence near 0.24, so SEC-4.2
(documented and dated) asks for review; India regions for Freshworks and LiveKit were never found. The four guesses (a verdict
of PASS or FAIL where the gold set says NEEDS_REVIEW) are the most important finding: AI-3.4 for Freshworks and Notion (a partial
sub-processor list passes), AI-2.1 for OpenAI (PASS resting on a third-party claim of confidence 0.02) and SEC-7.1 for Ringg
(FAIL because the page does not mention SSO). Limits: one run per vendor; live runs are non-deterministic; the web has moved
since 2026-11-01 (Lorikeet DP-5.4 may be drift: the vendor states 60 days, the gold set says PASS). The gold set was not edited.

**Final live run (2026-11-20, code `2130b77`; same harness, same gold set, Tavily + Firecrawl read-through, caches off).**
Everything below is as measured; the 80% gate was **not** met.

| Gate (pre-registered) | Result | |
|---|---|---|
| Correct cells >= 64/80 (80%) | **38/80 = 47.5%** | FAIL |
| Guesses (PASS/FAIL where gold says NEEDS_REVIEW) = 0 | 0 | PASS |
| False PASS on a gold-FAIL cell = 0 | 1 (OpenAI DP-3.1) | FAIL |
| Every run inside 90 s, no deadline hit | median 50 s (in-run, upper median 54.8 s), max 69 s | PASS |
| Integrity (provider as configured, no local model, no key in files) | OK | PASS |

Progress on the same ten vendors: 32/80 (2026-11-20 re-measure of the baseline code), 38/80 (final). Only about one cell of the
gain is attributable to the matcher changes of this session (the guess guard); the rest is live variance (the web and the
extractor change between runs), so a gain of under about 4 cells should not be read as an improvement. Cost $0.13, 155 Tavily
searches.

**Why 80% is out of reach here.** The 42 misses by cause: not found on any page read 15; certificate stated without a date 5
(the public pages say "SOC 2 Type II certified" and nothing else, so a dated, documented certificate does not exist to find);
wrong value 4; about 12 cells where the rules send the vendor to a person although the gold says PASS: SEC-9.3 (real
security-incident search results name Zoho, LiveKit, Notion, Anthropic and OpenAI), AI-3.4 (a third-party model provider is
listed), DP-5.4 (retention terms), DP-2.2 ("DPA on request"); and 6 more spread over unknown-by-rule, conflicting-source and SEC-9.3 cells that could not be established. Those are the policy working as designed against a gold set that
is optimistic: the gold is 65 PASS, 10 NEEDS_REVIEW and 5 FAIL, so a system that always answered PASS would score 81%. Moving the
rules to match the gold would be tuning to the data, so it was not done. The achievable ceiling without that is about 51/80 (64%).

**Known problems, not hidden.**
- OpenAI DP-3.1 is a false PASS: a *claimed* (confidence 0.43) fact from `openai.com/business-data` lists India. The gold says
  FAIL, which is probably stale because OpenAI now states India residency; this is unverified. A "documented evidence only" rule
  for DP-3.1 would remove it but cost about 4 correct cells, so it is left off.
- SEC-9.3 has a bounded "searched, nothing found" rule (`assessor/incident_scan.py`). It produced **no** PASS in the final run,
  so its safety on real PASS cells is unverified.
- Commit `c1cfeb3` (after the final run) makes a page's "no incidents disclosed" pass only when the incident search also came back
  clean. Rescoring the final profiles is unchanged (they carry no search record); the expected live effect is -2 cells (38 -> 36)
  and it has **not** been measured live.
- The extractor reads only the first 12,000 characters of a page (`EXTRACT_WINDOW_CHARS`); 22% of fetched characters are never
  shown to the model. A 20,000-character window failed its pre-registered stability gate and was not adopted.
- Extraction is noisy: 17.6% of claim elements differ between two identical runs.
- One run per vendor; live results are not reproducible cell for cell. The CLI cold start (1 to 9 s) is not in the in-run times.

**Latency work.** Slow runs came from single stalled extraction calls, not from search. The fix is a hedged duplicate request
after 10 s for extraction calls on cloud models and 6 page-reading slots (`LLM_HEDGE_AFTER_S`, `VRA_LLM_PARALLEL`; set to 0 and 4
to switch off). Tail of extract calls: p99 36.0 s (was 43.0 s), max 36.9 s (was 51.7 s); p95 got worse (21.0 s vs 17.9 s), so no
p95 gain is claimed.

**Kaggle "Vendor Verdict" (separate task).** Best submission 0.99881 (about 2 to 4 wrong rows; rank 12th to 15th as others
improved). Negative results that were measured and not adopted: claim verifiers (grounding, NLI, zero-shot) rejected correct
claims; a multi-agent system with a local router and decision maker failed its holdout gates (79 wrong rows against 3 for the
baseline); stage-B adjudication with a second model failed. Kaggle data is not in this repository.

**Docker.** `Dockerfile`, `docker-compose.yml` (`api`, `ui`, `llm`, `fetcher` sandbox, `obs` profile) and `ops/`
(Prometheus alerts, Grafana dashboards, Loki, promtail) are written but **have not been built or run**. Docker Desktop is now
installed, and `docker compose config` accepts every profile (checked through the Windows `docker.exe`), but that only proves the
file parses. Docker Desktop's WSL integration is not enabled for this Ubuntu distro, so `docker` does not work from inside it yet.
Treat the stack as untested until `make up` and `make up-obs` have run through it, and the `fetcher` sandbox checks have passed.

**Checks.** `LLM_PROVIDER=mock SEARCH_PROVIDER=offline python -m pytest -q`, also with `EMBEDDER=builtin RERANKER=builtin`,
then `python eval/run_eval.py --strict --real --ablations`.

