# Vendor Verdict multi-agent system (Plan B): design

Status: APPROVED by the user on 2026-11-19. Nothing is built.
User decisions: (1) time budget agreed: about one holdout and one test MiniMax run (each about 6-8M tokens, 1.5-3 h), plus local-model runs, possibly overnight; (2) safety net agreed: submission #2 (ref 58543825, 0.99881) stays one of the two finals whatever B scores.
Kaggle deadline: 2026-11-30 18:29:59 UTC (11 days). Submissions: 5 a day; up to 2 finals, chosen by the user in the web UI.

## 1. Intent

The Scaler capstone is judged on the Kaggle score, a presentation and an interview. The user wants a **multi-agent deep-research system** to show and explain: separate agent responsibilities, a graph, handovers between agents and a trace of each one. It should follow the course's deep-search agent (`study-material/shivank-deep-search-agent`: plan creator, plan reviewer, parallel workers, validator, merger, evaluator loop, LangGraph version in lab 12) and the Agentic RAG kit's graders (`Agentic_RAG_Study_Kit/code/agents.py`).

A second aim is a chance to close the last 2 to 4 wrong rows (score 0.99881; ten teams are at 1.000). The host states that correct facts plus `rules_lite` score exactly 1.000, so all the remaining error is in reading the pages.

Models: MiniMax (hosted, the only hosted option) is the main model and the reader. A local model (llama.cpp, Qwen3-4B Q4 on a GTX 1650 with 4 GB of VRAM) takes the classifier, grader and decision-maker roles.

## 2. Goals, non-goals, anti-goals

**Goals**
- G1. A graph for each case with distinct agents: Planner, Plan Reviewer, Source Validator, Page Router, four Field Specialists, Evidence Grader, Merger, Decision Maker, Evaluator (continue or stop), Verdict, Auditor.
- G2. The same node functions run as plain Python and as a LangGraph `StateGraph`. This is the repo convention (`assessor/graph.py`) and the course's lab 12 pattern.
- G3. Presentation outputs: a Mermaid drawing of the graph, a handover trace per case (JSON plus Markdown), an ablation table in the style of lab 11, and a cost and time table.
- G4. A candidate `submission.csv` written through the existing `write_submission` and self-check.
- G5. Every LLM call is cached, so a cache-only run reproduces the candidate and every demo exactly, with no network.

**Non-goals**
- Changing `rules_lite.py`, `metric.py` or the data.
- Changing the existing runner (`eval/run_kaggle.py`, `eval/consensus.py`). They stay as the baseline and the fallback.
- Live web search, CrewAI, fine-tuning, or the product's assessor graph.

**Anti-goals (a failure even if everything runs)**
- The final standing falls below 0.99881. Guard: submission #2 (ref 58543825) stays one of the two finals.
- Any prompt or rule fitted to train or dev page text or to labels, any probing of the leaderboard, or any test text printed. Only counts and case ids are reported.
- A local model deleting a claim. Stage B's damage came from dropping claims, so local agents may only flag, or choose among MiniMax readings.
- Results in the presentation that a cache-only run cannot reproduce.

## 3. Constraints

- The user's hard rules: pre-register the pass/fail rules before each run, run once, report as measured, get an explicit yes for every submission, and commit or push only when asked.
- Dev is contaminated: commit 2228f69 changed the prompt from dev misses. Dev is therefore used **only** for smoke tests (crashes, parse failures, schema errors), never to judge or reword a prompt.
- 7 holdout rows have exposed labels (c0060__SEC-4.2, c0312__SEC-4.2-T2, c0678__SEC-4.2-T2, c0062__DP-3.1, c0288__SEC-7.1, c0426__SEC-4.2, c0512__SEC-9.3). Every holdout rule excludes them.
- Throughput: one base-prompt test pass is about 3.7M input tokens in 41-45 minutes with 6 workers. A pass with a longer prompt took 97 minutes. MiniMax has a monthly allowance of 1.5B tokens, so time is the limit, not tokens. `--max-usd` is not a real bound, because `price_for` uses one fallback price.
- The local model's speed on the GTX 1650 has never been measured. It decides how much the local agents can do (Task 0 below).
- `data_kaggle/` is gitignored and is never published.
- Tests run on both backends (default, and `EMBEDDER=builtin RERANKER=builtin`), and `ruff` must stay clean.

## 4. Approaches considered

| | What it is | Trade-off |
|---|---|---|
| A | The existing pipeline as graph nodes, plus an escalation lane (local checker, then MiniMax specialist, then local decision maker) for unsure items only | Safest: the output equals submission #2 unless two model families agree on a change. Smaller story for the presentation |
| **B (chosen)** | **A full deep-research rewrite: planner, router, field specialists, graders, merger, decision maker, evaluator loop** | **Closest to the course and the strongest presentation. New prompts mean a full re-extraction, and our local checks cannot confirm a gain at this level. The score risk is bounded only by keeping submission #2 as a final** |
| C | A graph wrapper only, with no new agents | No score risk and no score gain; a thin demo |

The user chose **B** on 2026-11-19, for the presentation value. CrewAI was dropped because the repo already uses LangGraph and CrewAI's free-form delegation is not reproducible. Fine-tuning was dropped because it conflicts with the no-fitting rule.

## 5. The graph

```
START → plan → review_plan → validate_sources → route_pages
      → specialists (fan-out: 4 field specialists in parallel)
      → grade_evidence → merge → decide_conflicts → evaluate ─┬─ continue (iteration < 2) → specialists (targeted re-read)
                                                               └─ stop → verdict → audit → END
```

One graph run handles one case. Cases run in parallel, and pages run in parallel within each specialist. The shared state is a `CaseState` TypedDict: the case, eligible pages, plan, routes, readings per specialist and iteration, grades, merged facts, conflicts and decisions, verdicts, trace and budget.

| # | Node / agent | Model | What it does | What it may change |
|---|---|---|---|---|
| 1 | **Planner** | MiniMax (main) | Reads the case metadata and the page *list* (url, title, source_type, publisher, dates; no page text). Decides which field families each page is likely to cover, and the priorities | Proposes routes only |
| 2 | **Plan Reviewer** | deterministic | Course `review_plan` pattern. Fixes gaps: a registry page always goes to Certifications, and the final routes are the union of the Planner and the Page Router (node 4). Records every fix | Adds routes, never removes them |
| 3 | **Source Validator** | deterministic (+ local flag) | The existing `kaggle_policy.classify_page` (official domain, registry, look-alike, user content) and `quarantine_hits`. The local injection classifier runs as a logged second opinion only | Excludes pages by the published evidence policy, unchanged |
| 4 | **Page Router** | local | A multi-label, recall-first classifier: which of the 4 families does this eligible page mention? | Adds routes (through the reviewer) |
| 5 | **Field Specialists** ×4 | MiniMax-M3.1-Flash | Each reads its routed pages one at a time and returns facts **in the `rules_lite` schema** with a verbatim quote per item. **Certifications:** `security_certifications` (name, status, date, valid_until, holder, issuer). **Data:** `data_regions`, `data_retention`, `dpa_available`. **AI:** `trains_on_customer_data`, `subprocessors`. **Access & incidents:** `sso_saml`, `security_incidents` | Produces readings |
| 6 | **Evidence Grader** | deterministic + local | Course hallucination-grader pattern. (a) Grounding: is the quote in the page text after normalisation? (b) Local: "does this quote state this value?", answering supported, unsupported or unsure | Flags only, never removes |
| 7 | **Merger** | deterministic | The existing `build_facts` rules (newer official page wins, registry holder check against the legal entity, documented vs claimed), through an adapter from specialist readings | The existing policy, unchanged |
| 8 | **Decision Maker** | local | For each conflict item (field + 2 or more candidate readings + quotes): picks one candidate, or abstains. Abstaining keeps the Merger's default | Chooses among MiniMax readings only; gated by rule R-DM |
| 9 | **Evaluator** | deterministic | Course `evaluate_progress` pattern. Continue if a field has an unsupported or unsure grade, or an unresolved conflict, and iteration < 2. Iteration 2 = a targeted re-read: the specialist sees **all** of that field's eligible pages of the case at once, and must cite the `page_id` of every item, so the Merger's "newer official page wins" and registry-holder rules still apply | Starts the loop |
| 10 | **Verdict** | deterministic | `rules_lite.evaluate` and `rules_lite.overall`, unchanged | none |
| 11 | **Auditor** | deterministic | Writes the handover trace (agent, inputs summary, outputs, tokens, seconds per node), the case Markdown and the Mermaid drawing | none |

Conflict items for node 8 come from: (i) iteration 1 and iteration 2 readings that differ; (ii) a reading graded unsupported against another reading of the same field; (iii) a specialist reading that differs from the baseline Flash reading of the same page (the cached submission #2 samples). That gives a second reader at no extra cost, and it is label-free.

## 6. Prompts

- The four specialist prompts, the Planner prompt and the local prompts are written by Claude **only** from published material: the `rules_lite.py` docstring (the fact schema), `requirements.json`, the competition's data description, and the evidence policy in `kaggle_policy.py`. No train or dev page text, no label.
- Each prompt gets a version hash. Prompts are frozen and their hashes recorded in the pre-registration **before** the first holdout run.
- Dev smoke may lead to fixes to **code or output schema** (a parse failure, a missing key). It may not lead to changes in prompt wording that target a reading. Every such fix is logged.

## 7. Caching, cost and time

- All calls (MiniMax and local) are cached in `data_kaggle/cache/agents/<role>/`, keyed by role + prompt version + model + input hash + salt. `--cache-only` reproduces any run.
- A run starts only after `--estimate` prints the pages, calls and tokens per role, and the user says go.
- Expected MiniMax volume per test pass: about 1.5-2x the baseline (pages read by more than one specialist, plus the planner and iteration 2), so about 6-8M tokens and about 1.5-3 h. Local volume is set by Task 0.

## 8. Pre-registered rules (written into the package docstring before the first holdout run)

| Rule | Measured on | Pass when |
|---|---|---|
| R-route | holdout and test, label-free | Every page-field on which the baseline Flash cache has a claim is routed to that field's specialist (100% recall versus baseline) |
| R-smoke | dev, 20 cases | 0 crashes, 0 unparsed outputs, 100% of eligible pages read |
| R-hold | holdout, the 7 exposed rows excluded | B's wrong rows ≤ the wrong rows of baseline sample s0 (prompt 9566ad4b6b95, unsalted), and false-confident ≤ s0's, both counted over the same rows |
| R-DM | holdout, exposed rows excluded | Rows the Decision Maker changes: wrong to right ≥ right to wrong. On fail, the Decision Maker is switched to always abstain for the candidate (the switch is part of the rule, not a later choice) |
| R-test | test, label-free | B changes ≤ 30 rows versus submission #2 |

Candidate rule: R-route, R-smoke, R-hold and R-test pass gives the **B candidate**, which needs the user's yes to submit. If any of them fails, B is **not submitted**, and the system is still presented with its honest numbers. Final selection: submission #2 plus the B submission, if B was submitted (the user's call in the web UI).

Reporting: counts and case ids only. For analyses over a few rows, report pass/fail and min/max, never a distribution.

## 9. Ablations for the presentation (lab 11 style; holdout, cache only, counts)

Baseline runner; full B; B without the Page Router; B without the Evidence Grader; B with the Decision Maker always abstaining; B without iteration 2. They are cheap because every call is cached.

## 10. Code layout (builder's call, proposed)

A new top-level package `vendor_verdict/`: `state.py` (CaseState), `agents/` (planner, reviewer, router, specialists, grader, decision_maker), `nodes.py`, `graph.py` (plain and LangGraph engines), `cache.py`, `trace.py`, `prompts.py`, `rules.py` (pre-registration and gates), and `__main__.py` (`python -m vendor_verdict run --split test|holdout|dev --cache-only --estimate --salt`). It reuses `eval/kaggle_policy.py` (triage, merge) and `eval/run_kaggle.py` (`write_submission`, data loading) without changing them. Tests go in `tests/test_vv_*.py` on the mock provider and a fake local model.

## 11. Schedule

| Days (from 2026-11-20) | Work |
|---|---|
| 1 | Task 0 spike: measure the local model's speed and JSON reliability. Graph skeleton on the mock provider. Cache, trace |
| 2-4 | Specialists, adapter to the Merger, Planner and Reviewer, Router, Grader, Decision Maker, Evaluator loop, tests |
| 5 | Dev smoke (R-smoke), freeze prompts, write the pre-registration, estimates |
| 6 | Holdout run, R-route, R-hold, R-DM |
| 7 | Test run, R-test, candidate; submission only on the user's yes |
| 8-11 | Ablations, slides material, buffer. The deadline is the end of day 11 |

## 12. Left to the builder

- Exact prompt text (within section 6), the exact JSON schemas per specialist, and how many parallel workers.
- The local model's scope inside its roles, sized by the Task 0 measurement. Priority if it is slow: Grader on certifications and the enum fields first, then the Router, then the Decision Maker on conflicts only.
- How the adapter maps specialist facts into `PageClaims` so the existing `build_facts` works unchanged.
