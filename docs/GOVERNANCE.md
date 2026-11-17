# Governance

What the assistant is allowed to do, which code and test enforces each rule, which models see what data, and what is
still weak. Written from the code as of branch `prod-hardening`. A control listed here without a test is marked **no test**.

## 1. Controls matrix

| Rule | Where it is enforced | Test |
|---|---|---|
| The assistant never approves a vendor; a named person does | `approvals.py::apply_decision`; report wording filtered by `guardrails.enforce_output_policy` | `test_governance::test_t3_needs_both_signatures_from_two_people`, `test_matcher::test_overall_never_says_approved`, `test_router_guardrails::test_intent_check_blocks_approval_requests` |
| T3 needs Head of InfoSec **and** DPO (AI-5.1), two different people | `approvals.py` (roles from `auth.APPROVERS[tier]`) | `test_governance::test_one_person_cannot_sign_both_roles`, `test_api::test_approval_needs_identity_role_and_two_signatures` |
| The requester cannot approve their own request | `approvals.py` | `test_governance::test_requester_cannot_decide_own_request` |
| An approver must hold a role for the tier | `approvals.py` | `test_governance::test_approver_must_hold_a_role_for_the_tier` |
| A BLOCKED assessment cannot be approved (it can be rejected) | `approvals.py`, `app/api.py` (HTTP 409) | `test_governance::test_blocked_cannot_be_approved_but_can_be_rejected`, `test_api::test_blocked_run_cannot_be_approved` |
| A final decision cannot be changed | `approvals.py` | `test_governance::test_a_final_decision_cannot_be_changed` |
| Verdicts come from rules, not a model | `matcher.py` (no LLM import) | `tests/test_matcher.py` (all) |
| UNKNOWN beats a guess; a claim is not evidence | `matcher.py`, `vendor_research.py` (evidence levels), `claims.py` | `test_matcher::test_cert_claimed_only_is_not_a_pass`, `test_red_team_prod::test_attack_25_malformed_claim_values_are_rejected_not_repaired`, `test_matcher_survives_an_unreadable_fact_and_reports_unknown` |
| Identity is a gate: pages about a same-name company are rejected | `vendor_research.py` identity gate | `test_vendor_research::test_lookalike_company_rejected`, `eval/run_eval.py` ablation "No Identity Gate" |
| Pages that talk to the AI are quarantined whole | `injection.py` (regex, decoding, optional ONNX classifier) used by `vendor_research.py`, `policy_rag.py`, `guardrails.py` | `test_red_team.py` (attacks 1-15), `test_red_team_prod.py` (16-25) |
| The local injection classifier is never the only layer | `injection.py` (threshold 0.99, outage falls back to patterns) | `test_red_team_prod::test_attack_23_classifier_outage_does_not_disable_the_guardrail` |
| Access control is applied before ranking | `policy_rag.py` (ACL, status, trust filters) | `test_policy_rag.py`, `test_cache::test_restricted_answer_cached_for_security_is_invisible_to_employees` |
| Groups come from the token, not the request body (prod) | `auth.py`, `app/api.py` | `test_governance::test_prod_mode_requires_a_token_and_ignores_body_groups`, `test_api::test_groups_come_from_the_token_not_the_body`, `test_prod_mode_closes_every_door` |
| Restricted policy text is never sent to a cloud model | `policy_rag.py` data-egress rule for answer generation (local model or extractive answer only); `graders.py` + `LLM.local_only()` for the relevance and hallucination graders (local model or rule graders, no cloud fallback target) | `test_egress.py` (answer generation), `test_egress_pipeline.py` (whole policy path with the network faked: local server up or down, small role local or unset; also proves ordinary clauses still reach the cloud) |
| Assessments never come from the semantic cache; abstentions and refusals are not cached | `cache.py`, `graph.py` | `test_cache::test_assessments_never_use_the_semantic_cache`, `test_abstentions_and_refusals_are_not_cached` |
| Fetching is SSRF-safe (scheme, private/loopback/link-local ranges, redirects, size, content type) | `fetching.py` (and `fetcher/` when `VRA_FETCHER_URL` is set) | `test_fetching::test_unsafe_urls_are_refused`, `test_public_name_resolving_to_private_address_is_refused` |
| Research about companies, not people | `guardrails.py::scrub_person_data` | `test_red_team_prod::test_attack_21_person_data_is_removed_from_claims` |
| Every decision and security event is in a tamper-evident log | `db.py` (hash-chained `audit_log`), `scripts/verify_audit.py` | `test_governance::test_audit_chain_detects_edits_and_deletions`, `test_decision_is_persisted_with_an_audit_record` |
| Secrets are masked in logs and audit details | `errors.py::mask_secrets`, `observability.py` | `test_governance::test_audit_details_are_secret_masked` |
| Every report records what produced it | `graph.py` provenance stamp (policy / requirements / prompt hashes, models, backends, git commit) | `test_governance::test_report_carries_provenance_and_cli_gate_uses_the_same_rules` |
| Spend is capped per assessment, per day and per month | `tracing.py::Budget`, `costs.py::check_spend_caps` | `test_governance::test_budget_limits_and_stage_breakdown`; the daily/monthly caps: **no test** |
| Research stops at a deadline and reports UNKNOWN for what it did not establish | `vendor_research.py::live_iteration`, `fetching.py::_call_with_deadline` | `test_live_streaming.py` |
| Cache purges and retention runs are audited | `scripts/cache_admin.py`, `scripts/retention.py` | `test_ops_scripts.py` |

## 2. Model inventory

| Role | Model | Runs | What it sees | Decides verdicts? |
|---|---|---|---|---|
| main (planning when `VRA_PLANNER=llm`, answer synthesis) | MiniMax-M3 | cloud | redacted question, non-restricted policy chunks | No |
| extract (reading vendor pages) | MiniMax-M3.1-Flash | cloud | text of **public** web pages that passed the identity and injection gates | No: it proposes claims, `claims.py` validates their shape, `matcher.py` judges them |
| small (routing, graders, identity gate, injection LLM layer) | Qwen3-4B-Instruct-2507 Q4_K_M via llama.cpp | local GPU (about 3.1 GB VRAM); falls back to the cloud small model, then to rules, if the server is down. **Exception:** graders working on restricted policy text use `LLM.local_only()`: local model, or rules if it is down or not configured; never the cloud | redacted question, page excerpts, restricted policy text (only on the local model) | No |
| embedder | BAAI/bge-small-en-v1.5 (fastembed, ONNX) | local CPU | policy chunks, questions | No |
| reranker | Xenova/ms-marco-MiniLM-L-6-v2 cross-encoder (fastembed, ONNX) | local CPU | question + chunk or search snippet | No |
| injection classifier | protectai/deberta-v3-base-prompt-injection-v2 (ONNX) | local CPU | page text | No: one layer of several, threshold 0.99 |

Local-model selection (bake-off in `scripts/bakeoff_local_llm.py`): Qwen3-4B scored 30/30 routing, 18/18 injection, 33/33 identity,
8/8 grader; Phi-4-mini scored 28/30 on routing. These are small sets; they show the model is usable, not that it is robust.

## 3. Data flow and egress

| Data | Leaves the machine? | Where to |
|---|---|---|
| Employee question | Only after PII/secret redaction (`guardrails.py`), and routing runs on the local model | MiniMax for main-role calls |
| Restricted policy clauses (Approved Vendor List, security and procurement material) | **No**, on the policy-answer path (answer generation and both graders). The other LLM calls receive only the question, or public web pages. This was **not** true of the graders until it was fixed on 2026-11-17 (see the git log): they used the cloud small model with the full text of every cited clause. | n/a |
| Other policy clauses | Yes, when the main model writes an answer | MiniMax |
| Vendor name and search queries | Yes | Tavily and/or DuckDuckGo |
| Public vendor page text | Yes, for extraction | MiniMax |
| Page downloads | The fetch itself reaches the vendor's site from this machine's IP (or from the `fetcher` container) | vendor and third-party sites |
| Reports, approvals, audit log, caches | No | local SQLite under `runs/` |

## 4. Risk register

| Risk | Status | Mitigation / what remains |
|---|---|---|
| Prompt injection in a fetched page | Mitigated, not eliminated | Layered scan, whole-page quarantine, text cannot close the data tag, verdicts are rules. The ONNX classifier misses paraphrases and once flagged a clean page, so it is only an extra layer. |
| Wrong or stale vendor evidence | Open by nature | Evidence levels, freshness decay, conflicts recorded with provenance, UNKNOWN instead of a guess, human approval. A person must still read the sources. |
| Live results vary run to run | Known | Two clean live runs of the same question reached different overall outcomes because the web returned different pages. For identical facts the verdicts are identical. Reports list the pages used. |
| Research cut short by the 90 s deadline | Known, reduced | Before 69df346 one slow extraction call (45-82 s, likely provider latency plus the 60 s timeout and retries) used the whole budget; calls are now bounded by role and time left (unit tested; the cap was not reached in the live re-measurement). One of three later runs still stopped on the limit with a second pass not started; missing fields are UNKNOWN and become vendor questions. |
| Search provider unreliability | Known | DuckDuckGo timed out on roughly half of queries from the development network. Tavily is the intended primary where a key exists (none is configured yet). |
| Local model gives a wrong classification | Mitigated | Fallback chain, rules for the classifier-only role, output guard, red-team tests. Not evaluated beyond the bake-off sets. |
| Live extraction accuracy | **Not measured** | `data/gold_set_real.json` is evaluated against recorded facts, not against live extraction. A live accuracy check does not exist yet. |
| Authentication | Limited | Hashed tokens from a file, one machine. Not SSO; no token expiry. |
| Container hardening and observability stack | **Unverified** | Dockerfile, compose, `fetcher/` and `ops/` are written and reviewed but have never been built or started. The fetcher's isolation claims are untested until they have been. |
| Dependency vulnerabilities | Scanned once | `pip-audit` on `requirements-prod.lock` found none on 2026-11-17; CI repeats it. |
| Model download stalls | Mitigated | Cache-first loading with a deadline and a builtin fallback (`rag_core.load_fastembed`). |
