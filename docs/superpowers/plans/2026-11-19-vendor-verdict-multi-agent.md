# Vendor Verdict Multi-Agent System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A per-case multi-agent deep-research graph (planner, router, four field specialists, grader, decision maker, evaluator loop) that reads the Kaggle "Vendor Verdict" pages into `rules_lite` facts. It produces a presentable trace and, if the pre-registered gates pass, a candidate submission.

**Architecture:** A new package `vendor_verdict/`. Every agent is a plain function. Nodes wrap them over a shared `CaseState`, and one node list runs on two engines: plain Python and LangGraph `StateGraph` (repo convention, course lab 12). MiniMax-M3.1-Flash reads; the local llama.cpp model routes, grades and decides. The existing evidence policy, merge and verdict code (`eval/kaggle_policy.py`, `eval/run_kaggle.py`, `data_kaggle/rules_lite.py`) is reused unchanged. Every call is cached, so a cache-only run reproduces everything.

**Tech Stack:** Python 3.12, pydantic, langgraph (already in `requirements.txt`), the existing `assessor.llm.LLM`, pytest, ruff.

**Execution method (chosen by the user 2026-11-19):** subagent-driven (superpowers:subagent-driven-development), starting in a new chat.

**Spec:** `docs/superpowers/specs/2026-11-19-vendor-verdict-multi-agent-design.md` (approved 2026-11-19). Ledger for rulings during execution: `.superpowers/sdd/2026-11-19-vendor-verdict-multi-agent/progress.md` (git-excluded, as for the consensus experiment).

## Global Constraints

- Hard rules: no prompt or rule fitted to train/dev page text or labels; never print test claims, quotes or page text (counts and case ids only); no leaderboard probing; every submission needs the user's explicit yes; final selection is the user's, in the web UI.
- Commit or push only with the user's yes. The commit steps below are where a commit belongs once the user has allowed commits on `prod-hardening`.
- Before every paid run (any MiniMax call that is not cached), post `--estimate` (pages, calls, tokens per role) and wait for the user's go.
- Dev (`--split train --subset dev`) is for smoke only: crashes, parse failures, schema errors. It never justifies a change in prompt wording.
- 7 label-exposed holdout rows are excluded from every holdout rule: c0060__SEC-4.2, c0312__SEC-4.2-T2, c0678__SEC-4.2-T2, c0062__DP-3.1, c0288__SEC-7.1, c0426__SEC-4.2, c0512__SEC-9.3.
- Local roles may flag or choose among MiniMax readings. They never remove a claim.
- `rules_lite.py`, `metric.py`, `eval/kaggle_policy.py`, `eval/run_kaggle.py`, `eval/consensus.py`, gold sets and old tests are not edited.
- Tests use invented data only (no `data_kaggle/` read) and stub LLMs. They must pass on the default backend and on `EMBEDDER=builtin RERANKER=builtin`; `ruff check .` stays clean.
- Submission #2 (ref 58543825, `reports/kaggle_consensus/candidate/submission.csv`) stays one of the two finals.
- Deadline: 2026-11-30 18:29:59 UTC.

## Review Focus

1. **The local server is down or times out mid-run.** The run must continue: Router → every family (recall first), Grader → `unsure`, Decision Maker → abstain. Record a flag count in the report. Test in Tasks 4, 5, 6.
2. **A model returns malformed JSON, a field outside its family, or a value `normalise_claim` rejects.** The reading is dropped with a reason and the case continues. Test in Task 3.
3. **The Decision Maker answers with an index out of range, or with text that is not a candidate.** Treat it as abstain. Test in Task 6.
4. **An iteration-2 item cites a `page_id` that is not one of the case's readable pages for that family.** Drop it with a reason; it never reaches the merge. Test in Task 7.
5. **A run is cut short (budget, Ctrl-C, failed pages).** Mark the predictions `partial`; no `submission.csv` and no gate result may be built from them. Test in Task 8.

---

### Task 0: Measure the local model (spike, no product code)

**Files:**
- Create: `scripts/bench_local_llm.py` (throwaway measurement script, kept for the presentation's cost table)

- [ ] **Step 1:** Write `bench_local_llm.py`. It sends N=30 dev pages (`select_cases(..., "train", "dev", ...)`, readable official pages only) to `LLM(role="small", provider="llamacpp")`, using a fixed yes/no "does this page mention <topic>" JSON prompt and the existing `structured()`. It prints, as counts only: median and p90 seconds per call, the share of outputs that parse, and the context length used. No page text is printed.
- [ ] **Step 2:** Start the llama.cpp server (Qwen3-4B Q4) as the user normally does. Run `python scripts/bench_local_llm.py --n 30`. Expected: a table with s/call and parse rate.
- [ ] **Step 3:** Record the numbers in the ledger and decide the local scope with the spec's priority order. Estimate local calls per test pass for each role (Router: one call per readable page, about 2300; Grader: one call per grounded reading; Decision Maker: one per conflict). If the total exceeds about 8 h at the measured pace, shrink the scope in priority order: Grader limited to `security_certifications` + the 3 enum fields, then the Router's work handed to the Planner plus the fallback, then the Decision Maker on conflicts only. Write the chosen scope into the ledger. Tasks 4-6 read it as `LOCAL_SCOPE` in `vendor_verdict/config.py` (Task 1).

### Task 1: Package skeleton, state, cache, trace

**Files:**
- Create: `vendor_verdict/__init__.py`, `vendor_verdict/config.py`, `vendor_verdict/state.py`, `vendor_verdict/cache.py`, `vendor_verdict/trace.py`, `vendor_verdict/data.py`
- Test: `tests/test_vv_core.py`

**Interfaces:**
- Produces:
  - `config.FAMILIES: dict[str, tuple[str, ...]]` =
    `{"certifications": ("security_certifications",), "data": ("data_regions", "data_retention", "dpa_available"), "ai": ("trains_on_customer_data", "subprocessors"), "access_incidents": ("sso_saml", "security_incidents")}`
  - `config.FIELD_FAMILY: dict[str, str]` (the inverse map), `config.MAX_ITERATIONS = 2`, `config.LOCAL_SCOPE: frozenset[str]` (subset of `{"router", "grader", "decision_maker", "injection"}`, from Task 0), `config.EXPOSED_HOLDOUT_ROWS: frozenset[str]` (the 7 ids), `config.BASELINE_PROMPT = ("e42203d", "9566ad4b6b95")`.
  - `state.CaseState(TypedDict, total=False)`: `case: dict`, `recs: list[dict]` (from `run_kaggle.triage`), `plan: dict[str, list[str]]`, `router_tags: dict[str, list[str]]`, `routes: dict[str, list[str]]`, `review_fixes: list[str]`, `readings: list[dict]`, `grades: dict[str, str]`, `baseline: dict[str, list[dict]]`, `conflicts: list[dict]`, `decisions: dict[str, int | None]`, `facts: dict`, `verdicts: dict[str, str]`, `overall: str`, `iteration: int`, `targets: list[str]`, `notes: list[str]`, `flags: dict[str, int]`, `trace: Trace`.
  - A reading is the dict `{"id": str, "iteration": int, "family": str, "page_id": str, "field": str, "value": Any, "quote": str}`, with `id = f"{iteration}:{family}:{page_id}:{field}:{n}"`.
  - `cache.CallCache(root: Path)` with `key(role: str, prompt_version: str, model: str, payload: str, salt: str = "") -> str` (sha256 over the NUL-joined parts; an empty salt adds nothing), `get(key) -> dict | None`, `put(key, record: dict) -> None` (atomic write via `.tmp` + `replace`), and `hits`, `misses` counters. Default root: `data_kaggle/cache/agents/`.
  - `trace.Trace(case_id: str)` with `handover(agent: str, to: str, summary: dict, tokens_in: int = 0, tokens_out: int = 0, seconds: float = 0.0) -> None`, `to_json() -> dict`, `to_markdown() -> str`. Summaries hold counts and ids only.
  - `data.load(data_dir: Path, split: str, subset: str) -> tuple[dict[str, dict], dict[str, list[dict]], list[str]]` (cases, pages by case, case ids). It reuses `run_kaggle.load_cases`, `load_pages`, `select_cases` via `sys.path` on `eval/` (same import pattern as `tests/test_kaggle_runner.py`).

- [ ] **Step 1: Write the failing tests**

```python
def test_families_cover_every_rules_lite_field_once():
    fields = [f for fs in FAMILIES.values() for f in fs]
    assert sorted(fields) == sorted(["security_certifications", "data_regions", "data_retention", "dpa_available",
                                     "trains_on_customer_data", "subprocessors", "sso_saml", "security_incidents"])
    assert len(fields) == len(set(fields))

def test_cache_key_changes_with_every_part_and_empty_salt_adds_nothing(tmp_path):
    c = CallCache(tmp_path)
    k = c.key("specialist", "v1", "m", "payload")
    assert k == c.key("specialist", "v1", "m", "payload", "")
    assert len({k, c.key("router", "v1", "m", "payload"), c.key("specialist", "v2", "m", "payload"),
                c.key("specialist", "v1", "m2", "payload"), c.key("specialist", "v1", "m", "other"),
                c.key("specialist", "v1", "m", "payload", "s1")}) == 6

def test_cache_roundtrip_counts_hits(tmp_path):
    c = CallCache(tmp_path); k = c.key("r", "v", "m", "p")
    assert c.get(k) is None; c.put(k, {"out": 1}); assert c.get(k) == {"out": 1}; assert (c.hits, c.misses) == (1, 1)

def test_trace_records_handover_in_order():
    t = Trace("c9999"); t.handover("planner", "plan_reviewer", {"pages": 3}); t.handover("plan_reviewer", "router", {"fixes": 1})
    assert [h["agent"] for h in t.to_json()["handovers"]] == ["planner", "plan_reviewer"]
    assert "planner → plan_reviewer" in t.to_markdown()
```

- [ ] **Step 2:** Run `python -m pytest tests/test_vv_core.py -q`. Expected: FAIL (module `vendor_verdict` not found).
- [ ] **Step 3:** Implement the interfaces above.
- [ ] **Step 4:** Run `python -m pytest tests/test_vv_core.py -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `git add vendor_verdict tests/test_vv_core.py scripts/bench_local_llm.py && git commit -m "vendor_verdict: package skeleton, case state, call cache, handover trace"`

### Task 2: Prompts and output schemas

**Files:**
- Create: `vendor_verdict/prompts.py`, `vendor_verdict/schemas.py`
- Test: `tests/test_vv_prompts.py`

**Interfaces:**
- Produces:
  - `schemas.FieldReading(BaseModel)`: `page_id: str`, `field: str`, `value: Any`, `quote: str`.
  - `schemas.SpecialistOut(BaseModel)`: `readings: list[FieldReading]`.
  - `schemas.PlanOut(BaseModel)`: `routes: list[PageRoute]`, where `PageRoute` = `page_id: str`, `families: list[str]`.
  - `schemas.RouterOut(BaseModel)`: `families: list[str]`.
  - `schemas.GradeOut(BaseModel)`: `grade: Literal["supported", "unsupported", "unsure"]`.
  - `schemas.DecisionOut(BaseModel)`: `choice: int | None` (index into the candidate list; null = abstain).
  - `prompts.SPECIALIST_SYSTEM: dict[str, str]` (one per family), `PLANNER_SYSTEM`, `ROUTER_SYSTEM`, `GRADER_SYSTEM`, `DECISION_SYSTEM: str`, and `prompts.version(text: str) -> str` (sha256[:12]).
  - `prompts.page_user_prompt(page: dict) -> str` = `kaggle_policy.user_prompt(page)` (reuse it; untrusted-data wrapper included).
- Content rules for the prompt text (spec section 6):
  - Each specialist prompt names only its family's fields.
  - Each specialist prompt states the value vocabulary copied from the `rules_lite.py` docstring: cert names "SOC 2 Type II" | "SOC 2 Type I" | "SOC 2" | "ISO 27001" | other; status "certified" | "in_progress"; `trains_on_customer_data` "no" | "no_on_paid_plans" | "opt_out_available" | "yes"; `sso_saml` "all_plans" | "paid_plans" | "enterprise_only" | "none"; `dpa_available` "yes" | "on_request" | "no"; severity "major" | "critical" | "minor"; subprocessor categories hosting | email | analytics | support | model_provider.
  - Each specialist prompt includes the evidence policy lines from the `kaggle_policy.py` docstring (rules 3-5) and requires a verbatim quote per reading.
  - Certification readings carry `holder` and `issuer` when the page states them.
  - No example sentence may come from train or dev pages.

- [ ] **Step 1: Write the failing tests**

```python
def test_each_specialist_prompt_names_only_its_family_fields():
    for fam, fields in FAMILIES.items():
        text = SPECIALIST_SYSTEM[fam]
        assert all(f in text for f in fields)
        assert not any(f in text for other, ofs in FAMILIES.items() if other != fam for f in ofs)

def test_enum_vocabulary_is_copied_from_rules_lite():
    assert all(v in SPECIALIST_SYSTEM["ai"] for v in ("no_on_paid_plans", "opt_out_available"))
    assert all(v in SPECIALIST_SYSTEM["access_incidents"] for v in ("enterprise_only", "paid_plans", "all_plans"))
    assert "on_request" in SPECIALIST_SYSTEM["data"]

def test_versions_are_stable_and_distinct():
    vs = [version(t) for t in SPECIALIST_SYSTEM.values()] + [version(PLANNER_SYSTEM), version(ROUTER_SYSTEM), version(GRADER_SYSTEM), version(DECISION_SYSTEM)]
    assert len(set(vs)) == len(vs) and all(len(v) == 12 for v in vs)

def test_decision_out_accepts_abstain():
    assert DecisionOut.model_validate({"choice": None}).choice is None
```

- [ ] **Step 2:** Run `python -m pytest tests/test_vv_prompts.py -q`. Expected: FAIL (import error).
- [ ] **Step 3:** Write `schemas.py` and `prompts.py` to the content rules above.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `"vendor_verdict: agent prompts from the published schema, output schemas"`

### Task 3: Field Specialists and the adapter to the existing merge

**Files:**
- Create: `vendor_verdict/agents/__init__.py`, `vendor_verdict/agents/specialists.py`, `vendor_verdict/adapter.py`
- Test: `tests/test_vv_specialists.py`

**Interfaces:**
- Consumes: `CallCache`, `SPECIALIST_SYSTEM`, `SpecialistOut`, `FAMILIES`.
- Produces:
  - `specialists.read_page(llm, cache: CallCache, family: str, page: dict, iteration: int = 1, salt: str = "") -> tuple[list[dict], list[str]]` returns (readings, notes). Readings for fields outside `family` are dropped with a note. `page_id` is forced to the page's own id. A cached record is returned without a call. An `LLMError` after the existing retry pattern (`Extractor.run`: wait only on 429) returns `([], ["failed: …"])` and increments the caller's failure count.
  - `specialists.read_case(llm, cache, family, pages: list[dict], fields: list[str], salt: str = "") -> tuple[list[dict], list[str]]` (iteration 2): one call over several pages, each wrapped by `page_user_prompt`. Items whose `page_id` is not in `pages` are dropped with a note.
  - `adapter.to_page_claims(readings: list[dict], rec: dict, legal_entity: str) -> tuple[kaggle_policy.PageClaims, list[str]]`: one page's chosen readings → `Claim(field, value, display=str(value)[:120], quote)` → `kp.normalise_page_claims` → `kp.PageClaims(page_id, rec["role"], kp.page_date(rec["page"]), claims)` → `kp.check_holders`. Returns the notes of both.
  - `adapter.facts_for_case(case: dict, recs: list[dict], chosen: dict[tuple[str, str], list[dict]]) -> tuple[dict, list[str], list[str]]` (facts after `run_kaggle.strip_internal`, notes, conflicts). `chosen` maps (page_id, field) to that page-field's final readings; the function calls `kp.build_facts(items, as_of)` with the default strategies.

- [ ] **Step 1: Write the failing tests** (StubLLM as in `tests/test_kaggle_runner.py`, returning a fixed `SpecialistOut`)

```python
def test_reading_outside_family_is_dropped_with_note(tmp_path):
    out = SpecialistOut(readings=[FieldReading(page_id="p1", field="sso_saml", value="all_plans", quote="q"),
                                  FieldReading(page_id="p1", field="dpa_available", value="yes", quote="q")])
    r, notes = read_page(StubLLM(out), CallCache(tmp_path), "data", PAGE)
    assert [x["field"] for x in r] == ["dpa_available"] and any("sso_saml" in n for n in notes)

def test_second_read_hits_cache_without_call(tmp_path):
    llm = StubLLM(SpecialistOut(readings=[])); c = CallCache(tmp_path)
    read_page(llm, c, "ai", PAGE); read_page(llm, c, "ai", PAGE)
    assert len(llm.calls) == 1

def test_llm_error_returns_no_readings_and_a_failed_note(tmp_path):
    r, notes = read_page(RaisingLLM(), CallCache(tmp_path), "ai", PAGE)
    assert r == [] and notes[0].startswith("failed")

def test_invalid_enum_value_is_dropped_by_normalise_not_crash():
    pc, notes = to_page_claims([reading("trains_on_customer_data", "sometimes")], REC_OFFICIAL, "Acme Inc.")
    assert pc.claims == [] and notes

def test_adapter_matches_baseline_claim_path():
    # same statements as Claim objects through run_kaggle.assemble_case vs readings through facts_for_case -> identical facts
    assert facts_for_case(CASE, RECS, CHOSEN)[0] == assemble_case_facts_from_claims(CASE, RECS, CLAIMS)

def test_case_read_drops_item_citing_foreign_page(tmp_path):
    out = SpecialistOut(readings=[FieldReading(page_id="pX", field="sso_saml", value="none", quote="q")])
    r, notes = read_case(StubLLM(out), CallCache(tmp_path), "access_incidents", [PAGE], ["sso_saml"])
    assert r == [] and any("pX" in n for n in notes)
```

- [ ] **Step 2:** Run `python -m pytest tests/test_vv_specialists.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. Cache key payload = `page_user_prompt(page)` (or the joined pages for `read_case`) + the family; role `"specialist:<family>"`; version `prompts.version(SPECIALIST_SYSTEM[family])`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `"vendor_verdict: field specialists and adapter onto the existing evidence merge"`

### Task 4: Source Validator, Planner, Page Router, Plan Reviewer

**Files:**
- Create: `vendor_verdict/agents/planner.py`, `vendor_verdict/agents/router.py`, `vendor_verdict/agents/reviewer.py`
- Test: `tests/test_vv_planning.py`

**Interfaces:**
- Consumes: `run_kaggle.triage`, `run_kaggle.readable`, `PlanOut`, `RouterOut`, `CallCache`, `FAMILIES`, `LOCAL_SCOPE`.
- Produces:
  - `planner.plan_routes(llm, cache, case: dict, recs: list[dict]) -> tuple[dict[str, list[str]], list[str]]`. The prompt gets page metadata only (page_id, url, title, source_type, publisher, dates; no text) for readable pages. Unknown families and page_ids are dropped with notes. On `LLMError`: `({}, ["planner failed"])`.
  - `router.tag_pages(local_llm, cache, recs: list[dict]) -> tuple[dict[str, list[str]], int]` returns (page_id → families, failures). If `"router"` is not in `LOCAL_SCOPE`, or a call fails, the page maps to every family (recall first), and failures are counted.
  - `reviewer.review(plan, router_tags, recs) -> tuple[dict[str, list[str]], list[str]]` returns (family → page_ids, fixes). Rules:
    - (1) Only readable pages (`readable(rec)`); a quarantined or ignored page is never routed.
    - (2) Routes are the union of planner and router.
    - (3) A `registry` page always goes to `certifications`, and only there (a registry speaks only about certificates, as in `check_holders`).
    - (4) A readable official page with no route at all goes to every family.
    - Every fix is described in `fixes`.
  - `reviewer.route_recall(routes, baseline_fields: dict[str, set[str]]) -> tuple[int, int, list[str]]` returns (covered, total, missed "page_id:field" ids). This is the R-route measurement.
  - `router.injection_second_opinion(local_llm, recs) -> int` (the spec's Source Validator "+ local flag"): when `"injection"` is in `LOCAL_SCOPE` (the lowest priority in Task 0), it runs the existing `assessor.guardrails.classify_injection(text, local_llm)` on readable pages and returns how many it would flag. It is logged in the trace only and never changes `recs`.

- [ ] **Step 1: Write the failing tests**

```python
def test_quarantined_and_ignored_pages_are_never_routed():
    routes, _ = review({"ai": ["pq", "pi", "po"]}, {}, [rec("pq", quarantine=["x"]), rec("pi", role="ignored"), rec("po")])
    assert all("pq" not in v and "pi" not in v for v in routes.values())

def test_registry_page_goes_to_certifications_only():
    routes, fixes = review({"ai": ["pr"]}, {}, [rec("pr", role="registry")])
    assert [f for f, ids in routes.items() if "pr" in ids] == ["certifications"] and fixes

def test_unrouted_official_page_goes_to_every_family():
    routes, _ = review({}, {}, [rec("po")])
    assert all("po" in routes[f] for f in FAMILIES)

def test_router_failure_maps_page_to_every_family(tmp_path):
    tags, failures = tag_pages(RaisingLLM(), CallCache(tmp_path), [rec("po")])
    assert set(tags["po"]) == set(FAMILIES) and failures == 1

def test_planner_prompt_contains_no_page_text(tmp_path):
    llm = StubLLM(PlanOut(routes=[])); plan_routes(llm, CallCache(tmp_path), CASE, [rec("po", text="SECRET-BODY-TEXT")])
    assert "SECRET-BODY-TEXT" not in llm.last_user

def test_route_recall_counts_missed_page_fields():
    assert route_recall({"ai": ["p1"]}, {"p1": {"subprocessors"}, "p2": {"sso_saml"}}) == (1, 2, ["p2:sso_saml"])
```

- [ ] **Step 2:** Run `python -m pytest tests/test_vv_planning.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `"vendor_verdict: planner, page router and plan reviewer with recall-first fallbacks"`

### Task 5: Evidence Grader

**Files:**
- Create: `vendor_verdict/agents/grader.py`
- Test: `tests/test_vv_grader.py`

**Interfaces:**
- Produces:
  - `grader.grounded(quote: str, text: str) -> bool`: casefold, collapse whitespace, unify quote marks and dashes, then check for a substring; an empty quote → False.
  - `grader.grade(local_llm, cache, reading: dict, page: dict) -> str` returns one of `"supported" | "unsupported" | "unsure"`:
    - not grounded → `"unsupported"`, with no LLM call;
    - grounded and `"grader"` not in `LOCAL_SCOPE`, or the field is out of the Task 0 scope → `"unsure"`;
    - otherwise the local `GradeOut`, and an `LLMError` → `"unsure"`.
  - `grader.grade_all(local_llm, cache, readings, pages_by_id) -> tuple[dict[str, str], int]` returns (reading id → grade, failures). Grades never delete a reading.

- [ ] **Step 1: Write the failing tests**

```python
def test_grounded_ignores_case_whitespace_and_quote_style():
    assert grounded("We  are “SOC 2” certified", 'x we are "soc 2" certified. y')
def test_ungrounded_quote_is_unsupported_without_a_call(tmp_path):
    llm = StubLLM(GradeOut(grade="supported")); assert grade(llm, CallCache(tmp_path), reading(quote="not there"), PAGE) == "unsupported" and llm.calls == []
def test_local_failure_grades_unsure(tmp_path):
    assert grade(RaisingLLM(), CallCache(tmp_path), reading(quote=PAGE_SENTENCE), PAGE) == "unsure"
def test_grade_all_keeps_every_reading():
    grades, _ = grade_all(StubLLM(GradeOut(grade="unsupported")), cache, READINGS, PAGES)
    assert set(grades) == {r["id"] for r in READINGS}
```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `"vendor_verdict: evidence grader (grounding + local judge, flag only)"`

### Task 6: Conflicts and the Decision Maker

**Files:**
- Create: `vendor_verdict/agents/decision_maker.py`, `vendor_verdict/baseline.py`
- Test: `tests/test_vv_decisions.py`

**Interfaces:**
- Produces:
  - `baseline.baseline_readings(llm_model: str, recs: list[dict], cache_dir: Path) -> dict[str, list[dict]]` (page_id → readings in the reading-dict shape, `iteration=0`, `family=FIELD_FAMILY[field]`). It reads the cached baseline extraction (prompt from `kp.prompt_text_from_git(*BASELINE_PROMPT)`, `run_kaggle.Extractor(..., cache_only=True, system=…)`), and applies `kp.normalise_page_claims`. It never calls a model.
  - `decision_maker.value_key(field: str, value) -> str`: the comparable signature, reusing `kp.claim_signature` on a one-claim list.
  - `decision_maker.collect_conflicts(readings: list[dict], baseline: dict[str, list[dict]], grades: dict[str, str]) -> list[dict]`. For each (page_id, field) with 2 or more distinct `value_key`s among iteration 1, iteration 2 and baseline, it builds `{"id": "page_id:field", "page_id", "field", "candidates": [{"source": "iter1"|"iter2"|"baseline", "value", "quote", "grade"}], "default": index}`. `default` is the iteration-2 candidate if one exists, else iteration 1, else none (a baseline-only value is never the default).
  - `decision_maker.decide(local_llm, cache, conflict: dict) -> int | None`: `None` when `"decision_maker"` is not in `LOCAL_SCOPE`, on `LLMError`, on an index out of range, or on `null`.
  - `decision_maker.apply(readings, conflicts, decisions: dict[str, int | None]) -> dict[tuple[str, str], list[dict]]` is the `chosen` map for `adapter.facts_for_case`. With no conflict: the latest-iteration specialist readings. With a decision: the chosen candidate. With abstain: the default (or the latest-iteration readings when the default is none).

- [ ] **Step 1: Write the failing tests**

```python
def test_conflict_needs_two_distinct_values():
    assert collect_conflicts([r1("sso_saml", "all_plans")], {"p1": [b("sso_saml", "all_plans")]}, {}) == []
    assert len(collect_conflicts([r1("sso_saml", "all_plans")], {"p1": [b("sso_saml", "none")]}, {})) == 1

def test_baseline_only_value_is_never_default():
    c = collect_conflicts([], {"p1": [b("sso_saml", "none")]}, {})
    assert c == [] or c[0]["default"] is None

def test_out_of_range_and_failure_abstain(tmp_path):
    assert decide(StubLLM(DecisionOut(choice=7)), CallCache(tmp_path), CONFLICT_2) is None
    assert decide(RaisingLLM(), CallCache(tmp_path), CONFLICT_2) is None

def test_abstain_keeps_default_and_never_removes_a_field():
    chosen = apply(READINGS, [CONFLICT_2], {CONFLICT_2["id"]: None})
    assert chosen[("p1", "sso_saml")][0]["value"] == CONFLICT_2["candidates"][CONFLICT_2["default"]]["value"]

def test_decision_picks_candidate_value():
    chosen = apply(READINGS, [CONFLICT_2], {CONFLICT_2["id"]: 1})
    assert chosen[("p1", "sso_saml")][0]["value"] == CONFLICT_2["candidates"][1]["value"]
```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `"vendor_verdict: conflict collection and local decision maker (choose or abstain)"`

### Task 7: Nodes, Evaluator loop and the two engines

**Files:**
- Create: `vendor_verdict/nodes.py`, `vendor_verdict/graph.py`
- Test: `tests/test_vv_graph.py`

**Interfaces:**
- Consumes: everything above; `rules_lite` loaded with `run_kaggle.load_module`.
- Produces:
  - `nodes.Agents` (dataclass): `main_llm`, `reader_llm`, `local_llm`, `cache: CallCache`, `rl` (the rules_lite module), `reqs: list[dict]`, `baseline_dir: Path`, `baseline_model: str`, `salt: str`, `ablate: frozenset[str]` (subset of `{"router", "grader", "decision_maker", "iteration2"}`).
  - Node functions `(state: CaseState, ag: Agents) -> dict` (a partial state update), in graph order:
    - `plan`, `review_plan`, `validate_sources` (= `triage`; it runs first in practice, see the ordering note), `route_pages`, `specialists`, `grade_evidence`, `evaluate`, `decide_conflicts`, `merge`, `verdict`, `audit`.
    - Each node calls `state["trace"].handover(...)` once.
  - `nodes.evaluate`: when `iteration < MAX_ITERATIONS` and `"iteration2"` is not ablated, `targets` = fields with any `unsupported`/`unsure` grade or any (page, field) whose iteration-1 value differs from the baseline. It continues if `targets` is non-empty. `specialists` in iteration 2 calls `read_case` once per family, for the target fields, over that family's routed pages.
  - `graph.NODE_ORDER: list[str]`, `graph.run_plain(state, ag) -> CaseState`, `graph.build_langgraph(ag)` (a compiled `StateGraph` with the conditional edge `evaluate → specialists | decide_conflicts`), `graph.mermaid(ag) -> str` (`get_graph().draw_mermaid()`).
- Ordering note (deviation from the spec diagram, recorded here): `validate_sources` runs first (planning needs to know the readable pages), and `decide_conflicts` runs after the loop and before `merge`, so that decisions see both iterations. The node set and the roles are as in the spec.

- [ ] **Step 1: Write the failing tests** (stub LLMs; an invented case with 3 pages)

```python
def test_both_engines_give_identical_verdicts_and_handover_order():
    a, b = run_plain(new_state(CASE, PAGES), AG), build_langgraph(AG).invoke(new_state(CASE, PAGES))
    assert a["verdicts"] == b["verdicts"] and a["overall"] == b["overall"]
    assert [h["agent"] for h in a["trace"].to_json()["handovers"]] == [h["agent"] for h in b["trace"].to_json()["handovers"]]

def test_loop_stops_after_max_iterations():
    s = run_plain(new_state(CASE, PAGES), AG_ALWAYS_UNSURE)
    assert s["iteration"] == MAX_ITERATIONS

def test_no_flags_means_one_iteration():
    assert run_plain(new_state(CASE, PAGES), AG_ALL_SUPPORTED)["iteration"] == 1

def test_iteration2_ablation_runs_once():
    assert run_plain(new_state(CASE, PAGES), replace(AG_ALWAYS_UNSURE, ablate=frozenset({"iteration2"})))["iteration"] == 1

def test_verdicts_come_from_rules_lite_on_merged_facts():
    s = run_plain(new_state(CASE, PAGES), AG)
    assert s["verdicts"] == AG.rl.evaluate(s["facts"], AG.reqs, CASE["tier"], int(CASE["users"]), CASE["as_of"])

def test_mermaid_names_every_agent():
    assert all(n in mermaid(AG) for n in NODE_ORDER)
```

- [ ] **Step 2:** Run `python -m pytest tests/test_vv_graph.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `"vendor_verdict: agent nodes, evaluator loop, plain and LangGraph engines"`

### Task 8: Runner CLI, estimates, outputs

**Files:**
- Create: `vendor_verdict/__main__.py`, `vendor_verdict/runner.py`
- Test: `tests/test_vv_runner.py`

**Interfaces:**
- Produces:
  - `python -m vendor_verdict run --split test|train --subset all|dev|holdout [--limit N] [--only ids] [--cache-only] [--estimate] [--salt S] [--engine plain|langgraph] [--ablate router,grader,decision_maker,iteration2] [--workers 6] [--out DIR]`.
  - Default `--out`: `reports/vendor_verdict/<split>_<subset>[_ablate-…]/`.
  - Writes:
    - `preds.csv` (id, verdict, all rows of the selected cases);
    - `run.json` (counts per role: calls, cache hits, failures, tokens, seconds; `partial: bool`; prompt versions; `LOCAL_SCOPE`; ablations);
    - `traces/<case_id>.json` + `.md`;
    - `graph.mmd`;
    - for `--split test` with no salt and no ablation and not partial: `submission.csv` via `run_kaggle.write_submission`. It is never uploaded.
  - `runner.estimate(...) -> dict`: uncached calls and tokens per role, using `run_kaggle.estimate_call_tokens` and `OUT_PER_IN`, printed before any paid call. `--estimate` exits after printing.
  - Console output: counts and case ids only (a leak guard like `run_kaggle.main`).
  - `partial` is true when any readable page failed, or the run was interrupted, or budget stopped it. In that case there is no `submission.csv`, and gates refuse the run (Task 9).

- [ ] **Step 1: Write the failing tests** (invented data dir in `tmp_path`, stub LLMs injected through a `runner.run(..., agents=...)` seam)

```python
def test_partial_run_writes_no_submission(tmp_path):
    out = run_fixture(tmp_path, fail_pages={"p2"})
    assert json.loads((out / "run.json").read_text())["partial"] is True and not (out / "submission.csv").exists()

def test_salted_or_ablated_test_run_writes_no_submission(tmp_path):
    assert not (run_fixture(tmp_path, salt="s1") / "submission.csv").exists()
    assert not (run_fixture(tmp_path, ablate={"grader"}) / "submission.csv").exists()

def test_console_prints_no_page_text(tmp_path, capsys):
    run_fixture(tmp_path, page_text="CANARY-PAGE-TEXT"); assert "CANARY-PAGE-TEXT" not in capsys.readouterr().out

def test_cache_only_rerun_is_byte_identical(tmp_path):
    a = run_fixture(tmp_path / "a"); b = run_fixture(tmp_path / "b", cache_only=True, cache_from=a)
    assert (a / "preds.csv").read_bytes() == (b / "preds.csv").read_bytes()
```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests, then the full suite on both backends: `python -m pytest -q` and `EMBEDDER=builtin RERANKER=builtin python -m pytest -q`, then `ruff check .`. Expected: all pass, ruff clean.
- [ ] **Step 5:** Commit: `"vendor_verdict: runner CLI with estimates, traces, partial-run guard"`

### Task 9: Pre-registered gates

**Files:**
- Create: `vendor_verdict/gates.py` (its module docstring IS the pre-registration: rule texts, thresholds, frozen prompt versions)
- Test: `tests/test_vv_gates.py`

**Interfaces:**
- Produces (pure functions, then CLI glue `python -m vendor_verdict gates --target dev|holdout|test`, which writes `reports/vendor_verdict/gates.json`):
  - `r_smoke(run: dict) -> bool`: `partial` is false, failures are 0, readable pages read = 100%.
  - `r_route(covered: int, total: int) -> bool`: `covered == total`.
  - `wrong_counts(preds: dict[str, str], gold: dict[str, str], exclude: frozenset[str]) -> tuple[int, int]` returns (wrong rows, false-confident rows). False-confident means gold is UNKNOWN/NEEDS_REVIEW and the prediction is PASS/FAIL, on requirement rows. It is computed over `gold` ids minus `exclude`.
  - `r_hold(b: tuple[int, int], s0: tuple[int, int]) -> bool`: `b[0] <= s0[0] and b[1] <= s0[1]`.
  - `r_dm(preds_dm, preds_abstain, gold, exclude) -> tuple[bool, int, int]` returns (pass, wrong→right, right→wrong) over the rows that differ; pass when `w2r >= r2w`.
  - `r_test(preds: dict, sub2: dict) -> tuple[bool, int]`: rows changed ≤ 30.
  - `candidate(rules: dict[str, bool]) -> tuple[str, frozenset[str]]` returns (`"B"` or `"none"`, the ablations of the candidate run). It is `("B", frozenset())` when R-smoke, R-route, R-hold, R-test and R-DM pass; `("B", frozenset({"decision_maker"}))` when only R-DM fails (fixed in advance); and `("none", frozenset())` otherwise.
  - Holdout s0 baseline predictions: `run_kaggle.assemble_case` over the cached baseline extraction (the same path R0 proved reproduces s0). Submission #2: `reports/kaggle_consensus/candidate/submission.csv`.
  - Gates refuse a run whose `run.json` says `partial: true`, or whose prompt versions differ from the docstring's frozen ones.
  - The CLI prints pass/fail and counts only. For R-DM it prints pass/fail and the two counts, never a list of rows with labels.

- [ ] **Step 1: Write the failing tests**

```python
def test_wrong_counts_excludes_rows_and_counts_false_confidence():
    gold = {"c1__A": "UNKNOWN", "c1__B": "PASS", "c2__A": "FAIL"}
    assert wrong_counts({"c1__A": "PASS", "c1__B": "PASS", "c2__A": "PASS"}, gold, frozenset({"c2__A"})) == (1, 1)
def test_r_hold_needs_both_counts_not_worse():
    assert r_hold((2, 0), (3, 1)) and not r_hold((2, 2), (3, 1))
def test_r_dm_counts_only_changed_rows():
    assert r_dm({"x": "PASS", "y": "FAIL"}, {"x": "FAIL", "y": "FAIL"}, {"x": "PASS", "y": "FAIL"}, frozenset()) == (True, 1, 0)
def test_candidate_drops_dm_when_r_dm_fails():
    ok = {"R-smoke": True, "R-route": True, "R-hold": True, "R-test": True}
    assert candidate({**ok, "R-DM": True}) == ("B", frozenset())
    assert candidate({**ok, "R-DM": False}) == ("B", frozenset({"decision_maker"}))
    assert candidate({**ok, "R-hold": False, "R-DM": True}) == ("none", frozenset())
def test_partial_run_is_refused():
    with pytest.raises(GateRefused): check_run({"partial": True})
```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Write the docstring pre-registration (spec section 8 wording, thresholds as above).
- [ ] **Step 4:** Run the tests and the full suite on both backends, then ruff. Expected: all pass.
- [ ] **Step 5:** Commit: `"vendor_verdict: pre-registered gates (R-smoke, R-route, R-hold, R-DM, R-test)"`

### Task 10: Dev smoke and freezing the prompts (first paid run)

- [ ] **Step 1:** Run `python -m vendor_verdict run --split train --subset dev --limit 20 --estimate`. Post the estimate and wait for the user's go.
- [ ] **Step 2:** Run it without `--estimate`. Then run `python -m vendor_verdict gates --target dev`. Expected: R-smoke pass.
- [ ] **Step 3:** If R-smoke fails: fix code or schema only (never prompt wording aimed at a reading), log the fix in the ledger, and re-run Steps 1-2.
- [ ] **Step 4:** Record the prompt versions in the `gates.py` docstring (frozen). Run the full suite and ruff.
- [ ] **Step 5:** Commit: `"vendor_verdict: freeze prompt versions before the holdout run (pre-registration)"`. The commit is the timestamp that proves the rules came before the data.

### Task 11: Holdout run and its gates

- [ ] **Step 1:** `python -m vendor_verdict run --split train --subset holdout --estimate`. Post it and wait for the go.
- [ ] **Step 2:** Run it. Then run the ablation for R-DM from the cache: `--ablate decision_maker --cache-only` (free; every MiniMax call is cached).
- [ ] **Step 3:** `python -m vendor_verdict gates --target holdout`. Expected: R-route, R-hold, R-DM results in `gates.json`. Report them exactly as measured. Stop and report to the user.

### Task 12: Test run, R-test, candidate

- [ ] **Step 1:** `python -m vendor_verdict run --split test --estimate`. Post it and wait for the go.
- [ ] **Step 2:** Run it (with `--ablate decision_maker` if R-DM failed, as fixed in advance). Then run `python -m vendor_verdict gates --target test`. Expected: R-route and R-test results, and `candidate` = `B` or `none`.
- [ ] **Step 3:** If the result is `B`: report the rows changed versus submission #2 (count and ids only) and ask the user whether to submit. Never submit without the yes. If the result is `none`: say so plainly; nothing is submitted.

### Task 13: Ablations, presentation material, results note, final review

- [ ] **Step 1:** Holdout ablations, cache only (free): `--ablate router`, `grader`, `decision_maker`, `iteration2`, plus the baseline s0. Make a table of wrong and false-confident counts (exposed rows excluded), calls, tokens and seconds per role.
- [ ] **Step 2:** Presentation pack in `reports/vendor_verdict/presentation/`: `graph.mmd`, one dev case trace in Markdown (dev only, counts and ids; no page text), the ablation table, and the cost and time table (Task 0 numbers included).
- [ ] **Step 3:** Results note `docs/superpowers/results/2026-11-xx-vendor-verdict-multi-agent-results.md`: gates as measured, failures included, and what stayed unverified.
- [ ] **Step 4:** Verification (superpowers:verification-before-completion): both backends, ruff, a cache-only re-run reproduces `preds.csv` byte for byte.
- [ ] **Step 5:** Final whole-branch review by a fresh reviewer. Fix Important findings test-first. Commit with the user's yes.
