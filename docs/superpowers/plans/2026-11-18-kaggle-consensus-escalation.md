# Kaggle Consensus + Escalation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fewer wrong test rows for the Vendor Verdict submission by voting over several samples of the unchanged base prompt (stage A) and re-reading only the unsure page-fields with a stronger model (stage B), under rules written down before any new sample is drawn.

**Architecture:** One new module `eval/consensus.py` holds the pre-registration (module docstring), the pure vote/resolve logic, and a small CLI that reads cached extraction samples and feeds consensus claims into the unchanged `run_kaggle.assemble_case`. `run_kaggle.py` and `kaggle_policy.py` get only plumbing: choose a prompt from a git commit (text byte-identical, hash-checked), run the extractor with an explicit system text, estimate a run before it starts, skip scoring. No prompt wording, rules, matcher or gold set changes.

**Tech Stack:** Python 3.11, pytest, pydantic `Claim`, `eval/kaggle_policy.py`, `eval/run_kaggle.py`, pandas (via the host's `metric.py`), MiniMax through `assessor.llm.LLM`.

**Spec:** `docs/superpowers/specs/2026-11-18-kaggle-consensus-escalation-design.md` (approved, no changes). Read it with this plan; section numbers below (§) refer to it.

## Global Constraints

- No commits, no pushes (so this plan has no commit steps). The three untracked experiment files (`eval/claim_check.py`, `eval/nli_scorer.py`, `eval/laya_scorer.py`) stay untouched.
- No prompt wording changes: at the end `kp.PROMPT_VERSION == "f45b4b77c99d"` and `eval/kaggle_policy.py`'s `EXTRACT_SYSTEM` is byte-identical to HEAD. Base prompt `9566ad4b6b95` is only ever *selected* from commit `e42203d`.
- Pre-register first: the module docstring of `eval/consensus.py` carries §6 verbatim (plus decisions D1-D8 below) before any run that draws a sample or reads labels.
- Counts and case ids only on stdout and in report files. Never print or open claims, quotes, page text or `cases.jsonl` content. Files in `reports/` hold ids, field names and counts.
- One LLM run at a time. Before each run: post the `--estimate` output (tokens, time) in chat, then start it in the background; stop if the estimate exceeds the §7 row it belongs to, if wall time passes twice the upper estimate, if extraction failures exceed 1% of pages, or if rate limits or quota look tight. I cannot read the MiniMax meter: ask the user for the weekly and 5-hour readings before the first run and again after any run that was slow.
- Extraction stays on MiniMax (Flash for stage A, MiniMax-M3 for the adjudicator); never the local Qwen model; no page text to any other provider.
- Ask the user before every submission, before any spend beyond §7, and before anything that is not in this plan. A submission needs an explicit yes each time; final selection is the user's, in the web UI.
- Competition data stays in gitignored `data_kaggle/`. Tests use invented data only.
- Test commands: `source .venv/bin/activate`; `python -m pytest tests/test_kaggle_runner.py tests/test_kaggle_consensus.py -q`, then the full suite on the default backend and with `EMBEDDER=builtin RERANKER=builtin`; `ruff check .` (config in `ruff.toml`).

## Decisions the spec left open (check these; they go into the docstring as D1-D8; D2 is the one that most needs an explicit yes, because the spec says "plurality" and the tie rule is my reading of it)

- **D1** The adjudicator (MiniMax-M3) uses the base prompt `9566ad4b6b95`, so stage B changes the model only. §5.3 says "the same unchanged extraction prompt" without naming which.
- **D2** Ties: if Flash has no strict majority, the adjudicator's strict majority counts only when it equals one of the tied leaders (kind `tie_broken`); otherwise the item is `unresolved` and its claim is dropped. A confirmed majority changes nothing. B can therefore only break ties or drop claims; it never introduces a claim Flash did not produce.
- **D3** "Settled" counts agreeing votes in absolute terms (4 with 4 valid votes is settled). A final stage-A run needs every sample to cover 100% of readable pages (failed pages get re-run first); `--allow-partial` exists for dry runs only.
- **D4** RB0 pool = test page-fields on which every valid Flash vote agrees and the signature is not "absent" (otherwise the check is vacuous). Seed `20261118`. The adjudicator sample `adj1` only; the threshold is `agree*100 >= 95*n`.
- **D5** RB1 "untested" (B changes no holdout row) is not a pass: the candidate falls back to A-only.
- **D6** New holdout draws use `--no-score`, so no per-sample holdout score is printed and the holdout never judges `f45b4b77c99d`. Consensus numbers come from `consensus.py` only.
- **D7** If the candidate rule gives `A+B`, `candidate` also writes the A-only file (`submission_A_only.csv`, same self-check), so the user can decide yes/no on both files before any new score is read (§6: "at most two new submissions ... each decided before any new score is seen"). Both decisions go into the results note.
- **D8** The vote signature is the spec's `claim_signature`, which cannot see a certificate's `holder` (and `check_holders` acts on it for registry pages), and the consensus is built from raw claims carried per field, so `normalise_page_claims` can behave differently on the combined claims than on any single sample. Both gaps are measured, not fixed: stage A reports `holder_split` (a winning signature whose carriers disagree on holder) and `post_vote_mismatch` (the consensus record's signature differs from the voted winner). If either is non-empty, Task 11 stops and asks the user. Checked at plan time: `evidence_level` is not a hidden source of noise: `rules_lite.py` reads it only in the certificate rule, and `build_facts` derives it from issuer and date, both in the signature.
- Cost, measured from the run reports (not the spec's table): one test pass of the base prompt was 3.69M tokens and 41-45 min; the f45b prompt's 4.38M tokens and 97 min is where the spec's 4.4M / 1.7 h comes from. The seven draws below add up to about 19M tokens and 3.5-4 h, stage B under 1M: inside §7's 23M / 9 h. The adjudicator's cost and pace are unmeasured: `assessor.tracing.price_for` returns the same fallback price for every model name, so `--max-usd` does not bound MiniMax-M3 spend in real terms. The RB0 run (Task 12, step 2) is the MiniMax-M3 measurement; its wall time per page re-bases the estimates for the larger B runs, and I post that before they start.
- Sample salts, taken from the run headers in `reports/*.log`: test base prompt `s0` = no salt, `noise1` = `noise1`; test second opinion `v7` = no salt, `v7n1` = salt `v7n1` (not `noise1`). Task 9 confirms them by coverage (100% of readable pages found in the cache) before anything is drawn.

## File Structure

- Modify `eval/kaggle_policy.py`: `cache_key(..., system=None)`, `prompt_text_from_git(...)`.
- Modify `eval/run_kaggle.py`: `Extractor(..., system=None)`, `select_prompt(...)`, `estimate_call_tokens(...)`, `write_submission(...)` (extracted from `main`), flags `--prompt-commit --expect-prompt --estimate --no-score`.
- Create `eval/consensus.py`: docstring, constants, `tally`, per-page voting, stage B logic, rules, CLI (`check-r0`, `stage-a`, `plan-b`, `run-b`, `resolve`, `candidate`).
- Create `tests/test_kaggle_runner.py` (Tasks 1-2, 8) and `tests/test_kaggle_consensus.py` (Tasks 3-7). New files, so no existing test is edited.
- Outputs under `reports/kaggle_consensus/` (gitignored): `rules.json`, `<target>/stage_a.md`, `<target>/preds_A.csv`, `<target>/unsettled.json`, `<target>/b_plan.json`, `<target>/stage_b.md`, `<target>/preds_AB.csv`, `candidate/submission.csv`.

## Review Focus

Inputs the spec implies and nothing else exercises; each has a pinned test in the task that owns the code.

1. A sample that failed on a page gives no vote; it must never count as "absent" (Task 3, `test_failed_sample_is_not_an_absent_vote`).
2. A page that no sample extracted makes the case incomplete, and an incomplete case blocks the candidate file (Tasks 3 and 8).
3. Two samples with the same vote signature but a different certificate holder, and a region claim whose quote names a sub-processor that only another sample listed: the vote cannot see either, so both are measured (Task 4, D8).
4. B changes no holdout row: RB1 is `untested`, not `pass`, and the candidate is A-only (Task 5).
5. Reports and `unsettled.json` must not leak claim text, even for odd values (Task 4, canary test).

---

### Task 1: Select a prompt from a commit without editing it

**Files:**
- Modify: `eval/kaggle_policy.py` (`cache_key`, new `prompt_text_from_git`)
- Modify: `eval/run_kaggle.py` (`Extractor.__init__/run`, new `select_prompt`, argparse flags, report line that prints the prompt version)
- Test: `tests/test_kaggle_runner.py` (create)

**Interfaces:**
- Produces: `kp.cache_key(model: str, page: dict, salt: str = "", system: str | None = None) -> str` (`None` means `kp.EXTRACT_SYSTEM`; every existing key stays valid)
- Produces: `kp.prompt_text_from_git(rev: str, expect_version: str, repo: Path = ROOT) -> str` (executes `eval/kaggle_policy.py` as of `rev`, from this repo's own history, under a throwaway module name registered in `sys.modules`; returns its `EXTRACT_SYSTEM`; raises `ValueError` unless `sha256(text)[:12] == expect_version`)
- Produces: `rk.Extractor(llm, cache_dir, cache_only, salt="", system=None)`; its calls and cache keys use `system`
- Produces: `rk.select_prompt(prompt_commit: str, expect_prompt: str) -> tuple[str, str]` returning `(system_text, version)`; empty `prompt_commit` returns the current prompt; a commit without `expect_prompt` raises `SystemExit`
- Produces CLI flags `--prompt-commit REV --expect-prompt VERSION`; `main` passes `system` to `Extractor` and prints the selected version

- [x] **Step 1: Write the failing tests** in `tests/test_kaggle_runner.py` (import with `sys.path.insert(0, <repo>/eval)`, `import kaggle_policy as kp, run_kaggle as rk`; `PAGE = {"page_id": "p1", "url": "https://acme.example/s", "title": "t", "text": "x" * 80}`; `StubLLM` has `model = "stub-model"`, records `system` in `calls`, and `structured(...)` returns `ClaimList(claims=[])`)

```python
def test_default_cache_key_is_unchanged():          # old entries must stay findable
    h = hashlib.sha256()
    for part in ("m", kp.EXTRACT_SYSTEM, "ClaimList", kp.user_prompt(PAGE)):
        h.update(part.encode()); h.update(b"\x00")
    assert kp.cache_key("m", PAGE) == h.hexdigest()

def test_cache_key_depends_on_system_and_salt():
    a = kp.cache_key("m", PAGE, system="X")
    assert a != kp.cache_key("m", PAGE, system="Y") and a != kp.cache_key("m", PAGE, "s", system="X")
    assert kp.cache_key("m", PAGE, system=kp.EXTRACT_SYSTEM) == kp.cache_key("m", PAGE)

def test_prompt_text_from_git_checks_the_hash(tmp_path):
    # tmp git repo with eval/kaggle_policy.py containing only: EXTRACT_SYSTEM = "hello"
    assert kp.prompt_text_from_git("HEAD", hashlib.sha256(b"hello").hexdigest()[:12], repo=tmp_path) == "hello"
    with pytest.raises(ValueError):
        kp.prompt_text_from_git("HEAD", "000000000000", repo=tmp_path)

@pytest.mark.skipif(not _has_commit("e42203d"), reason="history not available")
def test_base_prompt_is_recoverable_byte_for_byte():
    assert hashlib.sha256(kp.prompt_text_from_git("e42203d", "9566ad4b6b95").encode()).hexdigest()[:12] == "9566ad4b6b95"

def test_extractor_uses_the_given_system_and_keys_the_cache_by_it(tmp_path):
    llm = StubLLM(); rk.Extractor(llm, tmp_path, False, "", system="SYS-A").run(PAGE)
    assert llm.calls == ["SYS-A"]
    assert (tmp_path / f"{kp.cache_key('stub-model', PAGE, '', system='SYS-A')}.json").exists()

def test_extractor_default_system_is_the_current_prompt(tmp_path):
    llm = StubLLM(); rk.Extractor(llm, tmp_path, False).run(PAGE)
    assert llm.calls == [kp.EXTRACT_SYSTEM]

def test_select_prompt():
    assert rk.select_prompt("", "") == (kp.EXTRACT_SYSTEM, kp.PROMPT_VERSION)
    with pytest.raises(SystemExit):
        rk.select_prompt("e42203d", "")
```

- [x] **Step 2: Run** `python -m pytest tests/test_kaggle_runner.py -q`. Expected: FAIL (`cache_key() got an unexpected keyword 'system'`, missing `prompt_text_from_git`).
- [x] **Step 3: Implement** the four signatures above. `cache_key` replaces the global lookup by `system or EXTRACT_SYSTEM` inside the same `(model, system, "ClaimList", user_prompt)` tuple. `Extractor` stores `self.system = system or kp.EXTRACT_SYSTEM` and uses it in both the `structured` call and the token estimate. Add the two flags to the parser (help text: selecting a prompt is not editing it).
- [x] **Step 4: Run** the same command. Expected: all pass. Then `ruff check eval tests`.
- [x] **Step 5: Free real-data check (cache only, no spend):** `python eval/run_kaggle.py --split train --subset dev --cache-only --prompt-commit e42203d --expect-prompt 9566ad4b6b95 --out reports/kaggle_dev_selectcheck`. Expected: header shows `prompt=9566ad4b6b95`, "cache hits" equals the pages to read, and the `score` line equals the one in `reports/kaggle_dev_v6/report.md`.

### Task 2: Estimate before a run, and skip scoring on request

**Files:**
- Modify: `eval/run_kaggle.py` (new `estimate_call_tokens`, constants, flags `--estimate`, `--no-score`)
- Test: `tests/test_kaggle_runner.py`

**Interfaces:**
- Produces: `rk.OUT_PER_IN = 282_140 / 3_408_090` (output tokens per input token, from `reports/kaggle_test_v6/report.md`); `rk.SEC_PER_PAGE = (1.1, 2.5)` (low from the base-prompt runs, high from the f45b runs, both in the run reports)
- Produces: `rk.estimate_call_tokens(pages: list[dict], system: str) -> tuple[int, int]` returning `(tokens_in, tokens_out)`; `tokens_in` is the sum of `estimate_tokens(system + kp.user_prompt(p))`, `tokens_out = round(tokens_in * OUT_PER_IN)`
- Produces: `--estimate` (triage, count pages already cached for this prompt+model+salt, print pages to call, tokens in/out and the time range, then exit before any LLM call or file write); `--no-score` (skip section 5's gold comparison and `misses.csv`)

- [x] **Step 1: Write the failing tests**

```python
def test_estimate_counts_input_exactly_and_output_by_the_measured_ratio():
    tin, tout = rk.estimate_call_tokens([PAGE, PAGE2], system="S")
    assert tin == estimate_tokens("S" + kp.user_prompt(PAGE)) + estimate_tokens("S" + kp.user_prompt(PAGE2))
    assert tout == round(tin * rk.OUT_PER_IN)
    assert rk.OUT_PER_IN == pytest.approx(282_140 / 3_408_090, rel=1e-3)

def test_estimate_of_no_pages_is_zero():
    assert rk.estimate_call_tokens([], system="S") == (0, 0)
```

- [x] **Step 2: Run** `python -m pytest tests/test_kaggle_runner.py -q -k estimate`. Expected: FAIL (no attribute).
- [x] **Step 3: Implement** `estimate_call_tokens`, the constants and the two flags. `--estimate` prints one line per run: `pages to call N (cached M); tokens in X / out Y; time A-B min at the measured pace`.
- [x] **Step 4: Run** the tests (PASS). Then `python eval/run_kaggle.py --split test --estimate --cache-salt p9s3 --prompt-commit e42203d --expect-prompt 9566ad4b6b95`. Expected: about 2304 pages, about 3.4M in tokens, no LLM call, no file written.
- [x] **Step 5:** `python eval/run_kaggle.py --split train --subset dev --limit 3 --cache-only --no-score --out reports/kaggle_dev_noscore`. Expected: `report.md` has no "Score on" section.

### Task 3: Pre-registration, `tally`, and per-page voting

**Files:**
- Create: `eval/consensus.py` (docstring first; `sys.path` and import preamble like `eval/compare_samples.py`)
- Test: `tests/test_kaggle_consensus.py` (create; helpers `claim(field, value, quote="q", evidence="claimed") -> dict` with `display=str(value)`, `ok(*claims) -> {"status": "ok", "claims": [...]}`; a failed sample is `None`; import with `sys.path.insert(0, <repo>/eval)`, `import consensus as cons`)

**Interfaces:**
- Produces constants: `MIN_VALID_VOTES = 3`, `SETTLED_MIN = {5: 4, 3: 3}` with `settled_min_for(k) -> int` (`SETTLED_MIN.get(k, k)`), `RA2_MAX_CHANGED = 30`, `RB0_MIN_AGREE_PCT = 95`, `RB0_SAMPLE = 100`, `RB0_SEED = 20261118`, `RB2_MAX_UNRESOLVED = 10`, `BASE_COMMIT = "e42203d"`, `BASE_VERSION = "9566ad4b6b95"`, `SECOND_VERSION = "f45b4b77c99d"`, `ADJUDICATOR_MODEL = "MiniMax-M3"`
- Produces: `Vote = Optional[str]` (a field signature, `None` = the sample says nothing about the field)
- Produces: `@dataclass(frozen=True) Tally(valid: int, has_winner: bool, winner: Vote, agree: int, settled: bool, leaders: frozenset)`; `leaders` = the votes with the highest count
- Produces: `tally(votes: Sequence[Vote], settled_min: int) -> Tally` (a winner needs `valid >= MIN_VALID_VOTES` and a strict majority of the valid votes; `settled` needs a winner and `agree >= settled_min`; `agree` counts votes equal to the winner, or the top count when there is none)
- Produces: `sample_signatures(extract: Optional[dict]) -> Optional[dict[str, str]]` (`None` for a failed or missing sample; otherwise `kp.claim_signature(kp.normalise_page_claims([Claim.model_validate(c) ...])[0])`)
- Produces: `page_tallies(extracts: Sequence[Optional[dict]], settled_min: int) -> dict[str, Tally]` (one entry per field any valid sample mentions; valid samples that do not mention it vote `None`; failed samples vote nothing)

- [x] **Step 1: Write the pre-registration.** The module docstring contains §6 (R0, RA1, RA2, RB0, RB1, RB2 and the candidate rule) verbatim, D1-D8 above, the sample lists of Task 4, and a line stating that labels are used only to report and to test these rules. Add a test that pins the constants:

```python
def test_preregistered_constants():
    assert (cons.MIN_VALID_VOTES, cons.settled_min_for(5), cons.settled_min_for(3)) == (3, 4, 3)
    assert (cons.RA2_MAX_CHANGED, cons.RB0_SAMPLE, cons.RB0_MIN_AGREE_PCT, cons.RB2_MAX_UNRESOLVED, cons.RB0_SEED) == (30, 100, 95, 10, 20261118)
    assert "RA1" in cons.__doc__ and "RB1" in cons.__doc__ and "D2" in cons.__doc__
```

- [x] **Step 2: Write the failing tally and voting tests**

```python
def test_tally_unanimous_five_is_settled():
    t = cons.tally(["a"] * 5, 4)
    assert (t.valid, t.has_winner, t.winner, t.agree, t.settled) == (5, True, "a", 5, True)

def test_tally_four_of_five_settled_three_of_five_not():
    assert cons.tally(["a"] * 4 + ["b"], 4).settled
    t = cons.tally(["a"] * 3 + ["b"] * 2, 4)
    assert t.has_winner and t.winner == "a" and not t.settled

def test_tally_two_two_one_has_no_winner_and_two_leaders():
    t = cons.tally(["a", "a", "b", "b", "c"], 4)
    assert not t.has_winner and t.leaders == frozenset({"a", "b"}) and not t.settled

def test_tally_fewer_than_three_valid_votes_never_wins():
    t = cons.tally(["a", "a"], 4)
    assert t.valid == 2 and not t.has_winner and not t.settled

def test_tally_absent_is_a_vote_and_can_win():
    t = cons.tally([None, None, None, "a", "a"], 4)
    assert t.has_winner and t.winner is None and t.agree == 3 and not t.settled

def test_tally_three_samples_need_three_of_three():
    assert cons.tally(["a"] * 3, 3).settled and not cons.tally(["a", "a", "b"], 3).settled

def test_failed_sample_is_not_an_absent_vote():
    sso = claim("sso_saml", "all_plans")
    t = cons.page_tallies([ok(sso), None, None, ok(sso), ok()], 4)["sso_saml"]
    assert t.valid == 3 and t.has_winner and t.winner is not None      # if failures counted as absent this would be None

def test_silence_is_a_vote_for_absent():
    t = cons.page_tallies([ok(claim("sso_saml", "all_plans"))] + [ok()] * 4, 4)["sso_saml"]
    assert t.has_winner and t.winner is None
```

- [x] **Step 3: Run** `python -m pytest tests/test_kaggle_consensus.py -q`. Expected: FAIL (module has no `tally`).
- [x] **Step 4: Implement** the signatures above. Order of the samples is the caller's list order and is the only tie-break anywhere.
- [x] **Step 5: Run** the tests (PASS).

### Task 4: Consensus claims, case records, stage-A numbers

**Files:**
- Modify: `eval/consensus.py`
- Test: `tests/test_kaggle_consensus.py`

**Interfaces:**
- Consumes: Task 3 (`Tally`, `tally`, `page_tallies`, `sample_signatures`), `rk.triage`, `rk.readable`, `rk.assemble_case`
- Produces: `@dataclass(frozen=True) Sample(label: str, system: str, model: str, salt: str)` with property `prompt_version` (`sha256(system)[:12]`); `require_one_prompt(samples: Sequence[Sample]) -> None` (raises `ValueError` if prompt versions or models differ)
- Produces: `TARGETS` (dict) with keys `test`, `holdout`, `dev`, each `{"split", "subset", "a": [(label, salt), ...], "second": [(label, salt), ...]}`. test: a = `s0 ""`, `noise1 noise1`, `p9s3 p9s3`, `p9s4 p9s4`, `p9s5 p9s5`; second = `v7 ""`, `v7n1 v7n1`. holdout: a = `s0 ""`, `p9h1 p9h1`, `p9h2 p9h2`; second = `f45 ""`. dev (R0 only): a = `s0 ""`; second = `[]`. Split/subset: test = (`test`, `all`), holdout = (`train`, `holdout`), dev = (`train`, `dev`)
- Produces: `load_extract(cache: Path, sample: Sample, page: dict) -> Optional[dict]` (reads `kp.cache_key(sample.model, page, sample.salt, system=sample.system)`; `None` if absent)
- Produces: `page_claims(extracts: Sequence[Optional[dict]], tallies: dict[str, Tally], overrides: Mapping[str, "Resolution"] | None = None) -> Optional[dict]` returning `{"status": "ok", "claims": [...]}` or `None` when no sample is valid. Per field: a winner of `None` carries nothing; a winner signature carries the raw claims of the first sample (list order) whose signature for the field equals it; no winner carries the first valid sample's raw claims for that field. `overrides` is used by Task 6 (a `Resolution` of kind `unresolved` drops the field; `tie_broken` replaces the winner)
- Produces: `case_records(recs: list[dict], per_sample: Sequence[Mapping[str, Optional[dict]]], settled_min: int, overrides: Mapping[tuple[str, str], "Resolution"] | None = None) -> tuple[list[dict], dict[tuple[str, str], Tally], dict[str, list[tuple[str, ...]]]]` (third element `gaps` per D8: `holder_split` = `(page_id, field)` for `security_certifications` winners whose carrier samples disagree on the sorted tuple of normalised holders; `post_vote_mismatch` = `(page_id, field)` where `sample_signatures` of the consensus record differs from the voted winner for a winner that was not overridden; one case: `per_sample[i][page_id]` is sample i's extract, `overrides` is keyed `(page_id, field)`; `stage_a_numbers` takes `per_sample[i][case_id][page_id]` and slices per case; returns copies of `recs` whose readable entries carry `extract`: the consensus dict, or `{"status": "failed", "error": "no sample extracted this page", "claims": [], "in": 0, "out": 0}`; tallies keyed `(page_id, field)`; unreadable recs untouched)
- Produces: `score_rows(pred: Mapping[str, str], gold: Mapping[str, str]) -> dict` with `rows`, `wrong` (pred != gold, every row) and `false_confident` (requirement rows, i.e. ids not ending `__OVERALL`, with gold in UNKNOWN/NEEDS_REVIEW and pred in PASS/FAIL; same definition as `metric.py` but a count)
- Produces: `changed_rows(a: Mapping[str, str], b: Mapping[str, str]) -> list[str]` (ids whose verdict differs)
- Produces: `stage_a_numbers(case_ids, recs_by_case, per_sample, settled_min, assemble) -> dict` where `assemble(case_id, recs) -> {"verdicts": dict, "overall": str, "bad": int, "ours_disagreements": int}` is injected. Returns `preds` (id -> verdict, OVERALL rows included), `singles` (per sample: preds), `tallies_summary` (counts `voted`, `settled`, `unsettled`, `per_field_unsettled`), `unsettled` (list of `{case_id, page_id, field, valid, agree}`), `changed_vs_first` (ids, consensus vs sample 0), `fragile` (ids that change when any one sample is left out), `incomplete_cases` (case ids with a failed page), `signature_gaps` (`holder_split` and `post_vote_mismatch`, each a list of `(case_id, page_id, field)`)

- [x] **Step 1: Write the failing tests**

```python
def test_consensus_carries_the_first_sample_whose_signature_wins():
    a, b = claim("sso_saml", "all_plans", quote="A"), claim("sso_saml", "all_plans", quote="B")
    exs = [ok(claim("sso_saml", "none")), ok(a), ok(b), ok(a), ok(b)]
    got = cons.page_claims(exs, cons.page_tallies(exs, 4))
    assert [x["quote"] for x in got["claims"]] == ["A"]        # the signature ignores the quote; sample 1 is the first carrying the winner

def test_no_winner_carries_the_first_valid_samples_claims():
    x, y = claim("sso_saml", "all_plans", quote="X"), claim("sso_saml", "none", quote="Y")
    exs = [None, ok(x), ok(x), ok(y), ok(y)]                   # 4 valid votes, 2-2: no strict majority
    got = cons.page_claims(exs, cons.page_tallies(exs, 4))
    assert [c["quote"] for c in got["claims"]] == ["X"]        # the first VALID sample is index 1

def test_no_winner_and_silent_first_sample_carries_nothing():
    x, y = claim("sso_saml", "all_plans"), claim("sso_saml", "none")
    exs = [ok(), ok(x), ok(x), ok(y), ok(y)]                   # absent 1, x 2, y 2: no strict majority
    assert cons.page_claims(exs, cons.page_tallies(exs, 4))["claims"] == []

def test_k1_reproduces_the_sample():                           # R0 on synthetic claims
    cert = {"name": "SOC 2 Type II", "status": "certified", "date": "2026-01-02", "issuer": "Acme Audit LLP", "valid_until": None}
    ex = ok(claim("sso_saml", "all_plans"), claim("dpa_available", "yes"), claim("security_certifications", [cert]))
    got = cons.page_claims([ex], cons.page_tallies([ex], 1))
    assert cons.sample_signatures(got) == cons.sample_signatures(ex) and len(got["claims"]) == 3

def test_page_with_no_valid_sample_is_failed_and_makes_the_case_incomplete():
    recs = [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None}]
    out, _, _ = cons.case_records(recs, [{"p1": None}, {}], 4)
    assert out[0]["extract"]["status"] == "failed"

def test_unreadable_pages_are_not_voted_on():
    recs = [{"role": "ignored", "quarantine": [], "page": {"page_id": "p1"}, "extract": None},
            {"role": "official", "quarantine": ["injection"], "page": {"page_id": "p2"}, "extract": None}]
    out, tallies, _ = cons.case_records(recs, [{"p1": ok(claim("sso_saml", "all_plans")), "p2": ok(claim("sso_saml", "all_plans"))}], 1)
    assert tallies == {} and all(r["extract"] is None for r in out)

def test_a_repeated_sample_reproduces_itself_through_the_winner_path():      # R0, winner path (k=5 identical samples)
    ex = ok(claim("sso_saml", "all_plans"), claim("dpa_available", "yes"))
    exs = [ex] * 5
    assert all(t.has_winner and t.settled for t in cons.page_tallies(exs, 4).values())
    assert cons.sample_signatures(cons.page_claims(exs, cons.page_tallies(exs, 4))) == cons.sample_signatures(ex)

PAGE1 = [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None}]

def test_equal_signatures_with_different_holders_are_reported():
    def cert(holder): return {"name": "ISO 27001", "status": "certified", "date": "2026-01-02", "issuer": "Acme Audit LLP", "valid_until": None, "holder": holder}
    exs = [ok(claim("security_certifications", [cert(h)])) for h in ("Acme Ltd", "Acme Ltd", "Acme Global Pty Ltd", "Acme Ltd", "Acme Ltd")]
    assert cons.page_tallies(exs, 4)["security_certifications"].settled              # the signature cannot see the holder
    _, _, gaps = cons.case_records(PAGE1, [{"p1": e} for e in exs], 4)
    assert gaps["holder_split"] == [("p1", "security_certifications")] and gaps["post_vote_mismatch"] == []

def test_cross_field_normalisation_after_the_vote_is_reported():
    sub = claim("subprocessors", [{"name": "Acme Cloud", "category": "hosting", "country": "Ireland"}])
    reg = lambda q: claim("data_regions", {"available": ["Ireland"], "planned": []}, quote=q)
    named, plain = "We host in Ireland with Acme Cloud.", "Data stays in Ireland."
    # regions: kept in samples 0, 3, 4 (wins 3 of 5); sub-processors: listed in 1-4 (wins 4 of 5). Sample 0 is the first carrier of the
    # winning region, and its quote names Acme Cloud, which only samples 1-4 list: on the combined claims the region is dropped.
    exs = [ok(reg(named)), ok(reg(named), sub), ok(reg(named), sub), ok(reg(plain), sub), ok(reg(plain), sub)]
    _, _, gaps = cons.case_records(PAGE1, [{"p1": e} for e in exs], 4)
    assert gaps["post_vote_mismatch"] == [("p1", "data_regions")]

def test_mixed_prompts_or_models_are_refused():
    with pytest.raises(ValueError):
        cons.require_one_prompt([cons.Sample("a", "SYS-1", "m", ""), cons.Sample("b", "SYS-2", "m", "")])
    with pytest.raises(ValueError):
        cons.require_one_prompt([cons.Sample("a", "SYS-1", "m1", ""), cons.Sample("b", "SYS-1", "m2", "")])

def test_score_rows_matches_the_metrics_definition():
    gold = {"c__R1": "UNKNOWN", "c__R2": "PASS", "c__OVERALL": "BLOCKED"}
    pred = {"c__R1": "PASS", "c__R2": "FAIL", "c__OVERALL": "BLOCKED"}
    assert cons.score_rows(pred, gold) == {"rows": 3, "wrong": 2, "false_confident": 1}

def _three_cases(quote="q"):
    sso = claim("sso_saml", "all_plans", quote=quote)
    recs = {c: [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None}] for c in ("c1", "c2", "c3")}
    says = {"c1": [1, 1, 1, 1, 1], "c2": [1, 0, 0, 0, 0], "c3": [1, 1, 0, 0, 0]}      # who says "sso" in each of 5 samples
    per_sample = [{c: {"p1": ok(sso) if says[c][i] else ok()} for c in says} for i in range(5)]       # per_sample[i][case_id][page_id]
    assemble = lambda cid, rs: {"verdicts": {"R1": "PASS" if rs[0]["extract"]["claims"] else "UNKNOWN"}, "overall": "CONDITIONAL",
                                "bad": 0, "ours_disagreements": 0}
    return cons.stage_a_numbers(["c1", "c2", "c3"], recs, per_sample, 4, assemble)

def test_stage_a_numbers_on_invented_cases():
    r = _three_cases()
    assert (r["tallies_summary"]["voted"], r["tallies_summary"]["settled"], r["tallies_summary"]["unsettled"]) == (3, 2, 1)   # c1 5/5, c2 4 of 5 absent, c3 3 of 5 absent
    assert [r["preds"][f"{c}__R1"] for c in ("c1", "c2", "c3")] == ["PASS", "UNKNOWN", "UNKNOWN"]
    assert r["changed_vs_first"] == ["c2__R1", "c3__R1"]      # sample 0 alone says "sso" in all three cases
    assert r["fragile"] == ["c3__R1"]                         # drop one "absent" sample: 2-2 tie, the first sample's claim is carried -> PASS
    assert r["incomplete_cases"] == [] and [u["case_id"] for u in r["unsettled"]] == ["c3"]

def test_stage_a_outputs_never_contain_claim_text():
    canary = "CANARY-QUOTE-7731"
    r = _three_cases(quote=canary)
    assert canary not in json.dumps(r, default=str) and canary not in cons.format_stage_a(r)
```

- [x] **Step 2: Run** `python -m pytest tests/test_kaggle_consensus.py -q`. Expected: FAIL.
- [x] **Step 3: Implement** the signatures above plus `format_stage_a(result) -> str` (counts and case ids only; the report text is built from `tallies_summary`, id lists and numbers, never from claims). `fragile` is computed by calling `stage_a_numbers`' own consensus step once per left-out sample.
- [x] **Step 4: Run** the tests (PASS).

### Task 5: Rules and the candidate rule

**Files:**
- Modify: `eval/consensus.py`
- Test: `tests/test_kaggle_consensus.py`

**Interfaces:**
- Produces: `Rule = dict` with `{"status": "pass" | "fail" | "untested" | "not_run", "detail": str}`
- Produces: `ra1(consensus: dict, singles: Sequence[dict]) -> Rule` (inputs are `score_rows` dicts; passes iff `consensus["wrong"] <= ceil(mean wrong of singles)` and `consensus["false_confident"] <= ceil(mean false_confident of singles)`, integer ceiling)
- Produces: `ra2(changed: int) -> Rule` (pass iff `changed <= RA2_MAX_CHANGED`)
- Produces: `rb0(agree: int, n: int) -> Rule` (pass iff `n >= 1` and `agree * 100 >= RB0_MIN_AGREE_PCT * n`)
- Produces: `rb1(changed: int, wrong_to_right: int, right_to_wrong: int) -> Rule` (`untested` if `changed == 0`; else pass iff `wrong_to_right > right_to_wrong`)
- Produces: `rb2(unresolved_on_test: int) -> Rule` (pass iff `<= RB2_MAX_UNRESOLVED`)
- Produces: `candidate_rule(rules: Mapping[str, Rule]) -> str` returning `"none"`, `"A"` or `"A+B"`: `A` needs R0, RA1, RA2 all `pass`; `A+B` also needs RB0, RB1, RB2 all `pass` (`untested`/`not_run`/`fail` never count)
- Produces: `rules.json` helpers `load_rules(path) -> dict` and `save_rule(path, key, rule) -> None` (read-modify-write; key one of `R0 RA1 RA2 RB0 RB1 RB2`)

- [x] **Step 1: Write the failing tests**

```python
def test_ra1_rounds_the_mean_up():
    s = [{"wrong": 3, "false_confident": 0}, {"wrong": 3, "false_confident": 1}, {"wrong": 4, "false_confident": 1}]   # means 3.33 -> 4, 0.67 -> 1
    assert cons.ra1({"wrong": 4, "false_confident": 1}, s)["status"] == "pass"
    assert cons.ra1({"wrong": 5, "false_confident": 1}, s)["status"] == "fail"
    assert cons.ra1({"wrong": 4, "false_confident": 2}, s)["status"] == "fail"

def test_ra2_boundary():  assert cons.ra2(30)["status"] == "pass" and cons.ra2(31)["status"] == "fail"
def test_rb0_boundary(): assert cons.rb0(95, 100)["status"] == "pass" and cons.rb0(94, 100)["status"] == "fail"
def test_rb2_boundary(): assert cons.rb2(10)["status"] == "pass" and cons.rb2(11)["status"] == "fail"

def test_rb1_untested_is_not_a_pass():
    assert cons.rb1(0, 0, 0)["status"] == "untested"
    assert cons.rb1(3, 2, 1)["status"] == "pass" and cons.rb1(3, 1, 1)["status"] == "fail" and cons.rb1(2, 0, 1)["status"] == "fail"

def test_candidate_rule():
    P, F, U, N = "pass", "fail", "untested", "not_run"
    def r(**kw): return {k: {"status": v, "detail": ""} for k, v in kw.items()}
    allp = dict(R0=P, RA1=P, RA2=P, RB0=P, RB1=P, RB2=P)
    assert cons.candidate_rule(r(**allp)) == "A+B"
    assert cons.candidate_rule(r(**{**allp, "RB1": U})) == "A"           # B untested -> A-only
    assert cons.candidate_rule(r(**{**allp, "RB0": N, "RB1": N, "RB2": N})) == "A"
    assert cons.candidate_rule(r(**{**allp, "RA1": F})) == "none"
    assert cons.candidate_rule(r(**{**allp, "R0": N})) == "none"

def test_rules_json_round_trip(tmp_path):
    p = tmp_path / "rules.json"; cons.save_rule(p, "RA2", cons.ra2(5)); cons.save_rule(p, "R0", {"status": "pass", "detail": ""})
    assert set(cons.load_rules(p)) == {"RA2", "R0"}
```

- [x] **Step 2: Run** `python -m pytest tests/test_kaggle_consensus.py -q -k "ra1 or ra2 or rb or candidate or rules_json"`. Expected: FAIL.
- [x] **Step 3: Implement** the signatures above.
- [x] **Step 4: Run** (PASS).

### Task 6: Stage B logic: triggers, resolution, plan, RB0 draw

**Files:**
- Modify: `eval/consensus.py`
- Test: `tests/test_kaggle_consensus.py`

**Interfaces:**
- Consumes: `Tally`, `tally`, `page_tallies`, `case_records(..., overrides)`
- Produces: `triggers(flash: Tally, second_votes: Sequence[Vote]) -> tuple[bool, bool]` returning `(unsettled, second_opinion)`. `second_opinion` is true iff there is at least one second-opinion vote, all of them are equal, `flash.has_winner`, and the vote differs from `flash.winner`
- Produces: `@dataclass(frozen=True) Resolution(kind: Literal["confirmed", "tie_broken", "unresolved"], winner: Vote)`
- Produces: `resolve(flash: Tally, adj_votes: Sequence[Vote]) -> Resolution`: `adj = tally(adj_votes, 3)`. No adjudicator winner -> `unresolved`. Winner not in `flash.leaders` -> `unresolved`. Otherwise `confirmed` when `flash.has_winner`, else `tie_broken` with `winner = adj.winner`
- Produces: `b_plan(tallies_by_case: Mapping[str, Mapping[tuple[str, str], Tally]], second_votes: Mapping[tuple[str, str, str], Sequence[Vote]]) -> list[dict]` returning one entry per page `{case_id, page_id, fields: [...], triggers: {field: [1] | [2] | [1, 2]}}`, pages de-duplicated, sorted by `(case_id, page_id)`
- Produces: `rb0_draw(tallies_by_case, n: int = RB0_SAMPLE, seed: int = RB0_SEED) -> list[tuple[str, str, str]]` (pool per D4, sorted, then `random.Random(seed).sample`; all of the pool if smaller than `n`)

- [x] **Step 1: Write the failing tests**

```python
def T(votes, k=5): return cons.tally(votes, cons.settled_min_for(k))

def test_trigger_one_is_unsettled_trigger_two_is_a_unanimous_disagreeing_second_opinion():
    assert cons.triggers(T(["a"] * 3 + ["b"] * 2), []) == (True, False)
    assert cons.triggers(T(["a"] * 5), ["b", "b"]) == (False, True)
    assert cons.triggers(T(["a"] * 5), ["b", "c"]) == (False, False)      # second opinion not unanimous
    assert cons.triggers(T(["a"] * 5), ["a", "a"]) == (False, False)
    assert cons.triggers(T(["a"] * 5), [None, None]) == (False, True)     # second opinion says "absent" unanimously
    assert cons.triggers(T(["a", "a", "b", "b", "c"]), ["b", "b"]) == (True, False)    # no flash winner: trigger two cannot fire

def test_resolve_confirmed_tie_broken_unresolved():
    assert cons.resolve(T(["a"] * 3 + ["b"] * 2), ["a", "a", "b"]).kind == "confirmed"
    r = cons.resolve(T(["a", "a", "b", "b", "c"]), ["b", "b", "c"]); assert (r.kind, r.winner) == ("tie_broken", "b")
    assert cons.resolve(T(["a", "a", "b", "b", "c"]), ["c", "c", "c"]).kind == "unresolved"     # c is not a tied leader
    assert cons.resolve(T(["a"] * 4 + ["b"]), ["b", "b", "b"]).kind == "unresolved"             # adjudicator disagrees with consensus
    assert cons.resolve(T(["a"] * 4 + ["b"]), ["a", "b", None]).kind == "unresolved"            # no adjudicator majority
    assert cons.resolve(T(["a"] * 4 + ["b"]), ["a", "a"]).kind == "unresolved"                  # fewer than three adjudicator votes

def test_plan_pages_are_deduplicated():
    tallies = {"c1": {("p1", "f1"): T(["a"] * 3 + ["b"] * 2), ("p1", "f2"): T(["a"] * 3 + ["b"] * 2), ("p2", "f1"): T(["a"] * 5)}}
    plan = cons.b_plan(tallies, {})
    assert [(e["case_id"], e["page_id"], e["fields"]) for e in plan] == [("c1", "p1", ["f1", "f2"])]

def test_resolutions_change_the_carried_claims():
    x, y = claim("sso_saml", "all_plans", quote="X"), claim("sso_saml", "none", quote="Y")
    exs = [ok(x), ok(x), ok(y), ok(y), ok()]                   # 2-2-1: no winner, placeholder is sample 0's claim
    t, ysig = cons.page_tallies(exs, 4), cons.sample_signatures(ok(y))["sso_saml"]
    quotes = lambda res: [c["quote"] for c in cons.page_claims(exs, t, {"sso_saml": res})["claims"]]
    assert quotes(cons.Resolution("tie_broken", ysig)) == ["Y"]
    assert quotes(cons.Resolution("unresolved", None)) == []
    assert quotes(cons.Resolution("confirmed", None)) == ["X"]

def test_rb0_draw_is_seeded_label_free_and_unanimous_only():
    tallies = {"c%d" % i: {("p", "f"): T(["a"] * 5)} for i in range(300)}
    tallies["c0"][("p", "g")] = T([None] * 5); tallies["c1"][("p", "h")] = T(["a"] * 4 + ["b"])
    d1, d2 = cons.rb0_draw(tallies), cons.rb0_draw(tallies)
    assert d1 == d2 and len(d1) == 100 and len(set(d1)) == 100
    assert all(f == "f" for _, _, f in d1)                  # not absent-unanimous, not 4-of-5
    assert len(cons.rb0_draw({"c": {("p", "f"): T(["a"] * 5)}})) == 1
```

- [x] **Step 2: Run** `python -m pytest tests/test_kaggle_consensus.py -q -k "trigger or resolve or plan or resolutions or rb0_draw"`. Expected: FAIL.
- [x] **Step 3: Implement** the signatures above, and make `page_claims` honor `overrides` as described in Task 4.
- [x] **Step 4: Run** (PASS).

### Task 7: Stage B and stage A commands

**Files:**
- Modify: `eval/consensus.py` (`main` with subcommands)
- Test: `tests/test_kaggle_consensus.py` (only the pure parts: `adjudicator_estimate`, `b_row_changes`)

**Interfaces:**
- Consumes: Tasks 1-6; `rk.load_cases`, `rk.load_pages`, `rk.select_cases`, `rk.triage`, `rk.assemble_case`, `rk.Extractor`, `rk.estimate_call_tokens`, `rk.load_module`; `LLM.extractor(budget=...)` for the Flash model name; `LLM(budget=budget, role="extract", model=ADJUDICATOR_MODEL)` for the adjudicator
- Produces: `b_row_changes(a: Mapping[str, str], ab: Mapping[str, str], gold: Mapping[str, str]) -> dict` with `changed`, `wrong_to_right` (a wrong, ab right), `right_to_wrong` (a right, ab wrong)
- Produces: `adjudicator_estimate(pages: list[dict], system: str, samples: int) -> dict` (`calls = len(pages) * samples`, tokens from `rk.estimate_call_tokens`, minutes from `rk.SEC_PER_PAGE`)
- Produces CLI (`python eval/consensus.py <cmd> --target test|holdout|dev [...]`), all writing under `reports/kaggle_consensus/<target>/` and printing counts and case ids only:
  - `check-r0`: for each target, consensus of sample `s0` alone (k=1) through `assemble_case` vs the stored `reports/kaggle_{dev,holdout,test}_v6/cases.jsonl` verdicts and overall (compared in code, nothing printed but counts); then the same with `s0` repeated k times (k = 5 test, 3 holdout and dev, so every field takes the winner path); saves rule `R0` as `pass` only if both variants differ in 0 rows on all three targets
  - `stage-a [--allow-partial]`: loads the flash samples (base prompt from `kp.prompt_text_from_git(BASE_COMMIT, BASE_VERSION)`, model from `LLM.extractor`), prints per-sample coverage for the `a` samples and the `second` samples, refuses to finish unless every sample covers 100% of readable pages (or `--allow-partial`), runs `stage_a_numbers`, writes `stage_a.md`, `preds_A.csv`, `unsettled.json`. holdout saves `RA1` (consensus vs each single sample, `score_rows` against `train_labels.csv`); test saves `RA2` (`changed_vs_first`)
  - `plan-b`: second-opinion votes from cached samples (system = current `kp.EXTRACT_SYSTEM`, asserted equal to `SECOND_VERSION`), `b_plan.json`; with `--rb0` on target test also writes `rb0_plan.json` from `rb0_draw`; prints page and call counts and `adjudicator_estimate` for 1 and 3 samples
  - `run-b --plan b_plan.json|rb0_plan.json --salts adj1,adj2,adj3 [--estimate] [--workers 6] [--max-usd 1.5]`: the pages of the plan only, base prompt, `Extractor(llm, cache, False, salt, system=base_system)` per salt; `--estimate` prints and exits
  - `resolve`: loads adjudicator samples from the cache, `resolve()` for each triggered page-field, `case_records(..., overrides)`, assembles A+B, writes `stage_b.md` and `preds_AB.csv`; on target test saves `RB2` (unresolved count) and, from `rb0_plan.json` and the `adj1` samples, `RB0`; on target holdout saves `RB1` from `b_row_changes(preds_A, preds_AB, gold)`; prints counts of confirmed / tie_broken / unresolved

- [x] **Step 1: Write the failing tests**

```python
def test_b_row_changes():
    gold = {"x": "PASS", "y": "FAIL", "z": "PASS", "w": "PASS"}
    a, ab = {"x": "FAIL", "y": "FAIL", "z": "PASS", "w": "FAIL"}, {"x": "PASS", "y": "PASS", "z": "PASS", "w": "UNKNOWN"}
    assert cons.b_row_changes(a, ab, gold) == {"changed": 3, "wrong_to_right": 1, "right_to_wrong": 1}     # w stays wrong

def test_adjudicator_estimate_counts_calls_per_sample():
    e = cons.adjudicator_estimate([PAGE, PAGE2], "S", 3)
    assert e["calls"] == 6 and e["tokens_in"] == 3 * sum(estimate_tokens("S" + kp.user_prompt(p)) for p in (PAGE, PAGE2))
```

- [x] **Step 2: Run** `python -m pytest tests/test_kaggle_consensus.py -q -k "b_row_changes or adjudicator_estimate"`. Expected: FAIL.
- [x] **Step 3: Implement** the helpers and the subcommands. The adjudicator run must reuse `rk.Extractor` so that 429 waits, retries and the budget behave as in the existing runner. Cache-only everywhere except `run-b`.
- [x] **Step 4: Run** the tests (PASS), then `python eval/consensus.py --help` and each subcommand's `--help` (exit 0).

### Task 8: Candidate file with the runner's self-check

**Files:**
- Modify: `eval/run_kaggle.py` (extract the self-check block of `main` into `write_submission`; `main` calls it)
- Modify: `eval/consensus.py` (`candidate` subcommand)
- Test: `tests/test_kaggle_runner.py`

**Interfaces:**
- Produces: `rk.write_submission(path: Path, test_rows: list[dict], preds: Mapping[str, str], metric) -> list[str]` (writes the CSV exactly as `main` does today: missing ids default to `UNKNOWN`, `CONDITIONAL` for OVERALL; reads it back; returns the problems from `kp.validate_submission` plus the `metric.components` check; empty list = OK). `main`'s printed lines and return codes stay as they are
- Produces: `python eval/consensus.py candidate`: loads `rules.json`, calls `candidate_rule`; `none` -> prints the rule table and writes nothing; `A` or `A+B` -> builds test preds from `preds_A.csv` or `preds_AB.csv`, refuses if `incomplete_cases` is non-empty (Task 4), calls `rk.write_submission` into `reports/kaggle_consensus/candidate/submission.csv`, prints the row count, the self-check result and the number of rows that differ from `reports/kaggle_test_v6/submission.csv`. For `A+B` it also writes `submission_A_only.csv` the same way (D7). It never uploads

- [x] **Step 1: Write the failing tests** (`StubMetric.components(solution, submission)` records its call and returns `{}`; `test_rows = [{"id": "c__R1"}, {"id": "c__OVERALL"}]`)

```python
def test_write_submission_writes_ids_in_order_with_defaults(tmp_path):
    problems = rk.write_submission(tmp_path / "s.csv", ROWS, {"c__R1": "PASS"}, StubMetric())
    assert problems == [] and (tmp_path / "s.csv").read_text().splitlines() == ["id,verdict", "c__R1,PASS", "c__OVERALL,CONDITIONAL"]

def test_write_submission_reports_an_invalid_label(tmp_path):
    assert rk.write_submission(tmp_path / "s.csv", ROWS, {"c__R1": "MAYBE"}, StubMetric())

def test_write_submission_reports_what_the_hosts_metric_rejects(tmp_path):
    class Bad(StubMetric):
        def components(self, *a): raise ValueError("nope")
    assert any("metric.py rejected" in p for p in rk.write_submission(tmp_path / "s.csv", ROWS, {}, Bad()))
```

- [x] **Step 2: Run** `python -m pytest tests/test_kaggle_runner.py -q -k write_submission`. Expected: FAIL.
- [x] **Step 3: Implement** `write_submission` by moving the block unchanged; add the `candidate` subcommand.
- [x] **Step 4: Run** the tests (PASS). Regression, free: `python eval/run_kaggle.py --split test --cache-only --prompt-commit e42203d --expect-prompt 9566ad4b6b95 --out reports/kaggle_test_refactor_check2`; its `submission.csv` must be byte-identical to `reports/kaggle_test_v6/submission.csv` (`cmp`).

### Task 9: Offline checks on cached data (no spend)

**Files:** none changed. Outputs in `reports/kaggle_consensus/`.

- [x] **Step 1:** `python eval/consensus.py check-r0 --target dev` then `holdout` and `test`. Expected: both variants (k=1, and `s0` repeated through the winner path) differ in 0 rows on each target, R0 `pass`. If any row differs: stop and use superpowers:systematic-debugging; do not continue.
- [x] **Step 2:** `python eval/consensus.py stage-a --target test --allow-partial` and `--target holdout --allow-partial` (only `s0` and `noise1` exist on test, only `s0` on holdout, so every page-field is unsettled by design). Expected: coverage lines per sample: 100% for `s0`, `noise1`, `v7`, `v7n1` on test and `s0` on holdout (this is what confirms the salts in D-notes above; a lower number means a wrong salt in `TARGETS`, fix it before anything is drawn, and tell the user because the docstring lists the sample names), 0% for the samples not drawn yet; counts, no crash, no claim text in stdout or `stage_a.md` (grep the files for a few field names only, never for values).
- [x] **Step 3:** Post the counts and the §7 estimates (from `--estimate` for each planned draw below) to the user.

### Task 10: Draw the new samples, one run at a time

**Files:** none changed. Logs in `reports/kaggle_consensus/logs/`.

Each draw is one iteration of: (a) run with `--estimate` and post the result in chat; (b) check the pre-run conditions in Global Constraints (ask the user for the meter readings before the first draw); (c) start the run in the background; (d) when it finishes, read only the counts in `report.md` (pages, cache hits, new calls, failures, wall time); (e) if failures > 0, re-run the same command (cached pages are free) until 0, or stop at the 1% rule; (f) stop and report if wall time passed twice the upper estimate.

Base command for test (change `--cache-salt`): `python eval/run_kaggle.py --split test --prompt-commit e42203d --expect-prompt 9566ad4b6b95 --cache-salt p9s3 --workers 6 --max-usd 1.5 --out reports/kaggle_test_p9s3`

- [x] **Draw 1-3 (test):** salts `p9s3`, `p9s4`, `p9s5`. Expected each: about 2304 pages, about 3.7M tokens, 41-45 min. No `submission.csv` (salted).
- [x] **Draw 4-5 (holdout, base prompt):** `python eval/run_kaggle.py --split train --subset holdout --prompt-commit e42203d --expect-prompt 9566ad4b6b95 --cache-salt p9h1 --no-score --workers 6 --max-usd 1.5 --out reports/kaggle_holdout_p9h1`, then `p9h2`. Expected each: about 1496 pages, about 2.4M tokens, about 26-30 min.
- [x] **Draw 6 (holdout, second opinion, current prompt, no salt):** `python eval/run_kaggle.py --split train --subset holdout --no-score --workers 6 --max-usd 1.5 --out reports/kaggle_holdout_f45`. Expected: about 1496 pages, about 2.9M tokens, about 35-55 min.
- [x] **Step 7:** After all draws, `python eval/consensus.py stage-a --target test` and `--target holdout` with no `--allow-partial`. Expected: coverage 100% for every sample.

### Task 11: Evaluate stage A (RA1, RA2)

**Files:** none changed.

- [x] **Step 1:** Read `reports/kaggle_consensus/rules.json` and the two `stage_a.md` files. Post to the user: settled/unsettled counts per target, per-field unsettled counts, rows changed vs submission #1 (RA2), fragile row count, holdout wrong and false-confident rows of the consensus and of each single sample (RA1), incomplete cases.
- [x] **Step 2:** If RA1 is `fail`: stop, report "A not adopted, #1 stays" with the numbers. If RA2 is `fail`: stop and investigate (counts by field and requirement only); do not submit. Neither failure is repaired by changing a threshold or a sample list. If `holder_split` or `post_vote_mismatch` is non-empty on either target (D8): stop, post the counts and the case ids, and ask the user how to proceed; do not choose.

### Task 12: Stage B

**Files:** none changed.

Same per-run protocol as Task 10 (estimate in chat, one run, counts only).

- [x] **Step 1:** `python eval/consensus.py plan-b --target test --rb0` and `--target holdout`. Post the page counts and the `--estimate` output for each run below.
- [x] **Step 2 (RB0 draw, 1 sample):** `python eval/consensus.py run-b --target test --plan rb0_plan.json --salts adj1 --max-usd 1.5`. Expected: at most 100 pages. This is the MiniMax-M3 measurement: read its pages, recorded tokens and real wall time, compute seconds per page, and re-estimate steps 3 and 4 from it (Flash's pace and price are not a basis for M3); post that estimate before step 3.
- [x] **Step 3 (test triggered pages, 3 samples):** `python eval/consensus.py run-b --target test --plan b_plan.json --salts adj1,adj2,adj3 --max-usd 1.5` (`adj1` is a cache hit where the RB0 pages overlap). Expected: about 60 pages x 3 samples, under 0.5M tokens.
- [x] **Step 4 (holdout triggered pages, 3 samples):** `python eval/consensus.py run-b --target holdout --plan b_plan.json --salts adj1,adj2,adj3 --max-usd 1.5`.
- [x] **Step 5:** `python eval/consensus.py resolve --target test` and `--target holdout`. Post RB0 (`agree of n`), RB1 (`changed`, `wrong_to_right`, `right_to_wrong`, or `untested`), RB2 (unresolved count), and the confirmed / tie_broken / unresolved counts. Expect little power on the holdout (3 wrong rows of 1580 with the base prompt); report it that way.

### Task 13: Candidate

**Files:** none changed.

- [x] **Step 1:** `python eval/consensus.py candidate`. Expected: a rule table and one of `none` / `A` / `A+B`; for `A` or `A+B`, `reports/kaggle_consensus/candidate/submission.csv` with "Self-check: OK" and the count of rows that differ from #1.
- [x] **Step 2:** Show the user the rule table, the candidate, the self-check line and the differing-row count. **Stop and ask for an explicit yes before any submission.** Do not submit, select or upload anything.

### Task 14: Results note and final verification

**Files:**
- Create: `docs/superpowers/results/2026-11-18-kaggle-consensus-results.md` (counts and case ids only; every rule with its number and verdict, negative results included, the cost actually spent, what stayed unverified)

- [x] **Step 1:** Write the note from `rules.json` and the two `stage_a.md` / `stage_b.md` files. State plainly: B untested or not decisive if so; errors every sample repeats are invisible; the holdout is small; the `holder_split` and `post_vote_mismatch` counts (D8) and what the user decided about them; the D7 decisions on both files; the signature cannot see a certificate's holder, and `evidence_level` is not a separate noise source (D8).
- [x] **Step 2 (superpowers:verification-before-completion):** run and read the output of each of: `python -m pytest -q` (default backend), `EMBEDDER=builtin RERANKER=builtin python -m pytest -q`, `ruff check .`, `python -c "import sys; sys.path.insert(0,'eval'); import kaggle_policy as kp; print(kp.PROMPT_VERSION)"` (must print `f45b4b77c99d`), `git diff --stat -- eval/kaggle_policy.py` (only `cache_key` and `prompt_text_from_git` hunks, no change inside `EXTRACT_SYSTEM`), `git status --short` (no commits made; the three experiment files still untracked and untouched).
- [x] **Step 3:** Report to the user: what was run, tokens and time actually spent against the estimates, the rule table, the candidate, and anything unverified.
