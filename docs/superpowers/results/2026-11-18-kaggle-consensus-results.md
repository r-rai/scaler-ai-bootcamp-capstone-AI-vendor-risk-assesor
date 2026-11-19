# Kaggle consensus + escalation: results

Run 2026-11-18 to 2026-11-19 on branch `prod-hardening` (nothing committed, nothing pushed, nothing submitted).
Spec: `docs/superpowers/specs/2026-11-18-kaggle-consensus-escalation-design.md`. Plan: `docs/superpowers/plans/2026-11-18-kaggle-consensus-escalation.md`.
Code: `eval/consensus.py` (the pre-registration is its module docstring), `eval/run_kaggle.py`, `eval/kaggle_policy.py`; tests `tests/test_kaggle_consensus.py`, `tests/test_kaggle_runner.py`.
This note holds counts and case ids only: no claim, quote or page text.

## Outcome

- **Stage A (vote over several samples of the unchanged prompt 9566ad4b6b95) passed every pre-registered rule.** It is the candidate: `reports/kaggle_consensus/candidate/submission.csv`, which differs from submission #1 in **4 of 2378 rows**. Self-check OK. **Not submitted**: a submission needs the user's explicit yes.
- **Stage B (MiniMax-M3 adjudicator on the unsure page-fields) failed RB1 and RB2 and is not adopted.** It did not damage the test file because it was never applied to it; on the holdout it changed 4 rows and all 4 went from right to wrong.
- The candidate rule therefore gives **A-only** (no A+B file, so there is no D7 decision on a second file).
- The expected effect of the candidate on the leaderboard is **unknown**: the test has no labels, and the holdout has too few wrong rows to say more than "no sign of harm" (below).

## Pre-registered rules and results

Rules were written before any new sample was drawn or any label read (docstring of `eval/consensus.py`); no threshold or sample list was moved afterwards.

| Rule | What it asks | Result | Number |
|---|---|---|---|
| R0 | k=1 and the winner path reproduce the stored single-sample verdicts | **pass** | 0 differing rows on dev (1605), holdout (1580), test (2378) |
| RA1 | holdout: consensus wrong rows and false-confident rows each <= the rounded-up mean of the 3 single samples | **pass** | wrong 1 (limit 3), false-confident 0 (limit 1); singles wrong [3, 3, 2], false-confident [1, 0, 0] |
| RA2 | test: k=5 consensus changes <= 30 rows vs submission #1 | **pass** | 4 rows |
| RB0 | adjudicator matches the unanimous Flash signature on >= 95% of a fixed-seed draw of 100 settled test page-fields | **pass** | 100 of 100 |
| RB1 | holdout: rows B changes, wrong to right must exceed right to wrong | **fail** | B changed 4 rows: wrong to right 0, right to wrong 4 |
| RB2 | test: <= 10 page-fields end unresolved | **fail** | 20 unresolved |

Candidate rule (fixed in advance): R0+RA1+RA2 pass gives A-only; RB0+RB1+RB2 must also pass for A+B. Result: **A**.

How to read these results (limits the table does not show):
- **RB0 is weak evidence.** Its draw comes from page-fields on which every valid Flash vote agrees and the signature is not "absent" (D4): the easiest items. 100 of 100 shows M3 reads easy items like Flash, not that it adjudicates the hard ones.
- **RB1 tested only dropped claims.** On the holdout `tie_broken` was 0, so all 4 changed rows came from unresolved (dropped) claims. B's 3 tie-breaks on test were never calibrated against labels.
- **Holdout "fragile" rows are not comparable to test's.** With k=3, leaving one sample out leaves 2 valid votes, below the 3 a winner needs, so every holdout leave-one-out falls back to the first remaining sample.
- **No recorded provenance for the rules.** `rules.json` stores no code version or input hash. The SHA-256 of the module docstring (the pre-registration, including the sections added later) is `84aa78dbe2c583632781b052fbba73df5b77d728d990427887501849d9423b8f` and of `eval/consensus.py` `20ed012f426812554e9b3c49b2e5b06dacddda5d84d52626b01883f9f3af3d9c`, both taken on 2026-11-19 after every analysis. They cannot prove the docstring preceded the data, because nothing is committed; committing the branch would make that checkable.

## Stage A

Samples (all MiniMax-M3.1-Flash, base prompt 9566ad4b6b95; the current prompt f45b4b77c99d only as a second opinion for stage B):
test k=5 (s0 = submission #1, noise1, p9s3, p9s4, p9s5), holdout k=3 (s0, p9h1, p9h2). All coverage was 100% of readable pages.

| | test (k=5) | holdout (k=3) |
|---|---|---|
| page-fields voted | 2560 | 1652 |
| settled | 2512 | 1598 |
| unsettled | 48 | 54 |
| unsettled by field | security_certifications 34, trains_on_customer_data 7, dpa_available 2, sso_saml 2, subprocessors 2, data_regions 1 | security_certifications 30, sso_saml 13, trains_on_customer_data 4, subprocessors 3, security_incidents 2, data_regions 1, dpa_available 1 |
| rows changed vs sample s0 | 4 (c0050__DP-3.2, c0255__DP-2.2, c0255__OVERALL, c0314__AI-2.1) | 4 (c0060__SEC-4.2, c0294__DP-2.2, c0294__OVERALL, c0327__AI-2.1) |
| fragile rows (change if any one sample is left out) | 3 (c0065__AI-2.1, c0258__AI-3.4, c0414__AI-2.1) | 6 (c0060__SEC-4.2, c0294__DP-2.2, c0294__OVERALL, c0312__SEC-4.2-T2, c0327__AI-2.1, c0678__SEC-4.2-T2) |
| incomplete cases | 0 | 0 |

The measured sampling noise this design targets: 14 of 2378 test verdicts flip between two samples of the same prompt (dev: 0 of 1605). The consensus moves 4.

**Candidate vs submission #1** (the 4 rows): c0050__DP-3.2 UNKNOWN -> PASS; c0255__DP-2.2 UNKNOWN -> FAIL; c0255__OVERALL CONDITIONAL -> BLOCKED (follows from DP-2.2); c0314__AI-2.1 FAIL -> NEEDS_REVIEW. Two of the four turn a silent UNKNOWN into a confident verdict, which the metric counts as false-confident whenever the gold is UNKNOWN or NEEDS_REVIEW. Whether the 4 changes help is not known.

**What the holdout can and cannot say.** The holdout had 3 wrong rows of 1580 with the base prompt (singles 3, 3, 2), so RA1 compares a 1 with a mean of 2.67: a difference of under two rows. It shows no sign of harm; it does not show benefit.

## D8: the two gaps the vote signature cannot see

The signature (the spec's `claim_signature`) cannot see a certificate's `holder`, and the consensus is assembled from raw claims carried per field. Measured, not fixed:

- `post_vote_mismatch` (the consensus record differs from the voted winner): **0** on both targets.
- `holder_split` (a winning signature whose carriers disagree on holder): **102** test page-fields (90 cases; 100 on official pages, 2 on registry pages) and **74** holdout (65 cases; 68 official, 6 registry), all `security_certifications`. `check_holders` acts only on registry pages.
- `evidence_level` is not a separate noise source: `rules_lite.py` reads it only in the certificate rule, and `build_facts` derives it from issuer and date, both inside the signature.

The user's decision (2026-11-19): first a free, label-free carrier-swap check, then a pre-registered, labelled RA1 sensitivity on the holdout; neither may change the carrier rule (the first carrier in sample-list order).

- **Carrier swap (label-free):** test 539 carrier combinations, **0 rows move** (the test candidate is carrier-invariant). Holdout 219 combinations, **3 rows move**: c0060__SEC-4.2, c0312__SEC-4.2-T2, c0678__SEC-4.2-T2 (all in registry-split cases).
- **RA1 carrier sensitivity (holdout, labelled, counts only):** 54 combinations over those 3 cases; wrong rows range **0 to 3** (limit 3), false-confident 0; **0 of 54 combinations fail RA1**; distribution (wrong: combinations) 0: 6, 1: 21, 2: 21, 3: 6. So RA1 holds under every carrier, **with no headroom in the worst case** (3 = the limit). Three gates passed before the sweep: two label-free (the default combination reproduces `preds_A.csv`; the moving rows and combination counts equal `carrier_swap.json`) and one aggregate (the default reproduces the recorded RA1 line). The sensitivity is a scratch script, not tested code: `reports/kaggle_consensus/holdout/ra1_carrier_sensitivity.py` (and `reports/kaggle_consensus/carrier_swap.py`).

## Label exposure on the holdout (found by the independent review)

Two published outputs let a reader work out the gold labels of 7 holdout rows, so these rows are **no longer blind**:

- **c0060__SEC-4.2, c0312__SEC-4.2-T2, c0678__SEC-4.2-T2:** the sensitivity's distribution of wrong rows (0: 6, 1: 21, 2: 21, 3: 6), together with the recorded default (1 wrong) and the per-row carrier verdict counts in the ledger, is consistent with exactly one gold assignment for these three rows (checked by enumeration from published aggregates only; no label file was read for the check).
- **c0062__DP-3.1, c0288__SEC-7.1, c0426__SEC-4.2, c0512__SEC-9.3:** RB1 reports 0 wrong-to-right and 4 right-to-wrong over exactly these changed rows, so A's verdict on each was the correct one.

Consequence: these 7 rows must not inform any later carrier, holder, certificate or adjudication rule, and a later check of such a rule on this holdout must exclude them or be read as non-blind. For any future analysis over a handful of rows, publish pass/fail and min/max only, never a distribution. This does not touch the candidate: the test side is carrier-invariant, RA1 holds under every carrier, and the 3 test labels remain unknown.

## Stage B

Adjudicator: MiniMax-M3 with the unchanged base prompt (D1: the model changes, nothing else). Resolution rule (D2): the adjudicator's strict majority of 3 samples must equal Flash's consensus (confirmed); it may break a Flash tie only toward a tied leader; otherwise the item is unresolved and its claim is dropped.

| | test | holdout |
|---|---|---|
| planned | 51 page-fields on 51 pages (trigger 1 only 40, trigger 2 only 3, both 8) | 69 on 67 pages (40 / 15 / 14) |
| confirmed / tie_broken / unresolved | 28 / 3 / 20 | 55 / 0 / 14 |
| unresolved by field | security_certifications 18, subprocessors 1, sso_saml 1 | security_certifications 11, data_regions 1, sso_saml 1, security_incidents 1 |
| rows changed by B vs A-only | 12 (c0065__OVERALL, c0065__SEC-4.2, c0105__SEC-4.2, c0198__OVERALL, c0198__SEC-4.2, c0211__SEC-4.2, c0241__SEC-4.2-T2, c0258__AI-3.4, c0414__AI-2.1, c0515__SEC-4.2-T2, c0655__OVERALL, c0655__SEC-7.1) | 4 (c0062__DP-3.1, c0288__SEC-7.1, c0426__SEC-4.2, c0512__SEC-9.3) |

Reading, stated plainly: B is **not decisive** (RB2) and **not calibrated** (RB1: 0 wrong to right, 4 right to wrong). Most unresolved items are certification fields, where Flash's own votes disagree and M3's majority differs from Flash's consensus; under D2 that drops the claim. The 12 test rows B would change were never put into the candidate. The RB1 power is low (4 changed rows), but the direction is one-sided and the pre-registered rule says fail; it was not reinterpreted. Errors that every sample repeats are invisible to any vote, and B did not look for them.

M3 measured (the plan said Flash's pace is not a basis): RB0 draw 99 pages, 0 failures, 221 s with 6 workers (2.2 s per page), output tokens 1.5x the Flash-calibrated estimate. The first test run (step 3) then hit **7 failures in 99 attempted calls (adj1 2 of 48, adj2 5 of 51) and a slower pace** under the default 30 s cap for the extract role (rate-limit count 0; the error kind was not classified, so "slow calls cut off at the cap" is the likely cause but unverified). The user approved raising `LLM_SMALL_TIMEOUT_S`; the retry and the holdout run used 60 (the effective cap is `min(LLM_TIMEOUT_S = 60, LLM_SMALL_TIMEOUT_S)`), both with 0 failures. The timeout is not part of the cache key; the prompt and model did not change. RB0 ran at the 30 s cap and is unaffected.

## Cost against the estimates

- Six Flash draws (Task 10): 3 test passes (about 2304 pages each) and 3 holdout passes (1496 each): about 18.8M tokens, **about $4.66 at the fallback price, about 3.5 h** (estimate 18.8M tokens, 3.5-7.9 h; low end). 8 connection errors in the first draw (0.35%) were re-run to 0 failures.
- Stage B (MiniMax-M3): about 460 calls (failed and cut-off ones included), about 0.65M tokens in, **about $0.40 at the fallback price** (estimates $0.11 for step 3 and $0.14 for step 4; the step 3 retry and step 4's longer outputs account for the difference). The plan's bound was "under 1M tokens".
- **`--max-usd` is not a real bound**: `assessor.tracing.price_for` returns the same fallback price for every model, so these dollar figures are the fallback price, not the provider's bill. The MiniMax meter was never reported tight (5-hour window 0% at the start, weekly 14%); I cannot read it myself.

## What stayed unverified or limited

- The candidate's effect on the test score is unknown (no test labels). A submission would be a measurement, and with only 4 rows different it is close to a direct read on those 4 rows, so it must feed no rule.
- The CLI glue in `eval/consensus.py` (`Ctx`, `check-r0`, `stage-a`, `plan-b`, `run-b`, `resolve`, `candidate`) has no unit test; its pure cores do. Its first real run was on this data, and R0 was re-run after the last change to the file.
- `run-b` counts failures but does not record their kind, which is why the 7 step-3 failures could not be classified. Not changed here (out of scope).
- The holdout is small: it can reject a harmful change but cannot confirm a small gain. D6: the new holdout draws used `--no-score`, so the holdout never judged prompt f45b4b77c99d.
- Docker, compose, ops and the CI docker-build remain unverified, as before.

## Final review and verification (2026-11-19)

**Review status: self-review by the implementer, one advisor pass, and then a fresh-context independent reviewer (Claude Opus, read-only, no access to the private data) at the user's request.** Verdict: **ready to merge with fixes**; it found no Critical issue and said the A-only candidate checks out (equals the stage-A predictions, differs from #1 in exactly the 4 reported rows, self-check OK, rules applied as pre-registered, nothing fitted to data). Its two Important findings are fixed below; its minors are deferred or turned into caveats in this note.

- **Fixed (1), test first:** a dry-run `resolve --allow-partial` overwrote `test/preds_AB.csv` with the claims of unread pages dropped, and `candidate` guarded only the stage-A `partial` flag, so an A+B candidate could have been built from it with a clean self-check. `resolve` now writes `<target>/stage_b.json` with a `partial` flag and `make_candidate` refuses A+B unless it says `partial: false` (missing = refused; A-only never reads it). 3 new tests; one older test (my own Task 8 test) was changed because it had encoded the unguarded behaviour. Unreachable on this data (A+B is not the candidate). The candidate file is **byte-identical** before and after.
- **Fixed (2), test first (independent reviewer, Important):** `resolve` checked adjudicator coverage against `b_plan.json` read from disk but resolved against a plan it recomputed from the samples, never comparing the two, and `plan-b` wrote the plan files even on a dry run. A missing, stale or partial plan file could therefore make `resolve` pass its coverage check while every page the adjudicator never read silently counted as unresolved (its claim dropped), with rules and an A+B `preds_AB.csv` built from that. Now: `plan-b` writes no plan file unless every sample covers 100% (`save_plans`); `resolve` computes coverage from the recomputed plan and refuses unless `b_plan.json` (and `rb0_plan.json`, test only, when it exists) equals the recomputed plan (`plan_file_problems`; counts only in its messages). 7 new tests (RED: the functions did not exist), 91 in the two Kaggle files. On the real cached data the plan files rewritten by `plan-b` are byte-identical to the ones the runs used, so this path was **not** hit in this experiment; `resolve` on test and holdout reproduces `rules.json`, both `preds_AB.csv` and the candidate exactly; a scratch copy of the reports without `b_plan.json` is refused (exit 2, no rule touched).
- **Fixed (3), documentation (independent reviewer, Important):** the 7 label-exposed holdout rows (section above).
- **Caveats added from the review's minors:** how to read RB0, RB1, the holdout "fragile" count and `rules.json` provenance (section "How to read these results").
- **Deferred minors (not fixed):** (1) `run-b` counts failures but does not record their kind (it cost a stop in step 3); (2) the `--estimate` text says "at the measured pace" for M3 runs, but that pace is Flash's (M3 measured 2.2 s per page in the RB0 draw and 3-7 s later); (3) `system or EXTRACT_SYSTEM` treats an empty string as "the current prompt"; (4) the CI tests job checks out with default depth, so the three prompt-recovery tests in `tests/test_kaggle_runner.py` (they need commit e42203d) always skip there; adding `fetch-depth: 0` to `.github/workflows/ci.yml` would run them; (5) the withheld-message exception handler in `main()` and the CLI print paths have no canary test (only `format_stage_a` and `format_stage_b` do); (6) the sensitivity's 54 is a reduced count (the 3 moving cases crossed), not all 219 per-case combinations crossed; min, max and pass/fail are unaffected.
- **Verification run, all on the final code (2026-11-19):** `python -m pytest -q` default backend **500 passed**; `EMBEDDER=builtin RERANKER=builtin python -m pytest -q` **500 passed** (409 earlier tests + 91 new, none skipped away); `ruff check .` clean; `PROMPT_VERSION` prints `f45b4b77c99d`; R0 re-run on dev, holdout and test after the last code change: 0 differing rows; `git status`: no commits made; `eval/consensus.py` SHA-256 unchanged since the hashes above were taken.
- **`eval/kaggle_policy.py` diff, exactly:** the imports `subprocess`, `sys`, `types`, `uuid`, `Path`; `ROOT`; the new `prompt_text_from_git`; and `cache_key` gaining the optional `system` argument. No hunk lies inside `EXTRACT_SYSTEM` (lines 91-140). `eval/run_kaggle.py` gained `Extractor(system=)`, the estimate helpers, `select_prompt`, `write_submission` and four flags (`--prompt-commit`, `--expect-prompt`, `--estimate`, `--no-score`); the test-split output is byte-identical to submission #1 (R0, and a cache-only regression run).
- **Experiment files untouched:** `eval/claim_check.py`, `eval/nli_scorer.py`, `eval/laya_scorer.py` are still untracked and were last modified 2026-11-18 between 19:59 and 20:08, before the spec was written (21:16).

## Reproduce (free, cache-only)

```
python eval/consensus.py check-r0 --target dev|holdout|test
python eval/consensus.py stage-a --target test|holdout
python eval/consensus.py plan-b --target test --rb0 | holdout
python eval/consensus.py resolve --target test | holdout
python eval/consensus.py candidate
```

Reports: `reports/kaggle_consensus/` (`rules.json`, `{test,holdout}/stage_a.md|stage_b.md|carrier_swap.json`, `holdout/ra1_carrier_sensitivity.json`, `candidate/submission.csv`, `logs/`). Every ruling and finding is in `.superpowers/sdd/2026-11-18-kaggle-consensus-escalation/progress.md` (git-excluded).

## Next, not done

- The user decides whether to submit the candidate (it would be submission #2) and, in the web UI only, which two submissions are final.
- The user asked to discuss a local-LLM decision maker now that B failed; nothing has been built.
