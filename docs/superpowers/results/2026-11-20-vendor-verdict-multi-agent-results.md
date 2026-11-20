# Vendor Verdict multi-agent system (Plan B): results

Built and run 2026-11-19 to 2026-11-20 on branch `prod-hardening` (local commits only, nothing pushed, **nothing submitted to Kaggle**).
Spec: `docs/superpowers/specs/2026-11-19-vendor-verdict-multi-agent-design.md`. Plan: `docs/superpowers/plans/2026-11-19-vendor-verdict-multi-agent.md`.
Code: package `vendor_verdict/` (the pre-registration is the docstring of `vendor_verdict/gates.py`); tests `tests/test_vv_*.py`.
This note holds counts and case ids only: no claim, quote or page text, no label of a named row. Corrected after the final whole-branch review (see "Rulings and process").

## Outcome

- **The multi-agent system (B) passed the build but failed the pre-registered accuracy gates, so it is not a submission candidate.** Nothing was submitted or uploaded. **Submission #2 (ref 58543825, 0.99881) is unaffected and stays the best file we have.**
- On the holdout (1573 rows, the 7 label-exposed rows excluded) the best B configuration has **79 wrong rows against 3 for the single-extractor baseline** (R-hold needs B <= baseline). On the test split B changes **95 rows** against submission #2 (R-test allows 30).
- What did work: the system runs end to end on both engines with identical output; every successful call is cached, so a cache-only replay of a run that had no failed call reproduces it byte for byte (shown for dev, for the LangGraph engine and for test); and the whole trace is presentable. Limits of those claims: failed calls are never cached, so `train_holdout` (3 planner failures, whose calls the later router-off run filled) can no longer be replayed exactly as it ran; and "a quota stop resumes exactly" is proven in tests only (no real quota stop occurred). Those are engineering results; the accuracy result is negative and is reported as measured.
- The cause of the accuracy gap is **not diagnosed** (see "Where B loses"). Any change to B now would be tuning to a holdout that has already been used as a gate; it would need a new pre-registered experiment and a fresh holdout, and we have none (dev is contaminated, see the spec).

## What was built

A per-case graph of distinct agents, run by one node list on two engines (plain Python and a LangGraph `StateGraph`): Source Validator (existing evidence policy), Planner (MiniMax-M3, page metadata only), Page Router (local Qwen3-4B), Plan Reviewer (deterministic union, recall first), four Field Specialists (MiniMax-M3.1-Flash: certifications, data, AI, access and incidents), Evidence Grader (grounding check then a local judge, flag only), Merger (the existing `build_facts`, through an adapter), Decision Maker (local, chooses among MiniMax readings or abstains), Evaluator (iteration 2 re-read of flagged fields), Verdict (`rules_lite`, unchanged) and Auditor (handover trace per case, Markdown and JSON, Mermaid drawing). The runner adds `--estimate`, `--cache-only`, `--ablate`, `--wait-quota` (a quota-window stop that resumes exactly), a shared local-server limiter, a health check, `--prewarm-local`, `selfcheck` (real `rules_lite`, no model call) and a `gates` subcommand. Tests: 1237 pass on the default and on the builtin backend, ruff clean. Local models may only flag or choose among MiniMax readings; they never delete a claim. The model names above come from the runs' console header (hosted models) and the Task 0 spike (the local model; the console names it by its server alias); run.json does not record them (see "What stayed unverified").

How it was built: subagent-driven, one fresh implementer per task, then a spec-and-quality reviewer, fix rounds until clean (Tasks 1, 2, 3, 5, 8A, 8C, GATE-P and 9 each needed one fix round), one local commit per task, then a final whole-branch review and one fix wave (see "Rulings and process").

## Pre-registered rules and results

The rules were committed before any holdout or test run (`7671846` at 00:36 and `0dac482` at 00:49 IST; prompt versions frozen at `eb3c380`). Run once, no threshold moved.
Timeline (IST, 2026-11-20): dev smoke finished 00:22; gates and pre-registration committed 00:36 and 00:49; runner rule for the router-off candidate file 00:56; prompt versions frozen 01:02:31; holdout full run about 01:03 to 02:35; holdout router-off run about 02:36 to 03:14; test run 1 about 03:16 to 06:24 and its resume to 06:26 (start times are approximate: run.json records an end time and a wall time, not a start time). What the files show: every holdout and test run.json has `prompt_versions` equal to the frozen ones in `gates.FROZEN_PROMPT_VERSIONS` (all 11 run directories checked), `vendor_verdict/prompts.py` is unchanged from `35d59bd` to `eb3c380`, and the freeze commit `eb3c380` changed only `vendor_verdict/gates.py` and `tests/test_vv_gates.py` (no runtime module). No prompt, rule or threshold changed after the freeze commit.

| Rule | Measured on | Result | Numbers |
|---|---|---|---|
| R-smoke | dev, 20 cases | **pass** | 20 of 20 cases, 0 failures, 149 of 149 readable pages read |
| R-route (full B, router on) | holdout, label-free | **fail** | 1624 of 1650 baseline page-fields routed (98.4%); 26 missed: trains_on_customer_data 21, dpa_available 2, data_regions 2, data_retention 1 |
| R-route (router off) | holdout and test | pass (by construction) | 1650 of 1650 and 2553 of 2553 |
| R-DM | holdout, 7 exposed rows excluded | **pass** | wrong to right 16, right to wrong 9 (router-off pair) |
| R-hold | holdout, 1573 rows | **fail** | B (router off, DM on): wrong 79, false-confident 5; baseline s0 wrong 3, false-confident 1 |
| R-test | test, label-free | **fail** | 95 rows changed versus submission #2 (limit 30) |
| Replay identical | test | pass | cache-only replay: `preds.csv` and `submission.csv` byte-identical |
| Candidate | | **none** | not submitted |

**An amendment made after seeing dev data, disclosed.** The dev smoke measured R-route on its 20 cases: 158 of 163 (all 5 misses on one field, `trains_on_customer_data`). R-route needs 100% on holdout and test, so full B would almost surely end as "none". We did not touch any routing prompt or add any field-specific rule (that would be fitting to those 5 misses). Instead, before any holdout or test run, we pre-registered a label-free fallback (ruling R53): if the holdout R-route fails, the candidate configuration turns the Page Router off (every readable page goes to every specialist), R-DM is then judged in that configuration, and R-hold and R-test are judged on the final configuration. The holdout R-route did fail (above), so the candidate configuration was "router off". The dev count that motivated the branch is a coverage count with no label and no page text.

## Ablations (holdout, 1573 rows, counts only)

"Model calls" and "tokens" are run.json `spend`, which counts the hosted (MiniMax) calls AND the local model's calls; the local share is run.json `local.calls`. The wrong-row columns are label-based counts computed by the controller with `gates.wrong_counts` (not re-verified by the final reviewer, who does not read labels).

| configuration | wrong rows | false-confident | rows differing from s0 | model calls (hosted + local) | of which local calls | tokens in / out, hosted + local (M) | wall (min) | run |
|---|---|---|---|---|---|---|---|---|
| baseline s0 (single extractor, prompt 9566ad4b6b95) | 3 | 1 | 0 | cached sample | | | | |
| B full (router on, DM on, grader on, iteration 2 on) | 100 | 3 | 102 | 8135 | 2991 | 6.06 / 1.04 | 93 | live |
| B minus Page Router | 79 | 5 | 81 | 2255 | 238 | 2.39 / 0.24 | 39 | live |
| B minus Page Router minus Decision Maker | 86 | 5 | 88 | 0 | 0 | 0 / 0 | 0 | cache-only replay |
| B minus Decision Maker | 98 | 3 | 100 | 15 | 6 | 0.02 / 0.00 | 1 | live |
| B minus Evidence Grader | 101 | 3 | 103 | 130 | 97 | 0.11 / 0.01 | 4 | live |
| B minus iteration 2 | 95 | 3 | 97 | 45 | 45 | 0.03 / 0.00 | 2 | live |

How to read it. Every configuration is far worse than the baseline. Removing the router, the grader, the Decision Maker or iteration 2 alone does not close the gap; the loss is in the specialists' reading and/or the merge of the per-family readings (hypothesis 1 below). The planner and the merge (the adapter and `build_facts` path) were not ablated, so this table cannot clear them. Removing iteration 2 improves B (100 to 95), so the re-read loop did not help. The Evidence Grader changes almost nothing (101 against 100). The Decision Maker helps with the router off (86 to 79) and hurts slightly with it on (98 to 100). Turning the router off removes 21 wrong rows and raises the iteration-1 specialist calls from 4245 to 5717 (1.35 times; 4947 against 6422 including iteration 2; counted from the traces, calls issued per node whether answered from the cache or paid; the estimate's worst case was 4 per page, the measured ratio is 1.35). Model calls for the ablations are small because they reuse the cached readings of the full run. The configuration with the fewest wrong rows (79) is the one the pre-registered rules selected: the router half of that selection (R53) used only the label-free R-route count; the Decision Maker half used R-DM's gold-based counts (wrong to right 16, right to wrong 9), as pre-registered.

Caveats on the ablations. (1) Spec section 9 planned them as cache-only; B minus Decision Maker, B minus Evidence Grader and B minus iteration 2 were LIVE runs and drew a few new samples: 9, 33 and 0 new hosted calls, plus 6, 97 and 45 new local calls. (2) Full B had 3 planner failures (c0109, c0471, c0665); the router-off run later filled those planner calls, so every router-on ablation ran with planner routes that full B did not have (the final reviewer found this changed at most 1 row, only for iteration 2; not re-derived here). (3) There is no noise estimate (no salted B run), so deltas of 1 to 5 rows are not distinguishable from noise.

## Where B loses (explanation only; nothing here fed a rule)

The counts in this section are label-based, **controller-computed, not re-verified by the reviewer** (a scratch script over the 1573 holdout rows, B = router off, DM on). By requirement on the holdout, B's wrong rows (cells under 5 shown as "<5"):

| requirement | what it checks | B wrong |
|---|---|---|
| OVERALL | the case verdict | 25 |
| DP-3.1 | customer data stored in India (region_includes) | 23 |
| SEC-4.2 | SOC 2 Type II or ISO 27001, documented (certification_valid) | 15 |
| SEC-4.2-T2 | same, Type I or questionnaire acceptable | 6 |
| AI-3.4 | sub-processors disclosed, model providers identified | 5 |
| others | | each <5 |

The baseline has 3 wrong rows in total over the 1573 rows (too few to spread over requirements). What was counted: B's prediction differs from the baseline's on 81 of the 1573 rows, and B has 79 wrong rows in total. A bound that needs no further label look: a row where B is wrong and equals the baseline is a row where the baseline is wrong too, and the baseline has only 3, so at least 76 of B's 79 wrong rows are among the 81 differing rows. By verdict pair (gold, then B's prediction; controller-computed): PASS predicted FAIL 22, PASS predicted NEEDS_REVIEW 21, and the overall verdict READY_FOR_APPROVAL to BLOCKED 10, CONDITIONAL to BLOCKED 8, READY_FOR_APPROVAL to CONDITIONAL 6. B is too pessimistic on the data-region, certificate and sub-processor families.

Hypotheses, **unverified** (diagnosing them needs page text, which the project rules do not allow us to print, and acting on them would be tuning to the holdout): (1) four independent specialists per page create extra or per-item split readings that the baseline's single extractor does not, and the merge rules then turn them into FAIL or NEEDS_REVIEW; (2) B's prompts were written only from published material (the `rules_lite` docstring, `requirements.json`, the evidence-policy docstring), while the baseline prompt was tuned on dev misses (commit 2228f69), so B starts without that tuning; (3) the certificate readings lose the issuer and date detail that the "documented" check needs.

**The Evidence Grader carried almost no signal.** Counted from the traces (both iterations, 200 holdout cases each): in the router-on full run it graded 4042 readings: unsupported 3421 (84.6%), supported 613, unsure 8; in the router-off run (the final configuration) it graded 4135: unsupported 3499 (84.6%), supported 631, unsure 5. The grounding check asks for the quote to appear verbatim in the page, and most model quotes do not (the earlier claim-check experiments found the same). So iteration 2 ran in 200 of 200 cases in both runs and was not selective. The grounding rule was deliberately not changed (ruling R43: no rule changes after the prompts were written).

## Mechanics and reproducibility

| check | result |
|---|---|
| dev smoke, 20 cases, live | 20 of 20, 0 failures, 149 of 149 pages read; 864 model calls (490 hosted + 374 local) |
| holdout full B, live | 200 of 200, 1496 of 1496 pages read, 0 page failures, 3 planner failures (c0109, c0471, c0665; the recall-first fallback applied), no quota stop; 8135 model calls (5144 hosted + 2991 local) |
| holdout router off, live | 200 of 200, 0 failures; 2255 model calls (2017 hosted + 238 local) |
| test router off, live | run 1: 300 of 300 cases but partial: 24 failures in total (21 page reads, 1 planner call, 2 iteration-2 case reads; transient); 12124 model calls of which 2016 local. The same command re-run once filled them (44 model calls, 31 hosted + 13 local, 90 s), then complete, 0 failures. Run 1's run.json was overwritten by the resume (same `--out`), so its counts come from its console log (failures, local calls) and the ledger (model calls, tokens, planner failure) |
| cache-only replays | dev, holdout (DM-off pair) and test: not partial, `not_cached` 0; dev and test `preds.csv` byte-identical to the live run (see the fingerprints below) |
| LangGraph engine | cache-only replay of the dev run: `preds.csv` byte-identical to the plain engine (never run live) |
| `selfcheck` | real `rules_lite`, 3 dev cases, 27 rows equal to the baseline path, 0 model or network calls (console output only, no artefact) |
| quota stop and resume | proven by tests (resume equivalence on both engines); **no real quota stop occurred**, so the live `--wait-quota` path and MiniMax's real quota error wording are unverified |

Cost and time are in `reports/vendor_verdict/presentation/cost_time_table.md` (gitignored; the user's plan is metered by time window and tokens, and the usd column uses one fallback price). The test pass in the final configuration took 3.1 h (run 1) plus 1.5 min (resume) and about 10.6M input tokens (run 1 10.55M plus the resume 0.04M); that figure includes the local model's tokens (run 1's local share is not recorded, its run.json having been overwritten). The local model (about 44 calls per minute) was not a bottleneck in the test pass: in test run 1 (router off) its summed queue wait was 105.6 s against a wall time of 11274 s. It was a co-bottleneck in the router-on holdout run: summed queue wait 1392 s against a wall time of 5563 s.

## Evidence fingerprints

`reports/` is gitignored, so the run outputs are not committed; their sha256 are recorded here (paths relative to `reports/vendor_verdict/`). `test_all_ablate-router` holds the resumed, final state of the test pass (run 1's files were overwritten and have no fingerprint). The matching hashes are the reproducibility evidence: the dev live run, its cache-only replay and the LangGraph replay share one `preds.csv` hash (`7b9c7a83...`), and the test live run and its replay share one `preds.csv` and `submission.csv` hash (`71db53ab...`).

| file | sha256 |
|---|---|
| `train_dev_limit-20/run.json` | `083aa2a93b096195fb628909f9c92e16a949dad5a216c08dcc04dfae1c16e0e4` |
| `train_dev_limit-20/preds.csv` | `7b9c7a836d74696e26b7b6623594aff388cdf85ef75b153d9756de1ea72a03ea` |
| `dev_replay_check/run.json` | `dd3705c21376dc5fd1321d0cafa2d1e8b495a413810e1c6ad3f6b6c1cd6cc401` |
| `dev_replay_check/preds.csv` | `7b9c7a836d74696e26b7b6623594aff388cdf85ef75b153d9756de1ea72a03ea` |
| `dev_replay_langgraph/run.json` | `af0d6f0a04ca08c3e519f5826d6bcdf2b77cd08db21652398f4c317f24508b01` |
| `dev_replay_langgraph/preds.csv` | `7b9c7a836d74696e26b7b6623594aff388cdf85ef75b153d9756de1ea72a03ea` |
| `train_holdout/run.json` | `b88a78008d8b0989b2f018611f43517b8861dfe9eec261de57a9e8005d4112cc` |
| `train_holdout/preds.csv` | `0017772190e3dfc02db40ad1b287eea075a6684d209f6a39f34fc380d00771d3` |
| `train_holdout_ablate-router/run.json` | `b356fb17d5a17146a99a1099e6d1fb6fd2ede40026cce0ca1ee4fa55d932fe86` |
| `train_holdout_ablate-router/preds.csv` | `d3f2497ce91922f646c0b2bf58949028b7b928818941b15cf33f8c2feccc2023` |
| `train_holdout_ablate-decision_maker+router/run.json` | `eb349778153f62a13ec3231a6a6161da13d7b66c3d70244deedc1b44e998542a` |
| `train_holdout_ablate-decision_maker+router/preds.csv` | `41234055493edf8943d5a371030661e792466517568a61a7a80cf3573edc78ea` |
| `train_holdout_ablate-decision_maker/run.json` | `3c5661e715781396c367af3755fcdaed638bb4ac07043beeedaeceba830c3ab1` |
| `train_holdout_ablate-decision_maker/preds.csv` | `2ca769d1145bbdf66275deac9f3d9aede8540dc9313987cb831de4782dff9162` |
| `train_holdout_ablate-grader/run.json` | `42032c9aea0d4c7cc4d7ac518f2109b633448d20fb6f58aa727924e463ffd989` |
| `train_holdout_ablate-grader/preds.csv` | `33d22f658a37fcb1d73e29de408a3928497392670f30794ef3ccdad7a3504159` |
| `train_holdout_ablate-iteration2/run.json` | `e86533bdf361b1881a0498125589a3bcf2ca0c422f0c57b04f40fd8131e5de9b` |
| `train_holdout_ablate-iteration2/preds.csv` | `e0fd280ff64bf4311f9bfe296e50b96c97c2f3325b1a878049d500fd5ce36bed` |
| `test_all_ablate-router/run.json` | `709601270ace24843f1266cf45c310d49dacb4e81778249bccfaab3564999fbf` |
| `test_all_ablate-router/preds.csv` | `71db53abfae519baac26ddb8dfdb3c6d2f83fbe0eb7447c38ca331d9a4a7c8b8` |
| `test_all_ablate-router/submission.csv` | `71db53abfae519baac26ddb8dfdb3c6d2f83fbe0eb7447c38ca331d9a4a7c8b8` |
| `test_replay_check/run.json` | `0139fca0a48186b34a2c88d91df6ae83b8a460d18c4b3a9f3ccaf33a70943dd7` |
| `test_replay_check/preds.csv` | `71db53abfae519baac26ddb8dfdb3c6d2f83fbe0eb7447c38ca331d9a4a7c8b8` |
| `test_replay_check/submission.csv` | `71db53abfae519baac26ddb8dfdb3c6d2f83fbe0eb7447c38ca331d9a4a7c8b8` |
| `gates.json` (3 history entries) | `9353353f45f9bcc8bfd4c5a2891fbaa7c873720ba9f32e4080207a8ad87f9e62` |

## What stayed unverified or not measured

- The cause of the accuracy gap (above). No page text was read for diagnosis.
- The live quota path: no quota window was exhausted, so `--wait-quota` and the keyword list for MiniMax's quota error were never exercised against the real service.
- The local-down or degraded path: 0 local fallbacks (router, grader, Decision Maker) in every run.
- The Decision Maker's invalid-answer path: `decision_invalid` 0 in every run.
- The LangGraph engine ran only as a cache-only replay of the dev run, never live.
- `--prewarm-local` was never run; the budget stop (`--max-usd`) never fired live.
- No salted B run, so no noise estimate for B (see the ablation caveats).
- Per-node tokens in the traces: not recorded (ruling R56; parallel cases share the global per-role counters). The trace Markdown prints seconds only; per-role token totals are in run.json `roles` (estimates for cached calls).
- Model names and the code commit are not recorded in run.json (ruling R57, deferred), and the gates do not check them; the model names in this note come from the console header only.
- The `selfcheck` "27 rows" result exists only on the console (no artefact).
- How often the adapter's silence guards (`trains_on_customer_data == "no"` and `security_incidents == []` read from the model's own `display`) fired, by reason: the traces do hold the aggregate merge-note count (138 in the router-on full run, 150 in the router-off run), but not the breakdown by reason. Ruling R50 forbids acting on the count anyway.
- Whether a stronger or different reader model would close the gap: not tried (the user keeps MiniMax-M3.1-Flash as the reader; M3 was judged weaker in the earlier experiment).
- Docker, compose and ops files remain unverified (unrelated to this work).
- The final selection of the two Kaggle finals is the user's call in the web UI and is unchanged by this work.

## Known errata in frozen or pre-registered text (not edited on purpose)

Editing either text would change a frozen prompt version or the pre-registration, and stale every gates section, so both stay as they are and are disclosed here:

- `DECISION_SYSTEM` (`vendor_verdict/prompts.py`, frozen) does not mention that a joined quote is clipped to 300 characters.
- The `vendor_verdict/gates.py` docstring (the pre-registration, "Fresh inputs", ruling R55a) says the hash of a missing run file is recorded as "absent"; the code records null.

## Deferred findings (later-version hardening, not fixed)

Non-blocking items from the task reviews and the final review: gates fingerprints omit the submission #2 file and its path; exit code 2 for a refused gates.json equals argparse's usage code; the `inputs: []` item and the relative replay path in gates.json noted by the final reviewer; planner failures, local fallbacks and `down_allowed` do not make a run partial and `check_run` accepts them; run.json records neither model names nor the code commit (R57); `quota.first_error` keeps up to 200 characters of provider text in run.json while its docstring says counts only; the estimate's usd prices local tokens; `parse_ablate` duplicates the `cli.refusal` check; a few test-hygiene items (a private-constant test in `tests/test_vv_quota.py`, one price-dependent runner test, an untested Ctrl-C branch in prewarm, `test_wait_mode_is_interruptible` can pass vacuously); two `from .nodes import` lines in `cli.py`; file sizes. The full list is in the ledger.

## Rulings and process

The decision log (rulings R1 to R59) is in the git-excluded ledger `.superpowers/sdd/2026-11-19-vendor-verdict-multi-agent/progress.md`. The final whole-branch review (24c5141..eb3c380) found 0 Critical, 6 Important and 7 Minor items; the controller ruled: R56, per-node trace tokens are not implemented and the deviation is disclosed (the trace Markdown no longer prints zero tokens); R57, model names and the code commit in run.json are deferred and disclosed; R58, the selfcheck tests no longer need git history (a depth-1 CI clone has none; the CI workflow is unchanged); R59, this note is corrected and the evidence fingerprints above are recorded. One fix wave made those changes (plus a deterministic version of the two race-dependent stop tests) in one local commit; gates were not re-run and no model was called.

## What this means for the presentation

The honest story: a pre-registered, reproducible multi-agent pipeline with per-agent traces, a Mermaid graph, two engines, quota-aware resumption (proven in tests) and an ablation table, evaluated against a hard baseline with rules fixed in advance; it **lost to the tuned single extractor**. Removing the router, the grader, the Decision Maker or iteration 2 alone does not close the gap; the loss is in the specialists' reading and/or the merge of their per-family readings (the planner and the merge were not ablated), and the process refused to submit it. That is a real result about where agent decomposition helps and where it does not.

## Addendum: field-by-field comparison (label-free, explanation only)

Added 2026-11-20 after the runs; **nothing here fed a prompt, a rule or a gate** (the holdout is spent as a gate). Counts only; no value, quote, claim or page text was read or printed (the host's `rules_lite.py` and `requirements.json` were read for which attributes the rules use). The label-free sections read no label; the last paragraph (labelled cross-tab) was run afterwards with the user's explicit yes, counts only, cells under 5 hidden, no case id next to a label-based figure, the 7 exposed rows excluded. Controller-computed, not re-verified by an independent reviewer. Full tables: `reports/vendor_verdict/presentation/field_comparison.md` (gitignored; with `field_comparison_ids.json`, the case ids, and the scratch scripts in `field_comparison_scripts/`).

**Method.** B = the configuration the gates picked (router off, Decision Maker on), regenerated cache-only (0 model calls, sockets blocked, 0 cache misses, 0 failure flags). Baseline = the cached extraction (prompt 9566ad4b6b95, unsalted) through `run_kaggle.assemble_case`, as `gates.s0_holdout_preds` does. Verification before any table: B's `preds.csv` byte-identical to the run; the same recomputation with the Decision Maker abstaining byte-identical to the DM-off run; the baseline replica equal to `s0_holdout_preds`; B differs from s0 on exactly 81 of 1573 rows (56 requirement rows + 25 OVERALL rows). The stages are recomputed from the cached readings: S1 = iteration 1 only (never run as a configuration), S2 = iteration 2 on and the DM abstaining, S3 = final. Equality is rules-relevant and order-insensitive. Each differing (case, field) pair is explained by the smallest swap of (page, field) readings for the baseline's that makes B's fact equal the adapter's output on the baseline's own readings (swaps are not additive).

| field | pairs that differ (of 200) | verdict rows they move | note |
|---|---|---|---|
| data_regions | 30 | 23 (DP-3.1) | only the `available` list; `planned` identical; B both adds and misses regions in 28 |
| security_certifications | 25 | 21 (SEC-4.2 15, SEC-4.2-T2 6) | names, status, issuer presence and `evidence_level` identical in every matched item; B sets `valid_until` where the baseline has none (23 items) and a different `date` (4) |
| data_retention | 6 | 3 | |
| subprocessors | 5 | 5 (AI-3.4) | |
| trains_on_customer_data, dpa_available, sso_saml, security_incidents | 2, 1, 1, 1 | 2, 1, 0, 1 | |
| total | 71 of 1600 (4.4%), in 62 of 200 cases | 56 requirement rows (+ 25 OVERALL = the 81 differing rows) | |

**Ranking of mechanisms** (pair counts, each pair once; a pair labelled "the baseline read it, B did not" is not the same count as the facts the baseline has and B lacks, 4): (1) the specialists read different content from a page both read, 55 (77%: certificates 25, data_regions 28, retention 1, sso 1); (2) extra readings from pages where the baseline had no claim for the field, 10 (subprocessors 5, regions 2, retention 2, dpa 1); (3) a page-level reading the baseline has and B lacks, 5; (4) the adapter's silence guard, 1; (5) per-item list splitting, **0** (B splits 209 of 679 list groups, the baseline 4 of 664, no fact differs because of it; the baseline path drops 0 list claims as invalid, so there is nothing for splitting to rescue); unexplained 0.

**Stage, two questions.** *When a difference first appears:* already at iteration 1 and still there at the end in 61 of 71 pairs; created by the iteration-2 override in 4; iteration 2 had fixed it and the Decision Maker broke it again in 6 (all certificates); created by a Decision Maker choice alone in 0. So 10 of the 71 are created or re-created by the later stages, while the net effect shrinks the gap: 91 differing pairs after iteration 1, 86 after iteration 2, 71 after the Decision Maker (it moved 21 facts toward the baseline, 6 away, 3 sideways). *Who stood behind the reading that stands:* in 54 of the 55 content-differs pairs the baseline's reading was a conflict candidate that the Decision Maker saw and passed over (the 55th stands on the default after it abstained); over all 134 conflicts it chose the baseline's candidate 15 times, an iteration-1 candidate 42, an iteration-2 candidate 74, and abstained 3. In 40 of the 56 differing page-field groups B's own iteration-1 and iteration-2 readings agree with each other and only the baseline differs; in the other 16 (12 certificates) they disagree. 69 pairs sit on official pages, 1 on a registry page (registry pages agree), 1 is the adapter-only pair. The holder filter is not a cause (registry drops by holder: 43 for B, 43 for the adapter on the baseline's readings, 43 for the baseline path). The derived-region rule does not create the region gap (switching it off raises it from 30 to 80 pairs). Optional router comparison (197 cases that replay as they ran; the run's 3 planner-failure cases are excluded, and the replay had 2 decision-maker cache misses): 92 differing pairs with the router on, 70 with it off; 19 of the 22 extra are in `trains_on_customer_data`.

**What this does to the hypotheses above.** (1) "per-item split readings create the gap": not supported (0 pairs). (3) "the certificate readings lose the issuer and date detail": not supported as worded; issuer presence and `evidence_level` are identical in all matched items, and the difference is B filling `valid_until` where the baseline leaves it empty. (2) "B's prompts were not tuned on dev misses": not tested here (this comparison cannot separate prompt wording from the model's sampling). The per-requirement moved-row counts equal the label-based B-wrong counts listed in "Where B loses" (DP-3.1 23, SEC-4.2 15, SEC-4.2-T2 6, AI-3.4 5), consistent with the 76-of-79 bound; the label-free sections do not say which side is right; the labelled cross-tab below does.

**Still unexplained.** Why the certificate specialist fills `valid_until` and why its region list differs: that needs page text and the prompts side by side, which the project rules do not allow here. Prompt effect and sampling noise are not separated (no salted B run; the baseline's own sampling noise was measured earlier at 14 of 2378 test verdicts, far below 81 of 1573).

**Labelled cross-tab** (run afterwards with the user's explicit yes; 1573 rows, the 7 exposed rows excluded; gate: the label totals reproduce this note's B 79 wrong / 5 false-confident and s0 3 / 1; cells under 5 shown as `<5`; explanation only, nothing here feeds a rule; tables in `field_comparison_labels.md`). Of the 81 rows where B and s0 differ, **s0 is right and B wrong on 78**, B right and s0 wrong on <5, both wrong on <5. Requirement rows (56): s0 right on 54; by field s0 is right on all 23 data_regions rows, all 21 certificate rows, all 5 sub-processor rows and 5 of the 7 rows of the other fields pooled; OVERALL rows 24 of 25. All 44 rows of the content-differs pairs, all 44 rows where the Decision Maker kept B's reading over the baseline candidate, and all 5 rows of the pairs that iteration 2 had brought back to the baseline's fact and the Decision Maker broke again, are rows where s0 is right. A cell says who is right on the ROW (the verdict), not why B reads differently, and the table covers only rows where the Decision Maker did not end on the baseline's candidate (it chose that candidate in 15 of 134 conflicts, which this table does not evaluate).

Fingerprints (sha256, first 16 hex; the report is the assembled file): `field_comparison.md` 6c9c48e5d07959ac, `field_comparison_labels.md` 5411c8548639136d, `field_comparison_ids.json` 9c3afab32f33b6f0, `fieldcmp.py` 8c06c905b0bc09c6, `fieldcmp_router.py` a6de6ffd21dbd6fe, `fieldcmp_labels.py` 7a4cf27d6b82be32, `assemble.py` 2036365491e3ecd4.
