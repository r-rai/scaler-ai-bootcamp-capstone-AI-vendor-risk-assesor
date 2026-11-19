# Kaggle score: consensus extraction, then escalation of the unsure rows

Status: DRAFT for the user's approval, 2026-11-18. Not committed (the user commits only when asked).
Competition: Vendor Verdict (`vendor-verdict-can-we-trust-this-ai-tool`), deadline 2026-11-30 18:29 UTC.

## 1. Intent and why

This is the capstone of the Scaler AI bootcamp. About 30 of 500+ candidates advance, chosen on the Kaggle score, a presentation and an
interview (the weights are not stated and keep changing). The user wants the best honest Kaggle score, and wants the experiments themselves
to be presentable: what was measured, what was tried, what failed, and why the result can be trusted.

Where we are (all numbers measured, none guessed):

- Submission #1 scored 0.99658, 12th. The board shows 26 scored teams; ten are at 1.00000, one at 0.99896. Ties appear ordered by earlier
  submission, so even 1.00000 would rank 11th (inferred from the order shown, not confirmed in the rules). The gap to 1.0 is roughly 5 to 11
  wrong rows of 2378 (an OVERALL row costs 0.00067, a requirement row 0.0003 to 0.0006).
- Extraction noise is the same size as the gap. Two samples of the same prompt flip 10 to 14 test verdicts. Four cached test samples
  (two per prompt) agree on 2356 of 2378 rows; 22 rows in 17 cases are not unanimous (20 have a clear >=3-of-4 majority, 2 are 2-2 ties),
  and #1 disagrees with a >=3-of-4 majority on 5 rows. Temperature is already 0, so the noise is the provider's.
- Dev has 0 wrong rows and 0 flips, but it is saturated (the prompt was fixed against dev misses), so it says nothing about test. The clean
  holdout had 3 wrong rows of 1580 with prompt 9566ad4b6b95 (the prompt behind #1).
- Most of the disagreement sits in `security_certifications` (39 of 470 pages differ between two samples) and `dpa_available` (10 of 316).

## 2. Goals, non-goals, anti-goals

Goals
- Fewer wrong rows on test, by removing run-to-run extraction noise (stage A) and by re-reading only the pages the model was unsure about
  with a stronger model (stage B).
- Every claim in the write-up backed by a pre-registered rule and a measurement, negative results included.

Non-goals
- No prompt changes, no new verifiers, no fine-tuning, no change to `rules_lite.py`, the matcher or the evidence policy.
- No work on the live pipeline (that is the later production-hardening phase).
- A perfect 1.00000 is not promised. Errors that every sample repeats are invisible without labels and may stay.

Anti-goals (a failure even if the score goes up)
- Choosing a candidate because of its leaderboard score, or probing the leaderboard to learn labels.
- Rewording any prompt after seeing dev/train text; hand-labelling or hand-adjudicating any test row (by a person or by Claude).
- Printing test claims, quotes or page text anywhere (counts and case ids only).
- Silently mixing prompts or models in one vote.

## 3. Constraints

- The hard rules in the user's memory: no tuning to data, pre-register before running, report scores exactly as measured, every
  submission and every final selection needs the user's explicit yes, unverified things are reported as unverified.
- MiniMax plan: 1.5 billion tokens a month (22.92M used), weekly quota 9% used, 5-hour window 0% used (what the windows count is not
  known). One full test pass is about 4.4M tokens and about 1.7 h at 6 workers, so time is the binding limit. Runs go one at a time
  with the meter checked between runs; the runner already waits on HTTP 429 and carries `--max-usd`.
- Extraction must NOT move to the local Qwen model (earlier decision).
- Competition data stays in the gitignored `data_kaggle/`. Sending pages to a provider other than MiniMax needs the user's explicit OK.
- 5 submissions a day; at most two finals, selected in the web UI only.

## 4. Approaches

| | What | Trade-off |
|---|---|---|
| A | Consensus vote over several independent samples of one prompt, then the same deterministic rules | Nearly free, fixes noise only; consistent errors stay |
| B | Re-read only the unsure pages with a stronger model, several samples, adjudicate | Cheap, can fix some consistent errors on hard pages; needs a calibration story |
| C | Keep #1 | Zero risk, zero gain |

Rejected: fine-tuning a verifier (breaks the no-fitting rule), more prompt edits against dev (dev is saturated), the claim-check
verifiers again (already failed their pre-registered rules), probing the leaderboard.

Chosen by the user: A, then B on the rows A cannot settle. It matches the course's own pattern (cheap method first, escalate only the hard
minority to the expensive model; a second model as a check, as with a judge from a different provider).

## 5. Design

### 5.1 Unit and signature

The unit of voting is a **page-field**: one field (for example `security_certifications`) as extracted from one readable page. Its
signature is the existing order-free `kaggle_policy.claim_signature` after `normalise_page_claims`. A sample that says nothing about the
field has the empty signature and **counts as a vote for "absent"** (no answer from silence). A page that failed to extract in a sample
gives no vote for any field.

Voting at page-field level (not verdict level) means the consensus claims go through the unchanged `assemble_case`, so every requirement
row and the OVERALL row stay consistent with each other.

### 5.2 Stage A: consensus

- Samples: prompt `9566ad4b6b95` (the prompt behind submission #1, the only prompt with a leaderboard score), model MiniMax-M3.1-Flash.
  Existing on test: the unsalted sample (which is #1 itself) and `noise1`. New test samples with salts `p9s3`, `p9s4`, `p9s5`, so k = 5 on
  test. Holdout: the existing unsalted sample plus two new salts, k = 3. The runner must be able to run this older prompt: the builder adds
  a way to load the prompt text from commit e42203d, byte-identical (selecting a prompt is not editing it, and the cache key contains the
  text, so the existing entries are found again).
- Vote: a signature wins with a strict majority of the valid votes (3 of 5, 2 of 3). The claims carried forward are those of the first
  sample, in fixed salt order, whose signature equals the winner.
- A page-field is **settled** if at least 4 of 5 votes agree (test) or 3 of 3 (holdout). Everything else is **unsettled**.
- With fewer than 3 valid votes, or no strict majority, the page-field is unsettled and the first sample's claims are carried forward
  as a placeholder for the A-only candidate.

### 5.3 Stage B: escalate the unsettled

Triggers (a page-field goes to B if either holds):
1. It is unsettled by 5.2.
2. All available samples of the second-opinion prompt `f45b4b77c99d` (test: `v7` and `v7n1`; holdout: one new sample) agree with each
   other and their signature differs from the consensus. These are the places where the wording of the prompt matters (the single
   samples of the two prompts differ on 7 test rows). Their claims come from the cached samples.

Adjudicator: MiniMax-M3 (the main role in `.env.example`), the **same unchanged extraction prompt**, 3 samples (salts `adj1..adj3`), run
only on the pages behind triggered page-fields. A different provider is possible only if the user supplies a key and agrees to the data
going there; it is not assumed.

Resolution per triggered page-field:
- Adjudicator signature = strict majority of its 3 samples AND equals the Flash plurality/consensus: use it.
- Adjudicator majority differs from the Flash consensus: **unresolved**. The claim is dropped for that page, so the evidence policy's "not
  stated means unknown" applies. The count of unresolved items is reported; it should be small.
- Adjudicator has no majority: unresolved as well.

### 5.4 Assembly and outputs

New module `eval/consensus.py` (name is the builder's call) reads cached samples, builds consensus records in the same shape
`run_kaggle.assemble_case` already takes, and writes per-case verdicts, `disagreements.csv`-style counts and a `submission.csv` only for
the pre-registered candidate. The submission passes the existing self-check (`kp.validate_submission` and `metric.py`) before it is
reported as written. Output contains counts and case ids only.

## 6. Pre-registered rules

These are written before any new sample is drawn, and the module docstring carries the same text. Labels are used only to report and to
test these rules; nothing is fitted to them. The numeric thresholds are my picks (flagged in section 9).

- **R0 plumbing.** With k = 1 the consensus path reproduces the single-sample pipeline verdict for verdict on dev, holdout and test
  (unit test on synthetic claims plus a cache-only regression run on existing data).
- **RA1 no harm on the clean holdout.** On the three holdout samples of prompt 9566ad4b6b95: wrong rows of the k = 3 consensus must be
  <= the mean wrong rows of the three single samples (rounded up), and false-confident rows of the consensus must be <= the mean of the
  three singles (rounded up). If this fails, A is not adopted: keep #1 and report the failure.
- **RA2 sanity on test.** The k = 5 consensus changes at most 30 verdict rows relative to the unsalted 9566 sample (submission #1). More means something
  other than noise is going on: investigate, do not submit.
- **RB0 adjudicator agrees on easy items.** A fixed-seed random draw of 100 settled test page-fields (label-blind): the adjudicator's
  signature equals the unanimous Flash signature on at least 95 of them. If not, the adjudicator reads the task differently and is not used.
- **RB1 calibration on the holdout.** Run the adjudicator on the holdout page-fields that trigger B (both triggers; the holdout gets one sample of the second-opinion prompt for trigger 2). Among rows B changes versus the A-only
  consensus, "wrong to right" must exceed "right to wrong" against gold. If B changes no holdout row, B is reported as **untested** and is
  not adopted unless the user decides otherwise with that stated. Expect little power: the holdout had 3 wrong rows of 1580.
- **RB2 decisiveness.** Report how many page-fields end unresolved; if more than 10 on test, B is reported as not decisive.

Candidate rule, fixed in advance (so no candidate is picked by its score): if R0, RA1, RA2 pass, the candidate is A-only; if RB0, RB1 and
RB2 also pass, the candidate is A+B; otherwise the candidate is none and #1 stays. At most two new submissions come out of this design,
each decided before any new score is seen, each needing the user's explicit yes. Final selection stays the user's call in the web UI.

## 7. Cost and time (estimates from the last full run)

| Step | Tokens | Wall time |
|---|---|---|
| 3 new test samples (3 x 2304 page reads) | about 13M | about 5 h |
| Holdout: 2 new samples of 9566 and 1 of f45b (3 x about 1540 pages) | about 9M | about 3.5 h |
| B: sanity draw of 100 pages, about 60 unsettled test pages x 3, holdout unsettled x 3 | under 1M | under 0.5 h |
| Total | about 23M, about 1.5% of the month | about 9 h in background runs |

Dev needs no new samples: it is only a cache-only regression check.

## 8. Order of work

1. Write the pre-registration (docstring) and `eval/consensus.py` test-first, with synthetic claims only; R0 regression on cached data. Add the option to run a prompt from a given commit (text unchanged).
2. Cache-only dry run on the existing samples: counts only, no spend.
3. Draw the 3 test samples, one run at a time; compute A on test (settled/unsettled counts, rows changed, leave-one-out fragile rows).
4. Draw the 3 holdout samples; evaluate RA1.
5. Stage B: RB0, then the adjudication runs; RB1, RB2.
6. Apply the candidate rule, write the candidate file, self-check it, show the counts, and ask the user before any submission.
7. Write the results note for the presentation, negative results included.

## 9. Calls I made (please check)

1. Vote at page-field level, not verdict level, so OVERALL stays consistent.
2. "Absent" is a vote.
3. Build the consensus on #1's prompt `9566ad4b6b95` and use `f45b4b77c99d` only as a second opinion (trigger 2). Reason: 9566 is the only
   prompt with a leaderboard score, and it was fixed from dev misses before the holdout was run, so the holdout is clean for it. The f45b
   commit (75142a7) records a label-free basis (test noise patterns and requirements.json), but its two changed fields feed AI-2.1 and DP-2.2,
   which is also where 2 of the 3 holdout misses fall, and the per-requirement holdout table existed before that commit. I cannot show the
   holdout did not influence it, so the holdout is never used to judge f45b. RA1 compares the consensus with single samples of the same
   prompt, so it is unaffected. The 7 rows where the two prompts differ are not decided by a coin flip: they go to the adjudicator.
   The fallback is still to keep #1 if RA1 fails.
4. k = 5 on test, k = 3 on holdout; settled means >= 4 of 5 (3 of 3).
5. Adjudicator = MiniMax-M3 with the same prompt, same provider; no second provider unless the user provides one.
6. Thresholds: 30 changed rows (RA2), 95 of 100 (RB0), 10 unresolved (RB2).
7. The unresolved rule drops the claim (leans to UNKNOWN) because the metric charges a confident wrong answer about twice a cautious one.
8. No commits and no pushes until asked. The three untracked experiment files (`eval/claim_check.py`, `nli_scorer.py`, `laya_scorer.py`)
   are untouched; whether to commit them as a negative-result record is the user's decision.

## 10. Left to the builder

Module and CLI layout, output directory names, how salts are named in the CLI, test structure, the exact shape of the leave-one-out
report, and how the run reports are parsed for the old-prompt samples.
