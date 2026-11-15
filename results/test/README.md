# Test runs: every system on the frozen test set (2026-11-15)

One Kaggle session (`notebooks/test_runs.ipynb`, T4 x2), commit `7b43458` (`commit.txt`,
`main` after PR #13), vLLM 0.32.0, torch 2.13.0, transformers 5.19.0 (`pip_freeze.txt`),
temperature 0, seed 0, one run per system. Code and prompts are the frozen version
(`docs/DECISIONS.md` sections 13 and 15); each test config is its dev config with only
name, cases and data root changed (`tests/test_test_configs.py`).

- **Test set:** 80 cases (20 situations x 4 tones, assignments T1 and T2), checked against
  its frozen manifest before the runs: 232 files listed, 0 missing, 0 changed
  (`test_set_check.json`, manifest sha256 `663e6b31...`).
- **All 11 runs finished with 0 crashed cases** (`run_summary.json`). The fine-tuned
  verifier was merged in the session from the saved adapter of the dev run
  (`verifier_merged_here: true`; same adapter as `results/verifier_ft_dev/`).
- **Only aggregates are kept here** (per run `config.yaml` and `metrics.json`, the
  statistics). Per-case logs contain test appeals and are not in the repo
  (`docs/DECISIONS.md` section 15); any error analysis on them is done by hand by a team
  member.

## Results

Rates as defined in `eval/metrics.py`. Wrongful change: share of decided cases given more
marks than deserved. Wrongful refusal: share of decided cases that deserved marks and got
fewer. Tone sensitivity: share of the 20 situations whose decision differs across the four
tones. Pushback flips: correct first decisions turned wrong by a pressure-only follow-up.
Missed correction: follow-ups with new evidence after which the system did not revise
when it had to. No decision (invalid output or out of tool calls) counts as wrong for
accuracy.

| System | Accuracy | Wrongful change | Wrongful refusal | Esc. precision | Esc. recall | Tone sens. | Pushback flips | Missed correction | No decision | s / appeal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 reject | 0.300 | 0.000 | 1.000 | - | 0.00 | 0.00 | 0.000 | 1.00 | 0 | 0 |
| B0 accept | 0.300 | 0.700 | 0.000 | - | 0.00 | 0.05 | 0.000 | 1.00 | 0 | 0 |
| B1 direct | 0.287 | 0.327 | 0.353 | 0.67 | 0.25 | 0.65 | 0.267 | 1.00 | 28 | 7 |
| B3 agent v0 (whole appeal) | 0.388 | 0.246 | 0.378 | 0.21 | 0.50 | 0.70 | 0.143 | 0.58 | 4 | 33 |
| **Full system (fine-tuned 3B verifier)** | **0.450** | **0.172** | **0.532** | **0.62** | **1.00** | **0.45** | **0.000** | **0.81** | **3** | **23** |
| A0 no claim splitting | 0.487 | 0.105 | 0.604 | 1.00 | 0.50 | 0.50 | 0.030 | 0.83 | 0 | 30 |
| A1 no code execution | 0.588 | 0.156 | 0.319 | 0.62 | 1.00 | 0.40 | 0.000 | 0.91 | 3 | 21 |
| A2 no retrieval | 0.438 | 0.141 | 0.596 | 0.62 | 1.00 | 0.50 | 0.000 | 0.84 | 3 | 24 |
| A6 no escalation | 0.412 | 0.208 | 0.532 | - | 0.00 | 0.55 | 0.000 | 0.75 | 3 | 24 |
| A4b prompted 7B verifier | 0.463 | 0.047 | 0.681 | 0.62 | 1.00 | 0.70 | 0.000 | 0.69 | 3 | 23 |
| A4a untuned 3B verifier | 0.200 | 0.368 | 0.286 | 0.14 | 1.00 | 0.60 | 0.000 | 0.83 | 3 | 22 |

Gold escalations: 8 cases (2 situations x 4 tones). The full system escalated 13 cases:
all 8 gold ones and 5 others (over-escalation rate 0.07).

Accuracy by planted error type, full system: arithmetic error 0.62, faulty test 0.15,
missed partial credit 0.25, none 0.62, rubric misapplied 0.62; by claim tier: T1 0.50,
T2 0.27, T3 1.00 (`metrics.json`, `breakdowns`).

## Statistics

Paired permutation tests by situation (20 situations), 10,000 resamples, bootstrap 95%
CIs, Holm correction within each family (`docs/DECISIONS.md` section 15). Difference =
full system minus the other system.

**Family 1, RQ1a (pre-registered, confirmatory).**

| vs | Metric | Full system [95% CI] | Other [95% CI] | Difference | p | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- |
| B0 reject | wrongful change | 0.172 [0.04, 0.33] | 0.000 | +0.172 | 0.002 | **0.012** |
| B0 reject | wrongful refusal | 0.532 [0.33, 0.75] | 1.000 | -0.468 | 0.007 | **0.036** |
| B0 accept | wrongful change | 0.172 | 0.700 [0.50, 0.90] | -0.528 | <0.001 | **0.002** |
| B0 accept | wrongful refusal | 0.532 | 0.000 | +0.532 | 0.001 | **0.008** |
| B1 | wrongful change | 0.172 | 0.327 [0.14, 0.56] | -0.155 | 0.033 | 0.134 |
| B1 | wrongful refusal | 0.532 | 0.353 [0.15, 0.56] | +0.179 | 0.224 | 0.673 |
| B3 | wrongful change | 0.172 | 0.246 [0.13, 0.38] | -0.074 | 0.434 | 0.673 |
| B3 | wrongful refusal | 0.532 | 0.378 [0.23, 0.53] | +0.154 | 0.237 | 0.673 |

**Family 2, RQ1b (design vs training).**

| vs | Tone sensitivity: other, difference, p (Holm) | Pushback flips: other, difference, p (Holm) |
| --- | --- | --- |
| A0 no claim splitting (design) | 0.50, -0.05, 1.00 | 0.030, -0.030, 1.00 |
| A4b prompted 7B verifier (training) | 0.70, -0.25, 0.76 | 0.000, 0.000, 1.00 |
| A4a untuned 3B verifier (training) | 0.60, -0.15, 1.00 | 0.000, 0.000, 1.00 |

**Family 3, exploratory.** No difference survives Holm correction except against B0
accept. The smallest uncorrected p-values: accuracy vs A4a +0.250 (p 0.006, Holm 0.15),
wrongful change vs A4a -0.197 (p 0.010, Holm 0.26), wrongful refusal vs A1 +0.213
(p 0.016, Holm 0.37), accuracy vs A1 -0.138 (p 0.026, Holm 0.59). Full numbers:
`stats_exploratory_accuracy.json`.

## What the results say

- **RQ1a is not supported.** The full system makes the fewest wrongful changes of the
  LLM baselines (0.17 vs 0.33 for B1 and 0.25 for B3), but it withholds deserved marks
  more often (0.53 vs 0.35 and 0.38). None of the differences against B1 or B3 is
  significant after correction; the only significant ones are against the trivial B0
  baselines. The direction is a trade-off: fewer wrongful changes, more wrongful refusals.
- **Claim splitting brings no measurable benefit here.** A0 is as accurate (0.49 vs 0.45)
  and makes fewer wrongful changes (0.11 vs 0.17); tone sensitivity and pushback flips
  are about the same.
- **Code execution hurt on this test set.** Without it the system is more accurate (0.59
  vs 0.45) and refuses less often (0.32 vs 0.53), mostly on faulty-test situations
  (accuracy 0.35 vs 0.15). The difference is not significant after correction (Holm
  0.59). A plausible cause, not yet checked, is that the failing test result is taken as
  evidence against the claim even when the test itself is wrong; checking it needs the
  per-case logs.
- **The fine-tuned verifier matters most against the untuned 3B**, not against the
  prompted 7B. A4a escalates 58 of 80 cases and reaches 0.20 accuracy; the fine-tuned
  3B reaches 0.45. The prompted 7B (A4b) is as accurate (0.46): it changes grades more
  rarely (wrongful change 0.05) and refuses more often (0.68). The large gain on dev
  claims (macro-F1 0.70 vs 0.41, `results/verifier_ft_dev/`) does not carry over to the
  system's decisions on test.
- **RQ1b, pushback:** every claim-pipeline variant has a pushback flip rate of 0.00 or
  0.03, against 0.14 for B3 and 0.27 for B1. This comes from the follow-up design, which
  revises only on new evidence, so it holds with every verifier. But the system also
  misses most genuine new evidence (missed correction 0.81, B3 0.58): it rarely revises
  at all, so its pushback robustness is partly a failure to revise.
- **RQ1b, tone:** decisions differ across tones in 45% of situations for the full system,
  against 50% (A0), 60% (A4a) and 70% (A4b, B3). This points to training rather than
  design as the source of tone robustness, but no difference is significant. Pressure did
  not raise wrongful changes: authority 0.13, the other tones 0.19.
- **RQ2:** within a situation the four tones are equivalent appeals, so tone sensitivity
  is the consistency measure: 9 of 20 situations are decided inconsistently. Precedent
  retrieval was not tested (the store was empty in every run).
- **Escalation works:** recall 1.00 and precision 0.62 for the full system. Without
  escalation (A6) accuracy drops only slightly (0.41).
- **B1 failed to give a valid decision in 28 of 80 cases** (0 on dev). Its accuracy
  (0.29) is mainly that format failure, so the B1 comparison says little about B1's
  judgement.

## Caveats

- **Small test set:** 20 situations, the unit of the paired tests, so the CIs are wide
  (full-system accuracy 0.28 to 0.62) and the tests have little power. "Not significant"
  here means "no evidence either way", not "no effect".
- **One run per system:** vLLM at temperature 0 is not exactly reproducible (random
  tool-call ids enter the prompt), and there was no time for repeated runs.
- **Dev results did not carry over:** the full system scored 0.67 on dev and 0.45 on
  test; B3 scored 0.67 on dev (`results/pipeline_dev_v2/`) and 0.39 on test. Prompts were
  tuned on 6 dev situations of one assignment; the test assignments are new.
- **The test set:** all 80 cases were written with an LLM's help, there are no blind
  cases. The second labelling was done by a second team member with an LLM's help, blind
  to the gold answers, from the information a TA sees (appeal, spec, rubric,
  announcements, student code, grade report), running the code (outcome kappa 1.00,
  claim-verdict agreement 1.00, tier kappa 0.86, 20 situations). The LLM help is from the
  same model family the author used, so errors shared by both labellings cannot be ruled
  out and the agreement may be optimistic. The only tier disagreement is on the
  arithmetic-error `total` claims, which follow the dev convention (T1). The full test-set review checklist (tone
  facts, follow-ups, grade reports against the autograder) was not run.
- **Not built or not run:** B2, B4, A3, A5, A7, the LLM judge, safety metrics, precedent
  runs.
