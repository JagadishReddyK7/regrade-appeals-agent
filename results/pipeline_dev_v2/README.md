# Second claim-pipeline run on dev, 24 cases (2026-11-12)

The pipeline after the fixes from `results/pipeline_dev_v1/`, with agent v0 rerun on the
same code (B1 was not rerun: nothing it uses changed). One Kaggle session
(`notebooks/pilot_dev.ipynb`), commit `95b4fb6` (`commit.txt`, branch
`feat/claim-pipeline`), vLLM 0.32.0, Tesla T4, Qwen2.5-7B-Instruct-AWQ, temperature 0,
seed 0, prompted 7B verifier (A4b). Every `metrics.json` was recomputed locally from
`predictions.jsonl` and matches. The vLLM log shows 11 hermes tool-parser errors (a reply
whose tool-call JSON did not parse); they appear as invalid tool calls in `events.jsonl`.

**The pipeline was changed again after this run (see below).** Same caveats as
`results/mid_dev_24/`: 6 situations x 4 tones, one assignment, LLM-written appeals,
prompts designed on this set, no comparison is statistically significant.

## Results

| System | Exact acc. | Wrongful change | Wrongful refusal | Esc. precision | Esc. recall | Over-esc. | Tone sens. | Pushback flip | Missed correction | No decision | Case errors | Wall s | Prompt tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 direct (v1 run) | 0.62 | 0.20 | 0.31 | 1.00 | 1.00 | 0.00 | 0.67 | 0.22 | 1.00 | 0 | 0 | 6.5 | 2,383 |
| Agent v0 (whole appeal) | 0.67 | 0.22 | 0.14 | 0.67 | 1.00 | 0.10 | 0.67 | 0.36 | 1.00 | 0 | 0 | 28.9 | 16,793 |
| Pipeline (full system) | 0.46 | 0.21 | 0.43 | 1.00 | 0.75 | 0.00 | 0.83 | 0.00 | 0.80 | 2 | 0 | 20.8 | 14,523 |
| Pipeline, A0 (no claim splitting) | 0.50 | 0.17 | 0.44 | - | 0.00 | 0.00 | 1.00 | 0.29 | 0.33 | 0 | 0 | 23.8 | 18,094 |

Against v1 the pipeline went from 0.33 to 0.46, over-escalation from 0.35 to 0.00 and
pushback flips from 0.38 to 0.00: the tier and follow-up fixes worked. It still trails
agent v0, mostly through wrongful refusals (0.43).

## What went wrong in the pipeline (from `events.jsonl`)

1. **Targets on items with no deducted marks** (situation 2, all four cases). The
   extractor linked "recursion was allowed" to `test_reverse_words_basic` (0 marks
   deducted); the verifier said supported and the aggregator correctly restored 3 marks on
   `rubric:approach`, which broke the target rule. Two cases ended without a decision.
2. **A claim was lost in extraction**: the emotional late-penalty appeal (dev-003) gave no
   claims ("the late penalty should be waived" read as feelings), so no escalation; dev-006
   lost its test claim.
3. **The submission was not read** in situation 5: the agent searched the docs but never
   opened the code, so nothing showed that `count_vowels` has a docstring; three of four
   style claims were refuted.
4. **The agent tested its own code**: in dev-015 it wrote its own `count_vowels` for
   `run_code`, which passed, and the verifier called the claim supported (+4, wrong).
5. **The prompted verifier was weak**: 13 of 23 claims with a gold verdict right, wrong in
   both directions (faulty test refuted in dev-005; "my code is readable" supported in
   dev-005 and dev-007; failing uppercase test supported in dev-015 and dev-016; the
   19 September partial-credit announcement refuted in dev-017 and dev-020 follow-ups).

## Changes made after this run (`docs/DECISIONS.md` section 12)

The extractor's targets are restricted to items with deducted marks, and a penalty waiver
counts as a claim; the pipeline loads the submission as E2 next to the grade report; the
evidence prompt says to test the student's functions through `run_code`, never a rewrite;
the verifier prompt has judging rules for test, rubric and arithmetic claims (shared with
verifier training).
