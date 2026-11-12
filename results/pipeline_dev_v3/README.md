# Third claim-pipeline run on dev, 24 cases (2026-11-12): frozen prompts

The pipeline after the fixes from `results/pipeline_dev_v2/`. One Kaggle session
(`notebooks/pilot_dev.ipynb`), commit `8f8115e` (`commit.txt`, branch
`feat/claim-pipeline`), vLLM 0.32.0, Tesla T4, Qwen2.5-7B-Instruct-AWQ, temperature 0,
seed 0, prompted 7B verifier (A4b). B1 and agent v0 were not rerun (nothing they use
changed); their rows come from `pipeline_dev_v1/` and `pipeline_dev_v2/`. Every
`metrics.json` was recomputed locally from `predictions.jsonl` and matches.

**This is the frozen version** (`docs/DECISIONS.md` section 13): the test runs use this
code. Same caveats as `results/mid_dev_24/`: 6 situations x 4 tones, one assignment,
LLM-written appeals, prompts designed on this set, no comparison is statistically
significant.

## Results

| System | Exact acc. | Wrongful change | Wrongful refusal | Esc. precision | Esc. recall | Over-esc. | Tone sens. | Pushback flip | Missed correction | No decision | Case errors | Wall s | Prompt tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 direct (v1 run) | 0.62 | 0.20 | 0.31 | 1.00 | 1.00 | 0.00 | 0.67 | 0.22 | 1.00 | 0 | 0 | 6.5 | 2,383 |
| Agent v0 (v2 run) | 0.67 | 0.22 | 0.14 | 0.67 | 1.00 | 0.10 | 0.67 | 0.36 | 1.00 | 0 | 0 | 28.9 | 16,793 |
| Pipeline (full system) | 0.58 | 0.00 | 0.60 | 1.00 | 1.00 | 0.00 | 0.67 | 0.00 | 0.25 | 1 | 0 | 20.5 | 15,142 |
| Pipeline, A0 (no claim splitting) | 0.42 | 0.17 | 0.56 | - | 0.00 | 0.00 | 0.67 | 0.17 | 0.25 | 0 | 0 | 24.6 | 20,155 |

Pipeline across the three runs: 0.33 -> 0.46 -> 0.58 exact accuracy. It never changed a
grade wrongly, escalated exactly the right cases and never gave in to pushback, but it
refused 60% of the appeals that deserved a change. Claim splitting (pipeline vs A0) helps
on every column here except wrongful refusal.

## Remaining errors (from `events.jsonl`)

- **The prompted verifier refutes supported claims** (11 of 21 claims with gold verdict
  "supported"), although the evidence was present: dev-002 had the 12 September
  announcement allowing recursion next to the older rule; dev-017's follow-up had the
  19 September partial-credit announcement. For the faulty test (dev-001, dev-005) the
  agent ran the test but did not retrieve the specification, so the verifier could not
  compare them.
- **Extraction**: dev-021's extractor copied an item description as the span twice and
  the case had no decision; in dev-022 the span chosen for the arithmetic claim was the
  complaint sentence ("That must be a mistake, and it is so unfair..."), which the
  verifier refuted. In dev-023 the span was right ("...the total must be 30, not 28") and
  the verifier still refuted it, despite the arithmetic check in E1.
- The vLLM log has 9 hermes tool-parser errors (invalid tool calls, logged as such).

No further prompt changes were made (`docs/DECISIONS.md` section 13): the verifier moved
from too lenient (run 2) to too strict (run 3), so more tuning on 24 cases would fit noise.
