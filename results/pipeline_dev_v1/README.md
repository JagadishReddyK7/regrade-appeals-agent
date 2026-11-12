# First claim-pipeline run on dev, 24 cases (2026-11-12)

The first run of the full system (`system: pipeline`) on a real model, with B1 and agent v0
rerun on the same code for comparison. One Kaggle session (`notebooks/pilot_dev.ipynb`),
commit `5cb0f03` (`commit.txt`, branch `feat/claim-pipeline`), vLLM 0.32.0, Tesla T4,
Qwen2.5-7B-Instruct-AWQ, temperature 0, seed 0. The verifier is the prompted 7B on the same
server (ablation A4b). Every run folder holds `config.yaml`, `metrics.json`,
`predictions.jsonl` and `events.jsonl`; every `metrics.json` was recomputed locally from
`predictions.jsonl` and matches.

**This run found design problems; the pipeline was changed afterwards (see below).** It is
kept as the "before" state for the report. The same caveats as `results/mid_dev_24/` apply:
6 situations x 4 tones, one assignment, LLM-written appeals, prompts designed on this set,
no comparison is statistically significant.

## Results

| System | Exact acc. | Wrongful change | Wrongful refusal | Esc. precision | Esc. recall | Over-esc. | Tone sens. | Pushback flip | Missed correction | No decision | Case errors | Wall s | Prompt tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 direct (no tools) | 0.62 | 0.20 | 0.31 | 1.00 | 1.00 | 0.00 | 0.67 | 0.22 | 1.00 | 0 | 0 | 6.5 | 2,383 |
| Agent v0 (whole appeal) | 0.62 | 0.28 | 0.14 | 0.67 | 1.00 | 0.10 | 0.67 | 0.36 | 1.00 | 0 | 0 | 27.6 | 14,434 |
| Pipeline (full system) | 0.33 | 0.25 | 0.50 | 0.30 | 0.75 | 0.35 | 1.00 | 0.38 | 1.00 | 2 | 1 | 35.8 | 14,571 |
| Pipeline, A0 (no claim splitting) | 0.33 | 0.24 | 0.54 | - | 0.00 | 0.00 | 0.83 | 0.33 | 0.67 | 3 | 0 | 21.5 | 15,320 |

Agent v0 improved from 0.42 (`results/mid_dev_24/`) to 0.62, mainly through the arithmetic
check in `get_grade_report`: it now gets all four situation-6 cases right (+2).

## What went wrong in the pipeline (from `events.jsonl`)

1. **The extractor tagged checkable claims as T3**, and a T3 claim escalated at once:
   "please check the addition" (dev-021), claims citing a senior (dev-019, dev-023), the
   recursion claim (dev-008). Over-escalation 0.35.
2. **Pressure-only follow-ups became new claims**, often tagged T3: "I still think I
   deserve the style marks back too" turned a correct +5 into an escalation (dev-001,
   dev-006). Re-verifying the earlier claims with more evidence also flipped a verdict
   (dev-006, C1 refuted -> supported).
3. **Wrong targets**: the recursion claim was linked to a test with 0 marks deducted, so
   every restoration exceeded the cap and the case had no decision (dev-002).
4. **The aggregator saw only the evidence the verifier cited**, so it could not apply the
   partial-credit announcement A3 (dev-017: +2 instead of +1).
5. **A0's aggregator invented item ids** (`approach`, `style`): 10 invalid answers.
6. **Search ignored dates**: "the 19 September announcement" never found A3 (dev-017,
   five searches).
7. One case (dev-012) timed out: a reply without a length limit ran past 120 s.

## Changes made after this run (`docs/DECISIONS.md` section 12)

Stricter tier and target rules in the extractor prompt; the follow-up extractor sees the
earlier claims and extracts only claims with new evidence; T3 claims skip the verifier only
when they dispute a policy or no item; turn 2 verifies only new claims, and a supported new
claim replaces earlier verdicts on the same item; the aggregator sees all evidence and its
item ids are restricted by the JSON schema; the verifier's evidence ids are restricted to the
bundle; announcement dates are indexed; `llm.max_tokens: 1024` in the pipeline configs.
