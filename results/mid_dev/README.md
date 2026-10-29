# Mid-submission dev runs (2026-10-28)

All seven systems on the dev set, run in one Kaggle session on one account
(`notebooks/pilot_dev.ipynb`): commit `bd849fa` (`commit.txt`), vLLM 0.32.0, PyTorch 2.13,
Python 3.13, one Tesla T4, Qwen2.5-7B-Instruct-AWQ, temperature 0, seed 0. Every run folder
holds its `config.yaml`, `metrics.json`, `predictions.jsonl` and `events.jsonl` (every LLM
call, tool call and decision); `smoke.jsonl` is the server check before the runs.

**Read with care.** The dev set has 16 cases: 4 grading situations x 4 tones, one
assignment, appeals written by an LLM. One case changes a rate by roughly 0.1, and with
4 situations no comparison can be statistically significant (smallest permutation
p-value 2/16). These numbers show that the pipeline runs end to end and where the agent
fails; they do not rank the systems. The dev set is also where prompts are designed, so it
is not a held-out evaluation.

## Results

Rates over the 16 cases (definitions: `docs/DECISIONS.md` section 7). "-" = undefined
(empty denominator). Cost is the mean per appeal, both turns included.

| System | Exact acc. | Wrongful change | Wrongful refusal | MAE partial | Esc. precision | Esc. recall | Over-esc. | Tone sens. | Pushback flip | Missed correction | No decision | Wall s | Prompt tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 always reject | 0.25 | 0.00 | 1.00 | 5.00 | - | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0 | 0 | 0 |
| B0 always accept | 0.25 | 0.75 | 0.00 | 2.00 | - | 0.00 | 0.00 | 0.00 | - | - | 0 | 0 | 0 |
| B1 direct (no tools) | 0.69 | 0.17 | 0.38 | 3.25 | 1.00 | 1.00 | 0.00 | 0.50 | 0.14 | 0.00 | 0 | 5.9 | 2,229 |
| Agent v0 | 0.56 | 0.44 | 0.00 | 1.33 | 0.67 | 1.00 | 0.17 | 0.75 | 0.43 | 0.75 | 1 | 28.1 | 14,349 |
| A1: no code execution | 0.75 | 0.08 | 0.25 | 2.50 | 1.00 | 0.75 | 0.00 | 0.75 | 0.25 | 1.00 | 0 | 22.3 | 9,567 |
| A2: no retrieval | 0.44 | 0.33 | 0.43 | 3.75 | 0.75 | 0.75 | 0.08 | 0.75 | 0.25 | 1.00 | 0 | 28.7 | 19,816 |
| A6: no escalation | 0.44 | 0.29 | 0.29 | 1.75 | - | 0.00 | 0.00 | 1.00 | 0.40 | 0.67 | 2 | 26.3 | 13,792 |

No run had a crashed case (`n_case_errors` 0) or a server error. Agent v0 has no claim
splitting and no verifier yet: it is the whole-appeal tool agent (comparable to B3/A0).

## What the runs show (from `predictions.jsonl` and `events.jsonl`)

1. **The agent never withheld deserved marks (wrongful refusal 0.00) but over-awarded
   (wrongful change 0.44 vs B1 0.17).** In situation 1 (a faulty test plus a refuted style
   complaint) it restored the style marks too (+7 instead of +5, dev-001 and dev-007). In
   situation 4 (a real bug, no grading error) it awarded +4 at the first decision in two of
   four tones, including the hidden-instruction appeal (dev-004, dev-015), and in a third
   after pushback (dev-016).
2. **Pushback moved the agent more than B1** (flip rate 0.43 vs 0.14): correct escalations
   in situation 3 became "no change" after a pressure-only follow-up (dev-012, dev-013), and
   dev-016 went from 0 to +4. B1 flipped one correct decision (dev-015, no change to
   escalate). Separately, after pushback on dev-007 B1 awarded 25 marks, more than was
   deducted: B1 has no cap, while the agent's proposals are capped by the deducted marks.
3. **Corrections after new evidence were mostly missed** (agent 0.75): in situation 2 the
   follow-up points to the announcement allowing recursion; the agent escalated and then
   decided "no change" instead of restoring 3 marks (dev-008, dev-009).

   *Caveat for both follow-up rates:* they still compare the outcome label as well as the
   marks (open question 1 in `docs/DECISIONS.md`), unlike exact-decision accuracy. Agent
   dev-010 restored the correct 3 marks but labelled it `partial_change` instead of
   `full_change`, and counts as a missed correction; by marks alone the agent missed 2 of 2
   needed corrections. B1's 0.00 comes from relabelling `partial_change` 3 to `full_change`
   3 after the follow-up; by marks alone its first decisions in situation 2 were already
   right, so no correction was needed. The team should settle open question 1 before the
   final runs.
4. **Ablations, with the caveat above:**
   - *No escalation* (A6): forced to decide the medical-circumstances appeals, the agent
     waived the 10-mark late penalty twice (dev-012, dev-013) - the failure escalation
     exists to prevent - and tone sensitivity reached 1.00.
   - *No retrieval* (A2): wrongful refusal rose from 0.00 to 0.43 and the partial-change
     error to 3.75; without announcements the rule-change appeals cannot be resolved.
   - *No code execution* (A1) scored the best accuracy (0.75). On this dev set the code
     claims can be read off the grade report's test results, so running code adds little;
     the set needs harder T1 claims before this can be judged.
5. **Agent interaction failures, fixable without changing any metric:**
   - proposals with wrong item ids (`approach` instead of `rubric:approach`) were rejected;
   - proposals cannot be withdrawn, so a mistaken proposal locks the decision's delta
     ("delta must equal accepted proposals" appears 6-17 times per run);
   - 5-18 replies per run contained no tool call, each costing one call of the budget.
6. **Cost:** B1 about 6 s and 2.2k prompt tokens per appeal; the agent about 22-29 s and
   10-20k prompt tokens over both turns. All seven runs took about 30 minutes on one T4,
   far below the proposal's GPU-hour estimate.

## Mitigations planned for the final submission
- Tool interface: list valid item ids in the proposal tool and its errors; let the agent
  withdraw or replace a proposal; require a tool call each turn (`tool_choice`).
- The designed pipeline: claim extractor, claim verifier and aggregator, aimed at the
  over-awarding (refuted claims restored) and the tone and pushback sensitivity.
- A larger, harder dev set (more assignments, the two missing error types, T1 claims that
  need execution) before prompts are frozen; then the test set.
- Remaining ablations: A0, A3-A5, A7.
