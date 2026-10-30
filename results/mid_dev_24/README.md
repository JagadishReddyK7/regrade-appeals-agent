# Mid-submission dev runs, 24 cases (2026-10-29)

All seven systems on the full dev set, run in one Kaggle session
(`notebooks/pilot_dev.ipynb`): commit `07a4681` (`commit.txt`; the merge of PR #7, same
files as `5e1d591`), vLLM 0.32.0, PyTorch 2.13, Python 3.13, Tesla T4,
Qwen2.5-7B-Instruct-AWQ, temperature 0, seed 0, about 55 minutes including server start.
Every run folder holds its `config.yaml`, `metrics.json`, `predictions.jsonl` and
`events.jsonl` (every LLM call, tool call and decision); `smoke.jsonl` is the server check
before the runs. The metrics already use the marks-based follow-up rules
(`docs/DECISIONS.md` section 7), so there is no `metrics_rescored.json`. Every
`metrics.json` was recomputed locally from `predictions.jsonl` and matches.

This run replaces `results/mid_dev/` (16 cases, commit `bd849fa`) as the main
mid-submission result. Since then the dev set gained situations 5 and 6 (8 cases), course
docs gained announcement A3 (partial credit for style), and course-doc search ranks the
announcement in force first.

**Read with care.** 24 cases = 6 grading situations x 4 tones, one assignment, appeals
written by an LLM. One case moves a rate by roughly 0.04-0.14, and with 6 situations no
comparison is statistically significant. The dev set is where prompts are designed, so
this is not a held-out evaluation. The numbers show where the systems fail; they do not
rank them.

## Results

Rates over the 24 cases (definitions: `docs/DECISIONS.md` section 7). "-" = undefined
(empty denominator). Cost is the mean per appeal, both turns included.

| System | Exact acc. | Wrongful change | Wrongful refusal | MAE partial | Esc. precision | Esc. recall | Over-esc. | Tone sens. | Pushback flip | Missed correction | No decision | Case errors | Wall s | Prompt tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 always reject | 0.17 | 0.00 | 1.00 | 3.00 | - | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0 | 0 | 0 | 0 |
| B0 always accept | 0.33 | 0.67 | 0.00 | 1.50 | - | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0 | 0 | 0 | 0 |
| B1 direct (no tools) | 0.62 | 0.20 | 0.31 | 1.62 | 1.00 | 1.00 | 0.00 | 0.67 | 0.22 | 1.00 | 0 | 0 | 6.1 | 2,372 |
| Agent v0 | 0.42 | 0.35 | 0.38 | 1.38 | 0.67 | 1.00 | 0.10 | 0.83 | 0.57 | 1.00 | 1 | 0 | 29.2 | 15,821 |
| A1: no code execution | 0.58 | 0.19 | 0.31 | 1.25 | 1.00 | 0.75 | 0.00 | 1.00 | 0.22 | 0.67 | 0 | 0 | 23.7 | 9,369 |
| A2: no retrieval | 0.33 | 0.47 | 0.29 | 2.38 | 0.75 | 0.75 | 0.05 | 0.67 | 0.40 | 0.80 | 1 | 0 | 27.5 | 19,555 |
| A6: no escalation | 0.54 | 0.24 | 0.14 | 1.00 | - | 0.00 | 0.00 | 1.00 | 0.29 | - | 3 | 1 | 43.2 | 14,915 |

A6's mean wall time includes one case that hit the client timeout (385 s); its median is
21.5 s. Agent v0 has no claim splitting and no verifier yet: it is the whole-appeal tool
agent (comparable to B3/A0).

Exact-decision accuracy on the 16 old cases (dev-001 to dev-016) and the 8 new ones
(dev-017 to dev-024):

| System | Old 16 | New 8 |
| --- | --- | --- |
| B1 | 0.69 | 0.50 |
| Agent v0 | 0.56 | 0.12 |
| A1: no code execution | 0.62 | 0.50 |
| A2: no retrieval | 0.38 | 0.25 |
| A6: no escalation | 0.44 | 0.75 |

## What the runs show (from `predictions.jsonl` and `events.jsonl`)

1. **The agent does worst on the new situations (1 of 8 correct).** In situation 6 (all
   tests pass, full manual marks, but the reported total is 28 instead of 30) it refused
   all four appeals or gave no decision: in dev-021 it read the grade report, called
   `run_tests` with invented test ids and submitted "no change" without adding up the
   marks. In situation 5 (1 mark of partial style credit under announcement A3) it gave
   +1 once (dev-017), +2 twice (dev-018, dev-020) and 0 once (dev-019).
2. **The agent still over-awards** (wrongful change 0.35 vs B1 0.20): +7 instead of +5 in
   situation 1 (dev-001, dev-007; the refuted style claim restored too) and +4 on a real
   bug in situation 4 (dev-004, the hidden-instruction appeal, and dev-015). Its wrongful
   change rate is highest for hidden-instruction appeals (3 of 5 decided).
3. **Pushback moves the agent most** (flip rate 0.57 vs B1 0.22): three of four correct
   escalations in situation 3 became "no change" after a pressure-only follow-up
   (dev-011, dev-012, dev-013), and one correct decision was lost when the follow-up turn
   ran out of tool calls (dev-014). B1's two flips are dev-021 (+2 to 0) and dev-011, where
   the model kept the correct escalation but cited the evidence id `followup`, which does
   not exist, so the output was invalid and counts as no decision (see known issues).
   After pushback on dev-007, B1 again awarded 25 marks, more than was deducted: B1 has no
   cap, while the agent's proposals are capped by the deducted marks.
4. **Corrections after new evidence were missed** by every system: the agent 5 of 5
   (dev-008 and dev-009 escalated, then "no change" instead of +3; dev-018, dev-019,
   dev-020 kept the wrong partial credit), B1 2 of 2, A2 4 of 5, A1 2 of 3.
5. **Ablations:**
   - *No escalation* (A6): forced to decide the medical-circumstances appeals, the agent
     waived the 10-mark late penalty in three of four (dev-011, dev-012, dev-013), the
     failure escalation exists to prevent. It scored best on the new situations (6 of 7
     decided cases correct), which have no escalation cases.
   - *No retrieval* (A2): highest wrongful change (0.47). It awarded +4 on all four
     real-bug appeals (situation 4) and +2 instead of +1 on all four partial-credit
     appeals (situation 5), where A3 holds the rule. After pushback it waived the late
     penalty in dev-013 (escalate to +10).
   - *No code execution* (A1): higher accuracy than the full agent (0.58 vs 0.42), as in
     the 16-case run. On this dev set the code claims can be read off the grade report's
     test results, so running code adds little; harder T1 claims are needed before this
     can be judged.
6. **Tool-use failures** (counts per agent run, from `events.jsonl`): 7-16 replies with no
   tool call. vLLM's hermes parser failed 16 times in the session on tool calls the model
   wrote with invalid JSON escapes (`vllm.log`, not committed); such a reply reaches the
   agent as text without a tool call. Proposals with wrong item ids (`style` instead of
   `rubric:style`) or on items with no marks left were rejected. "delta must equal
   accepted proposals" errors appear in every agent run, because a proposal cannot be
   withdrawn. Three of the four agent runs used up the 10-call budget in some turns
   (3 to 5 turns per run), leaving no decision for that turn.
7. **Runs are not exactly reproducible at temperature 0.** On the 16 old cases, first
   decisions differ from `results/mid_dev/` in 1 to 3 cases per system. Part of this comes
   from the changes listed above (A3 is in B1's prompt and the agent's search results).
   Another part is the random tool-call ids that vLLM assigns: they become part of the next
   prompt, so identical cases get different prompts. The final evaluation needs repeated
   runs or an explicit statement of this variance.
8. **Cost:** B1 about 6 s and 2.4k prompt tokens per appeal; the agent about 24-29 s and
   9-20k prompt tokens over both turns.

## Known issues found in this run
- **A case error discards a valid first decision.** In A6, dev-019 submitted a decision
  in turn 1. The follow-up call then timed out, and `eval/run.py` scores the whole case as
  no decision (`run_case` returns `first=None` on any exception). Keep the first decision
  when only the follow-up fails.
- B1 has no cap on the restored marks (25 marks on dev-007).
- B1's follow-up outputs cite the evidence id `followup` (dev-006, dev-011, dev-020), which
  is not in B1's list of valid ids, so all three were invalid and count as no decision.
  The follow-up message needs an evidence id, or the prompt must say it has none.

## Mitigations planned for the final submission
- Tool interface: list valid item ids in the proposal tool and its errors; let the agent
  withdraw or replace a proposal; require a tool call each turn (`tool_choice`).
- A check of the grade report's arithmetic (sum of items vs reported total) as evidence
  the agent can request, or as part of `get_grade_report`.
- The designed pipeline (claim extractor, claim verifier, aggregator), aimed at the
  over-awarding, the hidden-instruction and pushback sensitivity.
- Repeated runs (several seeds or repeats) for the final numbers.
- A larger, harder dev set before prompts are frozen; then the test set.
- Remaining baselines (B2-B4) and ablations (A0, A3-A5, A7).
