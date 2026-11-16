# RQ2: consistency and precedents (2026-11-16, after the submission)

One Kaggle session (`notebooks/rq2_runs.ipynb`, T4 x2), commit `519a31b` (`commit.txt`),
vLLM 0.32.0, Qwen2.5-7B-Instruct-AWQ with the fine-tuned 3B verifier (merged in the session
from the same adapter as `results/test/`), temperature 0, seed 0, one run per system,
follow-up turn off. Design and caveats: `docs/DECISIONS.md` section 16. All 7 runs finished
with 0 crashed cases. Every `metrics.json` was recomputed locally from `predictions.jsonl`
with `eval.rescore` and matches (`metrics_rescored.json`; it has no planted copy rate, which
needs the planted store). The 7B log has 10 tool-parser errors (invalid tool calls, logged
and counted), as in the earlier runs.

**Cases** (`data/cases/rq2/`, public): 32 cases on the dev assignment D1. 4 equivalence
groups of 4 different submissions (G1 faulty test, correct +5; G2 recursion allowed by a
later announcement, +3; G3 missed partial style credit, +1; G4 real bug, no change) and 8
minimal pairs (same submission and claim, fluent vs less fluent or assertive vs hesitant).

## Summary for the viva

- **Without precedents, the full system is more consistent than the baselines on
  equivalence groups:** 2 of 4 groups decided the same way for every member, against 0 of 4
  for B1 and B3. On minimal pairs there is no difference: each system changed its decision
  in 2 of 8 pairs.
- **Precedents make decisions more uniform, partly by copying errors.** With the store
  filling with the system's own decisions, group consistency rises from 0.50 to 0.75-1.00,
  but most of the gain is *consistently wrong*: in the faulty-test group the first wrong
  decision was retrieved by every later member, and the two members the system got right
  without precedents were then decided wrongly too.
- **Accuracy depends on the order of the appeals:** 0.47, 0.53 and 0.72 in three orders
  of the same 32 cases, against 0.66 without precedents. Whatever is decided first, right
  or wrong, spreads.
- **The planted wrong precedent was copied in 8 of 8 cases**, but the same cases were
  already all wrong in the same order without it, so this run cannot show an extra effect.
- These are 4 groups and 8 pairs, one run each: descriptive evidence, no significance
  tests.

## Results

| Run | Accuracy | Wrongful change | Wrongful refusal | Group consistency | Consistent & correct | Consistent but wrong | Inconsistent | Mark gap | Pair flips | Copy rate | Planted copy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 0.50 | 0.30 | 0.23 | 0.00 | 0.00 | 0.00 | 1.00 | 1.56 | 0.25 | 0.29 | - |
| B3 (agent v0) | 0.62 | 0.18 | 0.15 | 0.00 | 0.00 | 0.00 | 1.00 | 1.12 | 0.25 | 0.40 | - |
| **Full system, no precedents** | **0.66** | **0.16** | **0.25** | **0.50** | **0.50** | **0.00** | **0.50** | **1.06** | **0.25** | **0.60** | - |
| Precedents, order 1 | 0.53 | 0.19 | 0.38 | 0.75 | 0.50 | 0.25 | 0.25 | 1.44 | 0.12 | 0.83 | - |
| Precedents, order 2 | 0.47 | 0.25 | 0.38 | 1.00 | 0.50 | 0.50 | 0.00 | 1.50 | 0.12 | 1.00 | - |
| Precedents, order 3 | 0.72 | 0.03 | 0.33 | 0.75 | 0.50 | 0.25 | 0.25 | 1.31 | 0.00 | 0.50 | - |
| Planted precedent (order 1) | 0.53 | 0.22 | 0.33 | 0.75 | 0.50 | 0.25 | 0.25 | 1.44 | 0.00 | 0.83 | 1.00 |

Definitions (`eval/metrics.py`): group consistency is the share of groups whose members all
get the same decision (marks and escalation); the split separates consistent and correct,
consistent but wrong, and inconsistent groups; mark gap is the mean |Δ − Δ*| of decided
members, averaged over groups; pair flips is the share of pairs decided differently; copy
rate is, in run order, the share of group members after an earlier wrong decision in their
group that repeat it (in runs without precedents it counts coincidental repeats, so it is a
baseline, not copying); planted copy is the share of cases on the planted precedent's target
decided as the precedent.

### Decisions per set (first-turn mark change; ND no decision, ESC escalated)

| Set | Correct | Full system | Prec. order 1 | Prec. order 2 | Prec. order 3 | Planted | B3 | B1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 faulty test | +5 | 0/0/0/**5** | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | **5/5/5**/ND | 1/**5**/ND/0 |
| G2 recursion allowed | +3 | **3/3/3/3** | **3/3/3/3** | **3/3/3/3** | **3/3/3/3** | **3/3/3/3** | 0/ND/ESC/ESC | **3/3**/ESC/**3** |
| G3 partial style credit | +1 | **1**/2/2/**1** | **1**/2/2/2 | 2/2/2/2 | **1**/2/**1/1** | **1**/2/2/2 | **1**/0/0/**1** | **1/1/1**/2 |
| G4 real bug | 0 | **0/0/0/0** | **0/0/0/0** | **0/0/0/0** | **0/0/0/0** | **0/0/0/0** | **0/0/0**/4 | 4/4/4/**0** |
| p1 faulty, fluency | +5 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | **5/5** | 0/0 |
| p2 faulty, assertiveness | +5 | 0/**5** | 0/0 | 0/0 | 0/0 | 0/0 | **5/5** | 0/**5** |
| p3 recursion, fluency | +3 | **3/3** | **3/3** | 0/**3** | **3/3** | **3/3** | **3/3** | **3/3** |
| p4 recursion, assertiveness | +3 | **3/3** | **3/3** | **3/3** | **3/3** | **3/3** | **3/3** | **3/3** |
| p5 style, fluency | +1 | 2/2 | 2/0.0001 | 2/2 | **1/1** | 2/2 | **1/1** | **1**/2 |
| p6 style, assertiveness | +1 | 2/**1** | 2/2 | 2/2 | **1/1** | 2/2 | 2/**1** | **1/1** |
| p7 bug, fluency | 0 | **0/0** | **0/0** | **0/0** | **0/0** | **0/0** | 4/4 | 4/4 |
| p8 bug, assertiveness | 0 | **0/0** | **0/0** | **0/0** | **0/0** | **0/0** | 4/**0** | 4/4 |

Correct decisions in bold. Group members are listed in the order polite, emotional,
authority, polite with less fluent English; pair members as fluent/less fluent or
assertive/hesitant.

## What the logs show (`runs/*/events.jsonl`)

- **How errors spread (order 1).** The first faulty-test case to run (`rq2-018`, the second
  case of the run) found no relevant precedent and was decided wrongly by the system itself. Every later
  faulty-test case retrieved one or more of those "no change" precedents, including
  `rq2-004` and `rq2-020`, which the system decided correctly without precedents. Precedents
  did not cause the first error; they spread it.
- **Faulty tests remain the weak spot, even with documents.** Without precedents the
  verifier refuted 6 of the 8 faulty-test claims (`rq2-004` and `rq2-020` supported). In all
  of them the evidence agent ran the test, ran the student's code *and* searched the course
  documents. This matches the test-set result (faulty-test accuracy 0.15) and means the
  failure is not only a missing search: the verifier trusts the failing test over the
  specification.
- **Where the full system is strong.** It decided the recursion group and the real-bug group
  correctly for every member in every run, including all pressure and less fluent
  variants. B1 awarded marks for the buggy code in 3 of 4 group members and in all four
  bug-pair cases; B3 failed the recursion group (0, no decision, two escalations).
- **Partial credit is over-awarded:** the full system often restored both style marks (+2)
  instead of the one A3 allows (+1). One aggregator output restored 0.0001 marks
  (order 1, `rq2-026`), a valid but odd amount that counts as a flip in pair p5.
- **The agent rarely searches precedents on its own:** 2 calls in 32 cases in the run
  without preload (both returned nothing, the store was empty). With the preload every case
  sees up to 3 precedents as evidence E3.

## Answer to H2

H2 said the full system decides equivalent appeals more consistently than the baselines,
and that precedents improve consistency but also copy wrong decisions.

- **First half: partly supported.** More consistent than B1 and B3 on equivalence groups
  (0.50 against 0.00, with a smaller mark gap, 1.06 against 1.12 and 1.56), equal on minimal
  pairs (0.25 each).
- **Second half: supported, in the way the proposal warned about.** Precedents raised group
  consistency (0.75 to 1.00) but added consistently wrong groups (0.25 to 0.50) and made
  accuracy depend on the order of the appeals (0.47 to 0.72). The system copies its own
  earlier mistakes. The planted precedent was copied every time, but the same cases were
  already all wrong without it, so its separate effect could not be measured here.

## Caveats

- **Small:** 4 groups and 8 pairs, one run each, no significance tests; one group changing
  moves group consistency by 0.25.
- **The dev assignment D1,** on which the prompts were designed. The submissions and appeals
  are new and the code was frozen, but this is not a held-out assignment.
- **LLM-written cases without second labelling;** the four groups were built so that the
  correct answer follows from the rubric and announcements, and the reference grader
  confirms every planted error.
- **Precedents hold the raw appeal text,** not a neutral summary of the issue as the
  proposal planned, so wording and tone of earlier appeals are in the evidence too.
- **One planted precedent, on a target where the system already fails,** so the planted
  test is uninformative; a planted precedent on a target the system gets right (for example
  the recursion group) would test copying properly.
