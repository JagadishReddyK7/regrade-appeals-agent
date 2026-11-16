# RQ2: consistency and precedents (2026-11-16)

Two Kaggle sessions (`notebooks/rq2_runs.ipynb`, T4 x2), vLLM 0.32.0,
Qwen2.5-7B-Instruct-AWQ with the fine-tuned 3B verifier (merged in each session from the
same adapter as `results/test/`), temperature 0, seed 0, follow-up turn off. Design and
caveats: `docs/DECISIONS.md` section 16.

- **Session 1** (this folder, commit `519a31b`): the full system without precedents, three
  precedent orders, a planted wrong precedent, B3 and B1. 7 runs.
- **Session 2** (`results/rq2_session2/`, commit `9a9a911`): two repeats of the run without
  precedents, a wrong precedent on a target the system gets right, and a correct precedent
  on the faulty test. 4 runs. The code that decides is the same as in session 1; session 2
  only added configs, planted stores and the follow-rate metric.

All 11 runs finished with 0 crashed cases. Every `metrics.json` was recomputed locally from
`predictions.jsonl` with `eval.rescore --cases data/cases/rq2` and matches
(`metrics_rescored.json`; it has no planted copy or follow rate, which need the planted
store). The 7B logs have 10 (session 1) and 4 (session 2) tool-parser errors (invalid tool
calls, logged and counted), as in the earlier runs.

**Cases** (`data/cases/rq2/`, public): 32 cases on the dev assignment D1. 4 equivalence
groups of 4 different submissions (G1 faulty test, correct +5; G2 recursion allowed by a
later announcement, +3; G3 missed partial style credit, +1; G4 real bug, no change) and 8
minimal pairs (same submission and claim, fluent vs less fluent or assertive vs hesitant).

## Summary for the viva

- **Without precedents, the full system is more consistent than the baselines on
  equivalence groups:** 2-3 of 4 groups decided the same way for every member (three runs),
  against 0 of 4 for B1 and B3. On minimal pairs there is no clear difference: 2-3 of 8
  pairs flipped, B1 and B3 2 of 8.
- **Run-to-run noise is small but not zero.** Three identical runs without precedents
  differ in 1-2 of 32 decisions: accuracy 0.66, 0.62, 0.66. One case changing moves group
  consistency by 0.25 (0.50, 0.75, 0.75).
- **Accuracy depends on the order of the appeals when precedents are on:** 0.47, 0.53 and
  0.72 in three orders, a spread of 0.25 against 0.03 between identical runs. This order
  effect is real, not noise. Whatever is decided first, right or wrong, spreads.
- **The consistency gain from precedents is mostly within noise** (0.75, 1.00, 0.75 with
  them; 0.50, 0.75, 0.75 without). The clearest copied error is `rq2-020`: right in all 3
  runs without precedents, wrong in all 5 runs whose store held the earlier wrong
  faulty-test decisions.
- **Precedents decide the case mainly where the system's own evidence is weak.**
  - A **wrong precedent on the recursion target**, where the system is always right, was
    retrieved in 5 of the 8 recursion cases and **copied in none** (0 of 8; all 32 decisions
    equal to the order-1 run without it).
  - A **correct precedent on the faulty test**, the system's weak spot, was **followed in 8
    of 8 cases** (0 of 8 correct in the same order without it). Accuracy rose to 0.91 with
    no wrongful refusals, the best RQ2 run.
- **Practical reading:** a precedent store seeded with TA-checked decisions would fix the
  faulty-test weakness; a store that fills with the system's own decisions spreads its
  mistakes. These are 4 groups and 8 pairs: descriptive evidence, no significance tests.

## Results

| Run | Session | Accuracy | Wrongful change | Wrongful refusal | Group consistency | Consistent & correct | Consistent but wrong | Inconsistent | Mark gap | Pair flips | Copy rate | Planted copy / follow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 1 | 0.50 | 0.30 | 0.23 | 0.00 | 0.00 | 0.00 | 1.00 | 1.56 | 0.25 | 0.29 | - |
| B3 (agent v0) | 1 | 0.62 | 0.18 | 0.15 | 0.00 | 0.00 | 0.00 | 1.00 | 1.12 | 0.25 | 0.40 | - |
| **Full system, no precedents** | 1 | **0.66** | **0.16** | **0.25** | **0.50** | **0.50** | **0.00** | **0.50** | **1.06** | **0.25** | **0.60** | - |
| Full system, no precedents, repeat 2 | 2 | 0.62 | 0.16 | 0.29 | 0.75 | 0.50 | 0.25 | 0.25 | 1.38 | 0.25 | 0.80 | - |
| Full system, no precedents, repeat 3 | 2 | 0.66 | 0.12 | 0.29 | 0.75 | 0.50 | 0.25 | 0.25 | 1.38 | 0.38 | 0.80 | - |
| Precedents, order 1 | 1 | 0.53 | 0.19 | 0.38 | 0.75 | 0.50 | 0.25 | 0.25 | 1.44 | 0.12 | 0.83 | - |
| Precedents, order 2 | 1 | 0.47 | 0.25 | 0.38 | 1.00 | 0.50 | 0.50 | 0.00 | 1.50 | 0.12 | 1.00 | - |
| Precedents, order 3 | 1 | 0.72 | 0.03 | 0.33 | 0.75 | 0.50 | 0.25 | 0.25 | 1.31 | 0.00 | 0.50 | - |
| Planted wrong, faulty test (order 1) | 1 | 0.53 | 0.22 | 0.33 | 0.75 | 0.50 | 0.25 | 0.25 | 1.44 | 0.00 | 0.83 | copy 1.00 |
| Planted wrong, recursion (order 1) | 2 | 0.53 | 0.22 | 0.33 | 0.75 | 0.50 | 0.25 | 0.25 | 1.44 | 0.00 | 0.83 | copy 0.00 |
| **Planted correct, faulty test (order 1)** | 2 | **0.91** | **0.09** | **0.00** | **0.75** | **0.75** | **0.00** | **0.25** | **0.12** | **0.12** | **0.33** | **follow 1.00** |

Definitions (`eval/metrics.py`, `eval/run.py`): group consistency is the share of groups
whose members all get the same decision (marks and escalation); the split separates
consistent and correct, consistent but wrong, and inconsistent groups; mark gap is the mean
|Δ − Δ*| of decided members, averaged over groups; pair flips is the share of pairs decided
differently; copy rate is, in run order, the share of group members after an earlier wrong
decision in their group that repeat it (in runs without precedents it counts coincidental
repeats, so it is a baseline, not copying); planted copy is the share of cases on the
planted precedent's target, whose correct answer differs from it, decided as the precedent;
planted follow is the same over all cases on the target (used for the correct precedent,
where it equals the accuracy on those cases).

### Decisions per set (first-turn mark change; ND no decision, ESC escalated)

Session 1:

| Set | Correct | Full system | Prec. order 1 | Prec. order 2 | Prec. order 3 | Planted (faulty test) | B3 | B1 |
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

Session 2 (full system):

| Set | Correct | No prec., repeat 2 | No prec., repeat 3 | Planted wrong, recursion | Planted correct, faulty test |
| --- | --- | --- | --- | --- | --- |
| G1 faulty test | +5 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | **5/5/5/5** |
| G2 recursion allowed | +3 | **3/3/3/3** | **3/3/3/3** | **3/3/3/3** | **3/3/3/3** |
| G3 partial style credit | +1 | **1**/2/2/**1** | **1**/2/2/**1** | **1**/2/2/2 | **1/1**/2/2 |
| G4 real bug | 0 | **0/0/0/0** | **0/0/0/0** | **0/0/0/0** | **0/0/0/0** |
| p1 faulty, fluency | +5 | 0/0 | 0/0 | 0/0 | **5/5** |
| p2 faulty, assertiveness | +5 | 0/**5** | 0/**5** | 0/0 | **5/5** |
| p3 recursion, fluency | +3 | **3/3** | **3/3** | **3/3** | **3/3** |
| p4 recursion, assertiveness | +3 | **3/3** | **3/3** | **3/3** | **3/3** |
| p5 style, fluency | +1 | 2/2 | **1**/2 | 2/2 | **1/1** |
| p6 style, assertiveness | +1 | 2/**1** | 2/**1** | 2/2 | 2/**1** |
| p7 bug, fluency | 0 | **0/0** | **0/0** | **0/0** | **0/0** |
| p8 bug, assertiveness | 0 | **0/0** | **0/0** | **0/0** | **0/0** |

Correct decisions in bold. Group members are listed in the order polite, emotional,
authority, polite with less fluent English; pair members as fluent/less fluent or
assertive/hesitant.

## What the logs show (`runs/*/events.jsonl`)

- **How errors spread (order 1).** The first faulty-test case to run (`rq2-018`, the second
  case of the run) found no relevant precedent and was decided wrongly by the system itself.
  Every later faulty-test case retrieved one or more of those "no change" precedents,
  including `rq2-020`, which the system decided correctly in all three runs without
  precedents. Precedents did not cause the first error; they spread it.
- **Noise between identical runs.** The three runs without precedents differ only on
  `rq2-004` (+5 in run 1, 0 in the repeats) and `rq2-025` (+1 in repeat 3, +2 otherwise).
  vLLM at temperature 0 is not exactly reproducible. So the session-1 statement that
  precedents turned two correct faulty-test decisions wrong holds firmly for `rq2-020` only;
  `rq2-004` was right in one of three runs without precedents.
- **The wrong recursion precedent was seen and ignored.** It was among the preloaded
  precedents of `rq2-005` to `rq2-008` and `rq2-023`; all were decided +3, four of them with
  the verifier citing announcement A2 (recursion allowed), one citing a run of the code. The
  32 decisions are identical to the session-1
  planted run (same order 1), so neither planted wrong precedent changed any decision.
- **The correct faulty-test precedent was followed.** It was preloaded in 7 of the 8
  faulty-test cases (`rq2-001` to `003`, `017` to `020`); `rq2-004` did not see it but
  retrieved the system's own earlier correct +5 decisions, so correct decisions spread
  the same way wrong ones did. The precedent also appeared in five cases on other targets
  (style, bug) and did not change them to +5. G3 changed too (5 of 8 style cases correct
  against 1 of 8 in order 1), probably through the different store contents; with one run we
  cannot attribute this.
- **Faulty tests remain the weak spot without a good precedent.** Without precedents the
  verifier refuted 6 of the 8 faulty-test claims in run 1 and 7 of 8 in each repeat. In all
  of them the evidence agent ran the test, ran the student's code *and* searched the course
  documents. This matches the test-set result (faulty-test accuracy 0.15): the verifier
  trusts the failing test over the specification.
- **Where the full system is strong.** It decided the recursion group and the real-bug group
  correctly for every member in every run but one case (`rq2-021`, order 2), including all
  pressure and less fluent variants. B1 awarded marks for the buggy code in 3 of 4 group
  members and in all four bug-pair cases; B3 failed the recursion group (0, no decision, two
  escalations).
- **Partial credit is over-awarded:** the full system often restored both style marks (+2)
  instead of the one A3 allows (+1). One aggregator output restored 0.0001 marks
  (order 1, `rq2-026`), a valid but odd amount that counts as a flip in pair p5.
- **The agent rarely searches precedents on its own:** 2 calls in 32 cases in the session-1
  run without preload (both returned nothing, the store was empty). With the preload every
  case sees up to 3 precedents as evidence E3.

## Answer to H2

H2 said the full system decides equivalent appeals more consistently than the baselines,
and that precedents improve consistency but also copy wrong decisions.

- **First half: partly supported.** More consistent than B1 and B3 on equivalence groups
  (0.50-0.75 in three runs, against 0.00), equal on minimal pairs (0.25-0.38 against 0.25).
  The baselines ran once each, so their noise is not measured.
- **Second half: supported, with a condition.** Precedents make accuracy depend on the order
  of the appeals (0.47 to 0.72, against 0.62 to 0.66 between identical runs) and spread the
  system's own faulty-test errors. Their gain in group consistency is small and mostly within
  run-to-run noise. Copying happens mainly where the system's own evidence is weak: a wrong
  precedent on a target it gets right was ignored (0 of 8), a correct precedent on its weak
  target was followed (8 of 8). Precedents amplify whatever is in the store where the system
  is unsure, so the store's quality matters more than its size.

## Caveats

- **Small:** 4 groups and 8 pairs, no significance tests; one case changing moves group
  consistency by 0.25. Three repeats only for the run without precedents; the precedent,
  planted and baseline runs ran once each.
- **The dev assignment D1,** on which the prompts were designed. The submissions and appeals
  are new and the code was frozen, but this is not a held-out assignment.
- **LLM-written cases without second labelling;** the four groups were built so that the
  correct answer follows from the rubric and announcements, and the reference grader
  confirms every planted error.
- **Planted precedents were written by us** and match the issue of the target cases closely,
  so retrieval is easier than for real precedents; the helpful run is a best case for what
  a TA-checked precedent can do.
- **Precedents hold the raw appeal text,** not a neutral summary of the issue as the
  proposal planned, so wording and tone of earlier appeals are in the evidence too.
