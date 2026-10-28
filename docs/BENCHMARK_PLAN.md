# Benchmark plan (draft for team agreement)

Owner: benchmark lead. Written 2026-10-28 for the mid-submission; revise at the weekly sync.
The proposal (section 9) plans about 300 test cases on 6 assignments. With the final
submission on 2026-11-15 that does not fit, so the plan below is a reduced version. Every
cut is listed at the end so the report can state it openly.

## 1. Where the data stands (dev)
| | |
| --- | --- |
| Assignment | D1 (string utilities), 3 announcements (A1, A2 supersedes A1, A3), 2 autograder versions |
| Dev cases | 24 = 6 situations x 4 tones, built with `python -m data.build_cases`, all valid |
| Error types | faulty_test (sit01), rubric_misapplied (sit02), missed_partial_credit (sit05), arithmetic_error (sit06), none (sit03 T3, sit04 invalid appeal) |
| Outcomes | partial_change (sit01, sit05), full_change (sit02, sit06), escalate (sit03), no_change (sit04) |
| Follow-up | new evidence in sit02 and sit05; pressure only in the others |
| Source | all `generated` (LLM-written); hidden-instruction copies of every submission exist |

## 2. Test set (frozen around 2026-11-05)
Three test assignments (T1-T3, same structure as D1: about 6 tests, 2 manual items, a late
policy, 2-3 announcements, 2 autograder versions), about 9-10 grading situations each.

**28 situations x 4 tones = 112 core cases**, plus **2 equivalence groups of 4 (8 cases)** and
**3 minimal pairs (6 cases)**: about **126 cases**.

Quota per situation (28 in total):
| Correct outcome | Situations | Planted error types |
| --- | --- | --- |
| full_change | 9 | faulty_test 4, rubric_misapplied 3, arithmetic_error 2 |
| partial_change | 8 | missed_partial_credit 4, faulty_test 2 (plus a refuted claim), rubric_misapplied 2 (plus a refuted claim) |
| no_change (invalid appeal) | 6 | none |
| escalate (T3 claim decides) | 5 | none |

So 17 situations have a planted error (faulty_test 6, rubric_misapplied 5,
missed_partial_credit 4, arithmetic_error 2) and 11 do not. Per assignment aim for
the same mix (about 3 full, 3 partial, 2 invalid, 1-2 escalate).

Other quotas:
- **Tiers:** at least a third of the claims T1 (code checkable), a third T2 (rubric, policy,
  announcements; include at least 6 where a later announcement supersedes an earlier one),
  T3 only in the 5 escalation situations (plus a T3 claim next to a T1/T2 claim in 2 of them).
- **Follow-up:** about 25% of situations (7) with new evidence, the rest pressure only.
- **Tones:** every situation has all four; hidden-instruction cases use the `-hi` submission copy.
- **Source:** about 20% hand-written (6 situations, all four tones by hand); the rest LLM-generated and edited. The out-of-family generator set is cut (see section 5).
- **Blind cases** (evaluation lead, written before reading agent code): about 7 situations
  (28 cases), kept only in the private store, reported separately.
- **Equivalence groups:** same planted error and correct answer on 4 different submissions,
  built so the error is the only thing that affects the disputed marks. **Minimal pairs:** same
  submission and claims, appeal worded two ways (fluent vs less fluent, or assertive vs
  hesitant), checked by a second person. `data/build_cases.py` builds tone variants only, so
  groups and pairs need a small extension (a `group` / `pair` section in the situation file) or
  are written directly; decide on Nov 1.

## 3. Labelling and validation
1. **Automatic:** `python -m data.validate` for every case (schema, reference grader confirms
   the planted error and the correct mark, consistency across tones, groups and pairs).
2. **Independent second labelling:** a teammate who did not write the case labels claims,
   tiers and the correct outcome from the appeal, rubric and grade report without seeing the
   answer key. Report Cohen's kappa (outcome and tier). Drop cases without agreement; keep a
   T3 case only if both agree a human is needed.
3. **Realism check (only if time allows):** teammates try to tell hand-written from generated
   appeals; report the accuracy.

## 4. Where the data lives
Test assignments, submissions, answer keys and cases are **never committed**: they live in
`data/private/` (gitignored) and a private Kaggle dataset (`docs/DATA_FORMAT.md`, "Private
data"). Build them with `python -m data.build_cases ... --data-root data/private`. Validate
private data in a terminal, not through Claude Code. Freeze procedure: a manifest of file hashes
kept in the private dataset and checked before every main run.

## 5. Differences from the proposal
| Proposal | This plan | Reason |
| --- | --- | --- |
| 6 test assignments | 3 | about 16 days between the mid and final submissions |
| about 55 situations, 300 cases | 28 situations, about 126 cases | same |
| about 12 groups of 4, 30 minimal pairs | 2 groups, 3 pairs | RQ2 is reduced to a small check |
| 4 training-only assignments | decide with the fine-tuning lead (likely 1-2) | training data volume vs time |
| about 60 blind cases | about 28 | scaled with the test set |
| other-generator test set, realism test | dropped, or only if time allows | time |
| two seeds | one seed | GPU hours and time |

The effect on analysis: with about 28 situations the bootstrap by situation is coarse, so
report differences as exploratory unless they are large, and say so.

## 6. Schedule
| Dates | Work |
| --- | --- |
| Oct 28-29 | Dev set (24 cases) and the builder done (this PR) |
| Oct 30 | Quota plan agreed; mid slides |
| Oct 31-Nov 2 | T1-T3 specs, rubrics, reference solutions, autograders (correct and faulty), announcements; author submissions |
| Nov 2-4 | Situation files and appeals; second labelling; blind cases by the evaluation lead |
| Nov 5 | Freeze: validate, kappa, hash manifest |

Fallback if behind: drop to 2 test assignments (about 84 core cases) before dropping tones or
the second labelling.
