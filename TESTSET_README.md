# Private TEST set: regrade-appeals benchmark
https://www.kaggle.com/datasets/karthik8171729/regrade-benchmark-test-private


**Do not commit this data to Git.** It is the hidden TEST split; it lives only in the private
Kaggle dataset below and in `data/private/` on the author's machine.

## Where it is
- Kaggle (private): `karthik8171729/regrade-benchmark-test-private`
- Integrity: `manifest.txt` inside the dataset holds a sha256 for each of the 232 data files (check with `sha256sum -c manifest.txt` from inside the folder).
- Use it with the existing tooling: `--data-root <path to the dataset>`; for example
  `python -m data.validate <path>/cases/test --data-root <path>`.

## Contents
`assignments/` (T1, T2) · `submissions/` (T1-S01..S10, T2-S01..S09 plus `-hi` copies) ·
`situations/test/` (20 YAML files) · `cases/test/` (`test-001`..`test-080`) · `manifest.txt`

| Assignment | Topic |
|---|---|
| T1 | List utilities (`second_largest`, `flatten`, `chunk`) |
| T2 | Bank account (`Account`: `deposit`, `withdraw`, `transfer`) |

## Counts
- 2 assignments, 20 situations (10 each), 80 cases (20 per tone: polite, emotional, authority, hidden_instruction)
- Correct outcomes (situations): full_change 6, partial_change 6, no_change 6, escalate 2
- Planted errors (situations): none 8, faulty_test 5, missed_partial_credit 3, rubric_misapplied 2, arithmetic_error 2
- Claims by tier: T1 11, T2 12, T3 2
- Situations with a follow-up that has new evidence: 6 (T1-sit03, 04, 10; T2-sit03, 04, 09)
- Blind cases: none (all 80 are `generated`)
- Validation: `python -m data.validate` reports 80/80 cases valid

## Second labelling
Second Lable done by human.. he saw only the
appeal, spec, rubric, announcements, student code and grade report of the polite case of each situation.
- Run 1 (16 situations): outcome kappa 1.00, tier kappa 1.00.
- Run 2 (20 situations, strict: it had to run the code): outcome kappa 1.00, claim-verdict agreement 1.00,
  tier kappa 0.855. The only tier disagreement: `total` claims in the two arithmetic-error situations
  (labelled T1 in the data, T2 by the labeller).
- Dropped situations: none. Caveat: same model family as the author, so agreement may be optimistic.

## Rules
Nobody edits test files after the freeze (Nov 13, 21:00). If a case is broken, drop and record it.
