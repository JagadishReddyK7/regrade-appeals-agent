# Test set: step-by-step guide (Nov 11-13)

For whoever builds the test set. This replaces the schedule in `docs/BENCHMARK_PLAN.md`
section 6, which assumed we would start on Oct 31. The final submission is **Nov 15,
23:59**, the final runs are on **Nov 14**, so the test set must be **frozen by Nov 13,
21:00**. Everything below is cut to fit that.

Read first (30 minutes): `docs/DATA_FORMAT.md` (all of it), then look at the worked
example: `data/assignments/D1/`, `data/submissions/D1-S03/`, `data/situations/dev/D1-sit01.yaml`
and `data/cases/dev/dev-001.json`. A test assignment looks exactly like D1.

## Target

| | Target | Minimum if behind |
| --- | --- | --- |
| Test assignments | 2 (T1, T2) | 2 |
| Situations | 8 per assignment = 16 | 6 per assignment = 12 |
| Cases (x 4 tones) | 64 | 48 |
| Second labelling | every situation | every situation (do not cut this) |

Cut situations before cutting tones or the second labelling (same rule as the benchmark
plan). Equivalence groups and minimal pairs are **dropped** (the builder does not make them;
RQ2 becomes a small check on the dev set and precedents). Blind cases: Suresh writes 2-4
situations if he has time, otherwise none; say so in the report.

## Rules (read twice)

1. **Nothing goes into Git.** All test files live under `data/private/` (gitignored) and
   later in a private Kaggle dataset. Never `git add` anything from `data/private/`.
2. **Do not use Claude Code on test data**, and do not paste test files into a Claude Code
   session: the agent's prompts are designed in this repo and must not see the test set.
   Drafting appeal texts in a normal chat LLM (ChatGPT, Gemini, claude.ai) is fine; set
   `source: generated` for those, `hand_written` for appeals you write yourself.
3. **Do not copy D1.** New problems, new code, new announcements. Do not reuse dev
   appeal sentences.
4. Validate in your own terminal only.

## Mix per assignment (8 situations)

| # | Planted error | Correct outcome | Claims |
| --- | --- | --- | --- |
| 1 | `faulty_test` | `full_change` | 1 T1 claim, supported |
| 2 | `faulty_test` | `partial_change` | T1 supported + a T2 claim that is refuted (e.g. "my style was fine" but it wasn't) |
| 3 | `rubric_misapplied` | `full_change` | T2 claim, supported by an announcement that **supersedes** an earlier one (like D1 A1 -> A2) |
| 4 | `missed_partial_credit` | `partial_change` | T2 claim; the rule for partial credit is in an announcement |
| 5 | `arithmetic_error` | `full_change` | `target: total`; grade report total is wrong |
| 6 | none | `no_change` | T1 claim, refuted: the student says the code works but **it really has a bug** |
| 7 | none | `no_change` | T2 claim, refuted: asks for marks the rubric or policy does not give |
| 8 | none | `escalate` | a T3 claim (illness, extension, a TA's private promise) + optionally a T1/T2 claim |

Follow-up: give **2 of the 8** situations a follow-up with new evidence (e.g. "see
announcement A3" or "test X also fails on the reference input"), where `answer_after` differs
from `answer`. All others: pressure-only follow-up, `answer_after` = `answer`.

**Important, from the mid results:** on the dev set the code claims could be decided by
reading the grade report, so running code looked useless. In situations 1, 2 and 6, make the
claim need execution: e.g. the student's code is correct on the edge case the faulty test
checks wrongly, but you can only see that by running it; or in situation 6 the student
claims "my function handles negative numbers" and only running it shows it doesn't.

## Step 1: assignments (Nov 11-12, about 3 hours each)

For each of T1 and T2 under `data/private/assignments/<T1|T2>/`:
1. `spec.md`: a small Python task, 2-4 functions (e.g. list utilities, a simple bank
   account class, date parsing). Keep it at D1's size.
2. `reference/<module>.py`: a correct solution.
3. `autograder/standard/test_<module>.py`: **6 tests**, plain functions (no fixtures,
   no parametrize), one per test rubric item. Reference must pass all.
4. `autograder/faulty_<name>/...`: copy of `standard` with **one** test wrong (needed by
   situations 1-2). Two faulty versions if you want situations 1 and 2 to differ.
5. `announcements/A1.md, A2.md (supersedes A1), A3.md`: dated rules, e.g. a relaxed
   rule, a partial-credit rule.
6. `assignment.yaml`: rubric (6 test items + 2 manual items like `rubric:style`,
   `rubric:approach`), `policy:late_penalty`, autograder versions. Copy D1's structure.

## Step 2: submissions (Nov 12, about 3 hours per assignment)

Under `data/private/submissions/<T1-S01 ...>/`, one folder per student (about 5-6 per
assignment; one submission can serve several situations):
- `<module>.py`: the student's code (correct code for situations 1-5, buggy for 6).
- `grade_report.json`: what the student got, **including the planted error** (e.g. graded
  with `faulty_...`, a manual mark too low, or a wrong `total`). `tests` must match what the
  named autograder really reports.
- `reference_grade.json`: the correct manual marks and penalties.

Do not make the `-hi` copies yourself; the builder makes them.

## Step 3: situation files (Nov 12-13)

One YAML per situation under `data/private/situations/test/`, e.g. `T1-sit01.yaml`, copying
`data/situations/dev/D1-sit01.yaml`. Set `split: test`, a `hidden_instruction.comment`
(a comment line only), and four tones: `polite`, `emotional`, `authority`,
`hidden_instruction`, each with `appeal_text`, the **verbatim** `spans` of every claim, and
a `followup_text`. Leave out `case_ids`: the builder numbers them `test-001`, ...

Build and validate (your terminal):
```
python -m data.build_cases data/private/situations/test/T1-sit01.yaml --data-root data/private
python -m data.validate data/private/cases/test --data-root data/private
```
The validator runs the reference grader: it confirms that the planted error and marks add
up. It does **not** judge whether the appeal and answer make sense; that is your job and the
second labeller's.

## Step 4: second labelling (Nov 13, morning)

Someone who did not write the case (not Jagadish, who works on the agent) takes the
**polite** case of every situation and, from the appeal text, spec, rubric, announcements,
code and grade report only (not the YAML or answer), writes down: each claim's tier, its
verdict, and the outcome with delta. Put the two labellings in a spreadsheet; compute agreement
(Cohen's kappa on the outcome and on the tier, any online calculator is fine). Discuss
disagreements; fix or drop the situation. Keep a T3 situation only if both agree a human is
needed. Report: number of situations, kappa, number dropped.

## Step 5: freeze (Nov 13, by 21:00)

1. Validate one last time: `24/24`-style "all valid" line for the test folder.
2. Write a manifest of file hashes:
   ```
   python -c "import hashlib,pathlib;[print(hashlib.sha256(p.read_bytes()).hexdigest(),p.as_posix()) for p in sorted(pathlib.Path('data/private').rglob('*')) if p.is_file()]" > manifest.txt
   ```
   (run from the repo root; keep `manifest.txt` inside `data/private/`).
3. Zip `data/private/` and upload it as a **private** Kaggle dataset; share it with the
   team's Kaggle accounts only. Tell Jagadish the dataset name.
4. After the freeze nobody edits test files. If a case is broken, drop it and record why.

## Send to the team when done (no file contents, only counts)

Number of situations and cases per assignment, the outcome and error-type counts, the number
of claims per tier, kappa, dropped situations, and the Kaggle dataset name. These numbers go
into the report's dataset section.
