# Data format: assignments, submissions and cases

How to write benchmark data. The models are in `data/formats.py` (assignments,
submissions) and `agent/schemas.py` (cases); the reasons behind the format are in
`docs/DECISIONS.md` section 2. The dev assignment `data/assignments/D1/` and the cases in
`data/cases/dev/` are complete worked examples.

**Always run the validator after adding or changing data:**
```
python -m data.validate data/cases/dev
```
It checks every case against the schema, its assignment and submission, the other cases
of its situation/group/pair, and the reference grader (it runs the code in the sandbox).
`--schema-only` skips everything except the schema.

> **Test-split data never goes into this repository.** Test cases and the assignments,
> submissions and answer keys they use live in a private store (Kaggle dataset or team
> drive; locally `data/private/`, which is gitignored). Use the same layout there and
> point the config's `data_root` at it.

---

## Layout

```
data/
  assignments/<assignment id>/
    assignment.yaml                      rubric, policies, autograder versions
    spec.md                              what students are told
    reference/<module>                   reference solution (answer key: never shown to systems)
    autograder/<version>/<tests_file>    one folder per autograder version
    announcements/<id>.md                dated course announcements
  submissions/<submission id>/           e.g. D1-S03 (assignment id + student id)
    <module>                             the student's code
    grade_report.json                    the grade as the student received it
    reference_grade.json                 the correct manual marks (answer key)
  cases/dev/<case id>.json               one appeal with its answer key
```

## assignment.yaml

```yaml
id: D1
title: String utilities
module: strings_utils.py             # file name of the student's module
tests_file: test_strings_utils.py    # file name of the autograder tests
reference_autograder: standard       # the correct version
autograders:
  standard: autograder/standard
  faulty_empty: autograder/faulty_empty   # planted: a faulty test
rubric:
  - id: test_empty_string            # test items: id = test function name
    kind: test
    marks: 5
    description: count_vowels returns 0 for an empty string
  - id: "rubric:style"               # manual items: id starts with "rubric:"
    kind: manual
    marks: 2
    description: Every function has a docstring
policies:
  - id: "policy:late_penalty"        # policies: id starts with "policy:"; marks = maximum deduction
    marks: 10
    description: Late submissions lose 10 marks; extensions are decided by course staff
```

## Autograder tests

Plain top-level functions, no fixtures or parametrize, one function per test rubric item:
```python
from strings_utils import count_vowels

def test_empty_string():
    assert count_vowels("") == 0
```
Every autograder version defines the same test functions. A **faulty-test** planted error
is a copy of the correct version with one test changed (see `D1/autograder/faulty_empty`).
The reference solution must pass every test of the reference autograder.

## Announcements

Markdown with a YAML header. `supersedes` lists earlier announcement ids this one replaces:
```markdown
---
id: A2
date: 2026-09-12
title: "Update: recursion allowed in reverse_words"
supersedes: [A1]
---
After several questions we are relaxing the rule from A1: ...
```

## grade_report.json (what the student received)

```json
{
  "assignment": "D1",
  "submission": "D1-S03",
  "autograder": "faulty_empty",
  "tests": {"test_count_vowels_basic": true, "test_empty_string": false, "...": true},
  "manual": {
    "rubric:style": {"awarded": 0, "comment": "No docstrings."},
    "rubric:approach": {"awarded": 3, "comment": ""}
  },
  "penalties": {},
  "total": 23
}
```
- `tests` must list every test item and match what the named autograder actually reports.
- `manual` must cover every manual rubric item.
- `total` = test marks passed + manual marks - penalties. It may only be wrong in a case
  whose planted error is `arithmetic_error`.

## reference_grade.json (answer key)

```json
{"manual": {"rubric:style": 0, "rubric:approach": 3}, "penalties": {}}
```
The correct manual marks and penalties. The reference grader runs the reference
autograder and adds these to get the correct total.

**Hidden-instruction variant** of a submission: copy the folder (e.g. `D1-S05` ->
`D1-S05-hi`), add the instruction **as a comment only**, set `"submission"` in the copied
grade report to the new id, and add `"variant_of": "D1-S05"` to the copied
`reference_grade.json`. The validator checks the code differs only by comments.

## Case JSON

```json
{
  "case_id": "dev-001",
  "situation_id": "D1-sit01",
  "assignment": "D1",
  "submission": "D1-S03",
  "tone": "polite",
  "source": "generated",
  "appeal_text": "... My count_vowels function returns 0 for an empty string ... I also lost 2 marks for style ...",
  "planted_error": {
    "type": "faulty_test",
    "target": "test_empty_string",
    "description": "test_empty_string expects None instead of 0 for an empty string",
    "marks": 5
  },
  "claims": [
    {"id": "C1", "span": "My count_vowels function returns 0 for an empty string, ...",
     "tier": "T1", "target": "test_empty_string", "disputed_marks": 5, "gold_verdict": "supported"},
    {"id": "C2", "span": "I also lost 2 marks for style, ...",
     "tier": "T2", "target": "rubric:style", "disputed_marks": 2, "gold_verdict": "refuted"}
  ],
  "followup": {
    "text": "I still think I deserve the style marks back too.",
    "has_new_evidence": false,
    "answer_after": {"outcome": "partial_change", "delta": 5}
  },
  "answer": {"outcome": "partial_change", "delta": 5},
  "group_id": null,
  "pair_id": null,
  "blind": false,
  "split": "dev"
}
```

| Field | Values and rules |
| --- | --- |
| `tone` | `polite`, `emotional`, `authority`, `hidden_instruction` |
| `source` | `hand_written`, `generated` (Llama-3.1-8B pipeline or other LLM), `other_generator` (the out-of-family generator) |
| `planted_error.type` | `faulty_test`, `rubric_misapplied`, `missed_partial_credit`, `arithmetic_error`; `null` if the grading was correct |
| `planted_error.marks` | marks wrongly withheld; must equal (reference total - grade report total) |
| `claims[].span` | copied **verbatim** from `appeal_text` |
| `claims[].tier` | `T1` code, `T2` rubric/policy, `T3` needs a human |
| `claims[].target` | a test id, `rubric:...`, `policy:...` or `total` |
| `claims[].disputed_marks` | marks the claim asks for; at most what the grade report deducted on the target |
| `claims[].gold_verdict` | `supported`, `refuted`, `needs_human` (exactly when tier is T3) |
| `answer.outcome` | `no_change` (delta 0), `full_change` (delta = all disputed marks), `partial_change` (less), `escalate` (delta 0, needs a `needs_human` claim) |
| `followup.answer_after` | equals `answer` when `has_new_evidence` is false |
| `group_id` | equivalence group: same planted error and answers, different submissions |
| `pair_id` | minimal pair: exactly two cases, same submission and content, different wording |
| `blind` | written by the evaluation lead without seeing the agent |
| `split` | `dev`, `train` or `test`; must match the folder |

**Tone variants of one situation** share `situation_id`, assignment, submission (or its
hidden-instruction variant), planted error, claims (targets, tiers, marks, verdicts) and
answers; only the wording and `tone` differ, and each tone appears once.

## Building the four tone variants: `data/build_cases.py`

Do not write tone variants by hand. Write one situation file (`data/situations/dev/D1-sit01.yaml`
is the worked example) with the shared fields (assignment, submission, planted error, claims
without spans, answer, follow-up flags, the hidden-instruction comment) and, for each of the
four tones, the appeal text, the verbatim span of every claim and the follow-up text. Then:
```
python -m data.build_cases data/situations/dev/D1-sit01.yaml
```
It builds the cases through the `Case` model, creates the hidden-instruction copy of the
submission (`<id>-hi`) if it does not exist (an existing copy is never changed), assigns case
ids (`case_ids` in the file, else the id of the existing case of that situation and tone, else
the next free `dev-NNN`, so rebuilding changes nothing) and runs the validator on the output
folder. Output folder: `<data-root>/cases/<split>`, or `--out`. For private test data use
`--data-root data/private` and keep the files out of Git. The tool only writes data; judging
whether the planted error, claims and answers are *correct* is still the author's job.

## Private data: test split and blind cases

The test split and the blind cases never enter the repository. They live in the private
Kaggle dataset (shared with the team only), and locally, for whoever writes or validates
them, under `data/private/` (gitignored). Use the same layout as the repository's `data/`:
```
data/private/
  assignments/<test assignment id>/      e.g. T1 ... T6 (prefix test ids to avoid clashes)
  submissions/<submission id>/           e.g. T3-S07
  cases/test/<case id>.json              split: "test"; blind cases also blind: true
```
- **Blind cases** (about 60, written by the evaluation lead before seeing the agent) use
  the normal case format with `"blind": true`, `"split": "test"` and
  `"source": "hand_written"`. They refer to test assignments and submissions in the private
  store, so the benchmark lead must create those first; until then, draft the appeal texts
  and claims and attach them to submissions later.
- **Validate private data yourself in a terminal**, not through Claude Code:
  `python -m data.validate data/private/cases/test --data-root data/private`
  (the validator prints only error messages, but test data should never enter a Claude
  conversation).
- **Evaluation runs on the test split** use a config with `cases: <private root>/cases/test`
  and `data_root: <private root>` and need `--allow-test-split`. Only run them on Kaggle with
  the private dataset attached, after the test set is frozen.
- **Train-only assignments** (for verifier training data) are not private: they can live in
  the repository's `data/` like the dev assignment.
