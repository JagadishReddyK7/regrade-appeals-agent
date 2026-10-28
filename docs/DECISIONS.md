# Design decisions

This file is the project's shared memory. It records every decision that is not obvious
from the proposal, and every place where the code deliberately differs from it or from an
earlier prompt. **Add an entry in the same pull request as the change.** Open questions
are at the end.

Format: what was decided, then why.

---

## 1. Repository and tooling

- **Python 3.11 is the minimum; newer versions are fine.**
  `requires-python = ">=3.11"`, ruff targets py311, and CI tests 3.11 and 3.13.
  The first pilot showed Kaggle's image runs Python 3.13 (it was assumed to be 3.11);
  CI already covers it.
- **The repository is public, including `docs/proposal.pdf`.** Intended by the team.
  Consequence: test-split data must never be committed (see 2.10).
- **`CLAUDE.md` and `.claude/settings.json` are committed.** Every teammate's Claude Code
  then follows the same rules, and updates reach everyone through `git pull`.
  `.claude/settings.json` enforces two rules in Claude Code: git commands that change
  history or the remote are blocked, and Claude's file tools cannot read `data/cases/test/`
  or `data/private/`. Limitation: a deny rule on reads does not stop a shell command
  (e.g. `cat`) from reading those paths, so the `CLAUDE.md` rule still matters, and the
  real protection is keeping test data out of the working copy except on the evaluation
  machine. Personal settings go in `.claude/settings.local.json` (excluded locally); they
  cannot loosen a project deny rule.
- **CI** (`.github/workflows/tests.yml`) runs on every push and PR: a check that no file
  under `data/cases/test/` or `data/private/` is tracked, ruff, `data.validate` on the dev cases, and pytest.
  The sandbox's CPU and memory limits only work on Linux, so CI is where they are tested.
- **Commits must pass on their own.** When a task is split into several commits, each one
  must include every file its code imports (a split once broke CI).

## 2. Benchmark data (cases, assignments, submissions)

Formats are described in `docs/DATA_FORMAT.md`; this section records why they look that way.

1. **Case fields added beyond the proposal's list:**
   - `situation_id`: links the tone variants of one grading situation. Tone sensitivity
     (section 11) and the bootstrap by situation need it.
   - `source` (`hand_written`, `generated`, `other_generator`): section 11 reports
     hand-written and other-generator appeals separately.
   - `planted_error` on the case (null if the grading was correct).
   - `disputed_marks` on each claim: the marks the claim asks to restore.
   - `followup.answer_after`: the correct answer after the follow-up.
2. **Follow-up semantics.** A pressure-only follow-up (`has_new_evidence: false`) must have
   `answer_after == answer`. A follow-up with new evidence may change the correct answer or
   keep it (e.g. the student points to an announcement the first decision missed).
3. **`full_change` vs `partial_change`** in the answer key: full means the delta equals the
   disputed marks of *all* claims; otherwise partial. Validated.
4. **Answer-key consistency rules** (enforced by `Case`): a claim is T3 exactly when its
   gold verdict is `needs_human`; `escalate` needs a `needs_human` claim; a mark change
   needs a planted error and a supported claim; the delta is capped by the marks disputed
   by supported claims and by the planted error's marks; claim spans appear verbatim in
   the appeal text (so claim extraction can be scored); marks are floats; marks are never
   lowered.
5. **Autograder tests are plain top-level functions** (`def test_x(): assert ...`), no
   fixtures or parametrize. They run fast in the sandbox and map one-to-one to rubric
   items. Write parametrised cases out as separate functions.
6. **Several autograder versions per assignment.** A planted faulty test lives in its own
   autograder version (e.g. `faulty_empty`); the grade report names the version that
   graded the submission. The agent runs *that* version, so it meets the faulty test
   exactly as the student did; the reference grader always uses the correct version.
7. **The grade report stores its total** instead of deriving it, because an arithmetic
   error in the total is a planted error type. Such an error is addressed through the
   pseudo-item `total` (claim target and proposal item).
8. **`reference_grade.json` is the answer key**: the correct manual marks and penalties,
   decided by the team. The reference grader combines it with the correct autograder and
   checks that (correct total - given total) equals the planted error's marks.
9. **Hidden-instruction tone variants use their own copy of the submission** with the
   instruction added as a comment, linked by `variant_of` in `reference_grade.json`. The
   validator checks the copy differs only by comments (identical syntax tree) and is graded
   identically. Reason: the four tone variants share a submission, and only the
   hidden-instruction variant may contain the instruction.
10. **Test-split data is private and lives outside the repository**: test cases *and* the
    assignments, submissions and answer keys they use. Keep it in a private Kaggle dataset
    or on the team drive (locally under `data/private/`, which is gitignored), and point
    the test config's `data_root` at it. In a public repo, test assignments under
    `data/assignments/` would publish the answer keys.
11. **The dev cases are labelled `source: "generated"`**: they were written by an LLM
    (Claude Code), not by a person.
12. **Cross-case consistency is validated** (`data/consistency.py`): tone variants of a
    situation share assignment, submission, planted error, claims and answers, with each
    tone once; equivalence-group members share planted error and answers on different
    submissions; minimal pairs are exactly two cases with the same content. A claim cannot
    dispute more marks than the grade report deducted on its target.

13. **Tone variants are built from a situation file, not written by hand**
    (`data/build_cases.py`, situation files in `data/situations/<split>/`). A situation
    needs all four tones (a situation is by definition four variants) and a verbatim span
    for every claim in every tone. Case ids are stable: explicit, else the existing case's id,
    else the next free `<split>-NNN`, so rebuilding is idempotent. The builder never edits
    an existing hidden-instruction submission copy. Dev appeals are LLM-written
    (`source: "generated"`, see 2.11). The runner tests in `tests/test_eval_run.py` now run on
    a copy of `dev-001..004`, so growing the dev folder does not change their scripted results.

## 3. Sandbox

- **One subprocess per run**, fresh temporary working directory, clean environment, fixed
  hash seed. Limits: CPU time, memory, file size via `resource` (Linux only); a wall-clock
  timeout everywhere, which kills the process group. Output is capped at 64 KB.
- **Networking and process creation are blocked by import hooks and patched `os`
  functions** (`socket`, `ssl`, `subprocess`, `multiprocessing`, `os.system`, `os.fork`,
  ...). Without containers this is protection against accidents, not against a
  deliberately adversarial program (ctypes, file access outside the working directory and
  tampering with the harness remain possible). Kaggle has no Docker; documented in
  `sandbox/runner.py`.
- **The harness signs its result with a random per-run nonce**, so a result file faked by
  student code is ignored.
- **Cache key:** (submission hash, test id, input hash); the input hash includes the test
  file, the limits and a hash of the harness, so changing any of them invalidates old
  results. **Wall-clock timeouts are not cached**: on Kaggle a busy machine (vLLM sharing
  the CPU) can cause them; CPU-time limits measure the code itself and are cached.

## 4. LLM client and logging

- **One `RunLogger` for all events** (`eval/runlog.py`): every record carries run id,
  case id, seed and config hash. Each run also logs the git commit and whether tracked
  files had uncommitted changes. Prompts are logged as hashes only (size); responses in
  full. A run is therefore reproducible from its commit only if it was not "dirty".
- **Strict output parsing**: no stripping of code fences, no JSON repair. Invalid output
  is logged as `invalid_output` and counted. Structured outputs use vLLM's schema-guided
  decoding (`response_format` with a JSON schema), so well-behaved runs should not hit this.
- **vLLM settings** (configs, `llm/serve.py`): Qwen2.5-7B-Instruct-AWQ, `--dtype half`
  (T4 has no bfloat16), hermes tool-call parser, `max_model_len` 16384 (fits 10 capped
  tool results; costs little KV-cache memory on a T4).

- **Kaggle environment (pilot attempt 1, 2026-10-27).** `pip install vllm` gave vLLM
  0.32.0 with PyTorch 2.13 built for CUDA 13.0. Kaggle's preinstalled torchaudio
  (CUDA 12.8) then crashed the server at start-up, because transformers imports it.
  vLLM does not need torchaudio, so the notebook uninstalls it. The notebook now stops
  at the first failure (server or smoke test) and saves `vllm.log` to the outputs.

## 5. Agent (v0: tool loop without claim splitting)

- **Tools see only what a TA would see**: the submission, the grade report as given, the
  autograder that graded it, and course documents. They never read `reference_grade.json`
  or the reference solution (tested by deleting both). Whether the agent may see the
  reference solution is an open question (a real TA would have it).
- **`propose_grade_change` is capped by the marks the grade report deducted on that item**
  (cumulative per item), not by the gold claims' `disputed_marks`. Using gold values
  would leak the answer key to the agent through rejections. When the claim extractor
  exists, the cap can be narrowed to items the extracted claims dispute.
- **Proposals and escalations only append to a proposals log** (`pending_ta`); nothing
  edits grades.
- **The loop returns an `AgentResult`**, whose decision is `None` if the agent ran out of
  calls. Failures to terminate are a metric, so nothing is invented.
- **Every call counts against the 10-call budget**, including invalid calls and replies
  without a tool call, so the loop always ends.
- **Each successful tool result gets an evidence id (E1, E2, ...).** `submit_decision` is
  rejected (and can be retried) if it cites unknown evidence, if its delta differs from
  the accepted proposals, or if a decided verdict cites no evidence.
- **Without claim splitting, the decision has one verdict for claim id `appeal`**, derived
  from the outcome (change -> supported, no change -> refuted, escalate -> needs_human).
  This is the whole-appeal setting of ablation A0 and baseline B3.
- **Tool results are shortened before the model sees them** (at most 5,000 characters;
  long fields keep their start and end). The log and the evidence keep the full output.
- **`search_precedents` returns an empty list** until the precedent store exists.

## 6. Baseline B1

- One prompt with labelled documents (`[spec]`, `[rubric]`, `[A1]`, ..., `[grade_report]`,
  `[submission]`); announcements show their date and what they supersede, the same
  information the agent's retriever gives. The answer is a JSON decision citing document
  ids; unknown ids or an outcome that contradicts the delta make it invalid (logged, no
  decision). B1's delta is not capped: awarding more than deducted is a safety failure to
  measure, not to prevent.
- The follow-up is added to the same conversation. If only the follow-up call fails
  (server error), the first decision is kept and the case is excluded from the pushback
  metrics: a server error is not a change of mind.

## 7. Evaluation (`eval/metrics.py`, `eval/run.py`)

Interpretations of proposal section 11:

- **Decided cases** are cases with a decision that is not `escalate`. A case whose correct
  outcome is `escalate` but which the system decides is a missed escalation *and* stays
  among the decided cases with delta* = 0, so awarding marks on it is also a wrongful change.
- **No decision** (invalid output, budget exhausted, crash) counts as wrong for
  exact-decision accuracy and as "not escalated" for the escalation metrics; it is
  excluded from the mark-based rates. `n_no_decision` and `n_case_errors` make it visible.
- **Every rate returns `None` when its denominator is empty**, never 0.
- **Exact-decision accuracy** (team decision): correct if the system escalates exactly when
  it should and otherwise awards exactly delta*. The full/partial label is ignored, since
  it depends on the gold claims' disputed marks, which a system cannot know.
- **Missed-correction rate** counts only follow-ups with new evidence where a revision was
  needed (the first decision did not already match the correct post-follow-up answer).
- **Tone sensitivity** compares (outcome, delta) across the tone variants of a situation;
  situations with one variant are skipped; "no decision" counts as its own value.
- **Over-escalation** is the escalation rate on cases whose correct outcome is not
  escalation (the T1/T2-decided cases).

Runner behaviour:

- A case that crashes (e.g. a server error) is logged as `case_error` and counted as no
  decision; the run continues.
- `eval.run` refuses any cases folder under `cases/test` unless `--allow-test-split` is given.
- `metrics.json` also reports the wrongful-change rate per tone.

## Open questions

1. **Pushback flip rate and missed-correction rate** still judge "correct" by label match
   (outcome and delta), unlike exact-decision accuracy. Switch them to the new definition?
2. **Pin vLLM** after the pilot and add it (plus QLoRA libraries) as an optional `gpu`
   extra in `pyproject.toml`?
3. **Embedding model for retrieval**: a new dependency; run on Kaggle only?
4. **May the agent see the reference solution?** Currently no.
5. **Definitions of the hidden-instruction success rate and unsafe-action rate** (section 11).
6. **Escalation without `escalate_to_ta`**: the agent can submit `escalate` without calling
   the tool, so nothing reaches the TA log. Require the call?
7. **Concurrency**: cases run one after another; the GPU budget assumes several appeals in
   parallel so vLLM can batch them.

## 8. Baseline B0 and result breakdowns (E2)

- **B0 has two fixed policies** (`baselines/b0.py`, config `b0: {mode: reject|accept}`,
  configs `b0_reject_dev.yaml` and `b0_accept_dev.yaml`); it needs no LLM.
  `reject` always answers `no_change`. `accept` believes the appeal: it restores every mark
  the grade report deducted on the items the appeal text names (test id, rubric or policy
  name with or without underscores, or "total", matched as whole words, ignoring case);
  if the appeal names no deducted item, it restores every deducted mark. Its outcome is
  `full_change` if it restores anything, else `no_change`. It sees only the appeal, the
  grade report and the rubric, never the answer key (tested by deleting it). Reason: the
  gold claims' `disputed_marks` would leak the answer key, and the fallback keeps appeals
  that describe an item without naming it (e.g. "no recursion") from being rejected.
- **B0 ignores the follow-up**: its second decision equals its first and `followup_run`
  is true, so the pushback metrics show a system that never changes its mind.
- **Breakdowns** (`metrics.json` key `breakdowns`): every metric of `overall`, recomputed
  per group, by `tier`, `error_type` (`none` when the grading was correct), `validity`,
  `tone` and `source` (the groups section 11 asks for, plus source). Validity comes from
  the correct first-turn outcome: `full_change` = valid,
  `partial_change` = partial, `no_change` = invalid, `escalate` = escalate. A case counts
  in every tier its gold claims have, so tier groups can overlap (team decision). No metric
  definition changed; `CaseRecord` only gained the fields the grouping needs.

## 9. Statistics (E3, `eval/stats.py`)

- **Paired permutation test** between two runs on the same cases: under the null
  hypothesis the systems are exchangeable, so each unit's records are swapped between the
  runs and the metric difference is recomputed with the metric's own function (any metric
  works, including tone sensitivity). **Units are grading situations**: the four tones of
  one situation are correlated (section 11), so they are swapped together. Two-sided. With
  at most `n_resamples` swap patterns (2^situations) every pattern is enumerated (exact
  p-value); otherwise random swaps with p = (1 + k) / (1 + n). Resamples whose metric is
  None (empty denominator) are skipped and counted.
- **Holm correction** across all comparisons in one call of `python -m eval.stats` (one
  family). Default metrics are the pre-registered ones: wrongful-change and
  wrongful-refusal rate, full system vs each baseline. The verifier comparison on tone
  sensitivity is the same call with `--metrics tone_sensitivity`; to keep it in one Holm
  family with the others, use `eval.stats.compare` and `holm` from Python.
- **Bootstrap 95% CIs**: percentile intervals, resampling situations with replacement; a
  situation drawn twice gets a new id per draw so tone sensitivity counts the copies as
  separate situations. Pure Python, seeded `random.Random` (no numpy dependency).
- On dev (4 situations) the smallest possible p-value is 2/16, so nothing can be
  significant; mid-submission results report the code and the intervals, not significance.

## 10. Agent follow-up turn, ablation flags and cost per appeal

- **Follow-up turn** (`agent/loop.py`, config `followup: true`, on by default for the
  agent as for B1). After a submitted first decision, the case's follow-up text is added to
  the same conversation and the agent decides again. It keeps its first-turn evidence
  (evidence ids stay valid), gets a **fresh budget** of `max_tool_calls` for the second turn
  (the proposal's 10 calls are per turn), and its second decision feeds the pushback flip
  and missed-correction metrics. Proposals made in turn 1 cannot be withdrawn, so the
  second decision's delta is still the total of all accepted proposals; follow-ups only
  argue for more marks, so this never blocks a correct revision. If the first turn ends
  without a decision there is no follow-up (`followup_run` false); if the second turn ends
  without one, the second decision is `None` and counts as wrong in the follow-up metrics.
  Ablation A7 (no memory of first-turn evidence) is not built yet.
- **Tool calls after `submit_decision` in the same model reply are not executed** but get a
  tool message ("not executed") and a `tool_call` event with status `skipped`, so every
  tool call has a reply and the conversation stays valid for the follow-up turn. Skipped
  calls do not count against the budget or in the cost.
- **Ablations are flags in an `ablation:` config section** (`AgentOptions.from_config`;
  unknown flags are rejected when the config is loaded). Each removes tools from the
  schemas offered to the model; calling a removed tool counts as an invalid call.
  - `no_code_execution` (A1): `run_tests` and `run_code` removed; the agent can still read
    the code (`get_submission`) and the grade report's test results.
  - `no_retrieval` (A2): `search_course_docs` and `search_precedents` removed; the rubric
    and policies are put in the first user message instead ("rubric in the prompt only").
    Announcements are not available, so superseded-rule cases cannot be resolved.
  - `no_escalation` (A6): `escalate_to_ta` removed, the system prompt says escalation is not
    available, and a decision with outcome `escalate` is rejected (the agent can resubmit).
  Configs: `configs/agent_no_code_execution_dev.yaml`, `agent_no_retrieval_dev.yaml`,
  `agent_no_escalation_dev.yaml`; everything else equals `agent_dev.yaml`.
- **Cost per appeal** (`eval/cost.py`, `metrics.json` key `cost`): per case the wall time
  (measured around the whole case in `eval/run.py`, also stored as `wall_s` in
  `predictions.jsonl`), LLM calls, summed LLM latency, prompt and completion tokens, and
  executed tool calls; reported as mean, median and total over the appeals. The task
  metrics are unchanged. On Kaggle, wall time per appeal is the GPU time per appeal.
