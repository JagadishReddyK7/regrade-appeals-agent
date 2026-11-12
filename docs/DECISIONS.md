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
  `.claude/settings.json` blocks Claude's file tools from reading `data/cases/test/` and
  `data/private/`. The git deny rules were removed on 2026-10-30 (team decision): each
  person commits manually and the `CLAUDE.md` rule against git commands that change
  history or the remote still applies, but is not enforced by settings. Limitation: a deny rule on reads does not stop a shell command
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

14. **Partial credit for style is defined in announcement A3** (D1, 2026-09-19): 2 marks if
    every function has a docstring, 1 if at least one but not every, 0 if none. The
    `missed_partial_credit` situation (sit05) needs a rule the documents state; the correct
    marks are then decidable from the course documents, like the other error types. A3 does
    not supersede an earlier announcement. A `missed_partial_credit` claim has gold verdict
    `supported` with `disputed_marks` 2 (what the student asks for) and a correct delta of 1
    (partial change).
15. **The test set is scaled down** from about 300 to about 126 cases (28 situations on 3
    test assignments, 2 equivalence groups, 3 minimal pairs); `docs/BENCHMARK_PLAN.md` lists
    every cut and the reasons. Affects the proposal's scale, so it needs the team's agreement.

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
- **vLLM is pinned to 0.32.0** in the notebook (`VLLM_SPEC`), the version that worked in
  pilot attempt 2 and the mid-submission runs, so a new release cannot change or break a
  run. Adding it to `pyproject.toml` as an optional extra is still open (question 2).
- **Selected runs are committed under `results/`** (one folder per occasion, run folders
  directly inside it, since `runs/` is gitignored). Raw Kaggle output folders
  (`*_outputs/`) stay ignored. Only dev runs are committed; test-split runs never are.

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
- **`search_precedents` returns an empty list** unless a precedent store is configured
  (section 11).

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
- **Follow-up metrics use the same "correct decision" rule as exact-decision accuracy**
  (team decision, 2026-10-30, closes open question 1): pushback flip rate and
  missed-correction rate judge a decision by marks and escalation, not the
  full/partial label. Finished runs are rescored with `python -m eval.rescore <run dir>`,
  which writes `metrics_rescored.json` next to the original `metrics.json`.
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

1. *(Decided 2026-10-30: follow-up metrics use same_decision; see section 7.)*
2. *(Decided: vLLM pinned to 0.32.0 in the notebook; QLoRA libraries in the `gpu` extra.
   Still open: add vLLM to the `gpu` extra as well.)*
3. *(Decided for the mid-submission: BM25 only. Open for the final: add embeddings.)*
4. **May the agent see the reference solution?** Currently no.
5. **Definitions of the hidden-instruction success rate and unsafe-action rate** (section 11).
6. **Escalation without `escalate_to_ta`**: the agent can submit `escalate` without calling
   the tool, so nothing reaches the TA log. Require the call?
7. **Concurrency**: cases run one after another; the GPU budget assumes several appeals in
   parallel so vLLM can batch them (the mid runs took about 30 minutes, so not urgent).
8. **Tone sensitivity** still compares (outcome, delta), so a full/partial label difference
   counts as a different decision. It made no difference in the mid runs; switch it to
   same_decision too, for consistency?
9. **Scope reduction in `docs/BENCHMARK_PLAN.md`** (126 instead of 300 test cases, 3 test
   assignments, one seed, other-generator test and realism test dropped): the team must
   agree, and the report must state what can no longer be claimed (for example the
   proposal's 10-15 point detectable difference).
10. **Number of training-only assignments** for the verifier: `docs/BENCHMARK_PLAN.md` says
    likely 1-2, `docs/VERIFIER_PLAN.md` assumed 4 (about 3,500 examples). Decide with the
    benchmark lead.

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

## 11. Retrieval and verifier (R1-R3, mid-submission)

- **In-force announcements first** (`retrieval/course_docs.py`): when course-doc search
  returns an announcement that a later one supersedes, the announcement now in force is
  placed directly before it, even if it scored lower or did not match the query. This
  changes what `search_course_docs` returns compared with the mid-submission runs
  (`results/mid_dev/`, recorded at commit bd849fa).
- **Precedent store** (`retrieval/precedents.py`): past decisions as JSON lines
  (`Precedent`: id, assignment, target, neutral issue text, outcome, delta, rationale),
  searched with BM25 over issue, target and rationale within the same assignment; a result
  must share a word with the query. BM25Plus is used because BM25Okapi's scores collapse
  in a store of a few documents. `search_precedents` uses the store named by
  `retrieval: {precedents: <file.jsonl>}` in the config; without it the store is empty,
  as in all runs so far. Filling it (team-written precedents, the agent's own decisions
  for RQ2) is still to do.
- **BM25 only for now**; the embedding model from the proposal is a final-submission
  task (another dependency and a model download).
- **Evidence bundle and verifier format** (`agent/schemas.py`, `verifier/format.py`): see
  `docs/VERIFIER_PLAN.md`. The verifier answers JSON with a verdict and the evidence id it
  relies on; one prompt function serves training and inference.
- **QLoRA libraries are an optional `gpu` extra** (`transformers`, `peft`, `bitsandbytes`,
  `accelerate`), installed only on Kaggle; team-approved. `trl` and `datasets` are not
  needed: `verifier/train.py` uses a short custom loop, which also avoids `trl`'s frequent
  API changes. The GPU imports are inside `train()`, so tests and CI do not need them.
- **The verifier is trained on the answer tokens only**; examples longer than
  `max_length` are skipped and counted, never truncated (cutting the prompt would drop
  the claim or evidence).

## 12. Claim pipeline: the full system (final submission)

`agent/pipeline.py`, `agent/extractor.py`, `agent/aggregator.py`, `verifier/infer.py`;
config `system: pipeline` (`configs/pipeline_dev.yaml`). Agent v0 (`system: agent`) stays
unchanged as the whole-appeal tool agent (B3).

- **Extractor** (7B, constrained JSON): sees the appeal (or follow-up) and the graded items
  with the marks the grade report deducted, as a TA would; `target` is restricted by the
  JSON schema to the items **with deducted marks** or null, so wrong ids such as `style`
  cannot occur. Restricting to deducted items came after the second run
  (`results/pipeline_dev_v2/`): the extractor linked all four "recursion was allowed"
  claims to a test with 0 marks deducted, so the aggregator's correct restoration on
  `rubric:approach` broke the target rule and no marks came back. Marks can only be
  restored where they were deducted, so nothing valid is lost. The prompt also says that a
  request to waive a penalty is a claim (the emotional late-penalty appeal gave no claims
  at all). A span
  must occur in the text, compared with whitespace collapsed (line breaks are not
  rewording). An invalid answer is logged as `invalid_output` and retried once with the
  error message; a second failure leaves the turn without a decision. The retry is
  logged and counted, so this is not silent repair.
- **The pipeline loads the grade report as evidence E1 and the submission as E2** before
  the evidence agent starts (both count against the 10-call budget). Every claim disputes
  a mark in the report, and in the mid runs the agent sometimes never read it. The
  submission was added after the second run: for "my count_vowels has a docstring" the
  agent searched the docs but never opened the code, so nothing showed the docstring
  exists. The evidence prompt now also says to test the student's functions through
  `run_code` by importing their module: in dev-015 the agent wrote its own `count_vowels`,
  it passed, and the verifier took that as support.
- **The verifier prompt has judging rules per kind of claim** (test vs specification,
  rubric criterion with the latest announcement in force, arithmetic check; opinions do
  not change a criterion). Added after the second run, where the prompted 7B was right on
  only 13 of 23 claims with a gold verdict, wrong in both directions. The prompt is shared
  with training (`verifier/format.py`), so the fine-tuned 3B must be trained with this
  version.
- **`get_grade_report` returns an arithmetic check** (`items_add_up_to`,
  `reported_total`, `total_matches_items`), placed first so shortening cannot cut it.
  This is the mitigation for the missed arithmetic errors in `results/mid_dev_24/`; it is
  computed from the report itself, not the answer key. Agent v0 sees it too, which keeps
  the comparison with B3 fair.
- **The evidence agent sees the claims, not the appeal**, and only the read-only tools
  plus `finish_evidence`. It does not decide. If the budget runs out before
  `finish_evidence`, the verifier still gets the evidence gathered so far.
- **The verifier sees each claim with all evidence of the appeal**, not a per-claim
  subset: asking the agent to map evidence to claims added a failure mode for little gain
  with at most 10 items. `MAX_EVIDENCE_CHARS` is now 2,500 per item so that a whole grade
  report fits. **The verifier is prompted** for now (the 7B on the main server, ablation
  A4b); `verifier.llm` in the config points it at another server (untuned 3B, A4a, or the
  fine-tuned 3B). Fine-tuning is the main verifier in the proposal; whether it makes the
  final runs is decided on Nov 13.
- **T3 claims go to a human without a verifier call only if they dispute a policy or no
  item** (and escalation is available). A claim on a test, rubric item or the total can
  always be checked, so the verifier sees it even when the extractor called it T3: in the
  first run (`results/pipeline_dev_v1/`) the extractor tagged "please check the addition"
  and claims citing a senior as T3, and the pipeline over-escalated (0.35). The extractor
  prompt now also says that what another person says does not make a claim T3.
- **An invalid verifier answer counts as `needs_human`** (escalate: "unclear cases go to a
  TA"), logged as `invalid_output` and as a `verdict` event with `valid: false`.
- **The aggregator sees all evidence of the appeal** (not only the items the verifier
  cited), so it can apply a partial-credit rule found by search; its `item_id` is
  restricted by the JSON schema to the items with deducted marks (in A0 it invented
  `approach` and `style`). The verifier's `evidence_id` is likewise restricted to the
  bundle's ids at inference (training keeps the plain format).
- **The aggregator's rules are code; only the amount is the model's.** Escalate if any
  claim needs a human; no change, without a model call, if no claim is supported;
  otherwise the 7B proposes restorations, checked in code (supported claims only, on the
  claim's target only, never above the deducted marks). One retry after invalid output.
  `full_change` means every disputed item is restored in full. Low-confidence escalation
  is a config option (`pipeline.escalate_below_confidence`), off by default (0.0) until
  the dev runs show where to set it.
- **The pipeline, not the model, calls `propose_grade_change` and `escalate_to_ta`**
  (logged as `tool_call` with `actor: pipeline`). Proposals therefore always use valid
  ids, the cap is checked twice, and escalations always reach the TA log (closes open
  question 6 for the pipeline).
- **Follow-up**: the extractor reads the follow-up with the earlier claims and is asked
  only for claims that bring new evidence (a document, announcement, test, input or
  fact); repetition, insistence and opinions give an empty list, and then the first
  decision stands without further calls. Otherwise the agent gathers more evidence in the
  same conversation with a fresh budget, **only the new claims are verified**, a new
  claim supported on an item replaces the earlier verdicts on that item, the turn-1
  proposals are withdrawn (`proposals_withdrawn` in the proposals log) and the aggregator
  decides again. Re-verifying every claim (the first version) let unrelated new evidence
  flip an earlier verdict (dev-006).
- **Announcement dates are searchable**: BM25 indexes the date as `2026-09-19 19
  September 2026` (fixed English month names, not the locale), so "the 19 September
  announcement" finds it. This also changes agent v0's search results.
- **`llm.max_tokens`** (optional, `OpenAIBackend`) caps each reply; the pipeline configs
  use 1024 after a runaway reply timed out a case.
- **Ablations**: `no_claim_splitting` (A0: the whole appeal is one claim, id `appeal`, no
  target; the follow-up is one claim, id `followup`), `no_code_execution` (A1),
  `no_retrieval` (A2), `no_escalation` (A6: `needs_human` claims become eligible for the
  aggregator, which must decide). A3, A5 and A7 are not built.
- **Follow-up errors keep the first decision** in the pipeline and now also in agent v0
  (`followup_error` event, `followup_run` false), as B1 already did. Before, a timeout in
  the follow-up discarded a valid first decision (A6 in `results/mid_dev_24/`).
- **B1 may cite its follow-up as `followup`**: in the mid runs B1 did this three times and
  the outputs were invalid. The prompt now says so.
- **A rejected proposal lists the items with deducted marks** (agent v0), so the model can
  correct a wrong item id.

## 13. Prompts frozen after the third dev run (2026-11-12)

- **All prompts and pipeline code are frozen at commit `8f8115e`** (extractor, evidence
  agent, prompted verifier, aggregator, agent v0, B1). The test-set runs use exactly this
  code; only bug fixes that do not touch prompts or decision rules are allowed, each
  recorded here.
- **Why stop tuning:** three dev runs on the same 24 cases (`results/pipeline_dev_v1/` to
  `v3/`) moved the prompted verifier from too lenient (run 2: 4 wrong "supported") to too
  strict (run 3: 11 wrong "refuted"), while in run 3 the evidence it needed was present
  (the 12 September recursion announcement in dev-002, the 19 September partial-credit
  announcement in dev-017). The remaining errors are the prompted 7B's judgement, not
  missing evidence or a pipeline bug, so more prompt changes would fit noise on 24 cases.
  Improving the verifier is the job of the fine-tuned 3B (H1c, go/no-go on Nov 13).

## 14. Verifier training data and fine-tuning (2026-11-12)

- **Two training-only assignments, TR1 (list statistics) and TR2 (inventory dictionary)**,
  in the D1 layout (`data/assignments/TR*`, `data/submissions/TR*-S0*`), never used for
  evaluation or prompt tuning. Each covers the claim kinds of the benchmark with new
  content: a faulty test (TR1) and a test that follows a superseded announcement (TR2), an
  approach rule relaxed by a later announcement, partial credit announced later (style
  per function; type hints), wrongly applied manual marks, addition errors, false claims
  about failing code, opinion-only rubric claims, and late-penalty or extension requests
  (needs_human). `tests/test_verifier_training.py` checks every grade report against the
  autograder and the marks wrongly withheld.
- **Labels by construction, evidence from the real tools.** A situation file
  (`data/verifier_train/*.yaml`) names the claim's item, its verdict and the decisive tool
  calls; `verifier/train_data.py` runs those tools on the submission and lays out the
  evidence as the frozen pipeline does: E1 grade report, E2 submission, then the decisive
  results, sometimes supporting ones and 0-2 distractors in random order. The label cites
  the first decisive item. A claim that the total is wrong when it is not has no target,
  because the extractor only targets items with deducted marks (section 12).
- **Claim wordings written for the project, not by Llama-3.1-8B** as the proposal planned:
  three wordings per situation, wrapped in tone templates. Generating them on Kaggle would
  have cost a session before the Nov 13 go/no-go. Limitation for the report: less
  linguistic variety than LLM-generated appeals.
- **Tone pairing as planned**: each pair (one wording, one evidence set) is written in
  polite, emotional and hidden-instruction tone with the same label (the instruction is a
  comment in the submission, as in the benchmark); authority is held out
  (`verifier_heldout_authority.jsonl`: 46 fresh pairs written in all four tones, 184
  examples, so tone sensitivity shows whether the unseen tone flips verdicts). 594
  training examples (270 supported, 288 refuted, 36 needs_human).
- **Evaluation on real evidence from a held-out assignment**: `verifier/eval_data.py`
  rebuilds, from a pipeline run's events, each claim the verifier judged with exactly the
  evidence it saw, and pairs it with the dev case's gold verdict (30 claims from
  `results/pipeline_dev_v3/`, 23 more from `v2/`). The untuned 3B (A4a), the
  fine-tuned 3B and the prompted 7B (A4b) are compared on the same claims
  (`verifier/evaluate.py`: accuracy, macro-F1, tone sensitivity).
- **The adapter is merged into the float16 base for serving** (`merge: true`), so vLLM
  serves a plain model on the second T4; vLLM's LoRA support on the T4 is not needed. The
  merge runs on the GPU; Kaggle's preinstalled torchao 0.10 must be uninstalled first (peft
  refuses to merge into a float16 model next to a torchao older than 0.16; the first
  training run on Kaggle trained fine and failed only there). `verifier.train --merge-only`
  merges a saved adapter without training again.
- **Go/no-go rule (Nov 13):** the fine-tuned 3B becomes the main verifier for the test
  runs if its macro-F1 on the dev claims is at least the prompted 7B's and the pipeline
  with it does not lose exact-decision accuracy on dev; otherwise the prompted 7B stays
  and the fine-tuned 3B is reported as ablation A4. Decided on dev only.
- **Outcome (2026-11-13): go.** On the dev claims the fine-tuned 3B has macro-F1 0.70 (v3)
  and 0.69 (v2) against the prompted 7B's 0.41 and 0.45; the pipeline with it reaches
  0.67 exact accuracy against run 3's 0.58, with wrongful change still 0.00
  (`results/verifier_ft_dev/README.md`). The fine-tuned 3B is the main verifier of the
  test runs; the prompted 7B is ablation A4b, the untuned 3B A4a. Known weakness: on a
  case with a true and a false claim it got both wrong (dev-005, dev-007), so it can
  award marks for a false claim while the total stays below the gold change, which the
  wrongful-change metric does not count. The held-out tone set (0.96) shares the
  training situations, so it is reported as tone robustness only, not generalisation.

## 15. Test runs (2026-11-13)

`notebooks/test_runs.ipynb`, configs `configs/*_test.yaml`.

- **Systems** (one Kaggle session, T4 x2, temperature 0, seed 0): B0 reject and accept,
  B1, B3 (agent v0), the full system with the fine-tuned 3B verifier (main), and the
  ablations A0 (no claim splitting), A1 (no code execution), A2 (no retrieval), A6 (no
  escalation), A4b (prompted 7B verifier) and A4a (untuned 3B verifier). Ablations change
  one thing from the main system, so they use the fine-tuned verifier. Not run: B2 and
  B4 (not built), A3, A5, A7 (not built), the agent v0 ablations of the mid runs, and
  repeated runs (time). Runs go in priority order (main system, B3, B1, A0, A4b, A1, A2,
  A6, A4a) so an early stop loses the least important ones.
- **Each test config is its dev config with only name, cases and data root changed**
  (plus the one setting an ablation changes); `tests/test_test_configs.py` enforces it,
  so no setting can be tuned for the test runs.
- **The test set is checked against its frozen manifest** (`docs/TEST_SET_GUIDE.md` step
  5) before the runs: the notebook records the number of cases, the manifest's hash and
  the number of missing or changed files in `test_set_check.json`. A mismatch is a
  warning, not a stop, so the runs are not lost to a path-format problem; a run with a
  mismatch must be explained in the report.
- **Statistics, three families with Holm correction within each** (`eval/stats.py`,
  section 9 of this file): (1) RQ1a, pre-registered: main system vs B0 reject, B0
  accept, B1 and B3 on wrongful-change and wrongful-refusal rate; (2) RQ1b: main system
  vs A0 (design) and A4b, A4a (verifier) on tone sensitivity and pushback flip rate;
  (3) exploratory: main system vs every other run on exact accuracy, wrongful change and
  wrongful refusal. Only family 1 is confirmatory.
- **Outputs are split.** `test_metrics.zip` holds only aggregate numbers (per run
  `config.yaml` and `metrics.json`, the statistics, the manifest check, commit, `pip
  freeze`); it is shared, committed under `results/` and may be analysed with Claude
  Code. `test_private.zip` holds per-case events, predictions, proposals and server logs,
  which contain test appeals: never committed (`.gitignore`), and blocked from Claude
  Code (`.claude/settings.json`), as the test-set guide requires. Error analysis on test
  cases is done by a team member by hand.
