# Regrade Appeals Agent: project guide for Claude Code

## What this project is
Research prototype for the LMA Group Project 2026 (Team Attention Seekers).
An agent handles regrade appeals for Python programming assignments. It splits each
appeal into claims, checks each claim against evidence (running the student's code,
retrieving rubric/announcements/precedents), and changes the grade only when the
evidence supports it, escalating unclear cases to a TA. The full proposal is in
`docs/proposal.pdf`; treat it as the specification.

Research questions:
- RQ1a: does claim-by-claim evidence checking reduce wrongful grade changes without
  increasing wrongful refusals?
- RQ1b: are decisions robust to tone and follow-up pushback, and does robustness come
  from design (claim splitting) or training (tone-paired fine-tuned verifier)?
- RQ2: are equivalent appeals decided consistently, and does precedent retrieval help
  or propagate errors?

## Read before starting any task
- `docs/STATUS.md`: what is built, what is next, who owns what.
- `docs/DECISIONS.md`: design decisions and interpretations of the proposal. Where the
  code differs from the proposal or an old prompt, this file explains why. Do not undo a
  recorded decision without the team agreeing; add new decisions here in the same PR.
- `docs/DATA_FORMAT.md`: assignment, submission, grade-report and case formats.

## Architecture (see docs/proposal.pdf, Figure 1)
1. Claim extractor (Qwen2.5-7B-Instruct, 4-bit, via vLLM): appeal -> claims with
   original text span, linked test/rubric item, tier (T1 code, T2 rubric/policy, T3 human).
2. Evidence agent (same 7B): tool loop, max 10 tool calls per appeal.
3. Retriever: BM25 + small embedding model over course docs (dated announcements)
   and a precedent store.
4. Claim verifier (Qwen2.5-3B-Instruct + QLoRA): claim as written + evidence ->
   supported / refuted / needs_human.
5. Aggregator (7B): verdicts -> mark change (capped at disputed marks) or escalate.
6. One follow-up pushback turn; revise only on genuinely new evidence.

Tools: get_submission, get_grade_report, run_tests, run_code, search_course_docs,
search_precedents, propose_grade_change, escalate_to_ta, submit_decision.

## Repository layout
- `data/` assignments, submissions, cases/{dev,train}; `data/validate.py`,
  `data/grader.py` (reference grader), `data/consistency.py`, `data/formats.py`
- `sandbox/` sandboxed execution, test runner, result cache
- `llm/` model client (OpenAI-compatible for vLLM, plus a mock backend), vLLM launcher,
  smoke test
- `agent/` schemas, tools, tool-calling loop (extractor, aggregator to come)
- `retrieval/` course-doc index (precedent store to come)
- `verifier/` training-data generation, QLoRA training, inference
- `baselines/` B0-B4 (B1 done)
- `eval/` metrics, run log, evaluation runner (statistics and LLM judge to come)
- `configs/` YAML configs; every ablation (A0-A7) is a config flag, not a code fork
- `notebooks/` Kaggle notebooks (thin: clone repo, install, call scripts)
- `docs/` proposal, status, decisions, data format, implementation guide
- `tests/` pytest

## Environment
- Local development is CPU-only (VS Code). Use the mock LLM backend for tests.
- All GPU work runs on Kaggle T4 (16 GB): vLLM serves models with an OpenAI-compatible API.
  T4 has no bfloat16: use float16 everywhere.
- Code must run on Python 3.11 (Kaggle). Developing on a newer Python is fine; ruff
  targets py311 and CI tests 3.11 and 3.13. Dependencies in `pyproject.toml`.
- Windows: write files with `encoding="utf-8"` explicitly; the default encoding there
  cannot write characters such as the Greek letter delta.

## Commands
- `make test` (or `python -m pytest`) run tests
- `make lint` (or `python -m ruff check .`) run ruff
- `python -m data.validate data/cases/dev` validate cases and run the reference grader
- `python -m eval.run --config configs/<name>.yaml` run an evaluation
- `python -m llm.serve --config <config>` start vLLM; `python -m llm.smoke --config <config>`
  check chat, tool calling and constrained JSON against the server

## Rules (important)
- Test-split data is PRIVATE: never read, print, summarise or use `data/cases/test/` or
  `data/private/` (test assignments, submissions and answer keys). Develop and tune
  prompts only on `data/cases/dev/`. This protects the validity of the evaluation.
- Never expose the answer key to a system under evaluation: tools and prompts must not
  read `reference_grade.json`, the reference solution, or gold fields of a case
  (claims, planted_error, answer, followup.answer_after).
- Do not change metric definitions in `eval/metrics.py` without explicit instruction.
- Every LLM call, tool call, retrieval, verdict and decision must be logged as JSONL
  (one record per event) with a run id, case id, seed and config hash.
- Fix random seeds; cache code-execution results keyed by (submission hash, test id, input).
- Student code is untrusted: execute only through `sandbox/`, never with eval/exec directly.
- The agent may only PROPOSE grade changes; rubrics, tests and grade records are read-only.
- All structured outputs (claims, verdicts, decisions, tool calls) are validated with
  pydantic models in `agent/schemas.py`; invalid output is logged and counted, not silently fixed.
- Keep functions small and typed; add or update tests with every change.
- Ask before adding a new dependency.

## Git: each person commits and pushes manually
- NEVER run git commands that change history or the remote: no `git add`, `git commit`,
  `git push`, `git pull`, `git merge`, `git rebase`, `git reset`, `git checkout`,
  `git switch`, `git stash` or `git tag` (also enforced by `.claude/settings.json`,
  which also blocks file-tool reads of the private test paths; shell reads are not
  blocked, so never `cat`, `grep` or otherwise open them through Bash either).
  Read-only commands (`git status`, `git diff`, `git log`, `git show`) are fine.
- Team workflow: one branch and one pull request per task; `main` is protected and
  needs passing CI and one review. At the START of each task, tell the user which branch
  to create, e.g. `git checkout main && git pull && git checkout -b feat/<task>`.
- At the END of each task, after tests pass, give the user the exact commands to run,
  as one copy-paste block: list the files to stage explicitly (never `git add .`),
  a concise commit message in the imperative mood, and
  `git push -u origin feat/<task>`. Remind the user to open a pull request.
- If a task is large, suggest intermediate commits. Each commit must pass tests and lint
  on its own: check that every file a commit's code imports is in that commit.
- Never stage or suggest committing anything under `data/cases/test/` or `data/private/`,
  secrets, `.env` files, model weights or caches.
- `.claude/settings.local.json` and `.local/` are personal and excluded through
  `.git/info/exclude`; never stage them.
