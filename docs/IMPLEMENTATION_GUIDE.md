# Implementation guide: Claude Code in VS Code + GitHub + Kaggle

## How the pieces fit
- **VS Code + Claude Code**: write and test code locally (CPU only, mock LLM).
- **GitHub (public repo)**: the single source of truth for code, data formats, decisions
  and status. CI runs on every push and pull request.
- **Kaggle (T4 GPUs)**: notebooks clone the repo and run GPU jobs (vLLM serving,
  fine-tuning, evaluation runs). Results are saved as Kaggle outputs; summaries go into
  `docs/STATUS.md`.
- **Private store (Kaggle dataset or team drive)**: the test split, never on GitHub.

Claude Code sessions do not share memory. What one person's session learns is lost to
everyone else unless it is written into the repo: decisions into `docs/DECISIONS.md`,
progress into `docs/STATUS.md`. Each PR updates them.

---

## Step 1: Set up your machine (each teammate, once)

1. Accept the GitHub invitation, then:
   ```
   git clone https://github.com/JagadishReddyK7/regrade-appeals-agent.git
   cd regrade-appeals-agent
   python -m venv .venv
   .venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
   pip install -e ".[dev]"
   python -m pytest
   ```
2. Open the folder in VS Code and the Claude Code panel. `CLAUDE.md` and
   `.claude/settings.json` come with the repo: Claude Code reads the project rules, cannot
   run git commands that change anything, and its file tools cannot open the private test
   paths. Nothing else to set up. Only the evaluation machine should ever hold test data.
3. Optional personal settings go in `.claude/settings.local.json`; keep it out of Git:
   - Git Bash, macOS, Linux: `printf ".claude/settings.local.json\n.local/\n" >> .git/info/exclude`
   - Windows PowerShell: `Add-Content -Path .git\info\exclude -Value ".claude/settings.local.json", ".local/" -Encoding ascii`
4. Read `docs/STATUS.md`, `docs/DECISIONS.md` and (if you write data) `docs/DATA_FORMAT.md`.

## Step 2: Repository administration (repo owner, once)

1. Settings -> Collaborators: add the three teammates.
2. Settings -> Branches -> add a rule for `main`: require a pull request with 1 approval,
   and require status checks to pass: select `test (3.11)` and `test (3.13)` (they appear
   in the list after CI has run once).
3. Settings -> General -> Pull Requests: allow only **squash merging**, and tick
   "Automatically delete head branches".
4. Issues: create a label `decision`, then one issue per task in `docs/STATUS.md`
   (owners assigned at the kickoff).
5. Private test store: a private Kaggle dataset (shared with the team) holding
   `assignments/`, `submissions/` and `cases/test/` in the layout of
   `docs/DATA_FORMAT.md` ("Private data").

## Step 3: The loop for every task

1. Create a branch: `git checkout main && git pull && git checkout -b feat/<task>`
2. In Claude Code, start in **plan mode**, paste the task prompt (Step 6), review and
   comment on the plan, then approve.
3. Claude writes code and tests. Review every diff yourself; run `python -m pytest`.
4. Claude ends with a copy-paste block of git commands. Check the file list (nothing from
   `data/cases/test/` or `data/private/`, no secrets, no `.claude/settings.local.json`),
   then run it.
5. Open a pull request. CI must pass; a teammate reviews and merges.
6. In the same PR: add any design decision to `docs/DECISIONS.md`, and update
   `docs/STATUS.md` if a milestone-level item finished. Track progress in the issue.

## Step 3b: Working as a team

**Where things live (one source of truth each)**
| What | Where |
| --- | --- |
| Tasks, owners, progress, discussion of a task | GitHub Issues (one per task; link PRs with "Closes #n") |
| Milestone-level status, known limitations | `docs/STATUS.md` (update when a milestone item finishes) |
| Design decisions, open questions | `docs/DECISIONS.md` |
| How the code fits together | `docs/ARCHITECTURE.md` |
| Data formats | `docs/DATA_FORMAT.md` |
| Test split, blind cases | private Kaggle dataset only |

**Branches and pull requests**
- One task per branch and PR. Keep PRs small (roughly under 400 changed lines of code,
  data excluded); split big tasks into several PRs. Open a draft PR early if others
  depend on your interfaces.
- Bring `main` into your branch at least daily, and always before opening a PR:
  `git fetch origin` then `git merge origin/main`. Claude Code cannot run these
  commands; you run them yourself.
- Merge conflicts: resolve them in VS Code's merge editor, run `python -m pytest`, then
  commit. Files that almost every task touches and therefore conflict most:
  `agent/schemas.py`, `eval/run.py`, `docs/DECISIONS.md`, `docs/STATUS.md`. Add new
  schema models and decisions at the end of the relevant section, and keep unrelated
  edits to these files out of your PR.
- Merge with **squash and merge** so `main` has one commit per PR. The author merges
  after approval and green CI, then deletes the branch.

**Reviews**
- Every PR needs one approval. Review within one working day; if you cannot, say so in
  the PR so someone else takes it.
- Review against the checklist in the PR template (it appears automatically). Look in
  particular for answer-key leakage, test data, missing tests and unrecorded decisions.
- Ask questions in the PR; small fixes can go into the same PR.

**Decisions**
- Inside a role's area (e.g. data format details for the benchmark lead), the role lead
  decides, records it in `docs/DECISIONS.md` and mentions it at the next sync.
- Anything that affects other roles, the metrics, the benchmark's validity or the
  proposal's research questions is decided by the team: raise it as an issue labelled
  `decision`, settle it at the weekly sync (majority if needed), then record it.
- Open questions live at the end of `docs/DECISIONS.md` until decided.

**Communication**
- A weekly sync (30 min): issue board, blockers, decisions, next week's GPU plan.
- A team chat for quick questions; anything that matters later goes into an issue or
  `docs/DECISIONS.md`, not only the chat.

**Kaggle GPU time**
Each account has about 30 GPU hours a week; the proposal estimates 55-75 GPU hours in
total. Plan the week's GPU jobs at the sync, run each job type from one owner's account,
and note GPU hours used in the job's issue. Suggested split: agent lead runs pilots and
agent dev runs; fine-tuning lead runs data generation and verifier training; evaluation
lead runs the main evaluation and ablations (one account and one environment for all
main runs, so results are comparable).

## Kickoff agenda (first team meeting)
1. Walk through `docs/STATUS.md`, `docs/ARCHITECTURE.md` and the pilot results.
2. Assign the four roles (proposal section 14) and issue owners.
3. Agree the sync day, the chat channel and the GPU split.
4. Decide or assign the open questions in `docs/DECISIONS.md`.
5. Decide, or assign by week 2: the labelling protocol (two independent labellers,
   Cohen's kappa, dropping disagreements), the sources of openly licensed course material
   (recorded with their licences), and the test-freeze procedure (for example a manifest
   of file hashes committed at freeze time, checked before every main run).
6. Confirm the evaluation lead writes the blind cases before reading agent code or prompts.

## Step 4: Working with Claude Code efficiently (Pro has usage limits)

- One task per session; clear the context between unrelated tasks.
- Point it at the docs: every prompt starts with "Read CLAUDE.md, docs/STATUS.md and
  docs/DECISIONS.md" (and docs/ARCHITECTURE.md for code changes).
- Never paste test-split data or blind cases into a Claude conversation.
- Ask it to check its work against `docs/DECISIONS.md` before you approve a plan.
- Always ask for tests. You are responsible for the code in the report.

## Step 5: Kaggle

- Notebook settings: Accelerator **GPU T4 x2**, Internet **on**. The repo is public, so
  no token is needed to clone it.
- Attach the private test dataset only to evaluation notebooks, never to development runs.
- Keep notebooks thin: clone, install, call `python -m ...` scripts. All logic lives in the repo.
- Save outputs under `/kaggle/working`; record results and versions in `docs/STATUS.md`.

---

## Step 6: Prompts

### Done (up to the mid-submission)
Also done since: A2 (follow-up turn), A3 (cost per appeal), ablation flags A1/A2/A6, B2
(case builder), E2 (B0 and breakdowns), E3 (statistics), R1-R3 (evidence bundle, precedent
store, QLoRA smoke test). Their prompts below are kept for reference; do not re-run them.
The open items are A1 (claim extractor), A4 (concurrency) and the later prompts; the
current priorities are in `docs/STATUS.md`.

### Done (week 1, P0-P6)
Scaffold, case schema and validator, sandbox, LLM client, tools and agent v0, B1 and
metrics, Kaggle pilot notebook. Several prompts were adjusted while implementing (for
example, the proposal cap uses the grade report's deductions instead of the gold claims'
disputed marks); `docs/DECISIONS.md` records each change. Do not re-run these prompts.

### Next tasks (ids match `docs/STATUS.md`)

Every prompt below assumes the preamble: *Read CLAUDE.md, docs/STATUS.md and
docs/DECISIONS.md. Show me the plan first.*

**0. Pilot (no prompt).** Run `notebooks/pilot_dev.ipynb` on Kaggle. Record in
`docs/STATUS.md`: the three smoke-test results, the vLLM version, errors, and time and
tokens per appeal from `events.jsonl`. If vLLM does not start on the T4 (support for
compute capability 7.5 in vLLM's newer engine is not confirmed), try in this order: pin an
older vLLM release in `VLLM_SPEC`; lower `max_model_len`; a GPTQ-Int4 build of the model.
If tool calling fails the smoke test, the proposal's fallback is Qwen3-8B. Record what
worked in `docs/DECISIONS.md`.

**A1. Minimal claim extractor [feat/extractor]**
```
Implement agent/extractor.py: Qwen2.5-7B (via llm/client.py) turns an appeal into claims
with a verbatim span, target (test id, rubric:..., policy:... or total), tier T1/T2/T3 and
disputed marks, as schema-constrained JSON validated with a model in agent/schemas.py.
The extractor sees only the appeal, the rubric and the grade report, never gold fields.
Add eval/extraction.py: claim precision and recall against gold claims (a predicted claim
matches a gold claim if the targets are equal and the spans overlap), tier accuracy, and
a config + runner so it can run on data/cases/dev. Tests with MockLLM.
```

**A2. Agent follow-up turn [feat/agent-followup]**
```
Add the follow-up (pushback) turn to agent/loop.py: after the first decision, the case's
followup.text is added to the conversation and the agent continues with a fresh call
budget, keeping its first-turn evidence (ablation A7 will switch that memory off through a
config flag). Return both decisions in SystemOutput so pushback_flip_rate and
missed_correction_rate work for the agent. Tests with MockLLM: a pressure-only follow-up,
and one with new evidence.
```

**A3. Cost per appeal [feat/cost-metrics]**
```
Aggregate time and tokens per appeal from the run's events: wall time per case, LLM
latency, prompt and completion tokens, tool calls. Add them to metrics.json (mean, median,
total) without changing the existing metric definitions. Tests on a small hand-made
events file.
```

**A4. Concurrent cases [feat/concurrency]**
```
Let eval/run.py run several cases at once (config: concurrency, default 1) with a thread
pool, so vLLM can batch requests. Make RunLogger writes thread-safe, keep predictions in
case order, and keep results identical to a sequential run. Tests with MockLLM and
concurrency 4.
```

**B2. Case-building tool [feat/case-builder]**
```
Implement data/build_cases.py: from one "situation" YAML (assignment, submission, planted
error, claims with targets/tiers/marks/verdicts, answers, follow-ups) and four appeal
texts (one per tone), write the four case JSON files with consistent ids, situation_id and
claim spans, creating the hidden-instruction submission variant where needed. Then run the
validator on the output. Tests: a built situation passes data.validate.
```

**R1+R2. Evidence bundle and retriever [feat/retriever]**
```
Define the evidence bundle the verifier will receive (a pydantic model in
agent/schemas.py: claim as written plus a list of evidence items with id, source tool and
text) and record the format in docs/DECISIONS.md. Then implement retrieval/precedents.py:
a store of past decisions (assignment, target, issue, outcome, delta) with add/search, and
wire search_precedents to it. In retrieval/course_docs.py, rank an in-force announcement
above the announcement it supersedes. Embeddings are optional and need approval as a new
dependency. Tests: the in-force announcement ranks first; precedent search returns the
matching decision.
```

**R3. QLoRA smoke test [feat/verifier-smoke]**
```
Create verifier/train.py: QLoRA fine-tuning of Qwen2.5-3B-Instruct for 3-way
classification (supported/refuted/needs_human) from JSONL in the evidence bundle format.
Use float16 (T4 has no bfloat16), 4-bit loading, config-driven hyperparameters. The QLoRA
libraries are a new dependency: propose an optional "gpu" extra and ask first. Add
notebooks/verifier_smoke.ipynb that trains a few steps on 50 dummy examples on a Kaggle T4.
```

**E2. B0 and breakdowns [feat/b0-breakdowns]**
```
Implement baselines/b0.py: always reject (no_change), and always accept. Decide and
record in docs/DECISIONS.md what "accept" restores without access to gold claims (for
example, every mark the grade report deducted on items the appeal text mentions by test id
or rubric id). Then add result breakdowns to
metrics.json by tier, planted error type, validity (valid/partial/invalid/escalate) and
source, without changing metric definitions. Tests on hand-made records.
```

**E3. Statistics [feat/statistics]**
```
Implement eval/stats.py: paired permutation tests between two runs on the same cases,
Holm correction across the pre-registered comparisons (proposal section 11), and bootstrap
95% confidence intervals resampled by situation_id. Tests with fixed seeds on small
synthetic data.
```

### Later (expand when you get there)
- Benchmark generation with Llama-3.1-8B (appeals from planted claims), the other-generator
  test set, realism test, double labelling with Cohen's kappa.
- Verifier training data from the 4 training-only assignments: tone-paired, authority tone
  held out.
- Full pipeline: extractor -> evidence agent -> verifier -> aggregator -> pushback turn.
- Baselines B2 (majority vote), B3 (ReAct), B4 (blind regrade); ablations A0-A7 as config flags.
- Consistency metrics; precedent runs (3 orders) and the planted wrong precedent.
- Safety metrics (after definitions), tamper detection by hashes, capability metrics
  (citation accuracy, retrieval recall), LLM judge (Phi-3-medium) with human validation.

## Milestones
| Week | Done when |
|---|---|
| 1 | Pilot run; A1 for go/no-go; formats agreed; go/no-go decided |
| 2-3 | Benchmark built and double-labelled; blind cases written; verifier training data generated |
| 3-4 | Full system, verifier trained, baselines B0-B4 working on dev |
| 5 | Test set frozen; main evaluation runs complete |
| 6-8 | Ablations, analysis, report |
