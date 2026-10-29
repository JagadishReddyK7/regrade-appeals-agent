# Regrade Appeals Agent

Research prototype for the LMA Group Project 2026 (Team Attention Seekers). The agent
handles regrade appeals for Python programming assignments. It splits each appeal into
claims and checks every claim against evidence: it runs the student's code and retrieves
the rubric, course announcements and precedents. It proposes a grade change only when the
evidence supports one, and escalates unclear cases to a TA.

## Start here

| Document | What it is for |
| --- | --- |
| `docs/proposal.pdf` | the specification |
| `docs/STATUS.md` | what is built, what is next (tasks and owners: GitHub Issues) |
| `docs/ARCHITECTURE.md` | how the code fits together and how to extend it |
| `docs/DECISIONS.md` | design decisions and open questions; read before changing anything |
| `docs/DATA_FORMAT.md` | how to write assignments, submissions and cases |
| `docs/VERIFIER_PLAN.md` | claim verifier: design, training-data plan, timeline |
| `docs/IMPLEMENTATION_GUIDE.md` | setup, team workflow and rules, Claude Code usage, prompts for next tasks |
| `CLAUDE.md` | rules Claude Code follows in this repo (useful for humans too) |

## Setup

Requires Python 3.11 or newer; code must stay compatible with 3.11 (CI tests 3.11 and
3.13; Kaggle currently runs 3.13).

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest                # or: make test
python -m ruff check .          # or: make lint
```
On Windows two sandbox tests are skipped (CPU and memory limits need Linux); CI runs them.

## Commands

| Command | Does |
| --- | --- |
| `python -m data.validate data/cases/dev` | validate cases, their data and the reference grader |
| `python -m eval.run --config configs/b1_dev.yaml` | run a system on a case folder; writes `runs/<run id>/` |
| `python -m llm.serve --config configs/b1_dev.yaml` | start a vLLM server (GPU, Kaggle) |
| `python -m llm.smoke --config configs/b1_dev.yaml` | check chat, tool calling and constrained JSON on the server |

A run directory contains `config.yaml`, `events.jsonl` (every LLM call, tool call and
decision), `proposals.jsonl`, `predictions.jsonl` and `metrics.json`.

GPU work runs on Kaggle: `notebooks/pilot_dev.ipynb` clones this repo, starts vLLM and
runs the dev configs.

## Layout

| Path | Contents |
| --- | --- |
| `agent/` | schemas, tools, tool-calling loop |
| `baselines/` | baselines (B1 so far) |
| `configs/` | YAML configs; ablations A0-A7 will be config flags |
| `data/` | dev assignment, submissions and cases; formats, reference grader, validator |
| `eval/` | metrics, run log, evaluation runner |
| `llm/` | model client, vLLM launcher, smoke test |
| `notebooks/` | thin Kaggle notebooks |
| `retrieval/` | course-document search, precedent store |
| `sandbox/` | sandboxed code execution with limits and a result cache |
| `verifier/` | verifier prompt format, dummy data, QLoRA training (`pip install -e ".[gpu]"`, Kaggle) |
| `tests/` | pytest suite |

## Private test data

The test split (cases, and the assignments, submissions and answer keys they use) is never
committed: it lives in a private Kaggle dataset or on the team drive (locally under
`data/private/`, which is gitignored). Develop and tune only on `data/cases/dev/`.
