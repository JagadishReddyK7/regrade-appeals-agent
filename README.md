# Regrade Appeals Agent

Research prototype for the LMA Group Project 2026 (Team Attention Seekers). The agent
handles regrade appeals for Python programming assignments. It splits each appeal into
claims and checks every claim against evidence: it runs the student's code and retrieves
the rubric, course announcements and precedents. It proposes a grade change only when the
evidence supports one, and escalates unclear cases to a TA.

The full specification is in `docs/attention_seekers_final_proposal.pdf`.

## Layout

| Path | Contents |
| --- | --- |
| `data/` | assignments, submissions, cases (`dev`, `train`, `test`) |
| `sandbox/` | sandboxed execution, test runner, result cache |
| `llm/` | model client (OpenAI-compatible for vLLM, plus a mock backend) |
| `agent/` | claim extractor, evidence loop, aggregator, tools, schemas |
| `retrieval/` | course-doc index, precedent store |
| `verifier/` | training-data generation, QLoRA training, inference |
| `baselines/` | baselines B0-B4 |
| `eval/` | metrics, statistics, LLM judge, reports |
| `configs/` | YAML configs; ablations A0-A7 are config flags |
| `notebooks/` | thin Kaggle notebooks |
| `tests/` | pytest suite |

## Setup

Requires Python 3.11 or newer. Code must stay compatible with 3.11, the target on Kaggle.

```
pip install -e ".[dev]"
make test   # or: python -m pytest
make lint   # or: python -m ruff check .
```

Local development is CPU-only and uses the mock LLM backend. GPU work (vLLM, QLoRA)
runs on Kaggle T4 GPUs.

## Data

`data/cases/test/` is private and gitignored. Do not commit it or tune on it; develop
and tune only on `data/cases/dev/`.
