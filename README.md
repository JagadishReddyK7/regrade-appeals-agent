# Regrade Appeals Agent

Research prototype for the LMA Group Project 2026 (Team Attention Seekers: Jagadish Kollu,
Rishitha Erva, Nallala Karthik, Suresh Kumar Vuyala).

The agent handles regrade appeals for Python programming assignments. It splits each
appeal into claims and checks every claim against evidence: it runs the student's code and
retrieves the rubric and the dated course announcements. A fine-tuned 3B verifier judges
each claim, and the grade changes only when the evidence supports it. Unclear cases are
escalated to a TA. Grade changes are only proposed and logged, never written.

We test it on a benchmark we built, in which every appeal has a planted grading error (or
none) and a known correct outcome, and every grading situation is written in four tones
(polite, emotional, authority, hidden instruction) with one follow-up pushback message.

## Results in one table

Test set: 80 cases (20 situations x 4 tones, 2 new assignments), one run per system, Kaggle
T4 x2, temperature 0. Details, statistics and caveats: `results/test/README.md`.

| System | Exact accuracy | Wrongful change | Wrongful refusal | Pushback flips | Tone sensitivity |
| --- | --- | --- | --- | --- | --- |
| B0 always reject / always accept | 0.30 / 0.30 | 0.00 / 0.70 | 1.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.05 |
| B1 direct decision | 0.29 | 0.33 | 0.35 | 0.27 | 0.65 |
| B3 tool agent, whole appeal | 0.39 | 0.25 | 0.38 | 0.14 | 0.70 |
| **Full system** | **0.45** | **0.17** | **0.53** | **0.00** | **0.45** |
| A0 no claim splitting | 0.49 | 0.11 | 0.60 | 0.03 | 0.50 |
| A1 no code execution | 0.59 | 0.16 | 0.32 | 0.00 | 0.40 |
| A2 no retrieval | 0.44 | 0.14 | 0.60 | 0.00 | 0.50 |
| A4a untuned 3B verifier | 0.20 | 0.37 | 0.29 | 0.00 | 0.60 |
| A4b prompted 7B verifier | 0.46 | 0.05 | 0.68 | 0.00 | 0.70 |
| A6 no escalation | 0.41 | 0.21 | 0.53 | 0.00 | 0.55 |

RQ2 (consistency, 32 public cases on the dev assignment D1, `results/rq2/README.md`): the
full system decided 2-3 of 4 equivalence groups the same way for every member, B1 and B3
none. With a store of its own past decisions it copies its mistakes and its accuracy
depends on the order of the appeals (0.47-0.72, against 0.62-0.66 between identical runs);
a wrong precedent where its evidence is clear was ignored (0 of 8), a correct one on its
weakest case type was followed (8 of 8, accuracy 0.91).

## Start here

| Document | What it is for |
| --- | --- |
| `docs/proposal.pdf` | the original specification |
| `results/test/README.md` | the final test results, statistics and caveats |
| `results/rq2/README.md` | RQ2: consistency and precedent results |
| `docs/STATUS.md` | what is built and what is not |
| `docs/DECISIONS.md` | every design decision and where the code differs from the proposal |
| `docs/ARCHITECTURE.md` | how a case flows through the code, how to add a system, tool or metric |
| `docs/DATA_FORMAT.md` | assignments, submissions, grade reports and cases |
| `docs/VERIFIER_PLAN.md` | the claim verifier and its training data |
| `docs/TEST_SET_GUIDE.md` | how the private test set was built, labelled and frozen |
| `CLAUDE.md` | the rules our coding assistant followed in this repo (useful for humans too) |

## Setup (CPU, any machine)

Requires Python 3.11 or newer (CI tests 3.11 and 3.13; Kaggle runs 3.13).

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest                # about 330 tests, 1-2 minutes; or: make test
python -m ruff check .          # or: make lint
python -m data.validate data/cases/dev    # validate the dev cases with the reference grader
```
On Windows two sandbox tests are skipped (CPU and memory limits need Linux); CI runs them.
Everything that needs no GPU runs locally, for example the two B0 baselines on dev:
`python -m eval.run --config configs/b0_reject_dev.yaml`.

## Reproducing the results (Kaggle, GPU)

All GPU work ran on free Kaggle notebooks with two Tesla T4 GPUs, vLLM 0.32.0 and
Qwen2.5-7B-Instruct-AWQ (float16; the T4 has no bfloat16). The notebooks are thin: they
clone this repo, install it, start the vLLM servers and call the same commands as above.
For each notebook: import it into Kaggle, set *Accelerator: GPU T4 x2* and *Internet: on*,
add the inputs listed below, and use *Save Version -> Save & Run All*. Each one writes a
zip of its outputs; the committed copies are in `results/`.

| Result | Notebook | Inputs | Time |
| --- | --- | --- | --- |
| Dev runs of B0, B1, agent v0, the pipeline and A0 (`results/mid_dev_24/`, `results/pipeline_dev_v1/` to `v3/`) | `notebooks/pilot_dev.ipynb` | none | about 1 hour |
| Verifier training, verifier comparison and pipeline with the fine-tuned verifier on dev (`results/verifier_ft_dev/`) | `notebooks/verifier_ft.ipynb` | none (or an earlier version's output to reuse the adapter) | about 1.5 hours |
| Every system on the test set (`results/test/`) | `notebooks/test_runs.ipynb` | the private test dataset; the output of the `verifier_ft` notebook | about 5 hours |
| RQ2: consistency cases and precedent runs (`results/rq2/`: 7 runs; `results/rq2_session2/`: repeats and planted precedents, set `RUNS` in the notebook) | `notebooks/rq2_runs.ipynb` | the output of the `verifier_ft` notebook | about 2 hours + 1 hour |

Notes:
- **Prompts and pipeline code are frozen at commit `8f8115e`**; the test runs used commit
  `7b43458` (frozen code plus notebook fixes only). Each result folder records its commit
  in `commit.txt` and its packages in `pip_freeze.txt`.
- Each test config is its dev config with only the name, case folder and data root changed
  (`tests/test_test_configs.py` checks this).
- vLLM at temperature 0 is not exactly reproducible (random tool-call ids enter the next
  prompt), so a rerun can differ by a few cases.
- Statistics for any set of runs: `python -m eval.stats <main run dir> <other run dirs> --metrics wrongful_change_rate wrongful_refusal_rate`.

## The test set is private

The test cases (and the assignments, submissions and answer keys they use) are never
committed, so that no prompt could be tuned on them. They are in a private Kaggle dataset
(`TESTSET_README.md`; access on request to the team), checked against a frozen manifest of
file hashes before the runs. Only aggregate test metrics are in this repo. The dev set
(`data/cases/dev/`, 24 cases) and the training-only assignments (`data/assignments/TR1`,
`TR2`) are public.

## Commands

| Command | Does |
| --- | --- |
| `python -m data.validate data/cases/dev` | validate cases, their data and the reference grader |
| `python -m data.build_cases <situation.yaml>` | build the four tone variants of a situation |
| `python -m eval.run --config configs/<name>.yaml` | run a system on a case folder; writes `runs/<run id>/` |
| `python -m eval.stats <run> <run> ...` | paired permutation tests by situation, Holm correction, bootstrap CIs |
| `python -m llm.serve --config configs/<name>.yaml [--verifier]` | start vLLM for the main model or the verifier (GPU) |
| `python -m llm.smoke --config configs/<name>.yaml` | check chat, tool calling and constrained JSON on the server |
| `python -m verifier.train_data` / `verifier.train` / `verifier.evaluate` | build the verifier training data, train with QLoRA, evaluate |

A run directory contains `config.yaml`, `events.jsonl` (every LLM call, tool call,
retrieval, verdict and decision, with run id, case id, seed and config hash),
`proposals.jsonl`, `predictions.jsonl` and `metrics.json`.

## Systems and configs

| System | Config (`configs/`) | Code |
| --- | --- | --- |
| B0 always reject / accept | `b0_reject_*.yaml`, `b0_accept_*.yaml` | `baselines/b0.py` |
| B1 direct decision, no tools | `b1_*.yaml` | `baselines/b1_direct.py` |
| B3 tool agent on the whole appeal (agent v0) | `agent_*.yaml` | `agent/loop.py` |
| Full system: extractor, evidence agent, verifier, aggregator | `pipeline_test.yaml`, `pipeline_ft_dev.yaml` | `agent/pipeline.py`, `agent/extractor.py`, `agent/aggregator.py`, `verifier/infer.py` |
| Ablations A0, A1, A2, A4a, A4b, A6 | `pipeline_<ablation>_test.yaml` | config flags in the same code |

Not built: B2, B4, A3, A5, A7, the LLM judge and the precedent runs (see
`docs/STATUS.md`).

## Layout

| Path | Contents |
| --- | --- |
| `agent/` | schemas, tools, agent v0 loop, claim pipeline (extractor, aggregator) |
| `baselines/` | B0 and B1 |
| `configs/` | YAML configs; every ablation is a config flag |
| `data/` | dev and training assignments, submissions, situations and cases; formats, reference grader, validator, case builder |
| `eval/` | metrics, run log, evaluation runner, cost, statistics |
| `llm/` | model client (vLLM and mock), vLLM launcher, smoke test |
| `notebooks/` | thin Kaggle notebooks |
| `retrieval/` | BM25 course-document search with dated announcements, precedent store |
| `sandbox/` | sandboxed execution of student code with limits and a result cache |
| `verifier/` | verifier prompt format, training data, QLoRA training, inference, evaluation |
| `results/` | committed outputs of every run reported in the report |
| `report/` | mid-submission and final reports |
| `tests/` | pytest suite |
