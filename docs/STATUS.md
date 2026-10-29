# Project status

Last updated: 2026-10-27. Update this file in any PR that finishes or starts a task.

## Where we are
Week 1 of the plan in proposal section 14. The infrastructure for the pilot is built and
tested against a mock model; **nothing has run on a real model yet** (the Kaggle pilot is
the next step). The benchmark is the critical path and has not started beyond one dev
assignment.

## Pilot log
- **Attempt 1 (2026-10-27, commit 8080ae4): failed before the model loaded.** vLLM 0.32.0
  with PyTorch 2.13 (CUDA 13.0) clashed with Kaggle's torchaudio (CUDA 12.8). Fixed in
  the notebook (torchaudio uninstalled). The rest of the pipeline behaved as designed:
  the launcher reported the crash, the smoke test failed cleanly, and the runs recorded
  4 case errors with the commit. Environment: Python 3.13, 2x Tesla T4 (compute 7.5).

- **Attempt 2 (2026-10-28, commit ef34bee): worked.** vLLM 0.32.0, PyTorch 2.13 (CUDA
  13.0), Python 3.13, Tesla T4 (Triton attention backend, KV cache about 100k tokens).
  Smoke test passed (chat, tool call, constrained JSON). B1 and agent v0 (one turn) ran
  all 16 dev cases with no crashes and no invalid B1 outputs.
- **Mid-submission runs (2026-10-28, commit bd849fa):** B0 reject/accept, B1, the agent
  with follow-up turn and ablations A1, A2, A6, in one session on one account, about 30
  minutes in total. Results, caveats and observations: `results/mid_dev/README.md`.
  Per appeal: B1 about 6 s and 2.2k prompt tokens; the agent 22-29 s and 10-20k tokens.

## Built
| Area | Where | Notes |
| --- | --- | --- |
| Case and decision schemas | `agent/schemas.py` | strict pydantic; answer-key rules validated |
| Data formats, reference grader, validator | `data/` | `python -m data.validate <folder>`; see `docs/DATA_FORMAT.md` |
| Cross-case consistency checks | `data/consistency.py` | situations, equivalence groups, minimal pairs |
| Dev assignment D1 | `data/assignments/D1`, `data/submissions/D1-*` | 2 autograder versions, 2 announcements, 5 submissions |
| Dev cases | `data/cases/dev/`, `data/situations/dev/` | 24 cases (LLM-written): 6 situations x 4 tones covering all four error types, built with `data/build_cases.py` (B2, B3) |
| Benchmark plan | `docs/BENCHMARK_PLAN.md` | reduced quotas for the test set (B1); draft for team agreement |
| Sandbox | `sandbox/` | limits (Linux), network and process block, result cache |
| LLM client and run log | `llm/client.py`, `llm/parsing.py`, `eval/runlog.py` | vLLM and mock backends; JSONL events with run/case/seed/config hash/commit |
| vLLM launcher and smoke test | `llm/serve.py`, `llm/smoke.py` | checks chat, tool calls, constrained JSON |
| Agent v0 | `agent/tools.py`, `agent/loop.py` | 9 tools, 10-call budget per turn, evidence ids, proposal cap, follow-up turn, ablation flags A1/A2/A6; no claim splitting |
| Cost per appeal | `eval/cost.py` | wall time, LLM calls, latency, tokens, tool calls in `metrics.json` (`cost`) |
| Course-doc search | `retrieval/course_docs.py` | BM25; announcements carry date and `superseded_by` |
| Baseline B1 | `baselines/b1_direct.py` | one prompt, JSON decision, follow-up turn |
| Metrics and runner | `eval/metrics.py`, `eval/run.py` | section 11 task and robustness metrics; `python -m eval.run --config ...` |
| Baseline B0, result breakdowns | `baselines/b0.py`, `eval/metrics.py` | always reject / always accept, no LLM; `metrics.json` breakdowns by tier, error type, validity, tone, source (E2) |
| Statistics | `eval/stats.py` | paired permutation tests by situation, Holm, bootstrap CIs; `python -m eval.stats <run> <run> ...` (E3) |
| Kaggle dev-run notebook | `notebooks/pilot_dev.ipynb` | runs B0, B1, the agent and the three ablations from one account (vLLM pinned to 0.32.0) |
| Mid-submission results | `results/mid_dev/` | the seven dev runs with configs, metrics, predictions and full event logs |
| CI | `.github/workflows/tests.yml` | lint, validator, tests on Python 3.11 and 3.13 |

## Next: rest of week 1 (plan as of 2026-10-27)
**GitHub Issues are the task board** (one issue per row, with owner and progress). This
table is the initial plan and is not updated per task; update this file only when a
milestone-level item finishes or the plan changes. Roles follow proposal section 14.
Ready-to-paste prompts are in `docs/IMPLEMENTATION_GUIDE.md`.

| # | Task | Role | Depends on |
| --- | --- | --- | --- |
| 0 | Run the Kaggle pilot; record smoke-test results, vLLM version, time per appeal here (fallbacks in the guide) | agent lead | - |
| A1 | Minimal claim extractor + extraction precision/recall on dev | agent lead | pilot |
| A2 | Agent follow-up (pushback) turn | agent lead | - |
| A3 | Time and tokens per appeal in `metrics.json` | agent lead | - |
| A4 | Run cases concurrently (thread-safe logging) | agent lead | - |
| B1 | Quota plan (assignments x error types x tiers x tones) | benchmark lead | - |
| B2 | Case-building tool: one situation -> 4 tone variants (+ validator) | benchmark lead | - |
| B3 | Grow the dev set to ~40 cases with tone variants | benchmark lead | B2 |
| B4 | Start test assignments in the private store | benchmark lead | B1 |
| R1 | Evidence bundle format shared by agent and verifier | retrieval/FT lead + agent lead | - |
| R2 | Precedent store; rank in-force announcements above superseded ones | retrieval/FT lead | - |
| R3 | QLoRA smoke test on Kaggle | retrieval/FT lead | R1 |
| E1 | Write the ~60 blind cases **before reading agent code or prompts**; store them only in the private store (`docs/DATA_FORMAT.md`) | evaluation lead | test assignments (B4), or draft appeal texts first |
| E2 | Baseline B0; result breakdowns by tier, error type, validity, source | evaluation lead | - |
| E3 | Statistics: paired permutation tests, Holm, bootstrap by situation | evaluation lead | - |
| E4 | Definitions of safety metrics, then implementation | evaluation lead | open question 5 |

**End of week 1: go/no-go meeting.** (1) Does the 7B find about 80% of planted claims on
dev cases (needs A1)? (2) Does B1 make real mistakes on pressured and rubric-based
appeals? (3) What is the real GPU time per appeal? Update the plan before week 2.

## Known limitations of what is built
- Agent v0 over-awards (refuted claims restored), is moved by pushback, and often fails to
  use the proposal tool correctly (wrong item ids, proposals it cannot withdraw); see
  `results/mid_dev/README.md`.
- Dev has 4 situations (one assignment); `missed_partial_credit` and `arithmetic_error`
  situations are still missing (B3 continues).
- Cases run one at a time (A4).
- Ablations A0, A3, A4, A5 and A7 are not built (they need claim splitting, precedents or the verifier).
- The full test suite takes about 30-60 s on Windows (many sandbox subprocesses).

## Later (weeks 2-8)
Benchmark built and double-labelled; verifier training data and QLoRA training;
aggregator and full pipeline; baselines B2-B4; ablations A0-A7 as config flags;
consistency metrics and precedent runs; LLM judge with human validation; capability and
cost metrics; tamper detection; test-set freeze (week 5); main runs; analysis; report.

## Open questions
See the end of `docs/DECISIONS.md`.
