# Project status

Last updated: 2026-10-30 (mid-submission). Update this file when a milestone-level item
finishes or the plan changes; tasks and owners are tracked in GitHub Issues.

## Where we are
Mid-submission (Oct 30). The pipeline runs end to end on Kaggle: seven systems (B0 x2, B1,
agent v0 with follow-up turn, ablations A1, A2, A6) on the 24-case dev set
(`results/mid_dev_24/`; the earlier 16-case run is kept in `results/mid_dev/`), and the
verifier's QLoRA training pipeline passed its smoke test (`results/verifier_smoke/`).

**Not built yet, and central to the research questions:** claim splitting (claim
extractor, verifier inside the pipeline, aggregator). Agent v0 decides on the whole
appeal, so it is the whole-appeal condition (close to B3 / A0); RQ1a's comparison of
claim-by-claim against whole-appeal checking needs the full pipeline. Also not built:
baselines B2-B4, ablations A0, A3-A5, A7, consistency metrics (RQ2), safety metrics, the
LLM judge, the test set and human labelling. The test-set scope is reduced in
`docs/BENCHMARK_PLAN.md` (draft, needs the team's agreement).

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
- **Mid-submission runs, 24 cases (2026-10-29, commit 07a4681):** the same seven systems
  on the full dev set, about 55 minutes. One case error (a follow-up timeout in A6).
  Agent v0 got 1 of the 8 new cases right (it does not check the grade report's
  arithmetic). Results and observations: `results/mid_dev_24/README.md`.

- **Verifier smoke test (2026-10-29, commit e74a8cf): passed.** Qwen2.5-3B in 4-bit with
  LoRA trained 10 steps on 50 dummy examples in 12 s, peak GPU memory 4.0 GB, loss 1.30 to
  0.12; generated answers parse and match. Shows the pipeline runs, not verifier quality
  (`results/verifier_smoke/README.md`).

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
| Course-doc search | `retrieval/course_docs.py` | BM25; announcements carry date and `superseded_by`; the announcement in force ranks first (R2) |
| Precedent store | `retrieval/precedents.py` | JSONL store, BM25 search per assignment, wired to `search_precedents` via config; empty in all runs so far (R2) |
| Verifier format and training | `agent/schemas.py`, `verifier/` | evidence bundle (R1), prompt, dummy data, QLoRA training script, Kaggle smoke notebook (R3); smoke test passed on a T4 (`results/verifier_smoke/`); not trained on real data; see `docs/VERIFIER_PLAN.md` |
| Baseline B1 | `baselines/b1_direct.py` | one prompt, JSON decision, follow-up turn |
| Metrics and runner | `eval/metrics.py`, `eval/run.py` | section 11 task and robustness metrics; `python -m eval.run --config ...` |
| Baseline B0, result breakdowns | `baselines/b0.py`, `eval/metrics.py` | always reject / always accept, no LLM; `metrics.json` breakdowns by tier, error type, validity, tone, source (E2) |
| Statistics | `eval/stats.py` | paired permutation tests by situation, Holm, bootstrap CIs; `python -m eval.stats <run> <run> ...` (E3) |
| Kaggle dev-run notebook | `notebooks/pilot_dev.ipynb` | runs B0, B1, the agent and the three ablations from one account (vLLM pinned to 0.32.0) |
| Mid-submission results | `results/mid_dev/` | the seven dev runs with configs, metrics, predictions and full event logs |
| CI | `.github/workflows/tests.yml` | lint, validator, tests on Python 3.11 and 3.13 |

## Done for the mid-submission
Pilot (0); agent follow-up turn and cost per appeal (A2, A3); ablation flags A1, A2, A6;
case builder and dev set of 24 cases (B2, B3); test-set and labelling plan (B1, draft);
B0, breakdowns and statistics (E2, E3); evidence bundle, precedent store, in-force ranking
and QLoRA smoke test (R1-R3).

## Next: to the final submission (Nov 15)
In priority order; owners in GitHub Issues.

| # | Task | Why |
| --- | --- | --- |
| 1 | Claim extractor and aggregator; verifier in the pipeline | RQ1a and H1b cannot be tested without claim splitting |
| 2 | Test set built, double-labelled and frozen (about Nov 5, `docs/BENCHMARK_PLAN.md`) | no held-out evaluation exists yet |
| 3 | Verifier training data from the training-only assignments; real training | H1c |
| 4 | B3 (agent v0 as is), A0, A2 runs; then B2, B4, A4, A5 | baselines and the key ablations |
| 5 | Agent fixes: valid item ids in tool errors, withdrawable proposals, required tool calls | the failures seen in `results/mid_dev_24/` |
| 6 | RQ2 as a small check: consistency metrics, filled precedent store | scope reduced in the benchmark plan |
| 7 | Safety metrics (open question 5), LLM judge, tamper detection | section 11 |
| 8 | Blind cases (E1), in the private store, before reading agent code | validity |

## Known limitations of what is built
- Agent v0 over-awards (refuted claims restored), is moved by pushback, misses arithmetic
  errors in the grade report, and often fails to use the proposal tool correctly (wrong
  item ids, proposals it cannot withdraw); see `results/mid_dev_24/README.md`.
- A crash in the follow-up turn discards a valid first decision (`eval/run.py`
  `run_case`); B1's follow-up outputs cite a non-existent evidence id `followup`.
- Runs are not exactly reproducible at temperature 0 (vLLM's random tool-call ids enter
  the prompt); final numbers need repeated runs.
- Dev has 6 situations on one assignment.
- Cases run one at a time (A4).
- Ablations A0, A3, A4, A5 and A7 are not built (they need claim splitting, filled precedents or the trained verifier).
- The precedent store is empty in all runs; the verifier is not in the pipeline yet.
- The full test suite takes about 30-60 s on Windows (many sandbox subprocesses).

## Later (weeks 2-8)
Benchmark built and double-labelled; verifier training data and QLoRA training;
aggregator and full pipeline; baselines B2-B4; ablations A0-A7 as config flags;
consistency metrics and precedent runs; LLM judge with human validation; capability and
cost metrics; tamper detection; test-set freeze (week 5); main runs; analysis; report.

## Open questions
See the end of `docs/DECISIONS.md`.
