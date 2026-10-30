# Claim verifier: design, training data and plan

Status (mid-submission): the input/output format, the prompt, a dummy-data generator and
the QLoRA training script are built and tested; the Kaggle smoke test passed on a T4
(10 steps in 12 s, 4.0 GB peak GPU memory; `results/verifier_smoke/README.md`). The verifier is not yet trained on real data and
not yet part of the agent's pipeline.

## Role in the system (proposal section 7)

The claim extractor splits an appeal into claims; the evidence agent gathers evidence for
each claim with tools; the **verifier** reads one claim as the student wrote it plus that
evidence and answers `supported`, `refuted` or `needs_human`, citing the evidence item it
relies on; the aggregator turns the verdicts into a mark change or an escalation.

The verifier sees the claim **with its tone** on purpose: RQ1b asks whether robustness to
tone comes from design (claim splitting) or from training (a verifier trained on
tone-paired data), so the tone must reach the verifier for training to matter.

## Format (built: `agent/schemas.py`, `verifier/format.py`)

- **Input, `EvidenceBundle`:** `claim` (verbatim span), `target` (test id, `rubric:...`,
  `policy:...` or `total`), `evidence`: items with the agent's evidence id (`E3`), the
  source tool and the tool output (shortened to 1,500 characters).
- **Output, `VerifierOutput`:** `{"verdict": ..., "evidence_id": ...}`; a decided verdict
  must cite an evidence id that is in the bundle. Parsing is strict (no repair).
- **Training example, `VerifierExample`:** bundle + label + tone + assignment + `pair_id`
  linking the tone variants of the same claim and evidence.
- One prompt (`prompt_messages`) and one answer format (`answer_text`) serve training and
  inference, so they cannot drift apart. `bundle_for_claim` builds a bundle from the
  agent's evidence.

## Model and training (built: `verifier/train.py`, `configs/verifier_smoke.yaml`)

- Qwen2.5-3B-Instruct in 4-bit (NF4, double quantisation), float16 compute (T4: no
  bfloat16), LoRA r=16, alpha 32, dropout 0.05 on the attention projections, gradient
  checkpointing.
- Causal-LM fine-tuning on the answer JSON only (prompt tokens masked); examples that do
  not fit `max_length` are skipped and counted, never truncated.
- Seeded; every step's loss is logged as JSONL with run id and config hash; the run
  records the git commit. The LoRA adapter stays in the Kaggle output, never in Git.
- Libraries: optional `gpu` extra (`pip install -e ".[gpu]"`, Kaggle only).

## Training data plan (proposal section 9)

| Item | Plan |
| --- | --- |
| Source | the training-only assignments, never the test assignments (the proposal plans 4; `docs/BENCHMARK_PLAN.md` suggests 1-2 given the time: open question 10) |
| Labels | known by construction: generated with the same planted-error procedure as the benchmark, so each claim's verdict follows from the planted error and the reference grader |
| Evidence | tool outputs produced by running the real tools (sandbox, course-doc search) on the training submissions, in the bundle format above |
| Tone pairing | each claim and evidence written in 3 tones (polite, emotional, hidden instruction) with the same label; **authority held out** to test generalisation to an unseen tone |
| Generator | Llama-3.1-8B-Instruct on Kaggle; a second family (Gemma-2-9B or Mistral-7B) only for the out-of-distribution test |
| Size | about 3,500 examples with 4 assignments; fewer with 1-2 (to be decided), balanced across verdicts |
| Ablation A5 | the same data without tone pairing (one tone per claim) |

The dummy generator (`verifier/dummy_data.py`) already produces data in this shape (tone
pairs, held-out authority tone) for the smoke test.

## Evaluation (proposal sections 6 and 11)

Claim-verdict macro-F1 and tone sensitivity against the untuned 3B, a prompted 7B and a
few-shot prompted 7B (ablation A4); tested on held-out test assignments, the held-out
authority tone, hand-written appeals and the other generator's appeals.

## Timeline to the final submission (November 15)

| When | Step |
| --- | --- |
| Oct 29 | smoke test on Kaggle: passed (training pipeline runs) |
| Nov 1-4 | training-data generator on the training-only assignments; first real training run |
| Nov 4-7 | claim extractor and aggregator; verifier inside the pipeline |
| Nov 7-10 | ablations A4 and A5; verifier evaluation on held-out tone and generator |
| Nov 10-13 | final runs and analysis |
