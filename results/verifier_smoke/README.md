# Verifier QLoRA smoke test (2026-10-29)

`notebooks/verifier_smoke.ipynb` on a Kaggle Tesla T4, commit `e74a8cf` (`commit.txt`),
config `config.yaml`, step losses and generation checks in `events.jsonl`. The LoRA adapter
(29 MB) stayed in the Kaggle output and is not committed.

| | |
| --- | --- |
| Model | Qwen2.5-3B-Instruct, 4-bit NF4, float16 compute, LoRA r=16 on q/k/v/o |
| Data | 50 dummy tone-paired examples (`verifier/dummy_data.py`), none too long |
| Training | 10 steps, batch 2, 12 s; loss 1.30 -> 0.12 |
| Trainable parameters | 7.4M |
| Peak GPU memory | 4.0 GB |
| Generation check | 6/6 valid JSON, 6/6 matching the label |
| Libraries | torch 2.11 (CUDA 12.8), transformers 5.16.1, peft 0.20.0, bitsandbytes 0.50.2, accelerate 1.14 |

**What this shows:** the training pipeline (4-bit loading, LoRA, loss on the answer tokens,
saving the adapter, generating and strictly parsing the answer) runs on the T4 with ample
memory to spare. **What it does not show:** verifier quality. The check examples come from
the same 50 dummy examples (6 base claims repeated in three tones), so 6/6 only shows that
inference works end to end. Real training data comes from the training-only assignments
(`docs/VERIFIER_PLAN.md`).
