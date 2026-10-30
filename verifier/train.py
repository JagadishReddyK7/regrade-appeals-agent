"""QLoRA fine-tuning of the claim verifier (Qwen2.5-3B-Instruct) on VerifierExample JSONL.

Usage (GPU, Kaggle T4): python -m verifier.train --config configs/verifier_smoke.yaml

The model is loaded in 4-bit (NF4, float16 compute: the T4 has no bfloat16) and trained
with LoRA adapters to produce the verifier's JSON answer (verifier/format.py); the loss
covers only the answer tokens. Writes runs/<run id>/ with config.yaml, events.jsonl
(one train_step event per step, check results), metrics.json and the LoRA adapter.
The GPU libraries (the optional "gpu" extra) are imported only inside `train`, so the
helpers here are testable on CPU.
"""

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent.schemas import VerifierExample
from eval.runlog import RunContext, RunLogger, config_hash
from verifier.dummy_data import read_examples
from verifier.format import answer_text, parse_answer, prompt_messages

IGNORE_INDEX = -100  # label value the loss ignores


@dataclass(frozen=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class TrainConfig:
    name: str
    model: str
    data: str
    output_dir: str = "runs"
    seed: int = 0
    max_steps: int = 10
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_length: int = 1024
    check_examples: int = 6  # examples generated after training as a sanity check
    lora: LoraSettings = field(default_factory=LoraSettings)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        data = dict(data)
        lora = data.pop("lora", None) or {}
        if "target_modules" in lora:
            lora["target_modules"] = tuple(lora["target_modules"])
        config = cls(**data, lora=LoraSettings(**lora))
        if config.max_steps < 1 or config.batch_size < 1 or config.max_length < 16:
            raise ValueError("max_steps, batch_size and max_length must be positive")
        return config


def load_train_config(path: Path) -> TrainConfig:
    return TrainConfig.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    labels: list[int]


def encode(prompt_ids: list[int], answer_ids: list[int], max_length: int) -> Encoded | None:
    """Prompt + answer, with the loss masked on the prompt; None if it does not fit (the
    prompt is not truncated, since cutting it would drop the claim or the evidence)."""
    if len(prompt_ids) + len(answer_ids) > max_length or not answer_ids:
        return None
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(answer_ids)
    return Encoded(input_ids=list(prompt_ids) + list(answer_ids), labels=labels)


def pad_batch(batch: list[Encoded], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a batch; padding is masked in attention and loss."""
    width = max(len(e.input_ids) for e in batch)
    input_ids, labels, attention = [], [], []
    for e in batch:
        gap = width - len(e.input_ids)
        input_ids.append(e.input_ids + [pad_id] * gap)
        labels.append(e.labels + [IGNORE_INDEX] * gap)
        attention.append([1] * len(e.input_ids) + [0] * gap)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def batches(encoded: list[Encoded], batch_size: int, steps: int, seed: int) -> list[list[Encoded]]:
    """`steps` batches, cycling through shuffled epochs of the data."""
    rng = random.Random(seed)
    out: list[list[Encoded]] = []
    while len(out) < steps:
        order = list(encoded)
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            if len(out) == steps:
                break
            out.append(order[start : start + batch_size])
    return out


def tokenize_example(tokenizer: Any, example: VerifierExample) -> tuple[list[int], list[int]]:
    prompt = tokenizer.apply_chat_template(
        prompt_messages(example.bundle), tokenize=False, add_generation_prompt=True
    )
    answer = answer_text(example.label) + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return prompt_ids, answer_ids


def train(config: TrainConfig) -> Path:  # pragma: no cover - needs a GPU
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from eval.run import code_version

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    chash = config_hash(asdict(config))
    run_id = f"{config.name}-{time.strftime('%Y%m%d-%H%M%S')}-{chash[:6]}"
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
    logger = RunLogger(run_dir / "events.jsonl", RunContext(run_id, config.seed, chash))
    logger.log("run_start", code_version=code_version(), config=asdict(config))

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    examples = read_examples(Path(config.data))
    encoded = [encode(*tokenize_example(tokenizer, e), config.max_length) for e in examples]
    usable = [e for e in encoded if e is not None]
    logger.log("data", examples=len(examples), too_long=len(examples) - len(usable))
    if not usable:
        raise ValueError("no example fits max_length")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4: no bfloat16
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model, quantization_config=quantization, device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )

    model.train()
    start = time.perf_counter()
    losses = []
    for step, batch in enumerate(batches(usable, config.batch_size, config.max_steps, config.seed)):
        tensors = {k: torch.tensor(v, device="cuda") for k, v in pad_batch(batch, pad_id).items()}
        loss = model(**tensors).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        logger.log("train_step", step=step, loss=losses[-1], elapsed_s=time.perf_counter() - start)
    train_seconds = time.perf_counter() - start
    model.save_pretrained(run_dir / "adapter")

    checks = check_generation(model, tokenizer, examples[: config.check_examples], logger)
    metrics = {
        "run_id": run_id,
        "examples": len(examples),
        "too_long": len(examples) - len(usable),
        "trainable_parameters": trainable,
        "steps": len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "train_seconds": train_seconds,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "check_valid_json": checks["valid"],
        "check_correct": checks["correct"],
        "check_total": checks["total"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return run_dir


def check_generation(  # pragma: no cover - needs a GPU
    model: Any, tokenizer: Any, examples: list[VerifierExample], logger: RunLogger
) -> dict[str, int]:
    """Generate answers for a few examples: does the output parse, and match the label?
    On dummy data this only shows that inference works end to end."""
    import torch

    model.eval()
    valid = correct = 0
    for example in examples:
        prompt = tokenizer.apply_chat_template(
            prompt_messages(example.bundle), tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        raw = tokenizer.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
        result = parse_answer(raw, example.bundle)
        valid += result.ok
        correct += result.ok and result.value == example.label
        logger.log("check", example_id=example.example_id, raw=raw, error=result.error)
    return {"valid": valid, "correct": correct, "total": len(examples)}


def main(argv: list[str]) -> int:  # pragma: no cover - needs a GPU
    parser = argparse.ArgumentParser(prog="python -m verifier.train")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    run_dir = train(load_train_config(args.config))
    print(f"run written to {run_dir}")
    print((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
