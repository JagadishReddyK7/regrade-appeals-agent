"""Evaluate a verifier served by an OpenAI-compatible server on EvalExample JSONL.

Usage: python -m verifier.evaluate --examples runs/verifier_eval_dev.jsonl \
           --base-url http://localhost:8001/v1 --model verifier --name ft3b_dev

Writes runs/verifier_eval-<name>/ with predictions.jsonl, events.jsonl (every LLM call
and invalid output, with run id and seed) and metrics.json: accuracy, macro-F1 over the
gold classes, the confusion matrix, accuracy per tone and tone sensitivity (share of
groups whose tone variants got different verdicts). An invalid answer is counted as
wrong and reported separately (the pipeline treats it as needs_human).
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval.runlog import RunContext, RunLogger, config_hash
from llm.client import Backend, LLMClient, OpenAIBackend
from verifier.eval_data import EvalExample, read_eval
from verifier.infer import verify_claim

VERDICTS = ("supported", "refuted", "needs_human")
INVALID = "invalid"


def macro_f1(pairs: list[tuple[str, str]]) -> float | None:
    """Mean F1 over the classes that occur in the gold labels."""
    classes = sorted({gold for gold, _ in pairs})
    if not classes:
        return None
    scores = []
    for c in classes:
        tp = sum(g == c and p == c for g, p in pairs)
        fp = sum(g != c and p == c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores)


def score(examples: list[EvalExample], predictions: list[str]) -> dict[str, Any]:
    pairs = [(e.gold, p) for e, p in zip(examples, predictions, strict=True)]
    by_tone: dict[str, list[bool]] = defaultdict(list)
    by_group: dict[str, set[str]] = defaultdict(set)
    group_size: Counter[str] = Counter()
    for e, p in zip(examples, predictions, strict=True):
        by_tone[e.tone].append(e.gold == p)
        by_group[e.group].add(p)
        group_size[e.group] += 1
    multi = [g for g, n in group_size.items() if n > 1]
    confusion = Counter(f"{g}->{p}" for g, p in pairs)
    return {
        "n": len(pairs),
        "accuracy": sum(g == p for g, p in pairs) / len(pairs) if pairs else None,
        "macro_f1": macro_f1(pairs),
        "invalid": sum(p == INVALID for _, p in pairs),
        "confusion": dict(sorted(confusion.items())),
        "accuracy_by_tone": {t: sum(v) / len(v) for t, v in sorted(by_tone.items())},
        "tone_sensitivity": (
            sum(len(by_group[g]) > 1 for g in multi) / len(multi) if multi else None
        ),
        "groups_with_tone_variants": len(multi),
    }


def evaluate(examples: list[EvalExample], backend: Backend, logger: RunLogger) -> list[str]:
    """Verdict per example; INVALID for an invalid answer or a server error (logged)."""
    predictions = []
    for example in examples:
        log = logger.for_case(example.example_id)
        try:
            output = verify_claim(LLMClient(backend, log), example.bundle, log)
        except Exception as exc:  # one failing request must not end the evaluation
            log.log("case_error", error=repr(exc))
            output = None
        predictions.append(INVALID if output is None else output.verdict)
    return predictions


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m verifier.evaluate")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    settings = {"examples": str(args.examples), "model": args.model, "seed": args.seed}
    chash = config_hash(settings)
    run_id = f"verifier_eval-{args.name}-{time.strftime('%Y%m%d-%H%M%S')}-{chash[:6]}"
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True)
    logger = RunLogger(run_dir / "events.jsonl", RunContext(run_id, args.seed, chash))
    backend = OpenAIBackend(base_url=args.base_url, model=args.model, max_tokens=64)
    examples = read_eval(args.examples)
    predictions = evaluate(examples, backend, logger)
    rows = [
        {"example_id": e.example_id, "tone": e.tone, "gold": e.gold, "predicted": p}
        for e, p in zip(examples, predictions, strict=True)
    ]
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    metrics = {"run_id": run_id, **settings, **score(examples, predictions)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
