"""Verifier evaluation examples: one claim, its evidence and the gold verdict.

Two sources:
- `from_run`: the claims a pipeline run sent to the verifier, with exactly the evidence the
  verifier saw (rebuilt from the run's `events.jsonl`), and the gold verdict of the case's
  claim on the same item. Claims whose item has no gold claim are skipped. On the dev run
  this compares verifiers on real evidence from a held-out assignment.
- `from_examples`: generated examples (e.g. the held-out authority tone of
  verifier/train_data.py), gold verdict = label.

Usage: python -m verifier.eval_data --run results/pipeline_dev_v3/<run> \
           --cases data/cases/dev --out runs/verifier_eval_dev.jsonl
       python -m verifier.eval_data --examples runs/verifier_heldout_authority.jsonl \
           --out runs/eval_heldout_authority.jsonl
"""

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent.schemas import Case, EvidenceBundle, StrictModel, Tone, Verdict, VerifierExample
from verifier.dummy_data import read_examples
from verifier.format import evidence_items


class EvalExample(StrictModel):
    example_id: str
    bundle: EvidenceBundle
    gold: Verdict
    tone: Tone
    group: str  # examples that differ only in tone share a group (tone sensitivity)


def load_cases(folder: Path) -> dict[str, Case]:
    cases = (Case.model_validate_json(p.read_text("utf-8")) for p in folder.glob("*.json"))
    return {c.case_id: c for c in cases}


def read_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def from_run(run_dir: Path, cases: dict[str, Case]) -> list[EvalExample]:
    """Verifier calls of a pipeline run, with the evidence present at that point."""
    examples: list[EvalExample] = []
    evidence: dict[str, dict[str, dict[str, Any]]] = {}
    claims: dict[tuple[str, str], dict[str, Any]] = {}
    for event in read_events(run_dir / "events.jsonl"):
        case_id = event.get("case_id")
        if case_id not in cases:
            continue
        kind = event["event"]
        if kind == "claims":
            for claim in event["claims"]:
                claims[(case_id, claim["id"])] = claim
        elif kind == "tool_call" and event.get("status") == "ok":
            output = dict(event.get("output") or {})
            evidence_id = output.pop("evidence_id", None)
            if evidence_id is not None:
                evidence.setdefault(case_id, {})[evidence_id] = {
                    "tool": event["tool"],
                    "output": output,
                }
        elif kind == "verdict" and event.get("source") == "verifier":
            example = _example(cases[case_id], claims, evidence.get(case_id, {}), event)
            if example is not None:
                examples.append(example)
    return examples


def _example(
    case: Case,
    claims: dict[tuple[str, str], dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    event: dict[str, Any],
) -> EvalExample | None:
    claim = claims.get((case.case_id, event["claim_id"]))
    if claim is None or claim["target"] is None:
        return None
    gold = [c.gold_verdict for c in case.claims if c.target == claim["target"]]
    if len(gold) != 1:
        return None
    bundle = EvidenceBundle(
        claim=claim["span"], target=claim["target"], evidence=evidence_items(evidence)
    )
    return EvalExample(
        example_id=f"{case.case_id}-t{event['turn']}-{claim['id']}",
        bundle=bundle,
        gold=gold[0],
        tone=case.tone,
        group=f"{case.situation_id}-t{event['turn']}-{claim['target']}",
    )


def from_examples(examples: Iterable[VerifierExample]) -> list[EvalExample]:
    return [
        EvalExample(
            example_id=e.example_id,
            bundle=e.bundle,
            gold=e.label.verdict,
            tone=e.tone,
            group=e.pair_id or e.example_id,
        )
        for e in examples
    ]


def write_eval(examples: list[EvalExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(e.model_dump_json() + "\n" for e in examples), encoding="utf-8")


def read_eval(path: Path) -> list[EvalExample]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalExample.model_validate_json(line) for line in lines if line.strip()]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m verifier.eval_data")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="a pipeline run folder (needs --cases)")
    source.add_argument("--examples", type=Path, help="VerifierExample JSONL to convert")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.examples is not None:
        examples = from_examples(read_examples(args.examples))
    elif args.cases is None:
        parser.error("--run needs --cases")
    else:
        examples = from_run(args.run, load_cases(args.cases))
    write_eval(examples, args.out)
    print(f"wrote {len(examples)} evaluation examples to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
