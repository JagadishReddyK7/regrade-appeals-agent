"""Aggregator: claim verdicts -> mark change, or escalation.

The rules are code; only the amount of marks to restore is the model's judgement:
- a claim the verifier marked `needs_human` (or whose verdict was invalid) makes the
  appeal escalate, unless escalation is ablated (A6);
- with no supported claim, nothing is restored (`no_change`), without a model call;
- otherwise the model (the 7B) proposes restorations, checked here: only for supported
  claims (or, under A6, undecided ones), only on the claim's target if it has one, and
  never more than the marks deducted on an item (the cap). Invalid output is logged and
  retried once with the error; then the turn has no decision.
"""

import json
import math
from dataclasses import dataclass
from typing import Any

from agent.schemas import (
    AggregatorOutput,
    EvidenceItem,
    ExtractedClaim,
    Outcome,
    Restoration,
    Verdict,
)
from eval.runlog import RunLogger
from llm.client import LLMClient, Message
from llm.parsing import parse_json_output

MAX_ATTEMPTS = 2

SYSTEM_PROMPT = """\
You decide how many marks to give back on a regrade appeal for a Python programming \
course. Each claim has already been checked against the evidence. Give marks back only \
for the claims listed as eligible, only on the item the claim disputes, and never more \
than the marks deducted on that item. If the evidence (for example an announcement on \
partial credit) shows that only part of the deducted marks was wrong, give back only that \
part.

Answer with a JSON object: "restorations" (a list of {"claim_id", "item_id", "marks"}), \
"rationale" (one or two sentences) and "confidence" (0 to 1)."""


@dataclass(frozen=True)
class ClaimResult:
    claim: ExtractedClaim
    verdict: Verdict
    evidence_id: str | None  # the evidence the verifier relied on
    evidence_text: str | None  # that evidence as the verifier saw it
    valid: bool = True  # False: the verifier gave no valid answer (counted as needs_human)


@dataclass(frozen=True)
class Aggregation:
    outcome: Outcome | None  # None: the aggregator gave no valid answer
    restorations: list[Restoration]
    rationale: str
    confidence: float

    @property
    def delta(self) -> float:
        return sum(r.marks for r in self.restorations)


def eligible(results: list[ClaimResult], escalation: bool) -> list[ClaimResult]:
    allowed = {"supported"} if escalation else {"supported", "needs_human"}
    return [r for r in results if r.verdict in allowed]


def check_restorations(
    output: AggregatorOutput, claims: list[ClaimResult], deducted: dict[str, float]
) -> str | None:
    """Error message if a restoration breaks the rules, else None."""
    by_id = {r.claim.id: r.claim for r in claims}
    given: dict[str, float] = {}
    for r in output.restorations:
        if r.item_id not in deducted:
            return f"no marks were deducted on {r.item_id!r}; deducted per item: {deducted}"
        claim = by_id.get(r.claim_id)
        if claim is None:
            return f"claim {r.claim_id!r} is not eligible; eligible: {sorted(by_id)}"
        if claim.target is not None and r.item_id != claim.target:
            return f"claim {r.claim_id} disputes {claim.target!r}, not {r.item_id!r}"
        given[r.item_id] = given.get(r.item_id, 0.0) + r.marks
        cap = deducted.get(r.item_id, 0.0)
        if given[r.item_id] > cap and not math.isclose(given[r.item_id], cap):
            return f"{given[r.item_id]:g} marks on {r.item_id!r} exceed the {cap:g} deducted"
    return None


def output_schema(items: list[str]) -> dict[str, Any]:
    """AggregatorOutput's JSON schema with `item_id` restricted to the items with deducted
    marks, so the model cannot name an item that does not exist."""
    schema = AggregatorOutput.model_json_schema()
    schema["$defs"]["Restoration"]["properties"]["item_id"] = {"enum": items}
    return schema


def user_message(
    claims: list[ClaimResult], deducted: dict[str, float], evidence: list[EvidenceItem]
) -> str:
    lines = ["Eligible claims:"]
    for r in claims:
        entry = {
            "claim_id": r.claim.id,
            "item": r.claim.target,
            "claim": r.claim.span,
            "verdict": r.verdict,
            "evidence": r.evidence_text,
        }
        lines.append(json.dumps(entry))
    items = ", ".join(f"{item}: {marks:g}" for item, marks in deducted.items()) or "none"
    lines.append(f"\nMarks deducted per item in the grade report: {items}")
    lines.append("\nAll evidence gathered for the appeal:")
    lines.extend(f"[{item.id}] from {item.source}:\n{item.text}" for item in evidence)
    return "\n".join(lines)


def aggregate(
    client: LLMClient,
    results: list[ClaimResult],
    deducted: dict[str, float],
    logger: RunLogger,
    evidence: list[EvidenceItem] | None = None,
    escalation: bool = True,
    temperature: float = 0.0,
    seed: int = 0,
) -> Aggregation:
    if escalation and any(r.verdict == "needs_human" for r in results):
        reason = "a claim needs a human judgement"
        return Aggregation("escalate", [], reason, 1.0)
    claims = eligible(results, escalation)
    if not claims:
        return Aggregation("no_change", [], "no claim is supported by the evidence", 1.0)
    if not deducted:
        return Aggregation("no_change", [], "the grade report deducted no marks", 1.0)
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(claims, deducted, evidence or [])},
    ]
    for _ in range(MAX_ATTEMPTS):
        response = client.chat(
            messages,
            json_schema=output_schema(sorted(deducted)),
            temperature=temperature,
            seed=seed,
        )
        result = parse_json_output(response.content, AggregatorOutput)
        error = result.error
        if result.value is not None:
            error = check_restorations(result.value, claims, deducted)
            if error is None:
                return _aggregation(result.value, results, deducted)
        logger.log(
            "invalid_output",
            kind="aggregator",
            schema="AggregatorOutput",
            error=error,
            raw=response.content,
        )
        messages.append({"role": "assistant", "content": response.content or ""})
        messages.append({"role": "user", "content": f"Invalid answer: {error}. Answer again."})
    return Aggregation(None, [], "the aggregator gave no valid answer", 0.0)


def _aggregation(
    output: AggregatorOutput, results: list[ClaimResult], deducted: dict[str, float]
) -> Aggregation:
    """Outcome from the restorations: full_change if every disputed item is restored in
    full, partial_change if some marks are restored, else no_change."""
    delta = sum(r.marks for r in output.restorations)
    disputed = {r.claim.target for r in results}
    full = None not in disputed and math.isclose(
        delta, sum(deducted.get(item, 0.0) for item in disputed if item is not None)
    )
    outcome: Outcome = "no_change" if delta == 0 else "full_change" if full else "partial_change"
    return Aggregation(outcome, list(output.restorations), output.rationale, output.confidence)
