"""Verifier inference: one claim and its evidence bundle -> a verdict.

The same prompt serves a prompted model (ablation A4: untuned 3B or prompted 7B) and the
fine-tuned 3B (served by vLLM with the LoRA adapter as the model name). The answer is
constrained to the VerifierOutput schema and parsed strictly; invalid output is logged as
`invalid_output` (kind "verifier") and returned as None.
"""

from typing import Any

from agent.schemas import EvidenceBundle, VerifierOutput
from eval.runlog import RunLogger
from llm.client import LLMClient
from verifier.format import parse_answer, prompt_messages


def output_schema(bundle: EvidenceBundle) -> dict[str, Any]:
    """VerifierOutput's JSON schema with `evidence_id` restricted to the bundle's ids (or
    null), so constrained decoding cannot invent an id. Inference only: training uses
    the plain answer format."""
    schema = VerifierOutput.model_json_schema()
    ids = [item.id for item in bundle.evidence]
    allowed: list[dict[str, Any]] = [{"type": "null"}]
    if ids:
        allowed.insert(0, {"enum": ids})
    schema["properties"]["evidence_id"] = {"anyOf": allowed}
    return schema


def verify_claim(
    client: LLMClient,
    bundle: EvidenceBundle,
    logger: RunLogger,
    temperature: float = 0.0,
    seed: int = 0,
) -> VerifierOutput | None:
    response = client.chat(
        prompt_messages(bundle),
        json_schema=output_schema(bundle),
        temperature=temperature,
        seed=seed,
    )
    result = parse_answer(response.content, bundle)
    if not result.ok:
        logger.log(
            "invalid_output",
            kind="verifier",
            schema="VerifierOutput",
            error=result.error,
            raw=response.content,
        )
    return result.value
