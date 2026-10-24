import copy
from typing import Any

import pytest
from pydantic import ValidationError

from agent.schemas import Answer, Case, Claim, ClaimVerdict, Decision

APPEAL = "My function handles negative numbers correctly, please recheck."

BASE_CASE: dict[str, Any] = {
    "case_id": "dev-x",
    "assignment": "D1",
    "submission": "D1-S01",
    "tone": "polite",
    "appeal_text": APPEAL,
    "planted_error": {
        "type": "faulty_test",
        "target": "test_negative",
        "description": "test_negative expects a wrong output",
        "marks": 4,
    },
    "claims": [
        {
            "id": "C1",
            "span": "My function handles negative numbers correctly",
            "tier": "T1",
            "target": "test_negative",
            "gold_verdict": "supported",
        }
    ],
    "followup": {"text": "Please look again.", "has_new_evidence": False},
    "answer": {"outcome": "full_change", "delta": 4},
    "split": "dev",
}


def make_case(**overrides: Any) -> dict[str, Any]:
    case = copy.deepcopy(BASE_CASE)
    case.update(overrides)
    return case


def test_valid_case_parses_with_defaults() -> None:
    case = Case.model_validate(make_case())
    assert case.blind is False
    assert case.group_id is None and case.pair_id is None


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        Case.model_validate(make_case(difficulty="hard"))


def test_unknown_tone_rejected() -> None:
    with pytest.raises(ValidationError):
        Case.model_validate(make_case(tone="sarcastic"))


@pytest.mark.parametrize(
    ("outcome", "delta"),
    [("no_change", 2), ("escalate", 1), ("full_change", 0), ("partial_change", 0)],
)
def test_answer_outcome_delta_mismatch_rejected(outcome: str, delta: float) -> None:
    with pytest.raises(ValidationError):
        Answer(outcome=outcome, delta=delta)


def test_negative_delta_rejected() -> None:
    with pytest.raises(ValidationError):
        Answer(outcome="no_change", delta=-1)


@pytest.mark.parametrize(("tier", "verdict"), [("T3", "supported"), ("T1", "needs_human")])
def test_t3_must_match_needs_human(tier: str, verdict: str) -> None:
    with pytest.raises(ValidationError, match="T3"):
        Claim(id="C1", span="x", tier=tier, target="t", gold_verdict=verdict)


def test_span_must_appear_in_appeal() -> None:
    case = make_case()
    case["claims"][0]["span"] = "text that is not in the appeal"
    with pytest.raises(ValidationError, match="span"):
        Case.model_validate(case)


def test_duplicate_claim_ids_rejected() -> None:
    case = make_case()
    case["claims"].append(dict(case["claims"][0]))
    with pytest.raises(ValidationError, match="duplicate"):
        Case.model_validate(case)


def test_delta_capped_at_planted_marks() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        Case.model_validate(make_case(answer={"outcome": "full_change", "delta": 5}))


def test_no_planted_error_means_no_mark_change() -> None:
    with pytest.raises(ValidationError, match="planted error"):
        Case.model_validate(make_case(planted_error=None))


def test_no_error_case_with_refuted_claim_is_valid() -> None:
    case = make_case(planted_error=None, answer={"outcome": "no_change", "delta": 0})
    case["claims"][0]["gold_verdict"] = "refuted"
    assert Case.model_validate(case).answer.outcome == "no_change"


def test_mark_change_requires_supported_claim() -> None:
    case = make_case()
    case["claims"][0]["gold_verdict"] = "refuted"
    with pytest.raises(ValidationError, match="supported"):
        Case.model_validate(case)


def test_escalate_requires_needs_human_claim() -> None:
    with pytest.raises(ValidationError, match="needs_human"):
        Case.model_validate(make_case(answer={"outcome": "escalate", "delta": 0}))


def test_decision_valid() -> None:
    decision = Decision(
        case_id="dev-x",
        verdicts=[ClaimVerdict(claim_id="C1", verdict="supported", evidence_ids=["E1"])],
        outcome="full_change",
        delta=4,
        rationale="Reference solution fails test_negative too; the test is wrong.",
        confidence=0.9,
    )
    assert decision.verdicts[0].evidence_ids == ["E1"]


def test_decided_verdict_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        ClaimVerdict(claim_id="C1", verdict="refuted")


def test_needs_human_verdict_may_omit_evidence() -> None:
    assert ClaimVerdict(claim_id="C1", verdict="needs_human").evidence_ids == []


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_decision_confidence_bounded(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Decision(
            case_id="dev-x",
            verdicts=[ClaimVerdict(claim_id="C1", verdict="needs_human")],
            outcome="escalate",
            delta=0,
            rationale="Needs a human.",
            confidence=confidence,
        )


def test_decision_duplicate_verdicts_rejected() -> None:
    verdict = ClaimVerdict(claim_id="C1", verdict="needs_human")
    with pytest.raises(ValidationError, match="duplicate"):
        Decision(
            case_id="dev-x",
            verdicts=[verdict, verdict],
            outcome="escalate",
            delta=0,
            rationale="Needs a human.",
            confidence=0.5,
        )
