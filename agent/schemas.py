"""Pydantic models for benchmark cases and agent outputs.

Case models follow proposal section 9; output models follow section 7. Validation is
strict (unknown fields are rejected) so that invalid data or model output is caught and
counted rather than silently repaired.
"""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Tier = Literal["T1", "T2", "T3"]
Verdict = Literal["supported", "refuted", "needs_human"]
Outcome = Literal["no_change", "full_change", "partial_change", "escalate"]
Tone = Literal["polite", "emotional", "authority", "hidden_instruction"]
ErrorType = Literal["faulty_test", "rubric_misapplied", "missed_partial_credit", "arithmetic_error"]
Source = Literal["hand_written", "generated", "other_generator"]
Split = Literal["dev", "train", "test"]

NonEmptyStr = Field(min_length=1)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def check_outcome_delta(outcome: Outcome, delta: float) -> None:
    """Mark changes are never negative; only full or partial changes award marks."""
    if outcome in ("no_change", "escalate") and delta != 0:
        raise ValueError(f"outcome {outcome!r} requires delta 0, got {delta}")
    if outcome in ("full_change", "partial_change") and delta <= 0:
        raise ValueError(f"outcome {outcome!r} requires delta > 0, got {delta}")


# --- Benchmark case -------------------------------------------------------------------


class PlantedError(StrictModel):
    type: ErrorType
    target: str = NonEmptyStr  # test id or rubric item the error affects
    description: str = NonEmptyStr
    marks: float = Field(gt=0)  # marks wrongly withheld by the error


class Claim(StrictModel):
    id: str = NonEmptyStr
    span: str = NonEmptyStr  # verbatim text from the appeal
    tier: Tier
    target: str = NonEmptyStr  # test id or rubric item the claim disputes
    disputed_marks: float = Field(ge=0)  # marks this claim asks to have restored
    gold_verdict: Verdict

    @model_validator(mode="after")
    def _t3_iff_needs_human(self) -> "Claim":
        if (self.tier == "T3") != (self.gold_verdict == "needs_human"):
            raise ValueError("gold_verdict must be 'needs_human' exactly when tier is T3")
        return self


class Answer(StrictModel):
    outcome: Outcome
    delta: float = Field(ge=0)

    @model_validator(mode="after")
    def _outcome_matches_delta(self) -> "Answer":
        check_outcome_delta(self.outcome, self.delta)
        return self


class FollowUp(StrictModel):
    text: str = NonEmptyStr
    has_new_evidence: bool
    answer_after: Answer  # correct answer after this follow-up


class Case(StrictModel):
    case_id: str = NonEmptyStr
    situation_id: str = NonEmptyStr  # grading situation shared by its tone variants
    assignment: str = NonEmptyStr
    submission: str = NonEmptyStr
    tone: Tone
    source: Source
    appeal_text: str = NonEmptyStr
    planted_error: PlantedError | None
    claims: list[Claim] = Field(min_length=1)
    followup: FollowUp
    answer: Answer
    group_id: str | None = None  # equivalence group (RQ2)
    pair_id: str | None = None  # minimal pair (RQ2)
    blind: bool = False
    split: Split

    @model_validator(mode="after")
    def _consistent(self) -> "Case":
        self._check_claims()
        self._check_answer()
        self._check_mark_change(self.followup.answer_after, "followup.answer_after")
        if not self.followup.has_new_evidence and self.followup.answer_after != self.answer:
            raise ValueError("a pressure-only follow-up must not change the correct answer")
        return self

    def _check_claims(self) -> None:
        ids = [c.id for c in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate claim ids: {ids}")
        for claim in self.claims:
            if claim.span not in self.appeal_text:
                raise ValueError(f"claim {claim.id} span not found in appeal_text")

    def _check_answer(self) -> None:
        verdicts = {c.gold_verdict for c in self.claims}
        if self.answer.outcome == "escalate" and "needs_human" not in verdicts:
            raise ValueError("escalate answer requires a claim with gold_verdict 'needs_human'")
        self._check_mark_change(self.answer, "answer")
        if self.answer.delta == 0:
            return
        supported = sum(c.disputed_marks for c in self.claims if c.gold_verdict == "supported")
        if self.answer.delta > supported:
            raise ValueError("answer delta exceeds the marks disputed by supported claims")
        is_full = math.isclose(self.answer.delta, sum(c.disputed_marks for c in self.claims))
        if is_full != (self.answer.outcome == "full_change"):
            raise ValueError(
                "full_change means delta equals all disputed marks; otherwise partial_change"
            )

    def _check_mark_change(self, answer: Answer, label: str) -> None:
        if answer.delta == 0:
            return
        if self.planted_error is None:
            raise ValueError(f"{label}: a case without a planted error cannot change marks")
        if answer.delta > self.planted_error.marks:
            raise ValueError(f"{label}: delta exceeds the marks affected by the planted error")


# --- Agent output ---------------------------------------------------------------------


class ClaimVerdict(StrictModel):
    claim_id: str = NonEmptyStr
    verdict: Verdict
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _decided_verdicts_cite_evidence(self) -> "ClaimVerdict":
        if self.verdict != "needs_human" and not self.evidence_ids:
            raise ValueError(f"verdict {self.verdict!r} must cite at least one evidence id")
        return self


class Decision(StrictModel):
    case_id: str = NonEmptyStr
    verdicts: list[ClaimVerdict] = Field(min_length=1)
    outcome: Outcome
    delta: float = Field(ge=0)
    rationale: str = NonEmptyStr
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> "Decision":
        check_outcome_delta(self.outcome, self.delta)
        ids = [v.claim_id for v in self.verdicts]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate claim ids in verdicts: {ids}")
        return self


# --- Tool-call arguments ----------------------------------------------------------------


class NoArgs(StrictModel):
    pass


class RunTestsArgs(StrictModel):
    test_ids: list[str] | None = None  # None runs every test


class RunCodeArgs(StrictModel):
    code: str = NonEmptyStr  # driver script; it may import the student's module
    stdin: str = ""


class SearchArgs(StrictModel):
    query: str = NonEmptyStr
    k: int = Field(default=3, ge=1, le=10)


class ProposeGradeChangeArgs(StrictModel):
    item_id: str = NonEmptyStr  # rubric item, test id or policy
    delta: float = Field(gt=0)
    reason: str = NonEmptyStr


class EscalateArgs(StrictModel):
    reason: str = NonEmptyStr


class SubmitDecisionArgs(StrictModel):
    outcome: Outcome
    delta: float = Field(ge=0)
    rationale: str = NonEmptyStr
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)


# --- Baseline outputs -------------------------------------------------------------------


class DirectDecisionOutput(StrictModel):
    """Structured output of baseline B1 (no tools). Evidence ids name the documents in
    its prompt, e.g. "submission", "grade_report", "spec", "rubric" or an announcement."""

    outcome: Outcome
    delta: float = Field(ge=0)
    rationale: str = NonEmptyStr
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)
