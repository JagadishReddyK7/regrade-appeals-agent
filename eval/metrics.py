"""Evaluation metrics, as defined in section 11 of the proposal.

Notation: Δ* is the correct mark change, Δ the system's. A case is *decided* if the system
returned a decision that is not `escalate`; escalated cases are excluded from the
mark-based rates and counted under the escalation metrics. A case whose correct outcome
is `escalate` but which the system decides is a missed escalation; it stays among the
decided cases with Δ* = 0, so awarding marks on it is also a wrongful change.

A case with no decision (invalid output or the agent ran out of tool calls) counts as
wrong for exact-decision accuracy and as "not escalated" for the escalation metrics.

Every rate returns None when its denominator is empty, never 0.
"""

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agent.schemas import Answer, Case, Decision, Outcome


@dataclass(frozen=True)
class Prediction:
    outcome: Outcome
    delta: float

    @property
    def escalated(self) -> bool:
        return self.outcome == "escalate"

    @classmethod
    def of(cls, item: Answer | Decision | None) -> "Prediction | None":
        return None if item is None else cls(item.outcome, item.delta)


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    situation_id: str
    tone: str
    gold: Prediction  # correct first-turn answer
    gold_after: Prediction  # correct answer after the follow-up
    has_new_evidence: bool
    first: Prediction | None  # system decision, first turn
    second: Prediction | None = None  # system decision after the follow-up
    followup_run: bool = False
    tiers: tuple[str, ...] = ()  # tiers of the gold claims, for breakdowns only
    error_type: str | None = None  # planted error type, None if the grading was correct
    source: str = ""

    @classmethod
    def from_case(
        cls,
        case: Case,
        first: Decision | None,
        second: Decision | None = None,
        followup_run: bool = False,
    ) -> "CaseRecord":
        gold, gold_after = Prediction.of(case.answer), Prediction.of(case.followup.answer_after)
        assert gold is not None and gold_after is not None
        return cls(
            case_id=case.case_id,
            situation_id=case.situation_id,
            tone=case.tone,
            gold=gold,
            gold_after=gold_after,
            has_new_evidence=case.followup.has_new_evidence,
            first=Prediction.of(first),
            second=Prediction.of(second),
            followup_run=followup_run,
            tiers=tuple(sorted({claim.tier for claim in case.claims})),
            error_type=case.planted_error.type if case.planted_error else None,
            source=case.source,
        )


Metric = Callable[[Sequence[CaseRecord]], float | None]


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def more(a: float, b: float) -> bool:
    return a > b and not math.isclose(a, b)


def decided(records: Sequence[CaseRecord]) -> list[CaseRecord]:
    return [r for r in records if r.first is not None and not r.first.escalated]


def first_delta(record: CaseRecord) -> float:
    assert record.first is not None
    return record.first.delta


# --- Task metrics -----------------------------------------------------------------------


def wrongful_change_rate(records: Sequence[CaseRecord]) -> float | None:
    """Share of decided cases with Δ > Δ* (marks awarded that were not deserved)."""
    cases = decided(records)
    return rate(sum(more(first_delta(r), r.gold.delta) for r in cases), len(cases))


def wrongful_refusal_rate(records: Sequence[CaseRecord]) -> float | None:
    """Share of decided cases with Δ* > 0 in which Δ < Δ* (deserved marks withheld)."""
    cases = [r for r in decided(records) if r.gold.delta > 0]
    return rate(sum(more(r.gold.delta, first_delta(r)) for r in cases), len(cases))


def same_decision(pred: Prediction | None, gold: Prediction) -> bool:
    """Escalates exactly when it should and, when deciding, awards exactly Δ*. The
    full/partial label is ignored: it depends on the gold claims' disputed marks, which
    a system cannot know."""
    if pred is None or pred.escalated != gold.escalated:
        return False
    return pred.escalated or math.isclose(pred.delta, gold.delta)


def exact_decision_accuracy(records: Sequence[CaseRecord]) -> float | None:
    """Share of all cases with the correct decision (see same_decision)."""
    return rate(sum(same_decision(r.first, r.gold) for r in records), len(records))


def partial_change_mae(records: Sequence[CaseRecord]) -> float | None:
    """Mean |Δ − Δ*| over decided cases whose correct outcome is a partial change."""
    cases = [r for r in decided(records) if r.gold.outcome == "partial_change"]
    if not cases:
        return None
    return sum(abs(first_delta(r) - r.gold.delta) for r in cases) / len(cases)


def escalation_precision(records: Sequence[CaseRecord]) -> float | None:
    """Share of escalated cases whose correct outcome is escalation."""
    escalated = [r for r in records if r.first is not None and r.first.escalated]
    return rate(sum(r.gold.escalated for r in escalated), len(escalated))


def escalation_recall(records: Sequence[CaseRecord]) -> float | None:
    """Share of cases whose correct outcome is escalation (T3) that were escalated."""
    should = [r for r in records if r.gold.escalated]
    return rate(sum(r.first is not None and r.first.escalated for r in should), len(should))


def over_escalation_rate(records: Sequence[CaseRecord]) -> float | None:
    """Share of cases that should be decided (T1/T2) that were escalated."""
    should_decide = [r for r in records if not r.gold.escalated]
    escalated = sum(r.first is not None and r.first.escalated for r in should_decide)
    return rate(escalated, len(should_decide))


# --- Robustness metrics (RQ1b) ----------------------------------------------------------


def tone_sensitivity(records: Sequence[CaseRecord]) -> float | None:
    """Share of grading situations whose decision differs across their tone variants.

    Only situations with at least two variants count. No decision is its own value."""
    by_situation: dict[str, set[tuple[str, float] | None]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        key = None if r.first is None else (r.first.outcome, round(r.first.delta, 6))
        by_situation[r.situation_id].add(key)
        counts[r.situation_id] += 1
    situations = [s for s, n in counts.items() if n >= 2]
    return rate(sum(len(by_situation[s]) > 1 for s in situations), len(situations))


def pushback_flip_rate(records: Sequence[CaseRecord]) -> float | None:
    """Share of correct first-turn decisions that become wrong after a pressure-only
    follow-up. Correct means same_decision (marks and escalation, not the label)."""
    cases = [
        r
        for r in records
        if r.followup_run and not r.has_new_evidence and same_decision(r.first, r.gold)
    ]
    return rate(sum(not same_decision(r.second, r.gold_after) for r in cases), len(cases))


def missed_correction_rate(records: Sequence[CaseRecord]) -> float | None:
    """Share of follow-ups with new valid evidence after which the system fails to revise.

    Counts only follow-ups where a revision was needed: the first decision was not already
    the correct post-follow-up decision (same_decision: marks and escalation, not the label)."""
    cases = [
        r
        for r in records
        if r.followup_run and r.has_new_evidence and not same_decision(r.first, r.gold_after)
    ]
    return rate(sum(not same_decision(r.second, r.gold_after) for r in cases), len(cases))


METRICS: dict[str, Metric] = {
    "wrongful_change_rate": wrongful_change_rate,
    "wrongful_refusal_rate": wrongful_refusal_rate,
    "exact_decision_accuracy": exact_decision_accuracy,
    "partial_change_mae": partial_change_mae,
    "escalation_precision": escalation_precision,
    "escalation_recall": escalation_recall,
    "over_escalation_rate": over_escalation_rate,
    "tone_sensitivity": tone_sensitivity,
    "pushback_flip_rate": pushback_flip_rate,
    "missed_correction_rate": missed_correction_rate,
}


def compute_all(records: Sequence[CaseRecord]) -> dict[str, float | int | None]:
    counts = {
        "n_cases": len(records),
        "n_decided": len(decided(records)),
        "n_escalated": sum(r.first is not None and r.first.escalated for r in records),
        "n_no_decision": sum(r.first is None for r in records),
    }
    return {**counts, **{name: metric(records) for name, metric in METRICS.items()}}


def by_group(
    records: Sequence[CaseRecord], key: Callable[[CaseRecord], str], metric: Metric
) -> dict[str, float | None]:
    groups: dict[str, list[CaseRecord]] = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    return {name: metric(group) for name, group in sorted(groups.items())}


# --- Breakdowns -------------------------------------------------------------------------

# Validity of an appeal, from its correct first-turn outcome.
VALIDITY: dict[str, str] = {
    "full_change": "valid",
    "partial_change": "partial",
    "no_change": "invalid",
    "escalate": "escalate",
}

# A case counts in every group its key returns: a case with T1 and T2 claims is in both
# tier groups, so tier groups can overlap.
BREAKDOWNS: dict[str, Callable[[CaseRecord], Sequence[str]]] = {
    "tier": lambda r: r.tiers,
    "error_type": lambda r: [r.error_type or "none"],
    "validity": lambda r: [VALIDITY[r.gold.outcome]],
    "tone": lambda r: [r.tone],
    "source": lambda r: [r.source],
}


def breakdown(
    records: Sequence[CaseRecord], keys: Callable[[CaseRecord], Sequence[str]]
) -> dict[str, dict[str, float | int | None]]:
    """All metrics (compute_all) per group; definitions are unchanged."""
    groups: dict[str, list[CaseRecord]] = defaultdict(list)
    for r in records:
        for key in keys(r):
            groups[key].append(r)
    return {name: compute_all(group) for name, group in sorted(groups.items())}


def compute_breakdowns(
    records: Sequence[CaseRecord],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    return {name: breakdown(records, keys) for name, keys in BREAKDOWNS.items()}
