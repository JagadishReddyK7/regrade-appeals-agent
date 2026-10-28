"""Baseline B0: fixed policies without a model.

`reject` always answers no_change. `accept` believes the appeal: it restores every mark
the grade report deducted on the items the appeal text names (test id, rubric or policy
name, or "total"); if the appeal names no deducted item, it restores every deducted mark.
B0 sees only what a TA would see (the appeal, the grade report and the rubric), never the
answer key. It ignores the follow-up, so its second decision equals its first.
"""

import re
from pathlib import Path

from agent.schemas import Case, ClaimVerdict, Decision
from baselines.b1_direct import APPEAL_CLAIM_ID, VERDICT_FOR_OUTCOME
from baselines.base import SystemOutput
from data.formats import load_assignment, load_grade_report
from data.grader import deducted_marks
from eval.runlog import RunLogger

MODES = ("reject", "accept")


def aliases(item_id: str) -> set[str]:
    """Ways an appeal can name an item: the id, and for rubric and policy items the name
    after the prefix with or without underscores (policy:late_penalty -> "late penalty")."""
    name = item_id.split(":", 1)[-1]
    return {item_id, name, name.replace("_", " ")}


def mentions(text: str, item_id: str) -> bool:
    return any(
        re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE)
        for alias in aliases(item_id)
    )


def accepted_marks(appeal_text: str, deducted: dict[str, float]) -> float:
    named = [marks for item_id, marks in deducted.items() if mentions(appeal_text, item_id)]
    return sum(named) if named else sum(deducted.values())


class FixedBaseline:
    def __init__(self, data_root: Path, mode: str = "reject") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown B0 mode {mode!r}; choose from {MODES}")
        self.data_root = data_root
        self.mode = mode

    def delta(self, case: Case) -> float:
        if self.mode == "reject":
            return 0.0
        assignment = load_assignment(self.data_root / "assignments" / case.assignment)
        report = load_grade_report(self.data_root / "submissions" / case.submission)
        return accepted_marks(case.appeal_text, deducted_marks(assignment, report))

    def run(self, case: Case, logger: RunLogger) -> SystemOutput:
        delta = self.delta(case)
        outcome = "full_change" if delta > 0 else "no_change"
        decision = Decision(
            case_id=case.case_id,
            verdicts=[
                ClaimVerdict(
                    claim_id=APPEAL_CLAIM_ID,
                    verdict=VERDICT_FOR_OUTCOME[outcome],
                    evidence_ids=["grade_report"],
                )
            ],
            outcome=outcome,
            delta=delta,
            rationale=f"B0 {self.mode}: fixed policy",
            confidence=1.0,
        )
        logger.log("decision", system=f"b0_{self.mode}", turn=1, decision=decision.model_dump())
        return SystemOutput(first=decision, second=decision, followup_run=True)
