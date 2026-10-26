"""Tools the evidence agent can call, with JSON schemas for OpenAI-style tool calling.

The tools see only what a TA would see when handling the appeal: the submission, the
grade report as given, the autograder that produced it, and the course documents. They
never read the answer key (reference_grade.json) or the reference solution.

Grade changes are only proposed: `propose_grade_change` appends to a proposals log for TA
approval and never edits grades. A proposal may not exceed the marks deducted on that item
in the grade report, counting earlier proposals for the same item.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent.schemas import (
    EscalateArgs,
    NoArgs,
    ProposeGradeChangeArgs,
    RunCodeArgs,
    RunTestsArgs,
    SearchArgs,
    SubmitDecisionArgs,
)
from data.formats import load_assignment, load_grade_report
from data.grader import deducted_marks, tests_path
from eval.runlog import RunLogger
from retrieval.course_docs import CourseDocIndex
from sandbox.result import RunResult
from sandbox.runner import SandboxRunner

MAX_STDERR_CHARS = 1500


class ToolError(Exception):
    """A valid call that failed; the message is returned to the model."""


class RejectedAction(ToolError):
    """A call the safety layer refused, e.g. a proposal above the deducted marks."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]

    def schema(self) -> dict[str, Any]:
        parameters = self.args_model.model_json_schema()
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


TOOL_SPECS = [
    ToolSpec("get_submission", "Return the student's submitted source code.", NoArgs),
    ToolSpec(
        "get_grade_report",
        "Return the grade report the student received and the rubric with maximum marks.",
        NoArgs,
    ),
    ToolSpec(
        "run_tests",
        "Run autograder tests (all, or the given test ids) on the submission, using the "
        "same autograder that produced the grade report.",
        RunTestsArgs,
    ),
    ToolSpec(
        "run_code",
        "Run a Python driver script in the sandbox; it can import the student's module. "
        "Returns stdout, stderr and whether it ran without error.",
        RunCodeArgs,
    ),
    ToolSpec(
        "search_course_docs",
        "Search the specification, rubric and dated announcements. Check `date` and "
        "`superseded_by` to find the rule in force.",
        SearchArgs,
    ),
    ToolSpec("search_precedents", "Search past decisions on similar appeals.", SearchArgs),
    ToolSpec(
        "propose_grade_change",
        "Propose restoring marks on one rubric item, test or policy, for TA approval; "
        "use item_id 'total' for an addition error in the total. "
        "Cannot exceed the marks deducted on that item.",
        ProposeGradeChangeArgs,
    ),
    ToolSpec("escalate_to_ta", "Send the appeal to a human TA, with the reason.", EscalateArgs),
    ToolSpec(
        "submit_decision",
        "Submit the final decision. delta must equal the total of accepted proposals. "
        "Cite the evidence ids of the tool results the decision relies on.",
        SubmitDecisionArgs,
    ),
]
SPECS_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def summarise_run(result: RunResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "error_type": result.error_type,
        "stdout": result.stdout,
        "stderr": result.stderr[-MAX_STDERR_CHARS:],
    }


class ToolBox:
    """Tool implementations bound to one appeal (one assignment and submission)."""

    def __init__(
        self,
        data_root: Path,
        assignment_id: str,
        submission_id: str,
        runner: SandboxRunner,
        proposals_log: RunLogger,
        failures: dict[str, int] | None = None,
    ) -> None:
        self.folder = data_root / "assignments" / assignment_id
        self.submission_dir = data_root / "submissions" / submission_id
        self.assignment = load_assignment(self.folder)
        self.report = load_grade_report(self.submission_dir)
        self.code_path = self.submission_dir / self.assignment.module
        self.runner = runner
        self.proposals_log = proposals_log
        self.failures = dict(failures or {})  # tool name -> number of injected failures
        self.docs = CourseDocIndex.for_assignment(self.folder)
        self.deducted = deducted_marks(self.assignment, self.report)
        self.proposed: dict[str, float] = {}
        self._handlers: dict[str, Callable[[Any], dict[str, Any]]] = {
            "get_submission": self.get_submission,
            "get_grade_report": self.get_grade_report,
            "run_tests": self.run_tests,
            "run_code": self.run_code,
            "search_course_docs": self.search_course_docs,
            "search_precedents": self.search_precedents,
            "propose_grade_change": self.propose_grade_change,
            "escalate_to_ta": self.escalate_to_ta,
        }

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return [spec.schema() for spec in TOOL_SPECS]

    @property
    def proposed_total(self) -> float:
        return sum(self.proposed.values())

    def call(self, name: str, args: BaseModel) -> dict[str, Any]:
        """Run a non-terminal tool. Raises ToolError (or RejectedAction) on failure."""
        if self.failures.get(name, 0) > 0:
            self.failures[name] -= 1
            raise ToolError(f"{name} is temporarily unavailable (injected failure); retry")
        return self._handlers[name](args)

    def get_submission(self, _: NoArgs) -> dict[str, Any]:
        return {"filename": self.assignment.module, "code": self.code_path.read_text("utf-8")}

    def get_grade_report(self, _: NoArgs) -> dict[str, Any]:
        rubric = [
            item.model_dump() for item in [*self.assignment.rubric, *self.assignment.policies]
        ]
        return {"report": self.report.model_dump(), "rubric": rubric}

    def run_tests(self, args: RunTestsArgs) -> dict[str, Any]:
        test_ids = args.test_ids or self.assignment.test_ids()
        unknown = sorted(set(test_ids) - set(self.assignment.test_ids()))
        if unknown:
            raise ToolError(f"unknown test ids {unknown}; valid: {self.assignment.test_ids()}")
        tests = tests_path(self.assignment, self.folder, self.report.autograder)
        results = {tid: self.runner.run_test(self.code_path, tests, tid) for tid in test_ids}
        return {"results": {tid: summarise_run(r) for tid, r in results.items()}}

    def run_code(self, args: RunCodeArgs) -> dict[str, Any]:
        return summarise_run(self.runner.run_code(self.code_path, args.code, args.stdin))

    def search_course_docs(self, args: SearchArgs) -> dict[str, Any]:
        return {"documents": [doc.to_dict() for doc in self.docs.search(args.query, args.k)]}

    def search_precedents(self, args: SearchArgs) -> dict[str, Any]:
        return {"precedents": [], "note": "no precedents available"}

    def propose_grade_change(self, args: ProposeGradeChangeArgs) -> dict[str, Any]:
        available = self.deducted.get(args.item_id, 0.0) - self.proposed.get(args.item_id, 0.0)
        if args.delta > available and not math.isclose(args.delta, available):
            raise RejectedAction(
                f"cannot restore {args.delta} marks on {args.item_id!r}: "
                f"only {max(available, 0.0)} deducted marks remain"
            )
        self.proposed[args.item_id] = self.proposed.get(args.item_id, 0.0) + args.delta
        self.proposals_log.log("grade_change_proposal", **args.model_dump(), status="pending_ta")
        return {"accepted": True, "proposed_total": self.proposed_total}

    def escalate_to_ta(self, args: EscalateArgs) -> dict[str, Any]:
        self.proposals_log.log("escalation", reason=args.reason, status="pending_ta")
        return {"escalated": True}


def args_model(name: str) -> type[BaseModel] | None:
    spec = SPECS_BY_NAME.get(name)
    return spec.args_model if spec else None


def is_terminal(name: str) -> bool:
    return name == "submit_decision"
