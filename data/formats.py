"""File formats for assignments, announcements, grade reports and reference grades.

Layout:
    data/assignments/<id>/assignment.yaml        Assignment
    data/assignments/<id>/spec.md                specification given to students
    data/assignments/<id>/reference/<module>     reference solution
    data/assignments/<id>/<autograder dir>/<tests_file>   one folder per autograder version
    data/assignments/<id>/announcements/*.md     Announcement (YAML front matter + body)
    data/submissions/<submission id>/<module>    the student's code
    data/submissions/<submission id>/grade_report.json     GradeReport (what the student got)
    data/submissions/<submission id>/reference_grade.json  ReferenceGrade (answer key)

A hidden-instruction tone variant uses its own copy of a submission, identical except for
added comments; its reference_grade.json names the original in `variant_of`.

An assignment can have several autograder versions: the correct one (`reference_autograder`)
and planted faulty variants. Each grade report names the version that produced it.
`reference_grade.json` is part of the answer key and must never be exposed to the agent.
"""

import datetime as dt
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from agent.schemas import NonEmptyStr, StrictModel

TOTAL_ID = "total"  # claim target / proposal item for an addition error in the total


class RubricItem(StrictModel):
    id: str = NonEmptyStr  # test items: the test function name; manual items: "rubric:<name>"
    kind: Literal["test", "manual"]
    marks: float = Field(gt=0)
    description: str = NonEmptyStr

    @model_validator(mode="after")
    def _id_matches_kind(self) -> "RubricItem":
        prefix = "test_" if self.kind == "test" else "rubric:"
        if not self.id.startswith(prefix):
            raise ValueError(f"{self.kind} rubric item id must start with {prefix!r}")
        return self


class Policy(StrictModel):
    id: str = Field(pattern=r"^policy:")
    marks: float = Field(gt=0)  # maximum deduction
    description: str = NonEmptyStr


class Assignment(StrictModel):
    id: str = NonEmptyStr
    title: str = NonEmptyStr
    module: str = Field(pattern=r"^\w+\.py$")
    tests_file: str = Field(pattern=r"^test_\w+\.py$")
    autograders: dict[str, str] = Field(min_length=1)  # version -> folder
    reference_autograder: str
    rubric: list[RubricItem] = Field(min_length=1)
    policies: list[Policy] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "Assignment":
        if self.reference_autograder not in self.autograders:
            raise ValueError("reference_autograder must be one of the autograders")
        ids = [item.id for item in self.rubric] + [p.id for p in self.policies]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate rubric or policy ids: {ids}")
        return self

    @property
    def total_marks(self) -> float:
        return sum(item.marks for item in self.rubric)

    def test_ids(self) -> list[str]:
        return [item.id for item in self.rubric if item.kind == "test"]

    def manual_ids(self) -> list[str]:
        return [item.id for item in self.rubric if item.kind == "manual"]

    def target_ids(self) -> set[str]:
        """Everything a claim may dispute: rubric items, policies and the total."""
        return {item.id for item in self.rubric} | {p.id for p in self.policies} | {TOTAL_ID}

    def marks_of(self, item_id: str) -> float:
        for item in [*self.rubric, *self.policies]:
            if item.id == item_id:
                return item.marks
        raise KeyError(item_id)


class Announcement(StrictModel):
    id: str = NonEmptyStr
    date: dt.date
    title: str = NonEmptyStr
    supersedes: list[str] = Field(default_factory=list)
    body: str = NonEmptyStr


class ManualMark(StrictModel):
    awarded: float = Field(ge=0)
    comment: str = ""


class GradeReport(StrictModel):
    """The grade as given to the student. `total` is stored, not derived, because an
    arithmetic error in the total is one of the planted error types."""

    assignment: str = NonEmptyStr
    submission: str = NonEmptyStr
    autograder: str = NonEmptyStr
    tests: dict[str, bool]
    manual: dict[str, ManualMark]
    penalties: dict[str, float] = Field(default_factory=dict)
    total: float


class ReferenceGrade(StrictModel):
    """Correct manual marks and penalties, decided by the team (answer key)."""

    manual: dict[str, float]
    penalties: dict[str, float] = Field(default_factory=dict)
    variant_of: str | None = None  # submission this one copies, adding only comments


# --- Loading ----------------------------------------------------------------------------


def load_assignment(folder: Path) -> Assignment:
    data = yaml.safe_load((folder / "assignment.yaml").read_text(encoding="utf-8"))
    return Assignment.model_validate(data)


def load_announcement(path: Path) -> Announcement:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path.name}: missing YAML front matter")
    header, sep, body = text[4:].partition("\n---\n")
    if not sep:
        raise ValueError(f"{path.name}: unterminated YAML front matter")
    return Announcement.model_validate({**yaml.safe_load(header), "body": body.strip()})


def load_announcements(folder: Path) -> list[Announcement]:
    paths = sorted((folder / "announcements").glob("*.md"))
    return sorted((load_announcement(p) for p in paths), key=lambda a: a.date)


def load_grade_report(folder: Path) -> GradeReport:
    return GradeReport.model_validate_json((folder / "grade_report.json").read_text("utf-8"))


def load_reference_grade(folder: Path) -> ReferenceGrade:
    path = folder / "reference_grade.json"
    return ReferenceGrade.model_validate(json.loads(path.read_text(encoding="utf-8")))
