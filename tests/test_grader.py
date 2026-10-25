import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agent.schemas import Case
from data.formats import (
    Assignment,
    RubricItem,
    load_announcement,
    load_announcements,
    load_assignment,
)
from data.grader import check_assignment, check_case
from sandbox.cache import ResultCache
from sandbox.runner import SandboxRunner

DATA = Path(__file__).resolve().parents[1] / "data"
D1 = DATA / "assignments" / "D1"


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> SandboxRunner:
    return SandboxRunner(cache=ResultCache(tmp_path_factory.mktemp("cache")))


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A private copy of the assignments and submissions that tests may modify."""
    for part in ("assignments", "submissions"):
        shutil.copytree(DATA / part, tmp_path / part)
    return tmp_path


def load_case(case_id: str) -> Case:
    return Case.model_validate_json((DATA / "cases" / "dev" / f"{case_id}.json").read_text())


def edit_json(path: Path, **changes: Any) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_d1_assignment() -> None:
    assignment = load_assignment(D1)
    assert assignment.total_marks == 30
    assert len(assignment.test_ids()) == 6
    assert "policy:late_penalty" in assignment.target_ids()


def test_announcements_sorted_with_supersedes() -> None:
    a1, a2 = load_announcements(D1)
    assert a1.date < a2.date
    assert a2.supersedes == ["A1"]


def test_announcement_requires_front_matter(tmp_path: Path) -> None:
    path = tmp_path / "A9.md"
    path.write_text("no header here", encoding="utf-8")
    with pytest.raises(ValueError, match="front matter"):
        load_announcement(path)


def test_rubric_item_id_must_match_kind() -> None:
    with pytest.raises(ValidationError, match="rubric:"):
        RubricItem(id="style", kind="manual", marks=2, description="x")


def test_reference_autograder_must_exist() -> None:
    data = load_assignment(D1).model_dump()
    data["reference_autograder"] = "missing"
    with pytest.raises(ValidationError, match="reference_autograder"):
        Assignment.model_validate(data)


def test_d1_assignment_is_consistent(runner: SandboxRunner) -> None:
    assert check_assignment(D1, runner) == []


def test_broken_reference_solution_detected(runner: SandboxRunner, data_root: Path) -> None:
    folder = data_root / "assignments" / "D1"
    (folder / "reference" / "strings_utils.py").write_text(
        "def count_vowels(t):\n    return 0\n\ndef reverse_words(t):\n    return t\n"
    )
    errors = check_assignment(folder, runner)
    assert "reference solution fails test_count_vowels_basic" in errors


@pytest.mark.parametrize("case_id", ["dev-001", "dev-002", "dev-003", "dev-004"])
def test_dev_cases_match_reference_grader(runner: SandboxRunner, case_id: str) -> None:
    assert check_case(load_case(case_id), DATA, runner) == []


def test_wrong_total_detected(runner: SandboxRunner, data_root: Path) -> None:
    edit_json(data_root / "submissions" / "D1-S11" / "grade_report.json", total=26)
    errors = check_case(load_case("dev-002"), data_root, runner)
    assert any("does not add up" in e for e in errors)


def test_wrong_total_allowed_for_arithmetic_error(runner: SandboxRunner, data_root: Path) -> None:
    edit_json(data_root / "submissions" / "D1-S11" / "grade_report.json", total=24)
    case = load_case("dev-002").model_dump()
    case["planted_error"] = {
        "type": "arithmetic_error",
        "target": "rubric:approach",
        "description": "total summed wrongly",
        "marks": 3,
    }
    edit_json(
        data_root / "submissions" / "D1-S11" / "reference_grade.json",
        manual={"rubric:style": 2, "rubric:approach": 0},
    )
    assert check_case(Case.model_validate(case), data_root, runner) == []


def test_planted_marks_must_match_reference_grader(runner: SandboxRunner, data_root: Path) -> None:
    edit_json(
        data_root / "submissions" / "D1-S11" / "reference_grade.json",
        manual={"rubric:style": 2, "rubric:approach": 0},
    )
    errors = check_case(load_case("dev-002"), data_root, runner)
    assert any("wrongly withheld" in e for e in errors)


def test_report_must_match_autograder_run(runner: SandboxRunner, data_root: Path) -> None:
    path = data_root / "submissions" / "D1-S05" / "grade_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["tests"]["test_count_vowels_uppercase"] = True
    report["total"] = 30
    path.write_text(json.dumps(report), encoding="utf-8")
    errors = check_case(load_case("dev-004"), data_root, runner)
    assert any("differ from autograder run" in e for e in errors)


def test_unknown_claim_target_detected(runner: SandboxRunner) -> None:
    case = load_case("dev-004").model_dump()
    case["claims"][0]["target"] = "test_does_not_exist"
    errors = check_case(Case.model_validate(case), DATA, runner)
    assert any("not a rubric item" in e for e in errors)


def test_missing_submission_detected(runner: SandboxRunner, data_root: Path) -> None:
    shutil.rmtree(data_root / "submissions" / "D1-S07")
    errors = check_case(load_case("dev-003"), data_root, runner)
    assert errors and errors[0].startswith("cannot load referenced data")


def test_marks_out_of_range_detected(runner: SandboxRunner, data_root: Path) -> None:
    edit_json(
        data_root / "submissions" / "D1-S07" / "reference_grade.json",
        manual={"rubric:style": 5, "rubric:approach": 3},
    )
    errors = check_case(load_case("dev-003"), data_root, runner)
    assert any("rubric:style: 5" in e for e in errors)
