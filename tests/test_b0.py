import shutil
from pathlib import Path

import pytest

from agent.schemas import Case
from baselines.b0 import FixedBaseline, accepted_marks, mentions
from eval.runlog import RunContext, RunLogger, read_events

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEV = DATA / "cases" / "dev"


def load_case(case_id: str) -> Case:
    return Case.model_validate_json((DEV / f"{case_id}.json").read_text(encoding="utf-8"))


@pytest.fixture
def logger(tmp_path: Path) -> RunLogger:
    return RunLogger(tmp_path / "events.jsonl", RunContext("r", 0, "h"))


def test_mentions_ids_and_rubric_names() -> None:
    assert mentions("test_empty_string marked it wrong", "test_empty_string")
    assert mentions("I lost 2 marks for Style", "rubric:style")
    assert mentions("waive the late penalty", "policy:late_penalty")
    assert not mentions("test_empty_strings", "test_empty_string")
    assert not mentions("stylesheet", "rubric:style")


def test_accepted_marks_named_items_else_everything() -> None:
    deducted = {"test_a": 4.0, "rubric:style": 2.0}
    assert accepted_marks("test_a is wrong", deducted) == 4.0
    assert accepted_marks("test_a and my style", deducted) == 6.0
    assert accepted_marks("please recheck", deducted) == 6.0
    assert accepted_marks("anything", {}) == 0.0


@pytest.mark.parametrize(
    ("case_id", "outcome", "delta"),
    [
        ("dev-001", "full_change", 7.0),  # names the faulty test (5) and style (2)
        ("dev-002", "full_change", 3.0),  # names no item: everything deducted
        ("dev-003", "full_change", 10.0),  # names the late penalty
        ("dev-004", "full_change", 4.0),  # the test did fail: accept still believes it
    ],
)
def test_accept(case_id: str, outcome: str, delta: float, logger: RunLogger) -> None:
    output = FixedBaseline(DATA, "accept").run(load_case(case_id), logger.for_case(case_id))
    assert output.first is not None
    assert (output.first.outcome, output.first.delta) == (outcome, delta)
    assert output.second == output.first and output.followup_run


def test_accept_without_deductions_is_no_change(logger: RunLogger, tmp_path: Path) -> None:
    data = tmp_path / "data"
    shutil.copytree(DATA / "assignments", data / "assignments")
    shutil.copytree(DATA / "submissions" / "D1-S05", data / "submissions" / "D1-S05")
    report = data / "submissions" / "D1-S05" / "grade_report.json"
    text = report.read_text(encoding="utf-8")
    report.write_text(text.replace("false", "true").replace('"total": 26', '"total": 30'), "utf-8")
    case = load_case("dev-014")
    output = FixedBaseline(data, "accept").run(case, logger.for_case(case.case_id))
    assert output.first is not None
    assert (output.first.outcome, output.first.delta) == ("no_change", 0.0)


def test_reject_always_no_change(logger: RunLogger, tmp_path: Path) -> None:
    for case_id in ("dev-001", "dev-002", "dev-003"):
        output = FixedBaseline(DATA, "reject").run(load_case(case_id), logger.for_case(case_id))
        assert output.first is not None
        assert (output.first.outcome, output.first.delta) == ("no_change", 0.0)
    decisions = read_events(tmp_path / "events.jsonl", "decision")
    assert [d["system"] for d in decisions] == ["b0_reject"] * 3


def test_accept_never_reads_answer_key(logger: RunLogger, tmp_path: Path) -> None:
    data = tmp_path / "data"
    shutil.copytree(DATA / "assignments", data / "assignments")
    shutil.copytree(DATA / "submissions", data / "submissions")
    for path in data.rglob("reference_grade.json"):
        path.unlink()
    shutil.rmtree(data / "assignments" / "D1" / "reference")
    output = FixedBaseline(data, "accept").run(load_case("dev-001"), logger.for_case("dev-001"))
    assert output.first is not None and output.first.delta == 7.0


def test_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown B0 mode"):
        FixedBaseline(DATA, "maybe")
