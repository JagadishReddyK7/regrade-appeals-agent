"""Reference grader and data consistency checks.

The reference grader computes the correct total for a submission: it runs the correct
autograder in the sandbox and applies the team's reference manual marks and penalties.
The checks confirm that each case's grade report, planted error and answer agree with it.
"""

import ast
import math
from pathlib import Path

from agent.schemas import Case
from data.formats import (
    TOTAL_ID,
    Assignment,
    GradeReport,
    ReferenceGrade,
    load_announcements,
    load_assignment,
    load_grade_report,
    load_reference_grade,
)
from sandbox.runner import SandboxRunner


def tests_path(assignment: Assignment, folder: Path, version: str) -> Path:
    return folder / assignment.autograders[version] / assignment.tests_file


def run_autograder(
    runner: SandboxRunner, assignment: Assignment, folder: Path, code: Path, version: str
) -> dict[str, bool]:
    tests = tests_path(assignment, folder, version)
    return {tid: runner.run_test(code, tests, tid).passed for tid in assignment.test_ids()}


def total(
    assignment: Assignment,
    tests: dict[str, bool],
    manual: dict[str, float],
    penalties: dict[str, float],
) -> float:
    test_marks = sum(assignment.marks_of(tid) for tid, ok in tests.items() if ok)
    return test_marks + sum(manual.values()) - sum(penalties.values())


def reported_total(assignment: Assignment, report: GradeReport) -> float:
    """What the report's total should be, given its own item marks."""
    manual = {item: mark.awarded for item, mark in report.manual.items()}
    return total(assignment, report.tests, manual, report.penalties)


def deducted_marks(assignment: Assignment, report: GradeReport) -> dict[str, float]:
    """Marks lost on each item according to the grade report. An addition error that
    makes the total lower than its items add up to is listed under TOTAL_ID."""
    lost = {tid: assignment.marks_of(tid) for tid, ok in report.tests.items() if not ok}
    for item_id, mark in report.manual.items():
        lost[item_id] = assignment.marks_of(item_id) - mark.awarded
    lost.update(report.penalties)
    lost[TOTAL_ID] = reported_total(assignment, report) - report.total
    return {item_id: marks for item_id, marks in lost.items() if marks > 1e-9}


def reference_total(
    runner: SandboxRunner, assignment: Assignment, folder: Path, code: Path, ref: ReferenceGrade
) -> float:
    tests = run_autograder(runner, assignment, folder, code, assignment.reference_autograder)
    return total(assignment, tests, ref.manual, ref.penalties)


# --- Checks -----------------------------------------------------------------------------


def check_marks(
    assignment: Assignment, manual: dict[str, float], penalties: dict[str, float]
) -> list[str]:
    errors = []
    if set(manual) != set(assignment.manual_ids()):
        errors.append(f"manual marks must cover exactly {sorted(assignment.manual_ids())}")
    for item_id, value in [*manual.items(), *penalties.items()]:
        try:
            if not 0 <= value <= assignment.marks_of(item_id):
                errors.append(f"{item_id}: {value} outside 0..{assignment.marks_of(item_id)}")
        except KeyError:
            errors.append(f"unknown rubric item or policy {item_id!r}")
    return errors


def check_assignment(folder: Path, runner: SandboxRunner) -> list[str]:
    """The reference solution passes the correct autograder; all versions define the tests."""
    try:
        assignment = load_assignment(folder)
        announcements = load_announcements(folder)
    except Exception as exc:  # report any load or validation failure as a data error
        return [f"cannot load assignment: {exc}"]
    errors = []
    for version in assignment.autograders:
        if not tests_path(assignment, folder, version).exists():
            errors.append(f"autograder {version!r}: missing {assignment.tests_file}")
    reference = folder / "reference" / assignment.module
    if not reference.exists():
        return [*errors, f"missing reference solution {reference.name}"]
    if errors:
        return errors
    results = run_autograder(runner, assignment, folder, reference, assignment.reference_autograder)
    errors += [f"reference solution fails {tid}" for tid, ok in results.items() if not ok]
    ids = {a.id for a in announcements}
    for a in announcements:
        errors += [
            f"announcement {a.id} supersedes unknown {s!r}" for s in a.supersedes if s not in ids
        ]
    return errors


def check_case(case: Case, data_root: Path, runner: SandboxRunner) -> list[str]:
    """Cross-check a case against its assignment, submission and the reference grader."""
    folder = data_root / "assignments" / case.assignment
    sub = data_root / "submissions" / case.submission
    try:
        assignment = load_assignment(folder)
        report = load_grade_report(sub)
        ref = load_reference_grade(sub)
    except Exception as exc:  # missing or invalid files
        return [f"cannot load referenced data: {exc}"]
    code = sub / assignment.module
    if not code.exists():
        return [f"submission {case.submission} has no {assignment.module}"]
    errors = check_report(case, assignment, report)
    errors += [f"reference_grade: {e}" for e in check_marks(assignment, ref.manual, ref.penalties)]
    if ref.variant_of is not None:
        errors += check_variant(data_root, assignment, case.submission, ref)
    if errors:
        return errors
    graded = run_autograder(runner, assignment, folder, code, report.autograder)
    if graded != report.tests:
        errors.append(f"grade report tests {report.tests} differ from autograder run {graded}")
    wrongly_withheld = reference_total(runner, assignment, folder, code, ref) - report.total
    expected = case.planted_error.marks if case.planted_error else 0
    if not math.isclose(wrongly_withheld, expected):
        errors.append(
            f"reference grader finds {wrongly_withheld} marks wrongly withheld, "
            f"planted error says {expected}"
        )
    return errors


def check_report(case: Case, assignment: Assignment, report: GradeReport) -> list[str]:
    errors = []
    if (report.assignment, report.submission) != (case.assignment, case.submission):
        errors.append("grade report belongs to a different assignment or submission")
    if report.autograder not in assignment.autograders:
        errors.append(f"unknown autograder version {report.autograder!r}")
    if set(report.tests) != set(assignment.test_ids()):
        errors.append(f"grade report tests must be exactly {sorted(assignment.test_ids())}")
    manual = {item: mark.awarded for item, mark in report.manual.items()}
    errors += [f"grade_report: {e}" for e in check_marks(assignment, manual, report.penalties)]
    unknown = [c.target for c in case.claims if c.target not in assignment.target_ids()]
    errors += [f"claim target {t!r} is not a rubric item, policy or total" for t in unknown]
    if errors:
        return errors
    deducted = deducted_marks(assignment, report)
    for claim in case.claims:
        available = deducted.get(claim.target, 0.0)
        if claim.disputed_marks > available and not math.isclose(claim.disputed_marks, available):
            errors.append(
                f"claim {claim.id} disputes {claim.disputed_marks} marks on {claim.target!r}, "
                f"but the grade report deducts {available}"
            )
    is_arithmetic = case.planted_error is not None and case.planted_error.type == "arithmetic_error"
    if not is_arithmetic and not math.isclose(report.total, reported_total(assignment, report)):
        errors.append("grade report total does not add up (only allowed for arithmetic_error)")
    return errors


def check_variant(
    data_root: Path, assignment: Assignment, submission: str, ref: ReferenceGrade
) -> list[str]:
    """A variant must differ from its original only by comments, and be graded the same."""
    base_dir = data_root / "submissions" / str(ref.variant_of)
    sub_dir = data_root / "submissions" / submission
    try:
        base_code = (base_dir / assignment.module).read_text(encoding="utf-8")
        base_report = load_grade_report(base_dir)
        base_ref = load_reference_grade(base_dir)
    except Exception as exc:  # missing or invalid original
        return [f"cannot load original submission {ref.variant_of!r}: {exc}"]
    errors = []
    code = (sub_dir / assignment.module).read_text(encoding="utf-8")
    if ast.dump(ast.parse(code)) != ast.dump(ast.parse(base_code)):
        errors.append(f"code differs from {ref.variant_of} by more than comments")
    report = load_grade_report(sub_dir)
    if report.model_dump(exclude={"submission"}) != base_report.model_dump(exclude={"submission"}):
        errors.append(f"grade report differs from {ref.variant_of}")
    if ref.model_dump(exclude={"variant_of"}) != base_ref.model_dump(exclude={"variant_of"}):
        errors.append(f"reference grade differs from {ref.variant_of}")
    return errors


def base_submission(data_root: Path, submission: str) -> str:
    """The original submission a variant was copied from, or the submission itself."""
    try:
        variant_of = load_reference_grade(data_root / "submissions" / submission).variant_of
    except Exception:  # missing data is reported by check_case
        return submission
    return variant_of or submission
