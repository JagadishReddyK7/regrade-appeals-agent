"""Validate every case JSON file in a folder.

Usage: python -m data.validate <folder> [--data-root data] [--schema-only]

Without --schema-only, also checks each referenced assignment and submission and runs the
reference grader (student code runs in the sandbox). Exits with status 1 if anything is
invalid.
"""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from agent.schemas import Case
from data.consistency import check_consistency
from data.grader import base_submission, check_assignment, check_case
from sandbox.runner import SandboxRunner


def validate_file(path: Path) -> tuple[Case | None, list[str]]:
    """Return the parsed case, or None and a list of error messages."""
    try:
        return Case.model_validate_json(path.read_text(encoding="utf-8")), []
    except ValidationError as exc:
        return None, [_format_error(err) for err in exc.errors()]
    except (OSError, UnicodeDecodeError) as exc:
        return None, [str(exc)]


def _format_error(err: dict) -> str:
    loc = ".".join(str(part) for part in err["loc"]) or "<case>"
    return f"{loc}: {err['msg']}"


def validate_folder(
    folder: Path, data_root: Path | None = None, runner: SandboxRunner | None = None
) -> dict[Path, list[str]]:
    """Errors per file (empty if valid). With data_root, also cross-checks referenced data."""
    results: dict[Path, list[str]] = {}
    cases: dict[Path, Case] = {}
    seen_ids: dict[str, Path] = {}
    for path in sorted(folder.rglob("*.json")):
        case, errors = validate_file(path)
        if case is not None:
            if case.case_id in seen_ids:
                errors.append(f"case_id {case.case_id!r} already used in {seen_ids[case.case_id]}")
            else:
                seen_ids[case.case_id] = path
            if folder.name in ("dev", "train", "test") and case.split != folder.name:
                errors.append(f"split {case.split!r} does not match folder {folder.name!r}")
            cases[path] = case
        results[path] = errors
    if data_root is not None:
        cross_check(cases, results, data_root, runner or SandboxRunner())
        root = data_root
        base_of = lambda case: base_submission(root, case.submission)  # noqa: E731
    else:
        base_of = lambda case: case.submission  # noqa: E731
    for path, errors in check_consistency(cases, base_of).items():
        results[path] += errors
    return results


def cross_check(
    cases: dict[Path, Case], results: dict[Path, list[str]], data_root: Path, runner: SandboxRunner
) -> None:
    for assignment_id in sorted({case.assignment for case in cases.values()}):
        folder = data_root / "assignments" / assignment_id
        errors = check_assignment(folder, runner)
        if errors:
            results[folder / "assignment.yaml"] = errors
    for path, case in cases.items():
        results[path] += check_case(case, data_root, runner)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m data.validate")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 2
    data_root = None if args.schema_only else args.data_root
    results = validate_folder(args.folder, data_root)
    invalid = {path: errors for path, errors in results.items() if errors}
    for path, errors in invalid.items():
        print(f"INVALID {path}")
        for error in errors:
            print(f"  - {error}")
    n_cases = sum(1 for p in results if p.suffix == ".json")
    n_bad = sum(1 for p in invalid if p.suffix == ".json")
    print(f"{n_cases - n_bad}/{n_cases} cases valid")
    return 1 if invalid or not n_cases else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
