"""Validate every case JSON file in a folder.

Usage: python -m data.validate <folder>
Exits with status 1 if any file is invalid.
"""

import sys
from pathlib import Path

from pydantic import ValidationError

from agent.schemas import Case


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


def validate_folder(folder: Path) -> dict[Path, list[str]]:
    """Validate all *.json files under folder; return errors per file (empty if valid)."""
    results: dict[Path, list[str]] = {}
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
        results[path] = errors
    return results


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m data.validate <folder>", file=sys.stderr)
        return 2
    folder = Path(argv[0])
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2
    results = validate_folder(folder)
    invalid = {path: errors for path, errors in results.items() if errors}
    for path, errors in invalid.items():
        print(f"INVALID {path}")
        for error in errors:
            print(f"  - {error}")
    print(f"{len(results) - len(invalid)}/{len(results)} cases valid")
    return 1 if invalid or not results else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
