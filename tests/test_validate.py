import json
from pathlib import Path

import pytest

from data.validate import main, validate_folder
from tests.test_schemas import make_case

DEV_DIR = Path(__file__).resolve().parents[1] / "data" / "cases" / "dev"


def write_case(folder: Path, name: str, case: dict) -> Path:
    path = folder / name
    path.write_text(json.dumps(case), encoding="utf-8")
    return path


def test_dev_examples_are_valid() -> None:
    results = validate_folder(DEV_DIR)
    assert len(results) >= 3
    assert all(not errors for errors in results.values()), results


def test_invalid_file_reported(tmp_path: Path) -> None:
    write_case(tmp_path, "ok.json", make_case())
    bad = write_case(tmp_path, "bad.json", make_case(case_id="dev-y", tone="sarcastic"))
    results = validate_folder(tmp_path)
    assert results[bad] and results[bad][0].startswith("tone:")
    assert results[tmp_path / "ok.json"] == []


def test_malformed_json_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert validate_folder(tmp_path)[path]


def test_duplicate_case_id_reported(tmp_path: Path) -> None:
    write_case(tmp_path, "a.json", make_case())
    dup = write_case(tmp_path, "b.json", make_case())
    assert any("already used" in e for e in validate_folder(tmp_path)[dup])


def test_split_must_match_folder(tmp_path: Path) -> None:
    folder = tmp_path / "train"
    folder.mkdir()
    path = write_case(folder, "a.json", make_case())
    assert any("split" in e for e in validate_folder(folder)[path])


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(DEV_DIR)]) == 0
    write_case(tmp_path, "bad.json", make_case(tone="sarcastic"))
    assert main([str(tmp_path)]) == 1
    assert "INVALID" in capsys.readouterr().out
    assert main([]) == 2
    assert main([str(tmp_path / "missing")]) == 2
