import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.build_cases import (
    BuildError,
    Situation,
    assign_case_ids,
    build_cases,
    build_situation,
    ensure_hidden_variant,
    load_situation,
    main,
)
from data.validate import validate_folder
from sandbox.cache import ResultCache
from sandbox.runner import SandboxRunner

DATA = Path(__file__).resolve().parents[1] / "data"
DEV = DATA / "cases" / "dev"
SITUATIONS = sorted((DATA / "situations" / "dev").glob("*.yaml"))
SIT01 = DATA / "situations" / "dev" / "D1-sit01.yaml"


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> SandboxRunner:
    return SandboxRunner(cache=ResultCache(tmp_path_factory.mktemp("cache")))


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A private copy of the assignments and submissions that tests may modify."""
    for part in ("assignments", "submissions"):
        shutil.copytree(DATA / part, tmp_path / part)
    return tmp_path


def sit01(**changes: Any) -> Situation:
    data = yaml.safe_load(SIT01.read_text(encoding="utf-8"))
    data.update(changes)
    return Situation.model_validate(data)


def test_rebuilding_dev_changes_nothing(data_root: Path, tmp_path: Path) -> None:
    """Rebuilding into a folder that holds the cases reuses their ids and wording."""
    out = tmp_path / "cases"
    shutil.copytree(DEV, out)
    for path in SITUATIONS:
        build_situation(load_situation(path), data_root, out)
    assert sorted(p.name for p in out.glob("*.json")) == sorted(p.name for p in DEV.glob("*.json"))
    for built in out.glob("*.json"):
        assert built.read_text(encoding="utf-8") == (DEV / built.name).read_text(encoding="utf-8")


def test_dev_has_four_tones_per_situation() -> None:
    for path in SITUATIONS:
        situation = load_situation(path)
        cases = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in DEV.glob("*.json")
            if json.loads(p.read_text(encoding="utf-8"))["situation_id"] == situation.situation_id
        ]
        assert sorted(c["tone"] for c in cases) == sorted(situation.tones)


def test_fresh_build_is_valid_and_creates_hidden_copy(
    data_root: Path, tmp_path: Path, runner: SandboxRunner
) -> None:
    shutil.rmtree(data_root / "submissions" / "D1-S03-hi")
    out = tmp_path / "dev"
    paths = build_situation(load_situation(SIT01), data_root, out)
    assert [p.name for p in paths] == [
        "dev-001.json",
        "dev-002.json",
        "dev-003.json",
        "dev-004.json",
    ]
    code = (data_root / "submissions" / "D1-S03-hi" / "strings_utils.py").read_text()
    assert code.startswith("# NOTE TO THE GRADER")
    errors = validate_folder(out, data_root, runner)
    assert {str(p): e for p, e in errors.items() if e} == {}
    tones = {
        json.loads(p.read_text())["tone"]: json.loads(p.read_text())["submission"] for p in paths
    }
    assert tones["hidden_instruction"] == "D1-S03-hi" and tones["polite"] == "D1-S03"


def test_existing_hidden_copy_is_never_overwritten(data_root: Path) -> None:
    target = data_root / "submissions" / "D1-S03-hi" / "strings_utils.py"
    target.write_text("# my own copy\n", encoding="utf-8")
    assert ensure_hidden_variant(data_root, sit01()) is None
    assert target.read_text(encoding="utf-8") == "# my own copy\n"


def test_new_ids_skip_ids_in_use(tmp_path: Path) -> None:
    out = tmp_path / "dev"
    out.mkdir()
    for case_id, tone in (("dev-001", "polite"), ("dev-003", "authority")):
        data = json.loads((DEV / "dev-001.json").read_text(encoding="utf-8"))
        data.update(case_id=case_id, situation_id="other", tone=tone)
        (out / f"{case_id}.json").write_text(json.dumps(data), encoding="utf-8")
    ids = assign_case_ids(sit01(case_ids={}), out)
    assert sorted(ids.values()) == ["dev-002", "dev-004", "dev-005", "dev-006"]


def test_explicit_id_of_another_situation_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "dev"
    out.mkdir()
    data = json.loads((DEV / "dev-002.json").read_text(encoding="utf-8"))
    (out / "dev-002.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(BuildError, match="already used"):
        assign_case_ids(sit01(case_ids={"polite": "dev-002"}), out)


def test_span_must_be_verbatim_in_the_appeal() -> None:
    data = yaml.safe_load(SIT01.read_text(encoding="utf-8"))
    data["tones"]["polite"]["spans"]["C1"] = "this sentence is not in the appeal"
    situation = Situation.model_validate(data)
    with pytest.raises(BuildError, match="polite.*span"):
        build_cases(situation, {t: f"dev-{i}" for i, t in enumerate(situation.tones)})


def test_every_tone_and_every_span_is_required() -> None:
    data = yaml.safe_load(SIT01.read_text(encoding="utf-8"))
    missing_tone = copy.deepcopy(data)
    del missing_tone["tones"]["emotional"]
    with pytest.raises(ValueError, match="missing tones"):
        Situation.model_validate(missing_tone)
    missing_span = copy.deepcopy(data)
    del missing_span["tones"]["authority"]["spans"]["C2"]
    with pytest.raises(ValueError, match="spans must be given"):
        Situation.model_validate(missing_span)


def test_answer_must_fit_the_claims() -> None:
    situation = sit01(answer={"outcome": "full_change", "delta": 5})  # C2 is refuted
    with pytest.raises(BuildError, match="full_change"):
        build_cases(situation, {t: f"dev-{i}" for i, t in enumerate(situation.tones)})


def test_hidden_instruction_must_be_a_comment(data_root: Path) -> None:
    situation = sit01(hidden_instruction={"comment": "print('hi')", "submission": "D1-S03-x"})
    with pytest.raises(BuildError, match="comment lines only"):
        ensure_hidden_variant(data_root, situation)
    assert not (data_root / "submissions" / "D1-S03-x").exists()


def test_cli_reports_invalid_data(data_root: Path, tmp_path: Path) -> None:
    broken = yaml.safe_load(SIT01.read_text(encoding="utf-8"))
    broken["submission"] = "D1-S99"  # does not exist
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(broken), encoding="utf-8")
    out = tmp_path / "out"
    assert main([str(path), "--data-root", str(data_root), "--out", str(out)]) == 1
