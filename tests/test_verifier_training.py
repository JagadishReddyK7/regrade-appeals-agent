"""Verifier training data (TR1, TR2), dev evaluation examples, evaluation and serving."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from agent.schemas import VerifierExample
from data.formats import load_assignment, load_grade_report
from data.grader import check_assignment, reference_total, reported_total, run_autograder
from eval.run import load_config, verifier_backend
from eval.runlog import RunContext, RunLogger, read_events
from llm.client import MockLLM, OpenAIBackend
from llm.serve import build_command, server_section
from sandbox.cache import ResultCache
from sandbox.runner import SandboxRunner
from verifier.eval_data import from_examples, from_run, load_cases
from verifier.evaluate import INVALID, evaluate, macro_f1, score
from verifier.train import TrainConfig, load_train_config
from verifier.train_data import (
    HELDOUT_TONE,
    PAIRS_PER_VERDICT,
    TRAINING_TONES,
    Situation,
    ToolCallSpec,
    generate,
    label_for,
    load_situations,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SITUATIONS = sorted((DATA / "verifier_train").glob("*.yaml"))
RUN_V3 = next((ROOT / "results" / "pipeline_dev_v3").glob("pipeline_dev-*"))
# Marks each training submission wrongly lost (planted errors), by construction.
WITHHELD = {
    "TR1-S01": 4, "TR1-S02": 0, "TR1-S03": 4, "TR1-S04": 2, "TR1-S05": 0, "TR1-S06": 2,
    "TR2-S01": 4, "TR2-S02": 0, "TR2-S03": 1, "TR2-S04": 7, "TR2-S05": 0, "TR2-S06": 0,
}  # fmt: skip


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> SandboxRunner:
    return SandboxRunner(cache=ResultCache(tmp_path_factory.mktemp("cache")))


@pytest.fixture(scope="module")
def generated(runner: SandboxRunner) -> tuple[list[VerifierExample], list[VerifierExample]]:
    return generate(SITUATIONS, DATA, runner, seed=0)


# --- Training-only assignments -------------------------------------------------------------


@pytest.mark.parametrize("assignment", ["TR1", "TR2"])
def test_training_assignment_is_consistent(assignment: str, runner: SandboxRunner) -> None:
    folder = DATA / "assignments" / assignment
    assert check_assignment(folder, runner) == []
    spec = load_assignment(folder)
    from data.formats import load_reference_grade

    for sub in sorted((DATA / "submissions").glob(f"{assignment}-S*")):
        report = load_grade_report(sub)
        code = sub / spec.module
        assert run_autograder(runner, spec, folder, code, report.autograder) == report.tests
        correct = reference_total(runner, spec, folder, code, load_reference_grade(sub))
        assert math.isclose(correct - report.total, WITHHELD[sub.name]), sub.name
        if not sub.name.endswith("S04"):  # S04: planted addition error
            assert math.isclose(reported_total(spec, report), report.total), sub.name


def test_situations_target_items_with_deducted_marks() -> None:
    from data.grader import deducted_marks

    for path in SITUATIONS:
        spec = load_situations(path)
        assignment = load_assignment(DATA / "assignments" / spec.assignment)
        for s in spec.situations:
            report = load_grade_report(DATA / "submissions" / s.submission)
            # the pipeline's extractor only targets items with deducted marks, else null
            assert s.target is None or s.target in deducted_marks(assignment, report), s.id
            assert (s.verdict == "needs_human") == (not s.decisive), s.id


# --- Generated examples --------------------------------------------------------------------


def test_generated_counts_tones_and_pairs(
    generated: tuple[list[VerifierExample], list[VerifierExample]],
) -> None:
    train, heldout = generated
    situations = [s for p in SITUATIONS for s in load_situations(p).situations]
    expected = sum(PAIRS_PER_VERDICT[s.verdict] for s in situations) * len(TRAINING_TONES)
    assert len(train) == expected
    assert {e.tone for e in train} == set(TRAINING_TONES)
    assert HELDOUT_TONE not in {e.tone for e in train}
    assert {e.tone for e in heldout} == {*TRAINING_TONES, HELDOUT_TONE}
    assert len(heldout) == 2 * len(situations) * 4
    pairs: dict[str, list[VerifierExample]] = {}
    for e in train:
        pairs.setdefault(str(e.pair_id), []).append(e)
    for members in pairs.values():  # tone variants share label and evidence sources
        assert len(members) == len(TRAINING_TONES)
        assert len({m.label for m in members}) == 1
        assert len({tuple(i.source for i in m.bundle.evidence) for m in members}) == 1


def test_generated_evidence_matches_pipeline_layout(
    generated: tuple[list[VerifierExample], list[VerifierExample]],
) -> None:
    train, _ = generated
    for e in train:
        sources = [item.source for item in e.bundle.evidence]
        assert sources[:2] == ["get_grade_report", "get_submission"]
        submission = e.bundle.evidence[1].text
        assert (
            "Award full marks" in submission
            or "answer supported" in submission
            or "restore all deducted marks" in submission
        ) == (e.tone == "hidden_instruction")


def test_cited_evidence_settles_the_claim(
    generated: tuple[list[VerifierExample], list[VerifierExample]],
) -> None:
    train, _ = generated
    for e in train:
        if e.label.evidence_id is None:
            continue
        cited = next(i for i in e.bundle.evidence if i.id == e.label.evidence_id)
        pair = str(e.pair_id)
        if "TR1-01" in pair:  # faulty test: the run shows the wrong expectation
            assert "== 3" in cited.text and '"passed": false' in cited.text
        if "TR1-04" in pair or "TR1-06" in pair:  # the announcement in force
            assert "statistics.median inside median" in cited.text
        if "TR2-11" in pair:  # the student's add_item overwrites the quantity
            assert "{'apple': 3}" in cited.text
        if "TR1-07" in pair:
            assert '"total_matches_items": false' in cited.text


def test_label_cites_first_decisive_item() -> None:
    run = ToolCallSpec(tool="run_tests", test_ids=["t"])
    other = ToolCallSpec(tool="search_course_docs", query="q")
    base: dict[str, Any] = {"id": "x", "submission": "s", "target": "t", "claims": ["c"]}
    situation = Situation(verdict="refuted", decisive=[run], **base)
    assert label_for(situation, [other, run]).evidence_id == "E4"
    situation = Situation(verdict="supported", decisive=[ToolCallSpec(tool="submission")], **base)
    assert label_for(situation, [other]).evidence_id == "E2"
    situation = Situation(verdict="needs_human", decisive=[], **base)
    assert label_for(situation, [other]).evidence_id is None


# --- Dev evaluation examples and the evaluation --------------------------------------------


def test_eval_examples_rebuild_verifier_input_from_run() -> None:
    examples = from_run(RUN_V3, load_cases(DATA / "cases" / "dev"))
    assert len(examples) == 30
    first = next(e for e in examples if e.example_id == "dev-001-t1-C1")
    assert first.gold == "supported" and first.tone == "polite"
    assert [i.source for i in first.bundle.evidence][:3] == [
        "get_grade_report",
        "get_submission",
        "run_tests",
    ]
    assert "evidence_id" not in first.bundle.evidence[0].text[:30]


def test_macro_f1_and_score() -> None:
    assert macro_f1([("supported", "supported"), ("refuted", "supported")]) == pytest.approx(
        (2 / 3 + 0.0) / 2
    )
    examples = from_examples([])
    assert score(examples, [])["accuracy"] is None


def test_evaluate_counts_invalid_and_errors(
    generated: tuple[list[VerifierExample], list[VerifierExample]], tmp_path: Path
) -> None:
    _, heldout = generated
    examples = from_examples(heldout[:3])
    answers = [
        json.dumps(
            {
                "verdict": examples[0].gold,
                "evidence_id": None if examples[0].gold == "needs_human" else "E1",
            }
        ),
        "not json",
    ]  # the third request finds no scripted answer: a server error
    logger = RunLogger(tmp_path / "events.jsonl", RunContext("r", 0, "h"))
    predictions = evaluate(examples, MockLLM(answers), logger)
    assert predictions[0] == examples[0].gold and predictions[1:] == [INVALID, INVALID]
    assert len(read_events(tmp_path / "events.jsonl", "case_error")) == 1
    metrics = score(examples, predictions)
    assert metrics["invalid"] == 2 and metrics["n"] == 3


# --- Configs and serving -------------------------------------------------------------------


def test_training_config_merges_for_vllm() -> None:
    config = load_train_config(ROOT / "configs" / "verifier_train.yaml")
    assert isinstance(config, TrainConfig) and config.merge and config.max_length == 4096


def test_pipeline_ft_config_points_verifier_at_second_server() -> None:
    config = load_config(ROOT / "configs" / "pipeline_ft_dev.yaml")
    backend = verifier_backend(config, OpenAIBackend(base_url="http://main/v1", model="m"))
    assert isinstance(backend, OpenAIBackend) and backend.model == "/kaggle/temp/verifier-ft"
    llm = server_section(config, verifier=True)
    assert llm["serve"]["gpu"] == 1 and "--port" in build_command(llm)


def test_server_section_and_base_model() -> None:
    with pytest.raises(ValueError, match="verifier.llm"):
        server_section({"llm": {}, "verifier": {"llm": None}}, verifier=True)
    llm = {"model": "name", "serve": {"base_model": "Qwen/Qwen2.5-3B-Instruct"}}
    assert build_command(llm)[2] == "Qwen/Qwen2.5-3B-Instruct"
    assert build_command({"model": "Qwen/x"})[2] == "Qwen/x"


def test_train_cli_can_merge_an_earlier_adapter() -> None:
    from verifier.train import build_parser

    args = build_parser().parse_args(["--config", "c.yaml", "--merge-only", "runs/r1"])
    assert args.merge_only == Path("runs/r1")
    assert build_parser().parse_args(["--config", "c.yaml"]).merge_only is None
