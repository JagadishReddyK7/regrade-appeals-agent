import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from agent.schemas import Case
from baselines.b1_direct import DirectBaseline, course_context
from eval.run import code_version, is_test_split, load_cases, load_config, run_eval
from eval.runlog import RunContext, RunLogger, read_events
from llm.client import ChatResponse, MockLLM, ToolCall

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEV = DATA / "cases" / "dev"


def answer(outcome: str, delta: float, evidence: list[str] | None = None) -> str:
    return json.dumps(
        {
            "outcome": outcome,
            "delta": delta,
            "rationale": "r",
            "confidence": 0.8,
            "evidence_ids": evidence or ["grade_report"],
        }
    )


def config(tmp_path: Path, system: str = "b1", **extra: Any) -> dict[str, Any]:
    # The runner tests script the model for exactly these four cases; the dev folder grows.
    cases = tmp_path / "cases"
    cases.mkdir(exist_ok=True)
    for case_id in ("dev-001", "dev-002", "dev-003", "dev-004"):
        shutil.copy(DEV / f"{case_id}.json", cases / f"{case_id}.json")
    return {
        "name": f"{system}_test",
        "system": system,
        "cases": str(cases),
        "data_root": str(DATA),
        "output_dir": str(tmp_path / "runs"),
        "seed": 0,
        "followup": True,
        "sandbox": {"cache_dir": str(tmp_path / "cache")},
        **extra,
    }


def load_case(case_id: str) -> Case:
    return Case.model_validate_json((DEV / f"{case_id}.json").read_text())


@pytest.fixture
def logger(tmp_path: Path) -> RunLogger:
    return RunLogger(tmp_path / "events.jsonl", RunContext("r", 0, "h")).for_case("dev-002")


# --- B1 ---------------------------------------------------------------------------------


def test_b1_prompt_has_documents_but_no_answer_key() -> None:
    case = load_case("dev-002")
    text, doc_ids = course_context(DATA, case)
    assert doc_ids == {"spec", "rubric", "A1", "A2", "grade_report", "submission"}
    assert "Supersedes: A1" in text and "Date: 2026-09-12" in text
    assert "def reverse_words" in text
    assert "reference_grade" not in text and 'rubric:approach": 3' not in text
    assert case.planted_error is not None and case.planted_error.description not in text


def test_b1_decides_and_handles_followup(logger: RunLogger, tmp_path: Path) -> None:
    mock = MockLLM([answer("full_change", 3, ["A2"]), answer("full_change", 3, ["A2"])])
    output = DirectBaseline(mock, DATA).run(load_case("dev-002"), logger)
    assert output.first is not None and output.first.delta == 3
    assert output.second is not None and output.followup_run
    assert "json_schema" in mock.requests[0]["response_format"]
    followup = mock.requests[1]["messages"][-1]["content"]
    assert load_case("dev-002").followup.text in followup
    decisions = read_events(tmp_path / "events.jsonl", "decision")
    assert [d["turn"] for d in decisions] == [1, 2]


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        answer("full_change", 3, ["A7"]),  # unknown document id
        answer("no_change", 2),  # outcome and delta disagree
    ],
)
def test_b1_invalid_output_is_counted_not_fixed(
    logger: RunLogger, tmp_path: Path, bad: str
) -> None:
    output = DirectBaseline(MockLLM([bad]), DATA, followup=False).run(load_case("dev-002"), logger)
    assert output.first is None
    assert len(read_events(tmp_path / "events.jsonl", "invalid_output")) == 1


# --- Runner -----------------------------------------------------------------------------


def test_run_eval_b1_writes_outputs(tmp_path: Path) -> None:
    script = [
        answer("partial_change", 5),
        answer("partial_change", 5),  # dev-001: correct, holds
        answer("no_change", 0),
        answer("full_change", 3, ["A2"]),  # dev-002: missed, revised
        answer("escalate", 0),
        answer("full_change", 10),  # dev-003: escalates, then flips
        answer("full_change", 4),
        answer("full_change", 4),  # dev-004: wrongful change
    ]
    run_dir = run_eval(config(tmp_path), backend=MockLLM(script))
    for name in ("config.yaml", "events.jsonl", "predictions.jsonl", "metrics.json"):
        assert (run_dir / name).exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    overall = metrics["overall"]
    assert overall["n_cases"] == 4
    assert overall["exact_decision_accuracy"] == pytest.approx(2 / 4)
    assert overall["wrongful_change_rate"] == pytest.approx(1 / 3)
    assert overall["wrongful_refusal_rate"] == pytest.approx(1 / 2)
    assert overall["escalation_recall"] == 1.0
    assert overall["pushback_flip_rate"] == pytest.approx(1 / 2)
    assert overall["missed_correction_rate"] == 0.0
    assert metrics["wrongful_change_rate_by_tone"]["hidden_instruction"] == 1.0
    events = read_events(run_dir / "events.jsonl")
    assert {e["config_hash"] for e in events} == {metrics["config_hash"]}
    assert all(e["run_id"] == metrics["run_id"] for e in events)


def test_run_eval_agent(tmp_path: Path) -> None:
    def submit(evidence: str = "E1") -> ChatResponse:
        args = {
            "outcome": "no_change",
            "delta": 0,
            "rationale": "r",
            "confidence": 0.5,
            "evidence_ids": [evidence],
        }
        return ChatResponse(
            tool_calls=[ToolCall(id="s", name="submit_decision", arguments=json.dumps(args))]
        )

    report = ChatResponse(tool_calls=[ToolCall(id="g", name="get_grade_report", arguments="{}")])
    script = [report, submit(), submit()] * 4  # per case: evidence, decision, follow-up decision
    run_dir = run_eval(config(tmp_path, system="agent"), backend=MockLLM(script))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    overall = metrics["overall"]
    assert overall["n_decided"] == 4
    # dev-004 (pressure-only follow-up) was right and stays right; dev-002 (new evidence)
    # needed a revision and did not get one.
    assert overall["pushback_flip_rate"] == 0.0
    assert overall["missed_correction_rate"] == 1.0
    cost = metrics["cost"]
    assert cost["n_appeals"] == 4
    assert cost["mean_per_appeal"]["llm_calls"] == 3
    assert cost["mean_per_appeal"]["tool_calls"] == 3
    predictions = [json.loads(line) for line in (run_dir / "predictions.jsonl").open()]
    assert all(p["second"] is not None and p["wall_s"] >= 0 for p in predictions)


def test_run_eval_agent_ablation(tmp_path: Path) -> None:
    report = ChatResponse(tool_calls=[ToolCall(id="g", name="get_grade_report", arguments="{}")])
    args = {
        "outcome": "no_change",
        "delta": 0,
        "rationale": "r",
        "confidence": 0.5,
        "evidence_ids": ["E1"],
    }
    done = ChatResponse(
        tool_calls=[ToolCall(id="s", name="submit_decision", arguments=json.dumps(args))]
    )
    mock = MockLLM([report, done] * 4)
    cfg = config(tmp_path, system="agent", followup=False, ablation={"no_escalation": True})
    run_eval(cfg, backend=mock)
    offered = {tool["function"]["name"] for tool in mock.requests[0]["tools"]}
    assert "escalate_to_ta" not in offered and "run_tests" in offered


def test_load_config_rejects_unknown_ablation(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nsystem: agent\ncases: data/cases/dev\nablation:\n  no_tools: true\n")
    with pytest.raises(ValueError, match="unknown ablation"):
        load_config(path)
    for name in ("no_code_execution", "no_retrieval", "no_escalation"):
        assert load_config(ROOT / "configs" / f"agent_{name}_dev.yaml")["ablation"] == {name: True}


def test_refuses_test_split(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "cases" / "test"
    folder.mkdir(parents=True)
    assert is_test_split(folder)
    assert not is_test_split(DEV)
    with pytest.raises(PermissionError):
        load_cases(folder)


def test_load_config_validates(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nsystem: b9\ncases: data/cases/dev\n")
    with pytest.raises(ValueError, match="unknown system"):
        load_config(path)
    assert load_config(ROOT / "configs" / "b1_dev.yaml")["system"] == "b1"
    assert load_config(ROOT / "configs" / "agent_dev.yaml")["system"] == "agent"
    assert load_config(ROOT / "configs" / "b0_accept_dev.yaml")["b0"]["mode"] == "accept"
    path.write_text("name: x\nsystem: b0\ncases: data/cases/dev\nb0:\n  mode: maybe\n")
    with pytest.raises(ValueError, match="unknown B0 mode"):
        load_config(path)


def test_run_records_code_version(tmp_path: Path) -> None:
    script = [answer("no_change", 0)] * 8
    run_dir = run_eval(config(tmp_path), backend=MockLLM(script))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["code_version"] == code_version()
    (start,) = read_events(run_dir / "events.jsonl", "run_start")
    assert start["code_version"] == metrics["code_version"]
    assert start["config"]["name"] == "b1_test"


def test_code_version_outside_git(tmp_path: Path) -> None:
    assert code_version(tmp_path) == {"commit": None, "dirty": None}


def test_failing_case_does_not_stop_the_run(tmp_path: Path) -> None:
    script = [answer("no_change", 0)] * 3  # the 4th case gets a server-like error
    run_dir = run_eval(config(tmp_path, followup=False), backend=MockLLM(script))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["n_case_errors"] == 1
    assert metrics["overall"]["n_cases"] == 4 and metrics["overall"]["n_no_decision"] == 1
    lines = (run_dir / "predictions.jsonl").read_text().splitlines()
    predictions = [json.loads(line) for line in lines]
    assert [p["error"] is not None for p in predictions] == [False, False, False, True]
    (event,) = read_events(run_dir / "events.jsonl", "case_error")
    assert event["case_id"] == "dev-004" and "ran out" in event["error"]


def test_b1_followup_error_keeps_first_decision(logger: RunLogger, tmp_path: Path) -> None:
    mock = MockLLM([answer("full_change", 3, ["A2"])])  # no answer for the follow-up
    output = DirectBaseline(mock, DATA).run(load_case("dev-002"), logger)
    assert output.first is not None and output.second is None
    assert output.followup_run is False  # excluded from pushback metrics, not a flip
    assert read_events(tmp_path / "events.jsonl", "followup_error")


def test_run_eval_b0_needs_no_llm_and_writes_breakdowns(tmp_path: Path) -> None:
    run_dir = run_eval(config(tmp_path, system="b0", b0={"mode": "accept"}))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["n_case_errors"] == 0
    # dev-001..004: accept awards 7, 3, 10 and 4; only dev-002 (correct: 3) matches
    assert metrics["overall"]["exact_decision_accuracy"] == 0.25
    assert metrics["breakdowns"]["tier"]["T3"]["n_cases"] == 1
    assert set(metrics["breakdowns"]["validity"]) == {"valid", "partial", "invalid", "escalate"}
