import itertools
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from agent.loop import MAX_RESULT_CHARS, AgentResult, fit_for_model, run_agent
from agent.schemas import Case, ProposeGradeChangeArgs, RunTestsArgs, SearchArgs
from agent.tools import TOOL_SPECS, RejectedAction, ToolBox, ToolError
from data.formats import load_assignment, load_grade_report
from data.grader import deducted_marks
from eval.runlog import RunContext, RunLogger, read_events
from llm.client import ChatResponse, LLMClient, MockLLM, ToolCall
from sandbox.cache import ResultCache
from sandbox.runner import SandboxRunner

DATA = Path(__file__).resolve().parents[1] / "data"
TOOL_NAMES = [
    "get_submission",
    "get_grade_report",
    "run_tests",
    "run_code",
    "search_course_docs",
    "search_precedents",
    "propose_grade_change",
    "escalate_to_ta",
    "submit_decision",
]
_ids = itertools.count(1)


def call(name: str, **args: Any) -> ChatResponse:
    return ChatResponse(
        tool_calls=[ToolCall(id=f"call{next(_ids)}", name=name, arguments=json.dumps(args))]
    )


def raw_call(name: str, arguments: str) -> ChatResponse:
    return ChatResponse(
        tool_calls=[ToolCall(id=f"call{next(_ids)}", name=name, arguments=arguments)]
    )


def load_case(case_id: str) -> Case:
    return Case.model_validate_json((DATA / "cases" / "dev" / f"{case_id}.json").read_text())


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> SandboxRunner:
    return SandboxRunner(cache=ResultCache(tmp_path_factory.mktemp("cache")))


@pytest.fixture
def logs(tmp_path: Path) -> tuple[RunLogger, RunLogger]:
    context = RunContext(run_id="r1", seed=0, config_hash="test")
    run_log = RunLogger(tmp_path / "run.jsonl", context)
    proposals = RunLogger(tmp_path / "proposals.jsonl", context)
    return run_log, proposals


def make_toolbox(
    runner: SandboxRunner,
    logs: tuple[RunLogger, RunLogger],
    case: Case,
    data_root: Path = DATA,
    failures: dict[str, int] | None = None,
) -> ToolBox:
    proposals = logs[1].for_case(case.case_id)
    return ToolBox(data_root, case.assignment, case.submission, runner, proposals, failures)


def run(
    case_id: str,
    script: list[ChatResponse],
    runner: SandboxRunner,
    logs: tuple[RunLogger, RunLogger],
    failures: dict[str, int] | None = None,
) -> tuple[AgentResult, MockLLM]:
    case = load_case(case_id)
    logger = logs[0].for_case(case_id)
    mock = MockLLM(script)
    tools = make_toolbox(runner, logs, case, failures=failures)
    return run_agent(case, LLMClient(mock, logger), tools, logger), mock


# --- Tools ------------------------------------------------------------------------------


def test_tool_schemas_cover_all_tools() -> None:
    schemas = ToolBox.schemas()
    assert [s["function"]["name"] for s in schemas] == TOOL_NAMES
    for schema in schemas:
        assert schema["function"]["parameters"]["additionalProperties"] is False
    assert [spec.name for spec in TOOL_SPECS] == TOOL_NAMES


def test_read_tools(runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]) -> None:
    tools = make_toolbox(runner, logs, load_case("dev-001"))
    assert "def count_vowels" in tools.call("get_submission", None)["code"]
    report = tools.call("get_grade_report", None)
    assert report["report"]["total"] == 23
    assert any(item["id"] == "rubric:style" for item in report["rubric"])


def test_run_tests_uses_the_autograder_that_graded(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    tools = make_toolbox(runner, logs, load_case("dev-001"))  # graded by faulty_empty
    results = tools.call("run_tests", RunTestsArgs(test_ids=["test_empty_string"]))["results"]
    assert results["test_empty_string"]["passed"] is False


def test_run_tests_rejects_unknown_test(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    tools = make_toolbox(runner, logs, load_case("dev-001"))
    with pytest.raises(ToolError, match="unknown test"):
        tools.call("run_tests", RunTestsArgs(test_ids=["test_nope"]))


def test_search_course_docs_shows_superseded(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    tools = make_toolbox(runner, logs, load_case("dev-002"))
    docs = tools.call("search_course_docs", SearchArgs(query="recursion reverse_words", k=5))
    by_id = {d["doc_id"]: d for d in docs["documents"]}
    assert by_id["announcement:A1"]["superseded_by"] == "A2"
    assert by_id["announcement:A2"]["date"] == "2026-09-12"


def test_deducted_marks_from_report() -> None:
    case = load_case("dev-001")
    assignment = load_assignment(DATA / "assignments" / case.assignment)
    report = load_grade_report(DATA / "submissions" / case.submission)
    assert deducted_marks(assignment, report) == {"test_empty_string": 5, "rubric:style": 2}


def test_proposal_logged_for_ta(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    tools = make_toolbox(runner, logs, load_case("dev-001"))
    args = ProposeGradeChangeArgs(item_id="test_empty_string", delta=5, reason="faulty test")
    assert tools.call("propose_grade_change", args)["proposed_total"] == 5
    (record,) = read_events(tmp_path / "proposals.jsonl", "grade_change_proposal")
    assert record["item_id"] == "test_empty_string" and record["status"] == "pending_ta"
    assert record["case_id"] == "dev-001"


@pytest.mark.parametrize(
    ("item_id", "delta"),
    [("test_empty_string", 6), ("test_count_vowels_basic", 4), ("rubric:approach", 1)],
)
def test_proposal_above_deducted_marks_rejected(
    runner: SandboxRunner,
    logs: tuple[RunLogger, RunLogger],
    tmp_path: Path,
    item_id: str,
    delta: float,
) -> None:
    tools = make_toolbox(runner, logs, load_case("dev-001"))
    with pytest.raises(RejectedAction):
        tools.call(
            "propose_grade_change", ProposeGradeChangeArgs(item_id=item_id, delta=delta, reason="x")
        )
    assert read_events(tmp_path / "proposals.jsonl") == []


def test_proposals_capped_cumulatively(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    tools = make_toolbox(runner, logs, load_case("dev-001"))
    args = ProposeGradeChangeArgs(item_id="test_empty_string", delta=3, reason="x")
    tools.call("propose_grade_change", args)
    with pytest.raises(RejectedAction):
        tools.call("propose_grade_change", args)


def test_tools_never_need_the_answer_key(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    root = tmp_path / "data"
    for part in ("assignments", "submissions"):
        shutil.copytree(DATA / part, root / part)
    (root / "submissions" / "D1-S03" / "reference_grade.json").unlink()
    shutil.rmtree(root / "assignments" / "D1" / "reference")
    tools = make_toolbox(runner, logs, load_case("dev-001"), data_root=root)
    tools.call("get_submission", None)
    tools.call("get_grade_report", None)
    tools.call("run_tests", RunTestsArgs())
    tools.call("search_course_docs", SearchArgs(query="docstring"))


def test_injected_failure(runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]) -> None:
    tools = make_toolbox(runner, logs, load_case("dev-001"), failures={"get_submission": 1})
    with pytest.raises(ToolError, match="injected"):
        tools.call("get_submission", None)
    assert "code" in tools.call("get_submission", None)


# --- Loop -------------------------------------------------------------------------------


def test_partial_change_decision(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    script = [
        call("get_grade_report"),
        call("run_tests", test_ids=["test_empty_string"]),
        call("run_code", code="from strings_utils import count_vowels\nprint(count_vowels(''))"),
        call("propose_grade_change", item_id="test_empty_string", delta=5, reason="test is wrong"),
        call(
            "submit_decision",
            outcome="partial_change",
            delta=5,
            rationale="count_vowels('') returns 0 as the spec requires; the test expects None.",
            confidence=0.9,
            evidence_ids=["E2", "E3"],
        ),
    ]
    result, _ = run("dev-001", script, runner, logs)
    assert result.status == "submitted"
    assert result.decision is not None
    assert result.decision.outcome == "partial_change" and result.decision.delta == 5
    assert result.decision.verdicts[0].evidence_ids == ["E2", "E3"]
    assert result.evidence["E3"]["output"]["stdout"].strip() == "0"
    assert (result.tool_calls, result.invalid_calls, result.tool_errors) == (5, 0, 0)
    events = read_events(tmp_path / "run.jsonl")
    assert len([e for e in events if e["event"] == "tool_call"]) == 5
    assert len([e for e in events if e["event"] == "llm_call"]) == 5
    (decision,) = [e for e in events if e["event"] == "decision"]
    assert decision["case_id"] == "dev-001" and decision["status"] == "submitted"


def test_recovers_from_tool_error(runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]) -> None:
    script = [
        call("run_tests", test_ids=["test_uppercase"]),  # wrong id: tool error
        call("run_tests", test_ids=["test_count_vowels_uppercase"]),
        call(
            "submit_decision",
            outcome="no_change",
            delta=0,
            rationale="The test fails correctly.",
            confidence=0.8,
            evidence_ids=["E1"],
        ),
    ]
    result, mock = run("dev-004", script, runner, logs)
    assert result.tool_errors == 1
    assert result.decision is not None and result.decision.outcome == "no_change"
    error_message = mock.requests[1]["messages"][-1]
    assert error_message["role"] == "tool" and "unknown test ids" in error_message["content"]


def test_recovers_from_injected_failure(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    script = [
        call("get_submission"),
        call("get_submission"),
        call(
            "submit_decision",
            outcome="no_change",
            delta=0,
            rationale="Code has a bug.",
            confidence=0.7,
            evidence_ids=["E1"],
        ),
    ]
    result, _ = run("dev-004", script, runner, logs, failures={"get_submission": 1})
    assert result.tool_errors == 1 and result.status == "submitted"


def test_invalid_calls_counted(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    script = [
        call("delete_grades"),  # unknown tool
        raw_call("run_tests", "{not json"),  # malformed arguments
        call("run_code", stdin="x"),  # missing required argument
        ChatResponse(content="I think the grade is fine."),  # no tool call
        call("get_grade_report"),
        call(
            "submit_decision",
            outcome="no_change",
            delta=0,
            rationale="Deduction is correct.",
            confidence=0.6,
            evidence_ids=["E1"],
        ),
    ]
    result, _ = run("dev-004", script, runner, logs)
    assert result.invalid_calls == 4
    assert result.status == "submitted"
    invalid_outputs = read_events(tmp_path / "run.jsonl", "invalid_output")
    assert len(invalid_outputs) == 2  # the two argument-validation failures


def test_rejected_proposal_counted(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    script = [
        call(
            "propose_grade_change",
            item_id="test_count_vowels_uppercase",
            delta=10,
            reason="full marks",
        ),
        call("get_grade_report"),
        call(
            "submit_decision",
            outcome="no_change",
            delta=0,
            rationale="Deduction is correct.",
            confidence=0.6,
            evidence_ids=["E1"],
        ),
    ]
    result, _ = run("dev-004", script, runner, logs)
    assert result.rejected_actions == 1
    assert result.decision is not None and result.decision.delta == 0


@pytest.mark.parametrize(
    "bad_submit",
    [
        {"outcome": "no_change", "delta": 0, "evidence_ids": ["E9"]},  # unknown evidence
        {"outcome": "full_change", "delta": 4, "evidence_ids": ["E1"]},  # no proposal made
        {"outcome": "no_change", "delta": 0, "evidence_ids": []},  # refuted without evidence
    ],
)
def test_invalid_decision_can_be_resubmitted(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], bad_submit: dict[str, Any]
) -> None:
    script = [
        call("get_grade_report"),
        call("submit_decision", rationale="r", confidence=0.5, **bad_submit),
        call(
            "submit_decision",
            outcome="no_change",
            delta=0,
            rationale="r",
            confidence=0.5,
            evidence_ids=["E1"],
        ),
    ]
    result, _ = run("dev-004", script, runner, logs)
    assert result.invalid_calls == 1
    assert result.decision is not None and result.tool_calls == 3


def test_escalation(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    script = [
        call("search_course_docs", query="late penalty extension medical"),
        call("escalate_to_ta", reason="Medical circumstances need a staff decision."),
        call(
            "submit_decision",
            outcome="escalate",
            delta=0,
            rationale="Needs a TA.",
            confidence=0.9,
            evidence_ids=["E1"],
        ),
    ]
    result, _ = run("dev-003", script, runner, logs)
    assert result.decision is not None
    assert result.decision.verdicts[0].verdict == "needs_human"
    assert read_events(tmp_path / "proposals.jsonl", "escalation")


def test_budget_exhausted_returns_no_decision(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    script = [call("get_submission") for _ in range(12)]
    result, mock = run("dev-004", script, runner, logs)
    assert result.status == "budget_exhausted" and result.decision is None
    assert result.tool_calls == 10 and len(mock.requests) == 10


def test_prompt_contains_no_answer_key(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    script = [
        call("get_grade_report"),
        call(
            "submit_decision",
            outcome="no_change",
            delta=0,
            rationale="r",
            confidence=0.5,
            evidence_ids=["E1"],
        ),
    ]
    _, mock = run("dev-002", script, runner, logs)
    prompt = json.dumps(mock.requests[0]["messages"])
    case = load_case("dev-002")
    assert case.appeal_text in json.loads(prompt)[1]["content"]
    assert case.planted_error is not None and case.planted_error.description not in prompt
    for word in ("gold_verdict", "planted", "answer_after", 'full_change"'):
        assert word not in prompt


def test_addition_error_can_be_proposed_under_total(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    root = tmp_path / "data"
    for part in ("assignments", "submissions"):
        shutil.copytree(DATA / part, root / part)
    path = root / "submissions" / "D1-S07" / "grade_report.json"
    path.write_text(path.read_text().replace('"total": 20', '"total": 18'))
    tools = make_toolbox(runner, logs, load_case("dev-003"), data_root=root)
    args = ProposeGradeChangeArgs(item_id="total", delta=2, reason="items add up to 20")
    assert tools.call("propose_grade_change", args)["proposed_total"] == 2


# --- Output size ------------------------------------------------------------------------


def test_short_output_is_unchanged() -> None:
    assert fit_for_model({"a": "x" * 10}) == ('{"a": "' + "x" * 10 + '"}', False)


def test_long_field_keeps_start_and_end() -> None:
    text, truncated = fit_for_model({"stderr": "START" + "x" * 20000 + "END"})
    assert truncated and len(text) <= MAX_RESULT_CHARS
    value = json.loads(text)["stderr"]
    assert value.startswith("START") and value.endswith("END") and "truncated" in value


def test_many_fields_fit_the_limit() -> None:
    output = {f"t{i}": {"stderr": "e" * 3000} for i in range(10)}
    text, truncated = fit_for_model(output)
    assert truncated and len(text) <= MAX_RESULT_CHARS
    assert set(json.loads(text)) == set(output)  # every field still present


def test_many_tiny_fields_fall_back_to_plain_cut() -> None:
    text, truncated = fit_for_model({str(i): i for i in range(3000)})
    assert truncated and set(json.loads(text)) == {"truncated_result"}


def test_large_tool_output_truncated_for_model_only(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    script = [
        call("run_code", code="print('x' * 30000)"),
        call(
            "submit_decision",
            outcome="no_change",
            delta=0,
            rationale="r",
            confidence=0.5,
            evidence_ids=["E1"],
        ),
    ]
    result, mock = run("dev-004", script, runner, logs)
    tool_message = mock.requests[1]["messages"][-1]
    assert len(tool_message["content"]) <= MAX_RESULT_CHARS
    assert "characters truncated" in tool_message["content"]
    assert len(result.evidence["E1"]["output"]["stdout"]) > 30000  # full output kept
    (event,) = [
        e for e in read_events(tmp_path / "run.jsonl", "tool_call") if e["tool"] == "run_code"
    ]
    assert event["truncated_for_model"] is True
