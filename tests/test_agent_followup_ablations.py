"""Agent follow-up turn and ablation flags (no code execution, no retrieval, no escalation)."""

import json
from pathlib import Path
from typing import Any

import pytest

from agent.loop import ABLATION_FLAGS, AgentOptions, AgentResult, run_agent
from eval.runlog import RunContext, RunLogger, read_events
from llm.client import ChatResponse, LLMClient, MockLLM, ToolCall
from sandbox.cache import ResultCache
from sandbox.runner import SandboxRunner
from tests.test_agent_loop import call, load_case, make_toolbox

FOLLOWUP = AgentOptions(followup=True)


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> SandboxRunner:
    return SandboxRunner(cache=ResultCache(tmp_path_factory.mktemp("cache")))


@pytest.fixture
def logs(tmp_path: Path) -> tuple[RunLogger, RunLogger]:
    context = RunContext(run_id="r1", seed=0, config_hash="test")
    return RunLogger(tmp_path / "run.jsonl", context), RunLogger(
        tmp_path / "proposals.jsonl", context
    )


def submit(outcome: str = "no_change", delta: float = 0, evidence: str = "E1") -> ChatResponse:
    return call(
        "submit_decision",
        outcome=outcome,
        delta=delta,
        rationale="r",
        confidence=0.7,
        evidence_ids=[evidence],
    )


def run_with(
    case_id: str,
    script: list[ChatResponse],
    runner: SandboxRunner,
    logs: tuple[RunLogger, RunLogger],
    options: AgentOptions,
) -> tuple[AgentResult, MockLLM]:
    case = load_case(case_id)
    logger = logs[0].for_case(case_id)
    mock = MockLLM(script)
    tools = make_toolbox(runner, logs, case)
    return run_agent(case, LLMClient(mock, logger), tools, logger, options=options), mock


def offered_tools(request: dict[str, Any]) -> set[str]:
    return {tool["function"]["name"] for tool in request["tools"]}


# --- Follow-up turn ---------------------------------------------------------------------


def test_followup_pressure_only_keeps_decision(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    script = [call("get_grade_report"), submit(), submit()]
    result, mock = run_with("dev-004", script, runner, logs, FOLLOWUP)
    assert result.decision is not None and result.followup_run
    assert result.followup_decision is not None
    assert result.followup_decision.outcome == "no_change"
    assert result.followup_status == "submitted"
    followup_message = mock.requests[2]["messages"][-1]
    assert followup_message["role"] == "user"
    assert load_case("dev-004").followup.text in followup_message["content"]
    decisions = read_events(tmp_path / "run.jsonl", "decision")
    assert [d["turn"] for d in decisions] == [1, 2]


def test_followup_keeps_first_turn_evidence(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    script = [call("get_grade_report"), submit(), submit(evidence="E1")]
    result, mock = run_with("dev-004", script, runner, logs, FOLLOWUP)
    assert result.followup_decision is not None  # E1 from turn 1 is still a valid citation
    assert len(mock.requests[2]["messages"]) > len(mock.requests[0]["messages"])


def test_followup_gets_a_fresh_budget(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    options = AgentOptions(max_calls=2, followup=True)
    script = [call("get_grade_report"), submit(), call("get_submission"), submit()]
    result, _ = run_with("dev-004", script, runner, logs, options)
    assert result.followup_decision is not None
    assert result.tool_calls == 4


def test_followup_can_revise_with_new_proposal(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    script = [
        call("get_grade_report"),
        submit(),  # first decision: no change
        call("search_course_docs", query="recursion reverse_words"),
        call("propose_grade_change", item_id="rubric:approach", delta=3, reason="A2 allows it"),
        submit("full_change", 3, "E2"),
    ]
    result, _ = run_with("dev-002", script, runner, logs, FOLLOWUP)
    assert result.decision is not None and result.decision.delta == 0
    assert result.followup_decision is not None and result.followup_decision.delta == 3


def test_no_followup_after_failed_first_turn(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    options = AgentOptions(max_calls=1, followup=True)
    result, mock = run_with("dev-004", [call("get_submission")], runner, logs, options)
    assert result.decision is None and result.status == "budget_exhausted"
    assert not result.followup_run and result.followup_decision is None
    assert len(mock.requests) == 1


def test_failed_followup_turn_has_no_second_decision(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    options = AgentOptions(max_calls=2, followup=True)
    script = [call("get_grade_report"), submit(), call("get_submission"), call("get_submission")]
    result, _ = run_with("dev-004", script, runner, logs, options)
    assert result.followup_run and result.followup_decision is None
    assert result.followup_status == "budget_exhausted"


def test_calls_after_submit_are_skipped_but_answered(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    both = ChatResponse(
        tool_calls=[
            ToolCall(
                id="s1",
                name="submit_decision",
                arguments=json.dumps(
                    {
                        "outcome": "no_change",
                        "delta": 0,
                        "rationale": "r",
                        "confidence": 0.7,
                        "evidence_ids": ["E1"],
                    }
                ),
            ),
            ToolCall(id="g1", name="get_submission", arguments="{}"),
        ]
    )
    result, mock = run_with(
        "dev-004", [call("get_grade_report"), both, submit()], runner, logs, FOLLOWUP
    )
    answered = {m["tool_call_id"] for m in mock.requests[2]["messages"] if m["role"] == "tool"}
    assert {"s1", "g1"} <= answered  # every tool call gets a reply, so the chat stays valid
    assert result.tool_calls == 3  # the skipped call does not count
    skipped = [
        e for e in read_events(tmp_path / "run.jsonl", "tool_call") if e["status"] == "skipped"
    ]
    assert [e["tool"] for e in skipped] == ["get_submission"]


# --- Ablations --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "removed"),
    [
        (AgentOptions(code_execution=False), {"run_tests", "run_code"}),
        (AgentOptions(retrieval=False), {"search_course_docs", "search_precedents"}),
        (AgentOptions(escalation=False), {"escalate_to_ta"}),
    ],
)
def test_ablation_removes_tools(
    runner: SandboxRunner,
    logs: tuple[RunLogger, RunLogger],
    options: AgentOptions,
    removed: set[str],
) -> None:
    _, mock = run_with("dev-004", [call("get_grade_report"), submit()], runner, logs, options)
    offered = offered_tools(mock.requests[0])
    assert offered.isdisjoint(removed)
    assert {"get_submission", "get_grade_report", "submit_decision"} <= offered


def test_calling_a_removed_tool_is_invalid(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    options = AgentOptions(code_execution=False)
    script = [call("run_tests"), call("get_grade_report"), submit("no_change", 0, "E1")]
    result, _ = run_with("dev-004", script, runner, logs, options)
    assert result.invalid_calls == 1 and result.decision is not None


def test_no_retrieval_puts_rubric_in_prompt(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    script = [call("get_grade_report"), submit()]
    _, ablated = run_with("dev-002", script, runner, logs, AgentOptions(retrieval=False))
    prompt = ablated.requests[0]["messages"][1]["content"]
    assert "Rubric:" in prompt and "rubric:approach" in prompt
    assert "course documents" not in ablated.requests[0]["messages"][0]["content"]
    _, full = run_with(
        "dev-002", [call("get_grade_report"), submit()], runner, logs, AgentOptions()
    )
    assert "Rubric:" not in full.requests[0]["messages"][1]["content"]


def test_no_escalation_rejects_escalate_outcome(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    options = AgentOptions(escalation=False)
    script = [call("get_grade_report"), submit("escalate"), submit("no_change")]
    result, mock = run_with("dev-003", script, runner, logs, options)
    assert result.invalid_calls == 1
    assert result.decision is not None and result.decision.outcome == "no_change"
    assert "not available" in mock.requests[0]["messages"][0]["content"]


def test_options_from_config() -> None:
    options = AgentOptions.from_config(
        {"agent": {"max_tool_calls": 7}, "followup": False, "ablation": {"no_retrieval": True}}
    )
    assert options == AgentOptions(max_calls=7, followup=False, retrieval=False)
    assert AgentOptions.from_config({}) == AgentOptions(followup=True)
    assert AgentOptions.from_config({"ablation": None}).disabled_tools() == frozenset()


def test_unknown_ablation_flag_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ablation"):
        AgentOptions.from_config({"ablation": {"no_verifier": True}})
    assert set(ABLATION_FLAGS) == {"no_code_execution", "no_retrieval", "no_escalation"}
