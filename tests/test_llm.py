from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.schemas import ClaimVerdict
from eval.runlog import RunContext, RunLogger, config_hash, read_events
from llm.client import (
    ChatResponse,
    LLMClient,
    MockLLM,
    ToolCall,
    backend_from_config,
    build_request,
    prompt_hash,
    to_chat_response,
)
from llm.parsing import parse_json_output, validate_output, validate_tool_arguments

MESSAGES = [{"role": "user", "content": "Is claim C1 supported?"}]
VALID_VERDICT = '{"claim_id": "C1", "verdict": "supported", "evidence_ids": ["E1"]}'


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "run.jsonl"


@pytest.fixture
def logger(log_path: Path) -> RunLogger:
    context = RunContext(run_id="run-1", seed=7, config_hash=config_hash({"a": 1}))
    return RunLogger(log_path, context).for_case("dev-001")


def test_config_hash_ignores_key_order() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_log_record_carries_run_context(logger: RunLogger, log_path: Path) -> None:
    logger.log("tool_call", tool="run_tests")
    (record,) = read_events(log_path)
    assert record["event"] == "tool_call"
    assert record["run_id"] == "run-1"
    assert record["case_id"] == "dev-001"
    assert record["seed"] == 7
    assert record["config_hash"] == config_hash({"a": 1})
    assert record["tool"] == "run_tests"


def test_chat_returns_scripted_responses_in_order(logger: RunLogger) -> None:
    client = LLMClient(MockLLM(["first", "second"]), logger)
    assert client.chat(MESSAGES).content == "first"
    assert client.chat(MESSAGES).content == "second"


def test_mock_raises_when_script_exhausted(logger: RunLogger, log_path: Path) -> None:
    client = LLMClient(MockLLM([]), logger)
    with pytest.raises(RuntimeError, match="ran out"):
        client.chat(MESSAGES)
    (record,) = read_events(log_path, "llm_error")
    assert "ran out" in record["error"]


def test_chat_logs_every_call(logger: RunLogger, log_path: Path) -> None:
    response = ChatResponse(content="ok", prompt_tokens=12, completion_tokens=3)
    client = LLMClient(MockLLM([response]), logger)
    client.chat(MESSAGES, temperature=0.2, seed=3)
    (record,) = read_events(log_path, "llm_call")
    assert record["model"] == "mock"
    assert record["case_id"] == "dev-001"
    assert record["prompt_hash"] == prompt_hash(MESSAGES, None)
    assert record["prompt_tokens"] == 12 and record["completion_tokens"] == 3
    assert record["latency_s"] >= 0
    assert record["content"] == "ok"
    assert record["temperature"] == 0.2 and record["call_seed"] == 3


def test_prompt_hash_depends_on_tools() -> None:
    tool = {"type": "function", "function": {"name": "run_tests"}}
    assert prompt_hash(MESSAGES, None) != prompt_hash(MESSAGES, [tool])


def test_mock_records_requests(logger: RunLogger) -> None:
    mock = MockLLM(["{}"])
    schema = ClaimVerdict.model_json_schema()
    LLMClient(mock, logger).chat(MESSAGES, json_schema=schema, seed=5)
    (request,) = mock.requests
    assert request["seed"] == 5
    assert request["response_format"]["json_schema"]["schema"] == schema


def test_tool_calls_are_logged(logger: RunLogger, log_path: Path) -> None:
    call = ToolCall(id="t1", name="run_tests", arguments='{"test_id": "test_add"}')
    client = LLMClient(MockLLM([ChatResponse(tool_calls=[call])]), logger)
    assert client.chat(MESSAGES).tool_calls == [call]
    (record,) = read_events(log_path, "llm_call")
    assert record["tool_calls"][0]["name"] == "run_tests"


def test_build_request_with_tools_and_schema() -> None:
    tool = {"type": "function", "function": {"name": "run_tests"}}
    request = build_request("qwen", MESSAGES, [tool], {"title": "Verdict"}, 0.0, 1)
    assert request["tools"] == [tool] and request["tool_choice"] == "auto"
    assert request["response_format"]["json_schema"]["name"] == "Verdict"


def test_build_request_minimal() -> None:
    request = build_request("qwen", MESSAGES, None, None, 0.0, 1)
    assert "tools" not in request and "response_format" not in request


def test_to_chat_response_converts_openai_object() -> None:
    function = SimpleNamespace(name="run_code", arguments="{}")
    message = SimpleNamespace(
        content=None, tool_calls=[SimpleNamespace(id="c1", function=function)]
    )
    completion = SimpleNamespace(
        model="qwen",
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
    )
    response = to_chat_response(completion)
    assert response.tool_calls[0].name == "run_code"
    assert response.prompt_tokens == 10 and response.finish_reason == "tool_calls"


def test_backend_from_config_builds_openai_backend() -> None:
    backend = backend_from_config({"base_url": "http://localhost:8000/v1", "model": "qwen"})
    assert backend.model == "qwen"


def test_backend_from_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        backend_from_config({"backend": "mock"})


def test_parse_valid_json() -> None:
    result = parse_json_output(VALID_VERDICT, ClaimVerdict)
    assert result.ok and result.value is not None and result.value.verdict == "supported"


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (None, "empty"),
        ("not json", "invalid JSON"),
        ("```json\n" + VALID_VERDICT + "\n```", "invalid JSON"),  # fences are not stripped
        ('{"claim_id": "C1", "verdict": "maybe"}', "schema error"),
    ],
)
def test_parse_invalid_output(raw: str | None, error: str) -> None:
    result = parse_json_output(raw, ClaimVerdict)
    assert not result.ok and result.error is not None and error in result.error


def test_invalid_output_is_logged(logger: RunLogger, log_path: Path) -> None:
    assert validate_output("not json", ClaimVerdict, logger, kind="verdict") is None
    (record,) = read_events(log_path, "invalid_output")
    assert record["kind"] == "verdict"
    assert record["schema"] == "ClaimVerdict"
    assert record["raw"] == "not json"


def test_valid_output_is_not_logged(logger: RunLogger, log_path: Path) -> None:
    assert validate_output(VALID_VERDICT, ClaimVerdict, logger, kind="verdict") is not None
    assert read_events(log_path, "invalid_output") == []


def test_tool_arguments_validated(logger: RunLogger, log_path: Path) -> None:
    call = ToolCall(id="t1", name="submit_verdict", arguments='{"claim_id": "C1"}')
    assert validate_tool_arguments(call, ClaimVerdict, logger) is None
    (record,) = read_events(log_path, "invalid_output")
    assert record["kind"] == "tool_call:submit_verdict"
