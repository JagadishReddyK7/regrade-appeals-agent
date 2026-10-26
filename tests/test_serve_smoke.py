import json
from pathlib import Path

import pytest
import yaml

from eval.runlog import RunContext, RunLogger
from llm.client import ChatResponse, LLMClient, MockLLM, ToolCall
from llm.serve import build_command, server_env, wait_until_ready
from llm.smoke import run_checks

ROOT = Path(__file__).resolve().parents[1]

DECISION = json.dumps(
    {
        "outcome": "no_change",
        "delta": 0,
        "rationale": "The code fails the test.",
        "confidence": 0.9,
        "evidence_ids": ["grade_report"],
    }
)
RUN_CODE = ChatResponse(
    tool_calls=[ToolCall(id="c1", name="run_code", arguments='{"code": "print(2 + 3)"}')]
)


def llm_config(name: str = "b1_dev") -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))["llm"]


def client(tmp_path: Path, script: list) -> LLMClient:
    logger = RunLogger(tmp_path / "smoke.jsonl", RunContext("smoke", 0, "h"))
    return LLMClient(MockLLM(script), logger)


# --- serve ------------------------------------------------------------------------------


def test_build_command_for_t4() -> None:
    cmd = build_command(llm_config())
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen2.5-7B-Instruct-AWQ"]
    assert cmd[cmd.index("--dtype") + 1] == "half"  # T4 has no bfloat16
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--quantization") + 1] == "awq"


def test_both_dev_configs_share_one_server() -> None:
    assert build_command(llm_config("b1_dev")) == build_command(llm_config("agent_dev"))


def test_server_env_selects_gpu() -> None:
    assert server_env({"serve": {"gpu": 1}})["CUDA_VISIBLE_DEVICES"] == "1"


def test_wait_until_ready_polls() -> None:
    answers = iter([False, False, True])
    seen: list[str] = []

    def fetch(url: str) -> bool:
        seen.append(url)
        return next(answers)

    wait_until_ready("http://x/v1/", poll_s=0, fetch=fetch)
    assert seen == ["http://x/v1/models"] * 3


def test_wait_until_ready_detects_dead_server() -> None:
    with pytest.raises(RuntimeError, match="exited"):
        wait_until_ready("http://x/v1", poll_s=0, fetch=lambda _: False, is_alive=lambda: False)


def test_wait_until_ready_times_out() -> None:
    with pytest.raises(TimeoutError):
        wait_until_ready("http://x/v1", timeout_s=0, fetch=lambda _: False)


# --- smoke ------------------------------------------------------------------------------


def test_smoke_all_pass(tmp_path: Path) -> None:
    results = run_checks(client(tmp_path, ["Ready", RUN_CODE, DECISION]))
    assert [(r.name, r.ok) for r in results] == [
        ("chat", True),
        ("tool_call", True),
        ("json_schema", True),
    ]


def test_smoke_reports_each_failure(tmp_path: Path) -> None:
    script = [
        "",  # empty answer
        ChatResponse(content="I would run print(2 + 3)"),  # no tool call
        "```json\n" + DECISION + "\n```",  # not constrained to the schema
    ]
    results = run_checks(client(tmp_path, script))
    assert [r.ok for r in results] == [False, False, False]
    assert "no tool call" in results[1].detail


def test_smoke_invalid_tool_arguments_fail(tmp_path: Path) -> None:
    bad = ChatResponse(tool_calls=[ToolCall(id="c", name="run_code", arguments='{"cmd": "x"}')])
    results = run_checks(client(tmp_path, ["ready", bad, DECISION]))
    assert results[1].ok is False


def test_smoke_survives_crashing_check(tmp_path: Path) -> None:
    results = run_checks(client(tmp_path, ["ready"]))  # script runs out after one call
    assert [r.ok for r in results] == [True, False, False]
    assert "ran out" in results[1].detail
