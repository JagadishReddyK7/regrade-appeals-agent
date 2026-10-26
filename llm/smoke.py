"""Check that a model server supports what the systems rely on, before a real run.

Usage: python -m llm.smoke --config configs/b1_dev.yaml

Three checks: a plain chat answer, a tool call with schema-valid arguments, and a JSON
answer constrained to a schema. Each call is logged like any other LLM call. Exits with
status 1 if any check fails.
"""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from agent.schemas import DirectDecisionOutput, RunCodeArgs
from agent.tools import SPECS_BY_NAME
from eval.runlog import RunContext, RunLogger, config_hash
from llm.client import LLMClient, backend_from_config
from llm.parsing import parse_json_output


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_chat(client: LLMClient) -> CheckResult:
    response = client.chat([{"role": "user", "content": "Reply with the single word: ready"}])
    ok = bool(response.content) and "ready" in (response.content or "").lower()
    return CheckResult("chat", ok, repr(response.content))


def check_tool_call(client: LLMClient) -> CheckResult:
    messages = [
        {
            "role": "user",
            "content": "Use the run_code tool to run this Python code: print(2 + 3)",
        }
    ]
    response = client.chat(messages, tools=[SPECS_BY_NAME["run_code"].schema()])
    if not response.tool_calls:
        return CheckResult("tool_call", False, f"no tool call; content={response.content!r}")
    call = response.tool_calls[0]
    parsed = parse_json_output(call.arguments, RunCodeArgs)
    ok = call.name == "run_code" and parsed.ok
    return CheckResult("tool_call", ok, f"{call.name}({call.arguments}) {parsed.error or ''}")


def check_json_schema(client: LLMClient) -> CheckResult:
    messages = [
        {
            "role": "user",
            "content": "A student lost 4 marks on test_a, which their code really fails. "
            "Decide the appeal as JSON. Cite the document id 'grade_report'.",
        }
    ]
    response = client.chat(messages, json_schema=DirectDecisionOutput.model_json_schema())
    parsed = parse_json_output(response.content, DirectDecisionOutput)
    return CheckResult("json_schema", parsed.ok, parsed.error or str(response.content))


CHECKS: list[Callable[[LLMClient], CheckResult]] = [check_chat, check_tool_call, check_json_schema]


def run_checks(client: LLMClient) -> list[CheckResult]:
    results = []
    for check in CHECKS:
        try:
            results.append(check(client))
        except Exception as exc:  # report a crashing check as a failure, keep going
            results.append(CheckResult(check.__name__.removeprefix("check_"), False, repr(exc)))
    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm.smoke")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    context = RunContext(run_id="smoke", seed=0, config_hash=config_hash(config))
    logger = RunLogger(Path(config.get("output_dir", "runs")) / "smoke.jsonl", context)
    client = LLMClient(backend_from_config(config["llm"]), logger.for_case("smoke"))
    results = run_checks(client)
    for r in results:
        print(f"{'PASS' if r.ok else 'FAIL'}  {r.name}: {r.detail[:300]}")
    print(json.dumps({r.name: r.ok for r in results}))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
