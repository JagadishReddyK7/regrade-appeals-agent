"""Run one system over a folder of cases and compute the metrics.

Usage: python -m eval.run --config configs/<name>.yaml

Writes runs/<run id>/ with config.yaml, events.jsonl (every LLM call, tool call and
decision), proposals.jsonl (grade-change proposals and escalations for TA approval),
predictions.jsonl and metrics.json. Refuses the private test split unless
--allow-test-split is given.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from agent.loop import run_agent
from agent.schemas import Case
from agent.tools import ToolBox
from baselines.b1_direct import DirectBaseline
from baselines.base import System, SystemOutput
from eval.metrics import CaseRecord, by_group, compute_all, wrongful_change_rate
from eval.runlog import RunContext, RunLogger, config_hash
from llm.client import Backend, LLMClient, backend_from_config
from sandbox.cache import ResultCache
from sandbox.runner import SandboxRunner

SYSTEMS = ("b1", "agent")


class AgentSystem:
    """The tool-calling agent (no claim splitting or follow-up turn yet)."""

    def __init__(
        self,
        backend: Backend,
        data_root: Path,
        runner: SandboxRunner,
        proposals_log: RunLogger,
        config: dict[str, Any],
    ) -> None:
        self.backend = backend
        self.data_root = data_root
        self.runner = runner
        self.proposals_log = proposals_log
        self.max_calls = int(config.get("agent", {}).get("max_tool_calls", 10))
        self.temperature = float(config.get("temperature", 0.0))
        self.seed = int(config.get("seed", 0))

    def run(self, case: Case, logger: RunLogger) -> SystemOutput:
        tools = ToolBox(
            self.data_root,
            case.assignment,
            case.submission,
            self.runner,
            self.proposals_log.for_case(case.case_id),
        )
        client = LLMClient(self.backend, logger)
        result = run_agent(case, client, tools, logger, self.max_calls, self.temperature, self.seed)
        return SystemOutput(first=result.decision)


def code_version(repo: Path | None = None) -> dict[str, Any]:
    """The git commit the run used, and whether tracked files had uncommitted changes.
    Prompts are logged only as hashes, so this is what makes a run reproducible."""
    cwd = repo or Path(__file__).resolve().parents[1]

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("name", "system", "cases"):
        if key not in config:
            raise ValueError(f"config is missing {key!r}")
    if config["system"] not in SYSTEMS:
        raise ValueError(f"unknown system {config['system']!r}; choose from {SYSTEMS}")
    return config


def is_test_split(folder: Path) -> bool:
    parts = folder.resolve().parts
    return any(a == "cases" and b == "test" for a, b in zip(parts, parts[1:], strict=False))


def load_cases(folder: Path, allow_test_split: bool = False) -> list[Case]:
    if is_test_split(folder) and not allow_test_split:
        raise PermissionError("refusing to run on the private test split")
    paths = sorted(folder.rglob("*.json"))
    if not paths:
        raise ValueError(f"no case files in {folder}")
    return [Case.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]


def build_system(
    config: dict[str, Any], backend: Backend, data_root: Path, proposals_log: RunLogger
) -> System:
    if config["system"] == "b1":
        return DirectBaseline(
            backend,
            data_root,
            temperature=float(config.get("temperature", 0.0)),
            seed=int(config.get("seed", 0)),
            followup=bool(config.get("followup", True)),
        )
    cache_dir = Path(config.get("sandbox", {}).get("cache_dir", "sandbox/cache"))
    runner = SandboxRunner(cache=ResultCache(cache_dir))
    return AgentSystem(backend, data_root, runner, proposals_log, config)


def prediction_record(case: Case, output: SystemOutput) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "first": output.first.model_dump() if output.first else None,
        "second": output.second.model_dump() if output.second else None,
        "followup_run": output.followup_run,
    }


def run_eval(
    config: dict[str, Any], backend: Backend | None = None, allow_test_split: bool = False
) -> Path:
    """Run the configured system on every case; return the run directory."""
    cases = load_cases(Path(config["cases"]), allow_test_split)
    chash = config_hash(config)
    run_id = f"{config['name']}-{time.strftime('%Y%m%d-%H%M%S')}-{chash[:6]}"
    run_dir = Path(config.get("output_dir", "runs")) / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), "utf-8")
    context = RunContext(run_id=run_id, seed=int(config.get("seed", 0)), config_hash=chash)
    events = RunLogger(run_dir / "events.jsonl", context)
    proposals = RunLogger(run_dir / "proposals.jsonl", context)
    version = code_version()
    events.log("run_start", code_version=version, config=config)
    backend = backend or backend_from_config(config["llm"])
    system = build_system(config, backend, Path(config.get("data_root", "data")), proposals)
    records = []
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as predictions:
        for case in cases:
            output = system.run(case, events.for_case(case.case_id))
            predictions.write(json.dumps(prediction_record(case, output)) + "\n")
            records.append(
                CaseRecord.from_case(case, output.first, output.second, output.followup_run)
            )
    metrics = {
        "run_id": run_id,
        "config_hash": chash,
        "code_version": version,
        "overall": compute_all(records),
        "wrongful_change_rate_by_tone": by_group(records, lambda r: r.tone, wrongful_change_rate),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return run_dir


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-test-split", action="store_true")
    args = parser.parse_args(argv)
    run_dir = run_eval(load_config(args.config), allow_test_split=args.allow_test_split)
    print(f"run written to {run_dir}")
    print((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
