"""Run one system over a folder of cases and compute the metrics.

Usage: python -m eval.run --config configs/<name>.yaml

Writes runs/<run id>/ with config.yaml, events.jsonl (every LLM call, tool call and
decision), proposals.jsonl (grade-change proposals and escalations for TA approval),
predictions.jsonl and metrics.json. Refuses the private test split unless
--allow-test-split is given.
"""

import argparse
import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from agent.loop import AgentOptions, run_agent
from agent.pipeline import PipelineOptions, run_pipeline
from agent.schemas import Case
from agent.tools import ToolBox
from baselines.b0 import MODES as B0_MODES
from baselines.b0 import FixedBaseline
from baselines.b1_direct import DirectBaseline
from baselines.base import System, SystemOutput
from eval.cost import per_case_cost, summarise_cost
from eval.metrics import (
    CaseRecord,
    Prediction,
    by_group,
    compute_all,
    compute_breakdowns,
    compute_consistency,
    decision_key,
    rate,
    wrongful_change_rate,
)
from eval.runlog import RunContext, RunLogger, config_hash, read_events
from llm.client import Backend, LLMClient, backend_from_config
from retrieval.precedents import Precedent, PrecedentStore
from sandbox.cache import ResultCache
from sandbox.runner import SandboxRunner

SYSTEMS = ("b0", "b1", "agent", "pipeline")  # b0 needs no LLM


class AgentSystem:
    """The tool-calling agent (no claim splitting yet); options and ablations from config."""

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
        self.options = AgentOptions.from_config(config)
        self.precedents = load_precedents(config)
        self.temperature = float(config.get("temperature", 0.0))
        self.seed = int(config.get("seed", 0))

    def run(self, case: Case, logger: RunLogger) -> SystemOutput:
        tools = ToolBox(
            self.data_root,
            case.assignment,
            case.submission,
            self.runner,
            self.proposals_log.for_case(case.case_id),
            precedents=self.precedents,
        )
        client = LLMClient(self.backend, logger)
        result = run_agent(
            case,
            client,
            tools,
            logger,
            temperature=self.temperature,
            seed=self.seed,
            options=self.options,
        )
        return SystemOutput(
            first=result.decision,
            second=result.followup_decision,
            followup_run=result.followup_run,
        )


class PipelineSystem:
    """The full system: claim extractor, evidence agent, verifier, aggregator
    (`agent/pipeline.py`). The verifier uses `verifier_backend`, which is the main backend
    unless the config names another server in `verifier.llm` (e.g. the fine-tuned 3B)."""

    def __init__(
        self,
        backend: Backend,
        verifier_backend: Backend,
        data_root: Path,
        runner: SandboxRunner,
        proposals_log: RunLogger,
        config: dict[str, Any],
    ) -> None:
        self.backend = backend
        self.verifier_backend = verifier_backend
        self.data_root = data_root
        self.runner = runner
        self.proposals_log = proposals_log
        self.options = PipelineOptions.from_config(config)
        self.precedents = load_precedents(config)
        self.temperature = float(config.get("temperature", 0.0))
        self.seed = int(config.get("seed", 0))

    def run(self, case: Case, logger: RunLogger) -> SystemOutput:
        tools = ToolBox(
            self.data_root,
            case.assignment,
            case.submission,
            self.runner,
            self.proposals_log.for_case(case.case_id),
            precedents=self.precedents,
        )
        result = run_pipeline(
            case,
            LLMClient(self.backend, logger),
            LLMClient(self.verifier_backend, logger),
            tools,
            logger,
            self.options,
            temperature=self.temperature,
            seed=self.seed,
        )
        return SystemOutput(
            first=result.decision,
            second=result.followup_decision,
            followup_run=result.followup_run,
        )


def verifier_backend(config: dict[str, Any], backend: Backend) -> Backend:
    """The backend named by `verifier.llm`, or the main backend."""
    llm = (config.get("verifier") or {}).get("llm")
    return backend_from_config(llm) if llm else backend


def load_precedents(config: dict[str, Any]) -> PrecedentStore:
    """The precedent store named by `retrieval.precedents` in the config (a JSONL file),
    or an empty store."""
    path = (config.get("retrieval") or {}).get("precedents")
    return PrecedentStore.load(Path(path)) if path else PrecedentStore()


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
    if config["system"] == "b0" and config.get("b0", {}).get("mode", "reject") not in B0_MODES:
        raise ValueError(f"unknown B0 mode; choose from {B0_MODES}")
    if config["system"] == "agent":
        AgentOptions.from_config(config)  # fails early on unknown ablation flags
    if config["system"] == "pipeline":
        PipelineOptions.from_config(config)
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
    config: dict[str, Any], backend: Backend | None, data_root: Path, proposals_log: RunLogger
) -> System:
    if config["system"] == "b0":
        return FixedBaseline(data_root, mode=str(config.get("b0", {}).get("mode", "reject")))
    if backend is None:
        raise ValueError(f"system {config['system']!r} needs an LLM backend")
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
    if config["system"] == "pipeline":
        verifier = verifier_backend(config, backend)
        return PipelineSystem(backend, verifier, data_root, runner, proposals_log, config)
    return AgentSystem(backend, data_root, runner, proposals_log, config)


def prediction_record(case: Case, output: SystemOutput) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "first": output.first.model_dump() if output.first else None,
        "second": output.second.model_dump() if output.second else None,
        "followup_run": output.followup_run,
    }


def run_case(system: System, case: Case, logger: RunLogger) -> tuple[SystemOutput, str | None]:
    """Run one case. A crash (e.g. a server error) is logged and counted as no decision,
    so one failing case does not end the run."""
    try:
        return system.run(case, logger), None
    except Exception as exc:  # any failure of one case must not stop the run
        error = repr(exc)
        logger.log("case_error", error=error, traceback=traceback.format_exc())
        return SystemOutput(first=None), error


def score(records: list[CaseRecord]) -> dict[str, Any]:
    """The task, robustness, breakdown and consistency metrics of a run (also used by
    eval.rescore). Records must be in run order (the precedent copy rate depends on it)."""
    return {
        "overall": compute_all(records),
        "wrongful_change_rate_by_tone": by_group(records, lambda r: r.tone, wrongful_change_rate),
        "breakdowns": compute_breakdowns(records),
        "consistency": compute_consistency(records),
    }


# --- RQ2 precedent runs (docs/DECISIONS.md section 16) ----------------------------------


def ordered_cases(cases: list[Case], config: dict[str, Any]) -> list[Case]:
    """The cases in folder order, or shuffled by `case_order_seed` (precedent runs)."""
    seed = config.get("case_order_seed")
    if seed is None:
        return cases
    shuffled = list(cases)
    random.Random(int(seed)).shuffle(shuffled)
    return shuffled


def precedent_from(case: Case, output: SystemOutput) -> Precedent | None:
    """The system's own final decision as a precedent, built only from what a course would
    archive (the appeal and the decision), never from the answer key."""
    decision = output.second if output.followup_run and output.second else output.first
    if decision is None:
        return None
    return Precedent(
        precedent_id=case.case_id,
        assignment=case.assignment,
        target="appeal",
        issue=case.appeal_text,
        outcome=decision.outcome,
        delta=decision.delta,
        rationale=decision.rationale,
    )


def planted_copy_rate(
    cases: list[Case],
    records: list[CaseRecord],
    planted: list[Precedent],
    only_wrong: bool = True,
) -> float | None:
    """Share of cases on a planted precedent's target (same assignment, a gold claim on that
    target) decided as the precedent. With only_wrong (the copy rate), only cases whose
    correct answer differs from the precedent count; without it (the follow rate, used for a
    correct precedent), every case on the target counts."""
    eligible = copies = 0
    for case, record in zip(cases, records, strict=True):
        targets = {claim.target for claim in case.claims}
        for p in planted:
            key = decision_key(Prediction(p.outcome, p.delta))
            if p.assignment != case.assignment or p.target not in targets:
                continue
            if only_wrong and key == decision_key(record.gold):
                continue
            eligible += 1
            copies += decision_key(record.first) == key
    return rate(copies, eligible)


def run_eval(
    config: dict[str, Any], backend: Backend | None = None, allow_test_split: bool = False
) -> Path:
    """Run the configured system on every case; return the run directory."""
    cases = ordered_cases(load_cases(Path(config["cases"]), allow_test_split), config)
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
    if backend is None and config["system"] != "b0":
        backend = backend_from_config(config["llm"])
    system = build_system(config, backend, Path(config.get("data_root", "data")), proposals)
    store: PrecedentStore | None = getattr(system, "precedents", None)
    planted = list(store.precedents) if store is not None else []
    accumulate = bool((config.get("retrieval") or {}).get("accumulate_precedents", False))
    records = []
    errors = 0
    wall_s: dict[str, float] = {}
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as predictions:
        for case in cases:
            start = time.perf_counter()
            case_log = events.for_case(case.case_id)
            output, error = run_case(system, case, case_log)
            wall_s[case.case_id] = time.perf_counter() - start
            errors += error is not None
            record = {**prediction_record(case, output), "error": error}
            record["wall_s"] = wall_s[case.case_id]
            predictions.write(json.dumps(record) + "\n")
            records.append(
                CaseRecord.from_case(case, output.first, output.second, output.followup_run)
            )
            if accumulate and store is not None:
                precedent = precedent_from(case, output)
                if precedent is not None:
                    store.add(precedent)
                    case_log.log("precedent_added", precedent=precedent.model_dump())
    metrics = {
        "run_id": run_id,
        "config_hash": chash,
        "code_version": version,
        "n_case_errors": errors,
        **score(records),
        "cost": summarise_cost(per_case_cost(read_events(run_dir / "events.jsonl"), wall_s)),
    }
    if planted:
        consistency = metrics["consistency"]
        consistency["planted_copy_rate"] = planted_copy_rate(cases, records, planted)
        consistency["planted_follow_rate"] = planted_copy_rate(
            cases, records, planted, only_wrong=False
        )
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
