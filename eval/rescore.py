"""Recompute a finished run's metrics with the current metric definitions.

Usage: python -m eval.rescore <run dir> [--cases data/cases/dev]

Reads the run's predictions.jsonl and the cases it used, and writes
metrics_rescored.json next to the original metrics.json (which is kept unchanged). Cost,
run id and the original code version are copied over; `rescored_with` records the commit
of the metric code. The cases must be the ones the run used (check with git that they are
unchanged since the run's commit).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent.schemas import Case, Decision
from eval.metrics import CaseRecord
from eval.run import code_version, score


def load_records(run_dir: Path, cases_dir: Path) -> list[CaseRecord]:
    cases = {}
    for path in sorted(cases_dir.glob("*.json")):
        case = Case.model_validate_json(path.read_text(encoding="utf-8"))
        cases[case.case_id] = case
    records = []
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        prediction = json.loads(line)
        case = cases.get(prediction["case_id"])
        if case is None:
            raise ValueError(f"case {prediction['case_id']!r} not found in {cases_dir}")
        first, second = (
            Decision.model_validate(prediction[k]) if prediction[k] else None
            for k in ("first", "second")
        )
        records.append(CaseRecord.from_case(case, first, second, prediction["followup_run"]))
    return records


def rescore(run_dir: Path, cases_dir: Path) -> dict[str, Any]:
    original = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    rescored = {
        "run_id": original["run_id"],
        "config_hash": original.get("config_hash"),
        "code_version": original.get("code_version"),
        "rescored_with": code_version(),
        "n_case_errors": original.get("n_case_errors"),
        **score(load_records(run_dir, cases_dir)),
        "cost": original.get("cost"),
    }
    path = run_dir / "metrics_rescored.json"
    path.write_text(json.dumps(rescored, indent=2), encoding="utf-8")
    return rescored


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.rescore")
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--cases", type=Path, default=Path("data/cases/dev"))
    args = parser.parse_args(argv)
    for run_dir in args.run_dirs:
        overall = rescore(run_dir, args.cases)["overall"]
        print(run_dir.name, json.dumps({k: overall[k] for k in sorted(overall)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
