"""Shared JSONL event log: one record per LLM call, tool call, retrieval, verdict or decision.

Every record carries the run id, case id, seed and config hash, so any result can be traced
back to the exact configuration and inputs that produced it.
"""

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


def config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of a config (key order does not matter)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RunContext:
    run_id: str
    seed: int
    config_hash: str
    case_id: str | None = None


class RunLogger:
    """Appends JSON lines to `path`. Use `for_case` to get a logger bound to one case."""

    def __init__(self, path: Path, context: RunContext) -> None:
        self.path = path
        self.context = context

    def for_case(self, case_id: str) -> "RunLogger":
        return RunLogger(self.path, replace(self.context, case_id=case_id))

    def log(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "ts": time.time(),
            "event": event,
            "run_id": self.context.run_id,
            "case_id": self.context.case_id,
            "seed": self.context.seed,
            "config_hash": self.context.config_hash,
            **payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return record


def read_events(path: Path, event: str | None = None) -> list[dict[str, Any]]:
    """All records in a log file, optionally only those of one event type."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [r for r in records if event is None or r["event"] == event]
