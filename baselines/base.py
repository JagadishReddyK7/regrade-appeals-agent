"""Common interface for every evaluated system (baselines and the agent)."""

from dataclasses import dataclass
from typing import Protocol

from agent.schemas import Case, Decision
from eval.runlog import RunLogger


@dataclass(frozen=True)
class SystemOutput:
    first: Decision | None  # None: no valid decision (counted, never repaired)
    second: Decision | None = None  # decision after the follow-up turn
    followup_run: bool = False


class System(Protocol):
    def run(self, case: Case, logger: RunLogger) -> SystemOutput: ...
