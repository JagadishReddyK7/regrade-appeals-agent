"""Structured result of one sandboxed run."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ErrorType = Literal[
    "test_failed",  # an assertion in the named test failed
    "exception",  # student code raised (traceback in stderr)
    "timeout",  # wall-clock or CPU-time limit exceeded
    "memory_limit",  # memory limit exceeded
    "network_blocked",  # code tried to import a networking module
    "crashed",  # process exited without reporting a result
]


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    stdout: str
    stderr: str
    error_type: ErrorType | None
    runtime_s: float = Field(ge=0)
    test_id: str | None  # None for custom-input runs
