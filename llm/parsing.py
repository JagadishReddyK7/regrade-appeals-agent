"""Parse model output as JSON and validate it against a pydantic model.

Parsing is strict: no code-fence stripping, no repair. Invalid output is logged as an
`invalid_output` event so it can be counted, and the caller gets None.
"""

import json
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from eval.runlog import RunLogger
from llm.client import ToolCall

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ParseResult(Generic[T]):
    value: T | None
    error: str | None
    raw: str | None

    @property
    def ok(self) -> bool:
        return self.value is not None


def parse_json_output(raw: str | None, model: type[T]) -> ParseResult[T]:
    if raw is None:
        return ParseResult(None, "empty output", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ParseResult(None, f"invalid JSON: {exc}", raw)
    try:
        return ParseResult(model.model_validate(data), None, raw)
    except ValidationError as exc:
        return ParseResult(None, f"schema error: {exc.errors(include_url=False)}", raw)


def validate_output(raw: str | None, model: type[T], logger: RunLogger, kind: str) -> T | None:
    """Parsed model, or None after logging an `invalid_output` event."""
    result = parse_json_output(raw, model)
    if not result.ok:
        logger.log("invalid_output", kind=kind, schema=model.__name__, error=result.error, raw=raw)
    return result.value


def validate_tool_arguments(call: ToolCall, model: type[T], logger: RunLogger) -> T | None:
    return validate_output(call.arguments, model, logger, kind=f"tool_call:{call.name}")
