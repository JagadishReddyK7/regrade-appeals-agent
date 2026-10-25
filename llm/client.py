"""Chat client with two backends: an OpenAI-compatible server (vLLM) and a scripted mock.

All calls go through `LLMClient.chat`, which logs one `llm_call` event per call (or an
`llm_error` event if the backend raises). Prompts are logged as a hash only, to keep logs
small; responses are logged in full.

vLLM notes: for tool calling with Qwen2.5, start the server with
`--enable-auto-tool-choice --tool-call-parser hermes`. `json_schema` uses guided decoding
through `response_format`, so the output is constrained to the schema.
"""

import hashlib
import json
import time
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from eval.runlog import RunLogger

Message = dict[str, Any]


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: str  # raw JSON text from the model; validate with llm.parsing


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Backend(Protocol):
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        seed: int,
    ) -> ChatResponse: ...


def prompt_hash(messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None) -> str:
    canonical = json.dumps({"messages": messages, "tools": tools}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class LLMClient:
    def __init__(self, backend: Backend, logger: RunLogger) -> None:
        self.backend = backend
        self.logger = logger

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> ChatResponse:
        meta = {
            "model": self.backend.model,
            "prompt_hash": prompt_hash(messages, tools),
            "temperature": temperature,
            "call_seed": seed,
        }
        start = time.perf_counter()
        try:
            response = self.backend.complete(messages, tools, json_schema, temperature, seed)
        except Exception as exc:
            latency = time.perf_counter() - start
            self.logger.log("llm_error", **meta, latency_s=latency, error=repr(exc))
            raise
        self.logger.log(
            "llm_call",
            **meta,
            latency_s=time.perf_counter() - start,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            finish_reason=response.finish_reason,
            content=response.content,
            tool_calls=[call.model_dump() for call in response.tool_calls],
        )
        return response


# --- OpenAI-compatible backend (vLLM) -------------------------------------------------


def build_request(
    model: str,
    messages: Sequence[Message],
    tools: Sequence[dict[str, Any]] | None,
    json_schema: dict[str, Any] | None,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "seed": seed,
    }
    if tools:
        request["tools"] = list(tools)
        request["tool_choice"] = "auto"
    if json_schema is not None:
        name = str(json_schema.get("title", "output"))
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": json_schema, "strict": True},
        }
    return request


def to_chat_response(completion: Any) -> ChatResponse:
    """Convert an OpenAI chat completion object into a ChatResponse."""
    choice = completion.choices[0]
    calls = [
        ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
        for c in (choice.message.tool_calls or [])
    ]
    usage = completion.usage
    return ChatResponse(
        content=choice.message.content,
        tool_calls=calls,
        model=completion.model,
        finish_reason=choice.finish_reason,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )


class OpenAIBackend:
    def __init__(
        self, base_url: str, model: str, api_key: str = "EMPTY", timeout_s: float = 120.0
    ) -> None:
        from openai import OpenAI  # imported lazily so tests never need a server

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        seed: int,
    ) -> ChatResponse:
        request = build_request(self.model, messages, tools, json_schema, temperature, seed)
        return to_chat_response(self._client.chat.completions.create(**request))


# --- Mock backend ----------------------------------------------------------------------


class MockLLM:
    """Returns scripted responses in order and records every request it receives."""

    def __init__(self, responses: Sequence[ChatResponse | str], model: str = "mock") -> None:
        self.model = model
        self._responses = [
            r if isinstance(r, ChatResponse) else ChatResponse(content=r, model=model)
            for r in responses
        ]
        self.requests: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        seed: int,
    ) -> ChatResponse:
        self.requests.append(
            build_request(self.model, messages, tools, json_schema, temperature, seed)
        )
        if len(self.requests) > len(self._responses):
            raise RuntimeError(f"MockLLM ran out of scripted responses ({len(self._responses)})")
        return self._responses[len(self.requests) - 1]


def backend_from_config(config: dict[str, Any]) -> Backend:
    """Build the OpenAI-compatible backend from an `llm` config section.

    Expected keys: base_url, model; optional api_key, timeout_s. The mock backend needs a
    script, so tests construct MockLLM directly.
    """
    kind = config.get("backend", "openai")
    if kind != "openai":
        raise ValueError(f"unsupported backend {kind!r}; construct MockLLM directly in tests")
    return OpenAIBackend(
        base_url=config["base_url"],
        model=config["model"],
        api_key=config.get("api_key", "EMPTY"),
        timeout_s=float(config.get("timeout_s", 120.0)),
    )
