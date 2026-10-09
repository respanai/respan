"""Canonical mapping of native Messages responses and SSE events."""

from __future__ import annotations

from typing import Any

from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv._incubating.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import LOG_TYPE_AGENT, LOG_TYPE_CHAT
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SESSION_ID,
)

from respan_instrumentation_anthropic._privacy import CapturePolicy
from respan_instrumentation_anthropic._serialization import (
    json_string,
    json_value,
    native_dict,
    redact_text,
)


def tool_calls(blocks: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": block.get("id"),
            "type": "function",
            "function": {
                "name": block.get("name"),
                "arguments": json_string(block.get("input")),
            },
        }
        for block in blocks
        if type(block) is dict and block.get("type") in ("tool_use", "server_tool_use")
    ]


class CallState:
    def __init__(
        self, instrumentor: Any, kwargs: dict[str, Any], session_id: str | None = None
    ) -> None:
        self.policy = CapturePolicy(
            instrumentor._observer, instrumentor.capture_content, instrumentor.context
        )
        self.request = {}
        self.snapshot = None
        self.buffers: dict[int, bytes] = {}
        self.events: list[Any] = []
        self.session_id = session_id
        self.done = False
        self.content_keys: set[str] = set()
        self.error = None
        self.usage: dict[str, int] = {}
        self.request_model = (
            kwargs.get("model") if type(kwargs.get("model")) is str else None
        )
        self.span = instrumentor._tracer.start_span(
            "anthropic.managed_agent" if session_id else "anthropic.chat",
            context=self.policy.context,
            attributes={
                RESPAN_LOG_TYPE: LOG_TYPE_AGENT if session_id else LOG_TYPE_CHAT
            },
        )
        if session_id is not None:
            self.span.set_attribute(RESPAN_SESSION_ID, session_id)
        else:
            self.span.set_attribute(SpanAttributes.LLM_SYSTEM, "anthropic")
            self.span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "chat")
        if type(kwargs.get("model")) is str:
            self.span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, kwargs["model"])
        self.options = {
            key: kwargs[native]
            for native, key in (
                ("max_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
                ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
                ("top_p", SpanAttributes.LLM_REQUEST_TOP_P),
            )
            if type(kwargs.get(native)) in (int, float)
        }
        self.span.set_attributes(self.options)
        self.check()
        if self.policy.allowed and self.span.is_recording():
            self.request = json_value(kwargs)
        self.policy.observer.states.add(self)

    def safe(self, method: str, *args: Any) -> None:
        try:
            getattr(self, method)(*args)
        except Exception:  # noqa: BLE001
            self.abort()

    def abort(self) -> None:
        self.done = True
        self.policy.allowed = False
        self.request = {}
        self.snapshot = None
        self.buffers.clear()
        self.events.clear()
        self.error = None
        self.policy.observer.states.discard(self)
        try:
            attrs = getattr(self.span, "_attributes", None)
            if attrs is not None:
                for key in self.content_keys:
                    attrs.pop(key, None)
            self.content_keys.clear()
            if (
                getattr(getattr(self.span, "status", None), "status_code", None)
                == StatusCode.ERROR
            ):
                self.span.set_status(Status(StatusCode.ERROR))
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            self.span.end()
        except Exception:  # noqa: BLE001, S110
            pass

    def check(self) -> bool:
        allowed = self.policy.check() and self.span.is_recording()
        if not allowed:
            self.request = {}
            self.snapshot = None
            self.buffers.clear()
            self.events.clear()
            self.error = None
            attrs = getattr(self.span, "_attributes", None)
            if attrs is not None:
                for key in self.content_keys:
                    attrs.pop(key, None)
            self.content_keys.clear()
            if (
                getattr(getattr(self.span, "status", None), "status_code", None)
                == StatusCode.ERROR
            ):
                self.span.set_status(Status(StatusCode.ERROR))
        return allowed

    def content(self, key: str, value: Any) -> None:
        if self.check():
            self.span.set_attribute(key, value)
            self.content_keys.add(key)

    def event(self, event: Any) -> None:
        allowed = self.check()
        data = native_dict(event)
        if data is None:
            return
        if self.session_id is not None:
            if data.get("type") == "span.model_request_end":
                usage = native_dict(data.get("model_usage")) or {}
                for key, value in usage.items():
                    if type(value) is int and key.endswith("tokens"):
                        self.usage[key] = self.usage.get(key, 0) + value
            if allowed:
                self.events.append(json_value(event))
            if data.get("type") == "session.error":
                self.failure_event(data)
            return
        if allowed:
            from inspect import signature

            # These values are native structural observations, independent of
            # whether a later content veto discards the native snapshot.
            if data.get("type") == "message_start":
                self.structural(native_dict(data.get("message")) or {})
            if data.get("type") == "message_delta":
                self.structural(data)

            from anthropic import NOT_GIVEN
            from anthropic.lib.streaming._messages import accumulate_event

            options = {"event": event, "current_snapshot": self.snapshot}
            parameters = signature(accumulate_event).parameters
            if "json_bufs" in parameters:
                options["json_bufs"] = self.buffers
            if "output_format" in parameters:
                options["output_format"] = NOT_GIVEN
            self.snapshot = accumulate_event(**options)
        else:
            if data.get("type") == "message_start":
                self.structural(native_dict(data.get("message")) or {})
            if data.get("type") == "message_delta":
                self.structural(data)

    def structural(self, data: dict[str, Any]) -> None:
        model = data.get("model")
        if type(model) is str:
            self.span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, model)
            if SpanAttributes.LLM_REQUEST_MODEL not in (self.span.attributes or {}):
                self.span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
        usage = native_dict(data.get("usage")) or {}
        self.usage.update(
            {key: value for key, value in usage.items() if type(value) is int}
        )

    def failure_event(self, data: dict[str, Any]) -> None:
        self.span.set_status(Status(StatusCode.ERROR))
        error = native_dict(data.get("error")) or data
        if type(error.get("type")) is str and error.get("type") != "session.error":
            self.span.set_attribute(ERROR_TYPE, error["type"])
        if type(error.get("message")) is str and self.check():
            self.error = redact_text(error["message"])

    def failure(self, exc: BaseException) -> None:
        self.span.set_status(Status(StatusCode.ERROR))
        self.span.set_attribute(ERROR_TYPE, type(exc).__name__)
        if self.check():
            for arg in exc.args:
                if type(arg) is str:
                    self.error = redact_text(arg)
                    break
        if isinstance(exc, __import__("anthropic").APIStatusError):
            code = object.__getattribute__(exc, "__dict__").get("status_code")
            if type(code) is int:
                self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, code)

    def finish(self, message: Any = None) -> None:
        if self.done:
            return
        self.done = True
        try:
            data = native_dict(message if message is not None else self.snapshot)
            if data is not None:
                self.structural(data)
            self.check()
            if self.policy.allowed and self.span.is_recording():
                self.map_content(data)
                if self.error is not None:
                    self.content(ERROR_MESSAGE, self.error)
                    self.span.set_status(Status(StatusCode.ERROR, self.error))
            for native, keys in (
                (
                    "input_tokens",
                    (GEN_AI_USAGE_INPUT_TOKENS, SpanAttributes.LLM_USAGE_PROMPT_TOKENS),
                ),
                (
                    "output_tokens",
                    (
                        GEN_AI_USAGE_OUTPUT_TOKENS,
                        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                    ),
                ),
                (
                    "cache_read_input_tokens",
                    (SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,),
                ),
                (
                    "cache_creation_input_tokens",
                    (SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,),
                ),
            ):
                if native in self.usage:
                    for key in keys:
                        self.span.set_attribute(key, self.usage[native])
            self.span.set_attribute(
                RESPAN_LOG_TYPE, LOG_TYPE_AGENT if self.session_id else LOG_TYPE_CHAT
            )
            if not self.session_id:
                self.span.set_attribute(SpanAttributes.LLM_SYSTEM, "anthropic")
                self.span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "chat")
                if self.request_model is not None:
                    self.span.set_attribute(
                        SpanAttributes.LLM_REQUEST_MODEL, self.request_model
                    )
                if data is not None:
                    self.structural(data)
            self.check()
        finally:
            self.request = {}
            self.snapshot = None
            self.buffers.clear()
            self.events.clear()
            self.error = None
            self.policy.observer.states.discard(self)
            try:
                self.span.end()
            except Exception:  # noqa: BLE001, S110
                pass

    def map_content(self, data: dict[str, Any] | None) -> None:
        if self.session_id is not None:
            inputs = [
                event
                for event in self.events
                if event.get("type")
                in ("user.message", "user.tool_result", "user.custom_tool_result")
            ]
            outputs = [
                event
                for event in self.events
                if event.get("type")
                in (
                    "agent.message",
                    "agent.thinking",
                    "agent.tool_use",
                    "agent.tool_result",
                    "agent.mcp_tool_use",
                    "agent.mcp_tool_result",
                    "agent.custom_tool_use",
                )
            ]
            if inputs:
                self.content(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(inputs))
            if outputs:
                self.content(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(outputs)
                )
            return
        messages = self.request.get("messages", [])
        if "system" in self.request:
            messages = [
                {"role": "system", "content": self.request["system"]},
                *messages,
            ]
        for index, item in enumerate(messages):
            if type(item) is dict:
                self.content(
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.role", item.get("role", "")
                )
                self.content(
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                    json_string(item.get("content")),
                )
        if data is not None:
            output = json_value(data.get("content"))
            self.content(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))
            self.content(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
            self.content(
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content", json_string(output)
            )
            calls = tool_calls(output or [])
            if calls:
                self.content(
                    f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls", json_string(calls)
                )
        # Write complete canonical bodies after indexed projections so the
        # default OTel attribute limit cannot evict a long history or schema.
        self.content(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(messages))
        tools = self.request.get("tools")
        if tools is not None:
            definitions = []
            for tool in tools:
                if type(tool) is not dict:
                    continue
                function = dict(tool)
                if "input_schema" in function:
                    function["parameters"] = function.pop("input_schema")
                definitions.append({"type": "function", "function": function})
            self.content(SpanAttributes.LLM_REQUEST_FUNCTIONS, json_string(definitions))
        options = {
            key: value
            for key, value in self.request.items()
            if key not in ("messages", "system", "tools", "model")
        }
        if options:
            self.content(RESPAN_METADATA + ".anthropic.request", json_string(options))
        self.span.set_attributes(self.options)
