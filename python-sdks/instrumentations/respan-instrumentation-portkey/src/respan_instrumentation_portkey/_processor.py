"""Portkey's actual request/response values mapped to the canonical contract."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.exception_attributes import EXCEPTION_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import NonRecordingSpan, Status, StatusCode
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE

from respan_instrumentation_portkey._policy import content_allowed
from respan_instrumentation_portkey._serialization import (
    exception_message,
    exception_status,
    fields,
    json_dumps,
    safe_text,
    safe_type_name,
)


def usage_attributes(value):
    data = fields(value) or {}

    def count(name, owner=data):
        n = owner.get(name)
        return n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None

    result = {}
    for names, keys in [
        (
            ("input_tokens", "prompt_tokens"),
            (GEN_AI_USAGE_INPUT_TOKENS, SpanAttributes.LLM_USAGE_PROMPT_TOKENS),
        ),
        (
            ("output_tokens", "completion_tokens"),
            (GEN_AI_USAGE_OUTPUT_TOKENS, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS),
        ),
        (("total_tokens",), (SpanAttributes.LLM_USAGE_TOTAL_TOKENS,)),
    ]:
        actual = next((count(n) for n in names if count(n) is not None), None)
        if actual is not None:
            for key in keys:
                result[key] = actual
    details = (
        fields(data.get("input_tokens_details"))
        or fields(data.get("prompt_tokens_details"))
        or {}
    )
    actual = count("cache_read_input_tokens")
    if actual is None:
        actual = count("cached_tokens", details)
    if actual is not None:
        result[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = actual
    creation = count("cache_creation_input_tokens")
    if creation is None:
        creation = count("cache_write_tokens")
    if creation is None:
        creation = count("cache_write_tokens", details)
    if creation is not None:
        result[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] = creation
    details = (
        fields(data.get("output_tokens_details"))
        or fields(data.get("completion_tokens_details"))
        or {}
    )
    actual = count("reasoning_tokens", details)
    if actual is not None:
        result[SpanAttributes.LLM_USAGE_REASONING_TOKENS] = actual
    return result


def _messages(attrs, value, prefix):
    if not isinstance(value, list):
        return
    # Indexed fields must not evict entity JSON, usage or inherited metadata at
    # OTel's default128-attribute limit. The full message/choice payload remains
    # in entity input/output; only this redundant index projection is bounded.
    for index, raw in enumerate(value[:8]):
        message = fields(raw)
        if message is None:
            continue
        base = f"{prefix}.{index}"
        for source in ["role", "content", "tool_call_id", "name"]:
            item = message.get(source)
            if isinstance(item, str):
                attrs[f"{base}.{source}"] = safe_text(item, complete=True)
            elif source == "content" and isinstance(item, list):
                attrs[f"{base}.content"] = json_dumps(item, complete=True)
        if isinstance(message.get("tool_calls"), list):
            attrs[f"{base}.tool_calls"] = json_dumps(
                message["tool_calls"], complete=True
            )


def response_attributes(
    response, kind, capture=True, source_usage=None, source_seen=False
):
    data = fields(response)
    if data is None:
        return {}
    attrs = usage_attributes(source_usage if source_seen else data.get("usage"))
    if isinstance(data.get("model"), str):
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = safe_text(data["model"])
    if not capture:
        return attrs
    if kind == "embedding":
        rows = data.get("data")
        if isinstance(rows, list):
            vectors = [(fields(row) or {}).get("embedding") for row in rows]
            if vectors and all(isinstance(v, (list, str)) for v in vectors):
                attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(
                    vectors, complete=True
                )
        return attrs
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(response, complete=True)
    choices = data.get("choices")
    if isinstance(choices, list):
        messages = []
        for raw in choices:
            choice = fields(raw) or {}
            message = fields(choice.get("message"))
            if message is None and isinstance(choice.get("text"), str):
                message = {"role": "assistant", "content": choice["text"]}
            if message is not None:
                messages.append(message)
        _messages(attrs, messages, SpanAttributes.LLM_COMPLETIONS)
    output = data.get("output")
    if isinstance(output, list):
        messages = []
        calls = []
        for raw in output:
            item = fields(raw) or {}
            if item.get("type") == "message":
                content = item.get("content")
                text = []
                if isinstance(content, list):
                    for raw_part in content:
                        part = fields(raw_part) or {}
                        if part.get("type") == "output_text" and isinstance(
                            part.get("text"), str
                        ):
                            text.append(part["text"])
                message = {"role": item.get("role"), "content": "".join(text)}
                messages.append(message)
            elif item.get("type") == "function_call":
                calls.append(
                    {
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments"),
                        },
                    }
                )
        if calls:
            attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"] = json_dumps(
                calls, complete=True
            )
        _messages(attrs, messages, SpanAttributes.LLM_COMPLETIONS)
    return attrs


@dataclass(eq=False)
class Operation:
    span: Any
    request: dict
    kind: str
    capture: bool
    setting: bool
    parent: Any = None
    hide_inputs: bool = False
    hide_outputs: bool = False
    hide_tools: bool = False
    hide_vectors: bool = False
    done: bool = False
    on_finish: Any = None
    stream_accumulator: Any = None
    metadata: dict = field(default_factory=dict)
    lock: Any = field(default_factory=threading.RLock)
    source_seen: bool = False
    source_usage: Any = None
    policy: Any = None

    def checkpoint(self):
        if self.policy is not None:
            try:
                self.policy.observe(self.span)
                allowed = self.policy.allowed(self.span)
            except Exception:  # noqa: BLE001 - fail closed without replacing SDK results
                allowed = False
            if not allowed:
                self.capture = False
                self.request = {}
                if self.stream_accumulator is not None:
                    self.stream_accumulator.clear_content()
        if not content_allowed(self.setting):
            current = self
            while current is not None:
                current.capture = False
                current.request = {}
                if current.stream_accumulator is not None:
                    current.stream_accumulator.clear_content()
                current = current.parent
        if self.parent is not None and not self.parent.capture:
            self.capture = False
            self.request = {}
            if self.stream_accumulator is not None:
                self.stream_accumulator.clear_content()
        return self.capture and self.span.is_recording()

    def finish(self, response=None, error=None):
        with self.lock:
            if self.done:
                return
            self.done = True
        allowed = self.checkpoint()
        attrs = {
            RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
            RESPAN_LOG_TYPE: self.kind,
            SpanAttributes.TRACELOOP_ENTITY_NAME: f"portkey.{self.kind}",
            SpanAttributes.TRACELOOP_ENTITY_PATH: "",
        }
        if self.span.is_recording():
            # The upstream wrapper's context/error machinery is retained; canonical
            # source values replace its raw attributes without helper aliases.
            self.span.set_attributes(attrs)
            if self.kind in {"chat", "text", "embedding"}:
                self.span.set_attribute(SpanAttributes.LLM_SYSTEM, "portkey")
                self.span.set_attribute(
                    SpanAttributes.LLM_REQUEST_TYPE,
                    LLMRequestTypeValues.EMBEDDING.value
                    if self.kind == "embedding"
                    else LLMRequestTypeValues.CHAT.value,
                )
            model = self.metadata.get("model")
            if isinstance(model, str):
                self.span.set_attribute(
                    SpanAttributes.LLM_REQUEST_MODEL, safe_text(model)
                )
            if self.metadata.get("stream") is True:
                self.span.set_attribute(SpanAttributes.LLM_IS_STREAMING, True)
            if self.source_seen:
                self.span.set_attributes(usage_attributes(self.source_usage))
            if allowed and not self.hide_inputs:
                actual = (
                    self.request.get("input")
                    if self.kind == "embedding"
                    else {
                        k: v
                        for k, v in self.request.items()
                        if not self.hide_tools or k not in {"tools", "functions"}
                    }
                )
                self.span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                    json_dumps(actual, complete=True),
                )
                _attrs = {}
                _messages(
                    _attrs, self.request.get("messages"), SpanAttributes.LLM_PROMPTS
                )
                tools = self.request.get("tools", self.request.get("functions"))
                if not self.hide_tools and isinstance(tools, list):
                    _attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_dumps(
                        tools, complete=True, tool_definitions=True
                    )
                self.span.set_attributes(_attrs)
            if response is not None:
                native = fields(response) or {}
                failed = (
                    native.get("status") == "failed"
                    and fields(native.get("error")) is not None
                )
                output = response_attributes(
                    response,
                    self.kind,
                    allowed
                    and not self.hide_outputs
                    and not (self.kind == "embedding" and self.hide_vectors),
                    self.source_usage,
                    True,
                )
                for key, value in output.items():
                    content = (
                        key == SpanAttributes.TRACELOOP_ENTITY_OUTPUT
                        or key.startswith(SpanAttributes.LLM_COMPLETIONS + ".")
                    )
                    if content and (
                        not allowed
                        or self.hide_outputs
                        or (self.kind == "embedding" and self.hide_vectors)
                    ):
                        continue
                    if content and failed:
                        continue
                    self.span.set_attribute(key, value)
                if failed:
                    failure = fields(native["error"])
                    code = failure.get("type", failure.get("code"))
                    if isinstance(code, str):
                        self.span.set_attribute(ERROR_TYPE, safe_text(code))
                    message = failure.get("message") if allowed else None
                    if isinstance(message, str):
                        self.span.set_attribute(ERROR_MESSAGE, safe_text(message))
                    self.span.set_status(Status(StatusCode.ERROR))
            if error is not None and not isinstance(
                error, (GeneratorExit, asyncio.CancelledError)
            ):
                self.span.set_attribute(EXCEPTION_TYPE, safe_type_name(error))
                status = exception_status(error)
                if status is not None:
                    self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status)
                message = exception_message(error) if allowed else None
                if message:
                    self.span.set_attribute(ERROR_MESSAGE, message)
                self.span.set_status(Status(StatusCode.ERROR, message))
            elif error is None and (fields(response) or {}).get("status") != "failed":
                self.span.set_status(Status(StatusCode.OK))
        span_context = self.span.get_span_context()
        self.span.end()
        self.span = NonRecordingSpan(span_context)
        callback = self.on_finish
        self.request = {}
        self.parent = None
        self.stream_accumulator = None
        self.on_finish = None
        self.source_usage = None
        if callback:
            callback(self)
