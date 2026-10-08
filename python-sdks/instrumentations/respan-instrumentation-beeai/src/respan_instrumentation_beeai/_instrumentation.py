"""Translate BeeAI's public lifecycle events without patching SDK methods."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from threading import RLock
from typing import Any, ClassVar

from opentelemetry import context, trace
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_MESSAGE,
    EXCEPTION_TYPE,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_beeai._serialization import (
    MAX_CHARS,
    data,
    json_value,
    text,
)

logger = logging.getLogger(__name__)


@dataclass
class _Run:
    span: Any
    log_type: str
    capture: bool
    name: str
    input: str | None = None
    output: str | None = None
    messages: list[dict[str, Any]] | None = None
    completions: list[dict[str, Any]] | None = None
    tools: str | None = None
    streamed: bool = False


def _content_enabled(options: dict[str, Any]) -> bool:
    if context.get_value(ENABLE_CONTENT_TRACING_KEY) is False:
        return False
    if options.get("trace_content") is False:
        return False
    if os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False
    config = options.get("config")
    return not any(
        getattr(config, key, False)
        for key in (
            "hide_inputs",
            "hide_outputs",
            "hide_input_messages",
            "hide_output_messages",
            "hide_input_text",
            "hide_output_text",
            "hide_input_images",
            "hide_output_images",
            "hide_embedding_vectors",
            "hide_llm_tools",
            "hide_llm_invocation_parameters",
        )
    )


def _messages(messages: Any) -> list[dict[str, Any]]:
    from beeai_framework.backend import (
        MessageTextContent,
        MessageToolCallContent,
        MessageToolResultContent,
    )

    result = []
    for message in messages:
        item: dict[str, Any] = {"role": text(message.role, 64)}
        chunks = []
        calls = []
        for chunk in message.content:
            if isinstance(chunk, MessageTextContent):
                chunks.append(text(chunk.text))
            elif isinstance(chunk, MessageToolCallContent):
                call = {
                    "type": "function",
                    "function": {
                        "name": text(chunk.tool_name, None),
                        "arguments": (
                            data(chunk.args, complete=True)
                            if isinstance(chunk.args, str)
                            else json_value(chunk.args, complete=True)
                        ),
                    },
                }
                if chunk.id:
                    call["id"] = text(chunk.id, None)
                calls.append(call)
            elif isinstance(chunk, MessageToolResultContent):
                chunks.append(data(chunk.result, complete=True))
                item["tool_call_id"] = text(chunk.tool_call_id, None)
            else:
                chunks.append(data(chunk))
        if chunks:
            item["content"] = (
                text("".join(chunks), None if "tool_call_id" in item else MAX_CHARS)
                if all(isinstance(x, str) for x in chunks)
                else chunks
            )
        if calls:
            item["tool_calls"] = calls
        result.append(item)
    return result


def _usage(span: Any, usage: Any, *, embedding: bool = False) -> None:
    # BeeAI's usage models default to zero. An untouched default is not provider usage.
    fields = getattr(usage, "model_fields_set", set())
    mappings = (
        (
            "prompt_tokens",
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            GEN_AI_USAGE_INPUT_TOKENS,
        ),
        (
            "completion_tokens",
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            GEN_AI_USAGE_OUTPUT_TOKENS,
        ),
        ("total_tokens", SpanAttributes.LLM_USAGE_TOTAL_TOKENS, None),
        (
            "cached_prompt_tokens",
            SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            None,
        ),
        (
            "cached_creation_tokens",
            SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
            None,
        ),
    )
    for field, key, modern in mappings:
        if embedding and field != "prompt_tokens":
            continue
        value = getattr(usage, field, None)
        if (
            field in fields
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        ):
            span.set_attribute(key, value)
            if modern:
                span.set_attribute(modern, value)


def _kind(instance: Any) -> tuple[str, str]:
    from beeai_framework.agents import BaseAgent
    from beeai_framework.backend import ChatModel, EmbeddingModel
    from beeai_framework.tools.tool import Tool
    from beeai_framework.workflows import Workflow

    if isinstance(instance, ChatModel):
        return LOG_TYPE_CHAT, type(instance).__name__
    if isinstance(instance, EmbeddingModel):
        return LOG_TYPE_EMBEDDING, "CreateEmbeddings"
    if isinstance(instance, Tool):
        return LOG_TYPE_TOOL, instance.name
    if isinstance(instance, BaseAgent):
        return LOG_TYPE_AGENT, instance.meta.name or type(instance).__name__
    if isinstance(instance, Workflow):
        return LOG_TYPE_WORKFLOW, instance.name
    return LOG_TYPE_TASK, type(instance).__name__


class _Listener:
    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options
        self.tracer = trace.get_tracer(__name__)
        self.runs: dict[str, _Run | None] = {}
        self.cleanup: Any = None
        self.root: Any = None
        self.callback = self.handler

    async def handler(self, event: Any, meta: Any) -> None:
        try:
            self._handle(event, meta)
        except Exception:  # noqa: BLE001 - telemetry must not affect application results
            # Observers must never change SDK results or exceptions; do not log payloads.
            logger.debug("Could not translate a BeeAI event")

    def _handle(self, event: Any, meta: Any) -> None:
        from beeai_framework.context import RunContextFinishEvent, RunContextStartEvent

        if meta.trace is None:
            return
        run_id = meta.trace.run_id
        if isinstance(event, RunContextStartEvent):
            if run_id in self.runs:
                return
            parent_id = meta.trace.parent_run_id
            if (
                context.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
                or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
                or (parent_id in self.runs and self.runs[parent_id] is None)
            ):
                self.runs[run_id] = None
                return
            instance = meta.creator.instance
            log_type, name = _kind(instance)
            source_name = text(name, None)
            name = text(name, 512)
            parent = self.runs.get(parent_id)
            parent_context = (
                trace.set_span_in_context(parent.span)
                if parent
                else context.get_current()
            )
            span = self.tracer.start_span(
                name,
                context=parent_context,
                attributes={
                    RESPAN_LOG_TYPE: log_type,
                    SpanAttributes.TRACELOOP_ENTITY_NAME: name,
                    SpanAttributes.TRACELOOP_ENTITY_PATH: name if parent else "",
                },
            )
            run = _Run(
                span,
                log_type,
                span.is_recording() and _content_enabled(self.options),
                source_name,
            )
            self.runs[run_id] = run
            if not span.is_recording():
                return
            if log_type in {LOG_TYPE_CHAT, LOG_TYPE_EMBEDDING}:
                span.set_attribute(
                    SpanAttributes.LLM_REQUEST_MODEL, text(instance.model_id, 512)
                )
                span.set_attribute(
                    SpanAttributes.LLM_SYSTEM, text(instance.provider_id, 64).lower()
                )
                span.set_attribute(
                    SpanAttributes.LLM_REQUEST_TYPE,
                    "chat" if log_type == LOG_TYPE_CHAT else "embedding",
                )
            if log_type == LOG_TYPE_TOOL:
                call = meta.context.get("tool_call_msg")
                call_id = getattr(call, "id", None)
                if isinstance(call_id, str) and call_id:
                    span.set_attribute(GEN_AI_TOOL_CALL_ID, text(call_id, None))
            if run.capture:
                run.input = json_value(event.input, complete=log_type == LOG_TYPE_TOOL)
            return
        if run_id not in self.runs:
            return
        run = self.runs[run_id]
        if run is not None and not _content_enabled(self.options):
            # Lowering privacy at an observed event permanently vetoes content.
            run.capture = False
            run.input = run.output = run.tools = None
            run.messages = run.completions = None
        if isinstance(event, RunContextFinishEvent):
            try:
                if run is not None and run.span.is_recording():
                    if event.error is not None:
                        self._error(run, event.error)
                    elif run.capture and _content_enabled(self.options):
                        self._finish_output(run, event.output)
                    self._publish_content(run)
            finally:
                self.runs.pop(run_id, None)
                if run is not None:
                    run.span.end()
            return
        if run is None or not run.span.is_recording():
            return
        from beeai_framework.backend.events import (
            ChatModelNewTokenEvent,
            ChatModelStartEvent,
            ChatModelSuccessEvent,
            EmbeddingModelStartEvent,
            EmbeddingModelSuccessEvent,
        )

        if isinstance(event, ChatModelStartEvent):
            parameters = event.input
            for field, key in (
                ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
                ("top_p", SpanAttributes.LLM_REQUEST_TOP_P),
                ("max_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
                ("stream", SpanAttributes.LLM_IS_STREAMING),
                ("reasoning_effort", SpanAttributes.LLM_REQUEST_REASONING_EFFORT),
            ):
                value = getattr(parameters, field, None)
                if getattr(
                    self.options.get("config"), "hide_llm_invocation_parameters", False
                ):
                    continue
                if value is not None and isinstance(value, (str, bool, int, float)):
                    run.span.set_attribute(
                        key, text(value, 512) if isinstance(value, str) else value
                    )
            if run.capture and _content_enabled(self.options):
                run.messages = _messages(event.input.messages)
                definitions = []
                for tool in event.input.tools or []:
                    definitions.append(
                        {
                            "type": "function",
                            "function": {
                                "name": text(tool.name, None),
                                "description": text(tool.description, None),
                                "parameters": data(
                                    tool.input_schema.model_json_schema(), complete=True
                                ),
                            },
                        }
                    )
                if definitions:
                    run.tools = json_value(definitions, complete=True)
        elif isinstance(event, ChatModelSuccessEvent):
            if not run.streamed:
                _usage(run.span, event.value.usage)
            if event.value.finish_reason:
                run.span.set_attribute(
                    SpanAttributes.LLM_RESPONSE_FINISH_REASON,
                    (text(event.value.finish_reason, 64),),
                )
            if run.capture and _content_enabled(self.options):
                run.completions = _messages(event.value.output)
                run.output = json_value({"messages": run.completions})
        elif isinstance(event, ChatModelNewTokenEvent):
            # Read source chunks before BeeAI synthesizes defaults during merge.
            run.streamed = True
            _usage(run.span, event.value.usage)
            # Consume neither the SDK stream nor its mutable message objects.
            run.span.set_attribute(SpanAttributes.LLM_IS_STREAMING, True)
        elif isinstance(event, EmbeddingModelStartEvent):
            if run.capture and _content_enabled(self.options):
                run.input = json_value(event.input.values)
        elif isinstance(event, EmbeddingModelSuccessEvent):
            _usage(run.span, event.value.usage, embedding=True)
            if run.capture and _content_enabled(self.options):
                run.output = json_value(event.value.embeddings, complete=True)

    def _finish_output(self, run: _Run, output: Any) -> None:
        if run.output is not None or output is None:
            return
        from beeai_framework.tools.types import JSONToolOutput, StringToolOutput

        if isinstance(output, (StringToolOutput, JSONToolOutput)):
            run.output = json_value(output.result, complete=True)
        elif run.log_type == LOG_TYPE_AGENT:
            run.output = json_value({"messages": _messages(output.output)})
        elif run.log_type == LOG_TYPE_CHAT:
            run.completions = _messages(output.output)
            run.output = json_value({"messages": run.completions})
            if not run.streamed:
                _usage(run.span, output.usage)
        elif run.log_type == LOG_TYPE_WORKFLOW:
            run.output = json_value(
                output.result if output.result is not None else output.state
            )
        else:
            run.output = json_value(output)

    def _error(self, run: _Run, error: BaseException) -> None:
        run.span.set_status(Status(StatusCode.ERROR))
        capture = run.capture and _content_enabled(self.options)
        attrs = {EXCEPTION_TYPE: type(error).__name__}
        if capture:
            args = BaseException.args.__get__(error)
            message = "; ".join(text(value) for value in args if isinstance(value, str))
            attrs[EXCEPTION_MESSAGE] = message or type(error).__name__
        run.span.add_event("exception", attrs)
        run.span.set_status(Status(StatusCode.ERROR, attrs.get(EXCEPTION_MESSAGE)))
        run.output = None
        run.completions = None

    def _publish_content(self, run: _Run) -> None:
        if not run.capture or not _content_enabled(self.options):
            return
        if run.input is not None:
            value = run.input
            if run.log_type == LOG_TYPE_TOOL:
                import json

                raw = json.loads(value)
                arguments = raw.get("input", raw) if isinstance(raw, dict) else raw
                value = json_value(
                    {"name": run.name, "arguments": arguments}, complete=True
                )
            run.span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, value)
        if run.output is not None:
            run.span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, run.output)
        if run.tools:
            run.span.set_attribute(SpanAttributes.LLM_REQUEST_FUNCTIONS, run.tools)
        for messages, prefix in (
            (run.messages, SpanAttributes.LLM_PROMPTS),
            (run.completions, SpanAttributes.LLM_COMPLETIONS),
        ):
            for index, message in enumerate(messages or []):
                base = f"{prefix}.{index}"
                run.span.set_attribute(f"{base}.role", message["role"])
                if "content" in message:
                    content = message["content"]
                    run.span.set_attribute(
                        f"{base}.content",
                        text(
                            content, None if message.get("tool_call_id") else MAX_CHARS
                        )
                        if isinstance(content, str)
                        else json_value(
                            content, complete=bool(message.get("tool_call_id"))
                        ),
                    )
                if message.get("tool_calls"):
                    run.span.set_attribute(
                        f"{base}.tool_calls",
                        json_value(message["tool_calls"], complete=True),
                    )
                if message.get("tool_call_id"):
                    run.span.set_attribute(
                        f"{base}.tool_call_id", message["tool_call_id"]
                    )

    def close(self) -> None:
        try:
            if self.cleanup is not None:
                self.cleanup()
        finally:
            self.cleanup = None
            if self.root is not None:
                self.root.off(callback=self.callback)
            runs, self.runs = self.runs, {}
            for run in runs.values():
                if run is not None:
                    run.span.end()


class BeeAIInstrumentor:
    """Observe BeeAI runs with one shared, removable root Emitter listener."""

    name = "beeai"
    _lock: ClassVar[RLock] = RLock()
    _listener: ClassVar[_Listener | None] = None
    _owners: ClassVar[int] = 0
    _provider: ClassVar[Any] = None

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._options = instrumentor_kwargs
        self._is_instrumented = False

    def activate(self) -> None:
        cls = BeeAIInstrumentor
        with cls._lock:
            if self._is_instrumented:
                return
            tracer = getattr(RespanTracer, "_instance", None)
            if tracer is not None and not getattr(tracer, "is_enabled", True):
                return
            provider = trace.get_tracer_provider()
            if cls._listener is not None:
                if cls._provider is not provider:
                    raise RuntimeError(
                        "BeeAI instrumentation is active on another tracer provider"
                    )
                if cls._listener.options != self._options:
                    raise ValueError(
                        "BeeAI instrumentation is active with different settings"
                    )
                cls._owners += 1
                self._is_instrumented = True
                return
            from beeai_framework.emitter import Emitter, EmitterOptions

            listener = _Listener(dict(self._options))
            try:
                listener.root = Emitter.root()
                listener.cleanup = listener.root.on(
                    "*.*",
                    listener.callback,
                    EmitterOptions(match_nested=True, is_blocking=True),
                )
            except BaseException:
                listener.close()
                raise
            cls._listener = listener
            cls._provider = provider
            cls._owners = 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        cls = BeeAIInstrumentor
        with cls._lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            cls._owners -= 1
            if cls._owners:
                return
            listener, cls._listener = cls._listener, None
            cls._provider = None
            if listener is not None:
                listener.close()
