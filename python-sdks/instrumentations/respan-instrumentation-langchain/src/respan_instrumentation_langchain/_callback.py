"""Native LangChain callbacks into the active OpenTelemetry provider."""

from __future__ import annotations

import functools
import json
import logging
import os
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.messages import BaseMessage, ToolMessage
from opentelemetry import context, trace
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_PROVIDER_NAME,
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
    LOG_TYPE_TASK,
    LOG_TYPE_TEXT,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SPAN_ATTRIBUTES_MAP,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer

from ._constants import (
    LANGCHAIN_FRAMEWORK_ATTR,
    LANGCHAIN_METADATA_ATTR,
    LANGCHAIN_PARENT_RUN_ID_ATTR,
    LANGCHAIN_RUN_ID_ATTR,
    LANGCHAIN_SERIALIZED_ATTR,
    LANGCHAIN_TAGS_ATTR,
)
from ._serialization import MAX_CHARS, data, json_value, text

logger = logging.getLogger(__name__)
_MARKER_FIELDS = {
    "run_id",
    "example_run_id",
    "framework",
    "example",
    "workflow_name",
    "example_set",
    "example_name",
}


def _safe_callback(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:  # noqa: BLE001 - instrumentation cannot change application behavior
            logger.debug("Could not translate a LangChain callback")
            return None

    return call


def _key(value: Any) -> str:
    if isinstance(value, UUID):
        return value.hex
    return value if isinstance(value, str) else ""


def _content_enabled(handler) -> bool:
    return (
        handler.include_content
        and context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"0", "false", "off", "no"}
    )


def _suppressed() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    return bool(
        context.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
        or (tracer is not None and not getattr(tracer, "is_enabled", True))
    )


def _payload(value: Any) -> dict:
    return (
        value
        if isinstance(value, dict)
        else vars(value)
        if isinstance(value, BaseMessage)
        else {}
    )


def _arguments(value: Any) -> str:
    return (
        data(value, complete=True)
        if isinstance(value, str)
        else json_value(value, complete=True)
    )


def _calls(calls: Any) -> list[dict]:
    result = []
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        function = function if isinstance(function, dict) else call
        name = function.get("name")
        if not isinstance(name, str):
            continue
        item = {
            "type": "function",
            "function": {
                "name": text(name, None),
                "arguments": _arguments(
                    function.get("arguments", function.get("args", {}))
                ),
            },
        }
        if isinstance(call.get("id"), str):
            item["id"] = text(call["id"], None)
        result.append(item)
    return result


def _message(message: Any) -> dict:
    value = _payload(message)
    role = value.get("role") or value.get("type")
    role = {"ai": "assistant", "human": "user", "function": "tool"}.get(role, role)
    result = {"role": role if isinstance(role, str) else "unknown"}
    complete = role == "tool"
    if "content" in value:
        result["content"] = data(value["content"], complete=complete)
    extra = value.get("additional_kwargs") or {}
    calls = _calls(value.get("tool_calls") or extra.get("tool_calls"))
    legacy = extra.get("function_call")
    if not calls and isinstance(legacy, dict):
        calls = _calls([{"function": legacy}])
    if calls:
        result["tool_calls"] = calls
    call_id = value.get("tool_call_id")
    if isinstance(call_id, str):
        result["tool_call_id"] = text(call_id, None)
    return result


def _message_attributes(messages: list[dict], prefix: str) -> dict:
    attrs = {}
    for index, message in enumerate(messages):
        base = f"{prefix}.{index}"
        attrs[f"{base}.role"] = message["role"]
        complete = message.get("role") == "tool"
        if "content" in message:
            content = message["content"]
            attrs[f"{base}.content"] = (
                text(content, None if complete else MAX_CHARS)
                if isinstance(content, str)
                else json_value(content, complete=complete)
            )
        if message.get("tool_calls"):
            attrs[f"{base}.tool_calls"] = json_value(
                message["tool_calls"], complete=True
            )
        if message.get("tool_call_id"):
            attrs[f"{base}.tool_call_id"] = message["tool_call_id"]
    return attrs


def _model(serialized, metadata, response=None):
    sources = [
        metadata or {},
        (serialized or {}).get("kwargs", {}) if isinstance(serialized, dict) else {},
        serialized or {},
    ]
    if response is not None and isinstance(getattr(response, "llm_output", None), dict):
        sources.append(response.llm_output)
    for source in sources:
        if isinstance(source, dict):
            for key in ("ls_model_name", "model_name", "model", "repo_id"):
                if isinstance(source.get(key), str) and source[key]:
                    return text(source[key], 512)
    return None


def _name(serialized, kwargs, fallback):
    if isinstance(kwargs.get("name"), str):
        return text(kwargs["name"], 512)
    if isinstance(serialized, dict):
        value = serialized.get("name") or serialized.get("id")
        if isinstance(value, list) and value:
            value = value[-1]
        if isinstance(value, str):
            return text(value, 512)
    return fallback


def _usage(response) -> dict:
    sources = []
    output = getattr(response, "llm_output", None)
    if isinstance(output, dict):
        sources.extend(
            output.get(key) for key in ("token_usage", "usage", "usage_metadata")
        )
    for batch in getattr(response, "generations", []) or []:
        for generation in batch if isinstance(batch, list) else [batch]:
            message = getattr(generation, "message", None)
            if isinstance(message, BaseMessage):
                values = vars(message)
                sources.append(values.get("usage_metadata"))
                response_metadata = values.get("response_metadata") or {}
                sources.extend(
                    response_metadata.get(key)
                    for key in ("token_usage", "usage", "usage_metadata")
                )
    attrs = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for fields, keys in (
            (
                ("input_tokens", "prompt_tokens"),
                (GEN_AI_USAGE_INPUT_TOKENS, SpanAttributes.LLM_USAGE_PROMPT_TOKENS),
            ),
            (
                ("output_tokens", "completion_tokens"),
                (
                    GEN_AI_USAGE_OUTPUT_TOKENS,
                    SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                ),
            ),
            (("total_tokens",), (SpanAttributes.LLM_USAGE_TOTAL_TOKENS,)),
        ):
            value = next((source[key] for key in fields if key in source), None)
            if type(value) is int and value >= 0:
                for key in keys:
                    attrs.setdefault(key, value)
        for details, usage_field, keys in (
            (
                source.get("input_token_details")
                or source.get("input_tokens_details")
                or source.get("prompt_tokens_details"),
                "cache_read",
                (
                    SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                    SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
                ),
            ),
            (
                source.get("input_token_details")
                or source.get("input_tokens_details")
                or source.get("prompt_tokens_details"),
                "cache_creation",
                (
                    SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                    SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
                ),
            ),
            (
                source.get("output_token_details")
                or source.get("output_tokens_details")
                or source.get("completion_tokens_details"),
                "reasoning",
                (
                    SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
                    SpanAttributes.LLM_USAGE_REASONING_TOKENS,
                ),
            ),
        ):
            if isinstance(details, dict):
                value = details.get(
                    usage_field,
                    details.get(
                        "cached_tokens"
                        if usage_field == "cache_read"
                        else "reasoning_tokens"
                        if usage_field == "reasoning"
                        else "cache_write_tokens"
                    ),
                )
                if type(value) is int and value >= 0:
                    for key in keys:
                        attrs.setdefault(key, value)
    return attrs


def _completions(response) -> list[dict]:
    result = []
    for batch in getattr(response, "generations", []) or []:
        for generation in batch if isinstance(batch, list) else [batch]:
            message = getattr(generation, "message", None)
            if isinstance(message, BaseMessage):
                result.append(_message(message))
            elif isinstance(getattr(generation, "text", None), str):
                result.append({"role": "assistant", "content": text(generation.text)})
    return result


@dataclass
class _Run:
    span: Any
    name: str
    log_type: str
    capture: bool
    parent_key: str = ""
    metadata: dict = field(default_factory=dict)
    input: str | None = None
    extra: dict = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)


class RespanCallbackHandler(BaseCallbackHandler):
    """Real OTel spans driven by LangChain's public run callbacks."""

    raise_error = False
    run_inline = True

    def __init__(
        self,
        *,
        include_content=True,
        include_metadata=True,
        group_langflow_root_runs=False,
        max_cached_runs=4096,
    ):
        super().__init__()
        self.include_content = include_content
        self.include_metadata = include_metadata
        self.group_langflow_root_runs = group_langflow_root_runs
        self.max_cached_runs = max_cached_runs
        self._runs: dict[str, _Run | None] = {}
        self._parents: OrderedDict[str, Any] = OrderedDict()
        self._lock = RLock()
        self._enabled = True
        self._langflow_group = uuid4().hex

    def _start(
        self,
        *,
        run_id,
        parent_run_id,
        name,
        log_type,
        input_value=None,
        serialized=None,
        tags=None,
        metadata=None,
        extra=None,
    ):
        key = _key(run_id)
        parent_key = _key(parent_run_id)
        with self._lock:
            if not self._enabled or not key or key in self._runs:
                return None
            if _suppressed() or (
                parent_key in self._runs and self._runs[parent_key] is None
            ):
                self._runs[key] = None
                return None
            parent = self._record(parent_run_id) if parent_key else None
            cached = self._parents.get(parent_key)
            parent_span = (
                parent.span
                if parent
                else trace.NonRecordingSpan(cached[0])
                if cached
                else None
            )
            parent_context = (
                trace.set_span_in_context(parent_span)
                if parent_span
                else context.get_current()
            )
            attrs = {
                RESPAN_LOG_TYPE: log_type,
                SpanAttributes.TRACELOOP_ENTITY_NAME: name,
                SpanAttributes.TRACELOOP_ENTITY_PATH: "" if not parent_span else name,
                LANGCHAIN_RUN_ID_ATTR: key,
            }
            if parent_key:
                attrs[LANGCHAIN_PARENT_RUN_ID_ATTR] = parent_key
            framework = (
                "langgraph"
                if any(
                    isinstance(k, str) and k.startswith("langgraph_")
                    for k in (metadata or {})
                )
                else "langflow"
                if (metadata or {}).get("framework") == "langflow"
                else "langchain"
            )
            attrs[LANGCHAIN_FRAMEWORK_ATTR] = framework
            span = trace.get_tracer(__name__).start_span(
                name, context=parent_context, attributes=attrs
            )
            run = _Run(
                span,
                name,
                log_type,
                span.is_recording()
                and _content_enabled(self)
                and (parent.capture if parent else cached[1] if cached else True),
            )
            run.parent_key = parent_key
            self._runs[key] = run
            if not span.is_recording():
                return run
            params = (metadata or {}).get("respan_params") or {}
            if isinstance(params, dict):
                for field_name, value in params.items():
                    attr = RESPAN_SPAN_ATTRIBUTES_MAP.get(field_name)
                    if attr == RESPAN_METADATA and isinstance(value, dict):
                        run.metadata = {
                            k: data(v)
                            for k, v in value.items()
                            if run.capture or k in _MARKER_FIELDS
                        }
                    elif (
                        isinstance(attr, str)
                        and attr.startswith("respan.")
                        and attr != RESPAN_LOG_TYPE
                        and isinstance(value, (str, int, float, bool))
                    ):
                        span.set_attribute(
                            attr, text(value, None) if isinstance(value, str) else value
                        )
            if (
                framework == "langflow"
                and self.group_langflow_root_runs
                and not parent_key
            ):
                span.set_attribute(
                    RESPAN_SPAN_ATTRIBUTES_MAP["trace_group_identifier"],
                    self._langflow_group,
                )
            if framework == "langgraph" and isinstance(
                (metadata or {}).get("thread_id"), str
            ):
                span.set_attribute(
                    RESPAN_SPAN_ATTRIBUTES_MAP["thread_identifier"],
                    text(metadata["thread_id"], None),
                )
            if run.capture:
                run.input = json_value(input_value, complete=log_type == LOG_TYPE_TOOL)
                if self.include_metadata:
                    run.extra.update(
                        {
                            LANGCHAIN_METADATA_ATTR: json_value(metadata),
                            LANGCHAIN_SERIALIZED_ATTR: json_value(serialized),
                            LANGCHAIN_TAGS_ATTR: json_value(tags),
                        }
                    )
                if extra:
                    run.extra.update(extra)
            return run

    def _record(self, run_id):
        run = self._runs.get(_key(run_id))
        if not _content_enabled(self):
            key = _key(run_id)
            visited = set()
            while key and key not in visited:
                visited.add(key)
                active = self._runs.get(key)
                cached = self._parents.get(key)
                if cached:
                    self._parents[key] = (cached[0], False, cached[2])
                if active is None:
                    key = cached[2] if cached else ""
                    continue
                active.capture = False
                active.input = None
                active.extra.clear()
                active.tokens.clear()
                active.metadata = {
                    k: v for k, v in active.metadata.items() if k in _MARKER_FIELDS
                }
                key = active.parent_key
        return run

    def _end(self, run_id, *, output=None, error=None, extra=None):
        key = _key(run_id)
        with self._lock:
            run = self._record(run_id)
            if key not in self._runs:
                return
            self._runs.pop(key, None)
            if run is None:
                return
            self._parents[key] = (
                run.span.get_span_context(),
                run.capture,
                run.parent_key,
            )
            self._parents.move_to_end(key)
            while len(self._parents) > self.max_cached_runs:
                self._parents.popitem(last=False)
        try:
            if not run.span.is_recording():
                return
            if extra:
                for name, value in extra.items():
                    run.span.set_attribute(name, value)
            if error is not None:
                run.span.set_status(Status(StatusCode.ERROR))
                event = {EXCEPTION_TYPE: type(error).__name__}
                if run.capture:
                    args = BaseException.args.__get__(error)
                    message = "; ".join(
                        text(value) for value in args if isinstance(value, str)
                    )
                    if message:
                        event[EXCEPTION_MESSAGE] = message
                        run.span.set_status(Status(StatusCode.ERROR, message))
                run.span.add_event("exception", event)
            if run.capture:
                if run.input is not None:
                    run.span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_INPUT, run.input
                    )
                for name, value in run.extra.items():
                    run.span.set_attribute(name, value)
                if error is None and output is not None:
                    run.span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        json_value(output, complete=run.log_type == LOG_TYPE_TOOL),
                    )
            if run.metadata:
                run.span.set_attribute(RESPAN_METADATA, json_value(run.metadata))
                for name, value in run.metadata.items():
                    run.span.set_attribute(
                        f"{RESPAN_METADATA}.{name}",
                        value
                        if isinstance(value, (str, int, float, bool))
                        else json_value(value),
                    )
        finally:
            run.span.end()

    def _event(self, name, payload, run_id, metadata=None):
        parent = self._record(run_id)
        if not self._enabled or _suppressed() or parent is None:
            return
        with trace.get_tracer(__name__).start_as_current_span(
            name,
            context=trace.set_span_in_context(parent.span),
            attributes={
                RESPAN_LOG_TYPE: LOG_TYPE_TASK,
                SpanAttributes.TRACELOOP_ENTITY_NAME: name,
                SpanAttributes.TRACELOOP_ENTITY_PATH: name,
            },
        ) as span:
            if parent.capture and span.is_recording() and _content_enabled(self):
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT, json_value(payload)
                )

    @_safe_callback
    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        name = _name(serialized, kwargs, "chain")
        agent = (
            isinstance(serialized, dict)
            and (serialized.get("id") or [None])[-1] == "AgentExecutor"
        )
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            log_type=LOG_TYPE_AGENT
            if agent
            else LOG_TYPE_WORKFLOW
            if parent_run_id is None
            else LOG_TYPE_TASK,
            input_value=inputs,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )

    @_safe_callback
    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id, output=outputs)

    @_safe_callback
    def on_chain_error(self, error, *, run_id, **kwargs):
        try:
            from langgraph.errors import GraphInterrupt
        except ImportError:
            GraphInterrupt = None
        if GraphInterrupt is not None and isinstance(error, GraphInterrupt):
            self._event(
                "langgraph.interrupt", BaseException.args.__get__(error), run_id
            )
            self._end(run_id)
        else:
            self._end(run_id, error=error)

    @_safe_callback
    def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        model = _model(serialized, metadata)
        run = self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_name(serialized, kwargs, "chat_model"),
            log_type=LOG_TYPE_CHAT,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )
        if run is None or not run.span.is_recording():
            return
        run.span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "chat")
        if model:
            run.span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
        provider = (metadata or {}).get("ls_provider")
        if isinstance(provider, str):
            run.span.set_attribute(
                SpanAttributes.LLM_SYSTEM, text(provider, 64).lower()
            )
            run.span.set_attribute(GEN_AI_PROVIDER_NAME, text(provider, 64).lower())
        invocation = kwargs.get("invocation_params") or {}
        for field_name, key in (
            ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
            ("max_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
            ("stream", SpanAttributes.LLM_IS_STREAMING),
        ):
            value = invocation.get(field_name)
            if isinstance(value, (bool, int, float)):
                run.span.set_attribute(key, value)
        if run.capture:
            normalized = [
                [_message(m) for m in conversation] for conversation in messages
            ]
            run.input = json_value(normalized)
            run.extra.update(
                _message_attributes(
                    normalized[0] if normalized else [], SpanAttributes.LLM_PROMPTS
                )
            )
            tools = invocation.get("tools") or invocation.get("functions")
            if isinstance(tools, list):
                run.extra[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_value(
                    tools, complete=True
                )

    @_safe_callback
    def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        run = self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_name(serialized, kwargs, "llm"),
            log_type=LOG_TYPE_TEXT,
            input_value=prompts,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )
        if run is None or not run.span.is_recording():
            return
        run.span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "chat")
        model = _model(serialized, metadata)
        if model:
            run.span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
        if run.capture:
            run.extra.update(
                _message_attributes(
                    [{"role": "user", "content": data(p)} for p in prompts],
                    SpanAttributes.LLM_PROMPTS,
                )
            )

    @_safe_callback
    def on_llm_new_token(self, token, *, run_id, **kwargs):
        run = self._record(run_id)
        if run is not None and run.span.is_recording():
            run.span.set_attribute(SpanAttributes.LLM_IS_STREAMING, True)

    @_safe_callback
    def on_llm_end(self, response, *, run_id, **kwargs):
        run = self._record(run_id)
        if run is None:
            self._end(run_id)
            return
        extra = _usage(response) if run.span.is_recording() else {}
        if run.capture:
            messages = _completions(response)
            run.extra.update(
                _message_attributes(messages, SpanAttributes.LLM_COMPLETIONS)
            )
            output = {"messages": messages}
        else:
            output = None
        self._end(run_id, output=output, extra=extra)

    @_safe_callback
    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=error)

    @_safe_callback
    def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        inputs=None,
        **kwargs,
    ):
        name = _name(serialized, {}, "tool")
        value = inputs if inputs is not None else input_str
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                pass
        run = self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            log_type=LOG_TYPE_TOOL,
            input_value={"name": name, "arguments": value},
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )
        if run is not None and isinstance(kwargs.get("tool_call_id"), str):
            run.span.set_attribute(
                GEN_AI_TOOL_CALL_ID, text(kwargs["tool_call_id"], None)
            )

    @_safe_callback
    def on_tool_end(self, output, *, run_id, **kwargs):
        run = self._record(run_id)
        if run is not None and isinstance(output, ToolMessage):
            values = vars(output)
            call_id = values.get("tool_call_id")
            if isinstance(call_id, str):
                run.span.set_attribute(GEN_AI_TOOL_CALL_ID, text(call_id, None))
            output = (
                {"content": values.get("content"), "artifact": values.get("artifact")}
                if values.get("artifact") is not None
                else values.get("content")
            )
        self._end(run_id, output=output)

    @_safe_callback
    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=error)

    @_safe_callback
    def on_retriever_start(
        self,
        serialized,
        query,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_name(serialized, kwargs, "retriever"),
            log_type=LOG_TYPE_TASK,
            input_value=query,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )

    @_safe_callback
    def on_retriever_end(self, documents, *, run_id, **kwargs):
        self._end(run_id, output=list(documents))

    @_safe_callback
    def on_retriever_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=error)

    @_safe_callback
    def on_agent_action(self, action, *, run_id, **kwargs):
        self._event("agent_action", action, run_id)

    @_safe_callback
    def on_agent_finish(self, finish, *, run_id, **kwargs):
        self._event("agent_finish", finish, run_id)

    @_safe_callback
    def on_text(self, value, *, run_id, **kwargs):
        # on_text is diagnostic output, not a synthetic assistant completion.
        run = self._record(run_id)
        if run is not None and run.capture:
            self._event("text", value, run_id)

    @_safe_callback
    def on_retry(self, retry_state, *, run_id, **kwargs):
        run = self._record(run_id)
        if run is not None and run.span.is_recording():
            key = "langchain.retry_count"
            run.span.set_attribute(
                key, getattr(run.span, "attributes", {}).get(key, 0) + 1
            )

    @_safe_callback
    def on_custom_event(self, name, data, *, run_id, **kwargs):
        self._event(text(name, 512), data, run_id)

    @_safe_callback
    def on_interrupt(self, event):
        self._event("langgraph.interrupt", event, getattr(event, "run_id", None))

    @_safe_callback
    def on_resume(self, event):
        self._event("langgraph.resume", event, getattr(event, "run_id", None))

    def shutdown(self):
        with self._lock:
            self._enabled = False
            runs, self._runs = self._runs, {}
            self._parents.clear()
        for run in runs.values():
            if run is not None:
                run.span.end()


def _with_respan_callback(callbacks, handler, *, replace=False):
    if isinstance(callbacks, BaseCallbackManager):
        manager = callbacks.copy()
        existing = [x for x in manager.handlers if isinstance(x, RespanCallbackHandler)]
        if existing:
            if replace and handler not in existing:
                for previous in existing:
                    manager.remove_handler(previous)
                manager.add_handler(handler, inherit=True)
        else:
            manager.add_handler(handler, inherit=True)
        return manager
    values = (
        list(callbacks)
        if isinstance(callbacks, (list, tuple))
        else [callbacks]
        if callbacks is not None
        else []
    )
    existing = [x for x in values if isinstance(x, RespanCallbackHandler)]
    if existing:
        if replace and handler not in existing:
            return [x for x in values if not isinstance(x, RespanCallbackHandler)] + [
                handler
            ]
        return values
    return values + [handler]


def get_callback_handler(**kwargs):
    kwargs.setdefault("group_langflow_root_runs", True)
    return RespanCallbackHandler(**kwargs)


def add_respan_callback(
    config: Mapping[str, Any] | None = None,
    handler: RespanCallbackHandler | None = None,
) -> dict[str, Any]:
    result = dict(config or {})
    result["callbacks"] = _with_respan_callback(
        result.get("callbacks"),
        handler or get_callback_handler(),
        replace=handler is not None,
    )
    return result
