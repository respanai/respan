"""Native LlamaIndex handlers that emit Respan-compatible OTEL spans."""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any

from llama_index_instrumentation.dispatcher import active_instrument_tags
from llama_index_instrumentation.event_handlers import BaseEventHandler
from llama_index_instrumentation.span import BaseSpan
from llama_index_instrumentation.span_handlers import BaseSpanHandler
from opentelemetry import context
from opentelemetry import trace
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import Status, StatusCode
from pydantic import ConfigDict, PrivateAttr
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_COMPLETION,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
)
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_llama_index._constants import (
    CHAT_EVENT_KEY,
    COMPLETION_EVENT_KEY,
    EMBEDDING_EVENT_KEY,
    LLAMA_INDEX_CHAT_SPAN_NAME,
    LLAMA_INDEX_COMPLETION_SPAN_NAME,
    LLAMA_INDEX_DEFAULT_TOOL_NAME,
    LLAMA_INDEX_EMBEDDING_SPAN_NAME,
    LLAMA_INDEX_USAGE_INPUT_TOKENS,
    LLAMA_INDEX_USAGE_OUTPUT_TOKENS,
    LLAMA_INDEX_RUN_ID_TAG,
    LLAMA_INDEX_START_EVENT_TAG,
    LLAMA_INDEX_STEP_INPUT_EVENT_TAG,
    LLAMA_INDEX_STEP_INPUT_SUMMARY_TAG,
    LLAMA_INDEX_TOOL_SPAN_PREFIX,
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_USER,
    STATUS_CODE_ATTR,
)
from respan_instrumentation_llama_index._serialization import (
    chat_messages_to_dicts,
    chat_response_to_message_dict,
    completion_response_to_text,
    extract_usage,
    get_model_name,
    get_model_system,
    safe_json,
    to_jsonable,
)

logger = logging.getLogger(__name__)

_UUID_SUFFIX_RE = re.compile(
    r"-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class RespanLlamaIndexSpan(BaseSpan):
    """Bookkeeping object for an active LlamaIndex span."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    otel_span: Any
    context_token: Any
    entity_name: str
    log_type: str


class ActiveEventSpan:
    """Bookkeeping object for an active event-derived OTEL span."""

    def __init__(self, otel_span: Any, context_token: Any) -> None:
        self.otel_span = otel_span
        self.context_token = context_token


class RespanLlamaIndexSpanHandler(BaseSpanHandler[RespanLlamaIndexSpan]):
    """LlamaIndex span handler that creates workflow/task/tool OTEL spans."""

    capture_content: bool = True

    _span_contexts: dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, *, capture_content: bool = True) -> None:
        super().__init__()
        self.capture_content = capture_content

    @classmethod
    def class_name(cls) -> str:
        return "RespanLlamaIndexSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RespanLlamaIndexSpan:
        tags = tags or active_instrument_tags.get()
        entity_name = _span_entity_name(span_id=id_)
        log_type = _span_log_type(
            entity_name=entity_name,
            instance=instance,
            parent_span_id=parent_span_id,
        )
        attributes = _base_attributes(
            entity_name=entity_name,
            log_type=log_type,
            entity_path=entity_name,
        )
        if self.capture_content:
            attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
                _span_input_payload(bound_args=bound_args, tags=tags)
            )

        otel_span, context_token, span_context = _start_otel_span(
            span_name=entity_name,
            attributes=attributes,
            parent_context=self._parent_context(parent_span_id=parent_span_id),
        )
        self._span_contexts[id_] = span_context
        return RespanLlamaIndexSpan(
            id_=id_,
            parent_id=parent_span_id,
            tags=tags or {},
            otel_span=otel_span,
            context_token=context_token,
            entity_name=entity_name,
            log_type=log_type,
        )

    def _parent_context(self, *, parent_span_id: str | None) -> Any | None:
        if parent_span_id is None:
            return None

        active_parent = self.open_spans.get(parent_span_id)
        if active_parent is not None:
            parent_context = self._span_contexts.get(parent_span_id)
            if parent_context is None:
                parent_context = trace.set_span_in_context(active_parent.otel_span)
                self._span_contexts[parent_span_id] = parent_context
            return parent_context

        parent_context = self._span_contexts.get(parent_span_id)
        if parent_context is not None:
            return parent_context

        return self._create_synthetic_parent_context(parent_span_id=parent_span_id)

    def _create_synthetic_parent_context(self, *, parent_span_id: str) -> Any:
        entity_name = _span_entity_name(span_id=parent_span_id)
        attributes = _base_attributes(
            entity_name=entity_name,
            log_type=_span_log_type(
                entity_name=entity_name,
                instance=None,
                parent_span_id=None,
            ),
            entity_path=entity_name,
        )
        parent_context = context.get_current()
        otel_span = _start_detached_otel_span(
            span_name=entity_name,
            attributes=attributes,
            parent_context=parent_context,
        )
        span_context = trace.set_span_in_context(otel_span, parent_context)
        self._span_contexts[parent_span_id] = span_context
        otel_span.end()
        return span_context

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> RespanLlamaIndexSpan | None:
        active_span = self.open_spans.get(id_)
        if active_span is None:
            return None
        if self.capture_content:
            active_span.otel_span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                safe_json(
                    _span_output_payload(
                        entity_name=active_span.entity_name,
                        result=result,
                    )
                ),
            )
        _end_otel_span(active_span=active_span)
        with self.lock:
            self.completed_spans += [active_span]
        return active_span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        err: BaseException | None = None,
        **kwargs: Any,
    ) -> RespanLlamaIndexSpan | None:
        active_span = self.open_spans.get(id_)
        if active_span is None:
            return None
        if err is not None:
            error_message = str(err) if self.capture_content else type(err).__name__
            if self.capture_content and isinstance(err, Exception):
                active_span.otel_span.record_exception(err)
            active_span.otel_span.set_status(
                Status(status_code=StatusCode.ERROR, description=error_message)
            )
            active_span.otel_span.set_attribute(ERROR_MESSAGE_ATTR, error_message)
            active_span.otel_span.set_attribute(STATUS_CODE_ATTR, 500)
            error_output = {"status": "error", "error": type(err).__name__}
            if self.capture_content:
                error_output["message"] = error_message
            active_span.otel_span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                safe_json(error_output),
            )
        _end_otel_span(active_span=active_span)
        with self.lock:
            self.dropped_spans += [active_span]
        return active_span


class RespanLlamaIndexEventHandler(BaseEventHandler):
    """LlamaIndex event handler for LLM, embedding, and tool spans."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    capture_content: bool = True
    _open_event_spans: dict[tuple[str | None, str], list[ActiveEventSpan]] = (
        PrivateAttr(default_factory=dict)
    )

    def __init__(self, *, capture_content: bool = True) -> None:
        super().__init__(capture_content=capture_content)

    @classmethod
    def class_name(cls) -> str:
        return "RespanLlamaIndexEventHandler"

    def handle(self, event: Any, **kwargs: Any) -> Any:
        event_name = event.class_name()
        if event_name == "LLMChatStartEvent":
            self._handle_chat_start(event=event)
        elif event_name == "LLMChatEndEvent":
            self._handle_chat_end(event=event)
        elif event_name == "LLMCompletionStartEvent":
            self._handle_completion_start(event=event)
        elif event_name == "LLMCompletionEndEvent":
            self._handle_completion_end(event=event)
        elif event_name == "EmbeddingStartEvent":
            self._handle_embedding_start(event=event)
        elif event_name == "EmbeddingEndEvent":
            self._handle_embedding_end(event=event)
        elif event_name == "AgentToolCallEvent":
            self._handle_tool_call(event=event)
        elif event_name == "ExceptionEvent":
            self._handle_exception(event=event)

    def _handle_chat_start(self, *, event: Any) -> None:
        messages = chat_messages_to_dicts(getattr(event, "messages", []))
        model_dict = getattr(event, "model_dict", None)
        attributes = _llm_base_attributes(
            entity_name=LLAMA_INDEX_CHAT_SPAN_NAME,
            log_type=LOG_TYPE_CHAT,
            request_type=LLMRequestTypeValues.CHAT.value,
            model_dict=model_dict,
        )
        if self.capture_content:
            for message_index, message in enumerate(messages):
                attributes[f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role"] = (
                    message.get("role", "")
                )
                attributes[f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content"] = (
                    _content_attribute(value=message.get("content"))
                )
            attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(messages)
        self._push_event_span(
            span_id=getattr(event, "span_id", None),
            event_key=CHAT_EVENT_KEY,
            span_name=LLAMA_INDEX_CHAT_SPAN_NAME,
            attributes=attributes,
        )

    def _handle_chat_end(self, *, event: Any) -> None:
        active_event_span = self._pop_event_span(
            span_id=getattr(event, "span_id", None),
            event_key=CHAT_EVENT_KEY,
        )
        if active_event_span is None:
            return
        response_message = chat_response_to_message_dict(
            getattr(event, "response", None)
        )
        attributes: dict[str, Any] = {}
        if self.capture_content:
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = (
                response_message.get("role", MESSAGE_ROLE_ASSISTANT)
            )
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = (
                _content_attribute(value=response_message.get("content"))
            )
            attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(
                response_message
            )
        _set_usage_attributes(
            attributes=attributes,
            response=getattr(event, "response", None),
        )
        _finish_event_span(
            active_event_span=active_event_span,
            attributes=attributes,
        )

    def _handle_completion_start(self, *, event: Any) -> None:
        model_dict = getattr(event, "model_dict", None)
        prompt = getattr(event, "prompt", "")
        attributes = _llm_base_attributes(
            entity_name=LLAMA_INDEX_COMPLETION_SPAN_NAME,
            log_type=LOG_TYPE_COMPLETION,
            request_type=LLMRequestTypeValues.COMPLETION.value,
            model_dict=model_dict,
        )
        if self.capture_content:
            attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] = MESSAGE_ROLE_USER
            attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] = str(prompt)
            attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
                [{"role": MESSAGE_ROLE_USER, "content": prompt}]
            )
        self._push_event_span(
            span_id=getattr(event, "span_id", None),
            event_key=COMPLETION_EVENT_KEY,
            span_name=LLAMA_INDEX_COMPLETION_SPAN_NAME,
            attributes=attributes,
        )

    def _handle_completion_end(self, *, event: Any) -> None:
        active_event_span = self._pop_event_span(
            span_id=getattr(event, "span_id", None),
            event_key=COMPLETION_EVENT_KEY,
        )
        if active_event_span is None:
            return
        completion_text = completion_response_to_text(getattr(event, "response", None))
        attributes: dict[str, Any] = {}
        if self.capture_content:
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = (
                MESSAGE_ROLE_ASSISTANT
            )
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = completion_text
            attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(
                {"role": MESSAGE_ROLE_ASSISTANT, "content": completion_text}
            )
        _set_usage_attributes(
            attributes=attributes,
            response=getattr(event, "response", None),
        )
        _finish_event_span(
            active_event_span=active_event_span,
            attributes=attributes,
        )

    def _handle_embedding_start(self, *, event: Any) -> None:
        model_dict = getattr(event, "model_dict", None)
        attributes = _llm_base_attributes(
            entity_name=LLAMA_INDEX_EMBEDDING_SPAN_NAME,
            log_type=LOG_TYPE_EMBEDDING,
            request_type=LLMRequestTypeValues.EMBEDDING.value,
            model_dict=model_dict,
        )
        self._push_event_span(
            span_id=getattr(event, "span_id", None),
            event_key=EMBEDDING_EVENT_KEY,
            span_name=LLAMA_INDEX_EMBEDDING_SPAN_NAME,
            attributes=attributes,
        )

    def _handle_embedding_end(self, *, event: Any) -> None:
        active_event_span = self._pop_event_span(
            span_id=getattr(event, "span_id", None),
            event_key=EMBEDDING_EVENT_KEY,
        )
        if active_event_span is None:
            return
        chunks = getattr(event, "chunks", [])
        embeddings = getattr(event, "embeddings", []) or []
        attributes: dict[str, Any] = {}
        if self.capture_content:
            attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(chunks)
            attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(embeddings)
            if chunks:
                attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] = (
                    _embedding_input(chunks=chunks)
                )
        _finish_event_span(
            active_event_span=active_event_span,
            attributes=attributes,
        )

    def _handle_tool_call(self, *, event: Any) -> None:
        tool = getattr(event, "tool", None)
        tool_name = _tool_name(tool=tool)
        attributes = _base_attributes(
            entity_name=tool_name,
            log_type=LOG_TYPE_TOOL,
            entity_path=tool_name,
        )
        if self.capture_content:
            attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
                {
                    "name": tool_name,
                    "arguments": getattr(event, "arguments", ""),
                }
            )
        active_event_span = _start_event_span(
            span_name=f"{LLAMA_INDEX_TOOL_SPAN_PREFIX}{tool_name}",
            attributes=attributes,
        )
        _finish_event_span(active_event_span=active_event_span, attributes={})

    def _handle_exception(self, *, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        exception = getattr(event, "exception", None)
        if exception is None:
            return
        error_message = (
            str(exception) if self.capture_content else type(exception).__name__
        )
        error_output = {"status": "error", "error": type(exception).__name__}
        if self.capture_content:
            error_output["message"] = error_message
        matching_keys = [key for key in self._open_event_spans if key[0] == span_id]
        for key in matching_keys:
            active_event_spans = self._open_event_spans.pop(key)
            for active_event_span in reversed(active_event_spans):
                if self.capture_content and isinstance(exception, Exception):
                    active_event_span.otel_span.record_exception(exception)
                active_event_span.otel_span.set_status(
                    Status(
                        status_code=StatusCode.ERROR,
                        description=error_message,
                    )
                )
                active_event_span.otel_span.set_attribute(
                    ERROR_MESSAGE_ATTR, error_message
                )
                active_event_span.otel_span.set_attribute(STATUS_CODE_ATTR, 500)
                _finish_event_span(
                    active_event_span=active_event_span,
                    attributes={
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: safe_json(error_output)
                    },
                )

    def _push_event_span(
        self,
        *,
        span_id: str | None,
        event_key: str,
        span_name: str,
        attributes: dict[str, Any],
    ) -> None:
        active_event_span = _start_event_span(
            span_name=span_name,
            attributes=attributes,
        )
        key = (span_id, event_key)
        self._open_event_spans.setdefault(key, []).append(active_event_span)

    def _pop_event_span(
        self,
        *,
        span_id: str | None,
        event_key: str,
    ) -> ActiveEventSpan | None:
        key = (span_id, event_key)
        active_event_spans = self._open_event_spans.get(key)
        if not active_event_spans:
            return None
        active_event_span = active_event_spans.pop()
        if not active_event_spans:
            self._open_event_spans.pop(key, None)
        return active_event_span


def _start_otel_span(
    *,
    span_name: str,
    attributes: dict[str, Any],
    parent_context: Any | None = None,
) -> tuple[Any, Any, Any]:
    tracer = RespanTracer().get_tracer()
    otel_span = tracer.start_span(
        span_name,
        context=parent_context,
        attributes=_clean_attributes(attributes=attributes),
    )
    span_context = trace.set_span_in_context(
        otel_span,
        parent_context or context.get_current(),
    )
    context_token = context.attach(span_context)
    return otel_span, context_token, span_context


def _start_detached_otel_span(
    *,
    span_name: str,
    attributes: dict[str, Any],
    parent_context: Any | None = None,
) -> Any:
    tracer = RespanTracer().get_tracer()
    return tracer.start_span(
        span_name,
        context=parent_context,
        attributes=_clean_attributes(attributes=attributes),
    )


def _start_event_span(
    *,
    span_name: str,
    attributes: dict[str, Any],
) -> ActiveEventSpan:
    otel_span, context_token, _ = _start_otel_span(
        span_name=span_name,
        attributes=attributes,
    )
    return ActiveEventSpan(otel_span=otel_span, context_token=context_token)


def _end_otel_span(*, active_span: RespanLlamaIndexSpan) -> None:
    try:
        active_span.otel_span.end()
    finally:
        context.detach(active_span.context_token)


def _finish_event_span(
    *,
    active_event_span: ActiveEventSpan,
    attributes: dict[str, Any],
) -> None:
    for key, value in _clean_attributes(attributes=attributes).items():
        active_event_span.otel_span.set_attribute(key, value)
    try:
        active_event_span.otel_span.end()
    finally:
        context.detach(active_event_span.context_token)


def _base_attributes(
    *,
    entity_name: str,
    log_type: str,
    entity_path: str,
) -> dict[str, Any]:
    return {
        SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: entity_path,
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: log_type,
    }


def _llm_base_attributes(
    *,
    entity_name: str,
    log_type: str,
    request_type: str,
    model_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    attributes = _base_attributes(
        entity_name=entity_name,
        log_type=log_type,
        entity_path=entity_name,
    )
    attributes[SpanAttributes.LLM_REQUEST_TYPE] = request_type
    model_name = get_model_name(model_dict=model_dict)
    model_system = get_model_system(model_dict=model_dict)
    if model_name:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model_name
    if model_system:
        attributes[GEN_AI_SYSTEM] = model_system
    return attributes


def _set_usage_attributes(*, attributes: dict[str, Any], response: Any) -> None:
    prompt_tokens, completion_tokens, total_tokens = extract_usage(response=response)
    if prompt_tokens is not None:
        attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
        attributes[LLAMA_INDEX_USAGE_INPUT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
        attributes[LLAMA_INDEX_USAGE_OUTPUT_TOKENS] = completion_tokens
    if total_tokens is not None:
        attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens


def _clean_attributes(*, attributes: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        else:
            cleaned[key] = safe_json(value)
    return cleaned


def _content_attribute(*, value: Any) -> str:
    jsonable_value = to_jsonable(value)
    if isinstance(jsonable_value, str):
        return jsonable_value
    return safe_json(jsonable_value)


def _span_output_payload(*, entity_name: str, result: Any) -> Any:
    return result


def _span_input_payload(
    *,
    bound_args: inspect.BoundArguments,
    tags: dict[str, Any] | None,
) -> Any:
    """Prefer public Workflows event summaries over internal runtime state.

    Current llama-index-workflows run spans bind a recursive broker state
    object. Serializing that object can fail before the root span is created.
    The SDK emits stable input summaries in instrumentation tags specifically
    for integrations, so consume those and keep the raw vendor tags off the
    exported span.
    """

    tags = tags or {}
    if LLAMA_INDEX_START_EVENT_TAG in tags:
        payload: dict[str, Any] = {
            "event": tags[LLAMA_INDEX_START_EVENT_TAG],
        }
        if LLAMA_INDEX_RUN_ID_TAG in tags:
            payload["run_id"] = tags[LLAMA_INDEX_RUN_ID_TAG]
        return payload
    if LLAMA_INDEX_STEP_INPUT_SUMMARY_TAG in tags:
        payload = {
            "event": tags[LLAMA_INDEX_STEP_INPUT_SUMMARY_TAG],
        }
        if LLAMA_INDEX_STEP_INPUT_EVENT_TAG in tags:
            payload["event_type"] = tags[LLAMA_INDEX_STEP_INPUT_EVENT_TAG]
        if LLAMA_INDEX_RUN_ID_TAG in tags:
            payload["run_id"] = tags[LLAMA_INDEX_RUN_ID_TAG]
        return payload
    if LLAMA_INDEX_RUN_ID_TAG in tags:
        return {"run_id": tags[LLAMA_INDEX_RUN_ID_TAG]}
    return {
        "args": to_jsonable(bound_args.args),
        "kwargs": to_jsonable(bound_args.kwargs),
    }


def _embedding_input(*, chunks: Any) -> str:
    if isinstance(chunks, list) and len(chunks) == 1:
        return str(chunks[0])
    return safe_json(chunks)


def _span_entity_name(*, span_id: str) -> str:
    entity_name = _UUID_SUFFIX_RE.sub(repl="", string=span_id)
    if ".<locals>." in entity_name:
        entity_name = entity_name.rsplit(".<locals>.", maxsplit=1)[-1]
    return entity_name or span_id


def _span_log_type(
    *,
    entity_name: str,
    instance: Any | None,
    parent_span_id: str | None,
) -> str:
    normalized_name = entity_name.lower()
    instance_name = type(instance).__name__.lower() if instance is not None else ""
    if "tool" in normalized_name or "tool" in instance_name:
        return LOG_TYPE_TOOL
    if "agent" in normalized_name or "agent" in instance_name:
        return LOG_TYPE_AGENT
    if parent_span_id is None:
        return LOG_TYPE_WORKFLOW
    return LOG_TYPE_TASK


def _tool_name(*, tool: Any) -> str:
    for attr_name in ("name", "tool_name"):
        value = getattr(tool, attr_name, None)
        if value:
            return str(value)
    get_name = getattr(tool, "get_name", None)
    if callable(get_name):
        return str(get_name())
    return LLAMA_INDEX_DEFAULT_TOOL_NAME
