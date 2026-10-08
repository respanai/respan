"""Native LlamaIndex handlers that emit Respan-compatible OTEL spans."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections import OrderedDict
from threading import RLock
from typing import Any

from llama_index.core.tools.types import BaseTool
from llama_index_instrumentation.dispatcher import active_instrument_tags
from llama_index_instrumentation.event_handlers import BaseEventHandler
from llama_index_instrumentation.span import BaseSpan
from llama_index_instrumentation.span_handlers import BaseSpanHandler
from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_TOOL_CALL_ID,
)
from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_MESSAGE,
    EXCEPTION_TYPE,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import Status, StatusCode
from pydantic import ConfigDict, PrivateAttr
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TASK,
    LOG_TYPE_TEXT,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
)

from respan_instrumentation_llama_index._constants import (
    CHAT_EVENT_KEY,
    COMPLETION_EVENT_KEY,
    EMBEDDING_EVENT_KEY,
    LLAMA_INDEX_CHAT_SPAN_NAME,
    LLAMA_INDEX_COMPLETION_SPAN_NAME,
    LLAMA_INDEX_DEFAULT_TOOL_NAME,
    LLAMA_INDEX_EMBEDDING_SPAN_NAME,
    LLAMA_INDEX_RUN_ID_TAG,
    LLAMA_INDEX_START_EVENT_TAG,
    LLAMA_INDEX_STEP_INPUT_EVENT_TAG,
    LLAMA_INDEX_STEP_INPUT_SUMMARY_TAG,
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_USER,
)
from respan_instrumentation_llama_index._policy import (
    clear_content,
    content_allowed,
    suppressed,
)
from respan_instrumentation_llama_index._serialization import (
    chat_messages_to_dicts,
    chat_response_to_message_dict,
    completion_response_to_text,
    get_model_name,
    get_model_system,
    safe_json,
    safe_text,
    to_jsonable,
    usage_attributes,
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
    capture: bool = True
    deferred: bool = False
    tool_call_id: str | None = None


class ActiveEventSpan:
    """Bookkeeping object for an active event-derived OTEL span."""

    def __init__(
        self, otel_span: Any, context_token: Any, capture: bool = True
    ) -> None:
        self.otel_span = otel_span
        self.context_token = context_token
        self.capture = capture
        self.last_response = None


class RespanLlamaIndexSpanHandler(BaseSpanHandler[RespanLlamaIndexSpan]):
    """LlamaIndex span handler that creates workflow/task/tool OTEL spans."""

    capture_content: bool = True

    _event_handler: Any = PrivateAttr(default=None)
    _span_contexts: dict[str, Any] = PrivateAttr(default_factory=OrderedDict)
    _suppressed_ids: set[str] = PrivateAttr(default_factory=set)

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
    ) -> RespanLlamaIndexSpan | None:
        if suppressed() or parent_span_id in self._suppressed_ids:
            self._suppressed_ids.add(id_)
            return None
        if not content_allowed(self.capture_content):
            self.veto(parent_span_id)
        tags = tags or active_instrument_tags.get()
        source_entity_name = _span_entity_name(span_id=id_)
        is_tool_execution = _is_executable_tool_span(
            entity_name=source_entity_name,
            instance=instance,
        )
        entity_name = (
            _tool_name(tool=instance) if is_tool_execution else source_entity_name
        )
        log_type = _span_log_type(
            entity_name=source_entity_name,
            instance=instance,
            parent_span_id=parent_span_id,
        )
        attributes = _base_attributes(
            entity_name=entity_name,
            log_type=log_type,
            entity_path=entity_name if parent_span_id is not None else "",
        )
        otel_span, context_token, span_context = _start_otel_span(
            span_name=entity_name,
            attributes=attributes,
            parent_context=self._parent_context(parent_span_id=parent_span_id),
        )
        parent = self.open_spans.get(parent_span_id)
        capture = (
            otel_span.is_recording()
            and content_allowed(self.capture_content)
            and self.parent_capture(parent_span_id)
        )
        if capture:
            otel_span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                safe_json(
                    _span_input_payload(
                        bound_args=bound_args,
                        tags=tags,
                        tool_name=entity_name if is_tool_execution else None,
                    ),
                    complete=is_tool_execution,
                ),
            )
        tool_id = _native_tool_call_id(bound_args) or (
            parent.tool_call_id if parent else None
        )
        if is_tool_execution and tool_id:
            otel_span.set_attribute(GEN_AI_TOOL_CALL_ID, tool_id)
        self._span_contexts[id_] = (span_context, capture)
        while len(self._span_contexts) > 2048:
            self._span_contexts.pop(next(iter(self._span_contexts)))
        return RespanLlamaIndexSpan(
            id_=id_,
            parent_id=parent_span_id,
            tags={},
            capture=capture,
            tool_call_id=tool_id,
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
            cached = self._span_contexts.get(parent_span_id)
            if cached is None:
                parent_context = trace.set_span_in_context(active_parent.otel_span)
                self._span_contexts[parent_span_id] = (
                    parent_context,
                    active_parent.capture,
                )
                return parent_context
            return cached[0]

        parent_context = self._span_contexts.get(parent_span_id)
        if parent_context is not None:
            return parent_context[0]

        return None

    def parent_context(self, span_id: str | None) -> Any | None:
        return self._parent_context(parent_span_id=span_id) if span_id else None

    def parent_capture(self, span_id: str | None) -> bool:
        parent = self.open_spans.get(span_id)
        if parent is not None:
            return parent.capture
        cached = self._span_contexts.get(span_id)
        return cached[1] if cached is not None else True

    def veto(self, span_id: str | None) -> None:
        visited = set()
        while span_id and span_id not in visited:
            visited.add(span_id)
            active = self.open_spans.get(span_id)
            cached = self._span_contexts.get(span_id)
            if cached is not None:
                self._span_contexts[span_id] = (cached[0], False)
            if active is None:
                break
            active.capture = False
            clear_content(active.otel_span)
            span_id = active.parent_id

    def finish_deferred(
        self,
        span_id: str | None,
        response: Any = None,
        error: BaseException | None = None,
    ) -> None:
        if not content_allowed(self.capture_content):
            self.veto(span_id)
        active = self.open_spans.get(span_id)
        if active is None or not active.deferred:
            return
        if error is not None:
            _record_error(
                active.otel_span,
                error,
                active.capture and content_allowed(self.capture_content),
            )
        elif (
            response is not None
            and active.capture
            and content_allowed(self.capture_content)
        ):
            active.otel_span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, safe_json(response)
            )
        if not active.capture or not content_allowed(self.capture_content):
            clear_content(active.otel_span)
        active.otel_span.end()
        self.open_spans.pop(span_id, None)

    def close(self) -> None:
        for active in list(self.open_spans.values()):
            if not active.capture or not content_allowed(self.capture_content):
                clear_content(active.otel_span)
            active.otel_span.end()
        self.open_spans.clear()
        self.completed_spans.clear()
        self.dropped_spans.clear()
        self._span_contexts.clear()
        self._suppressed_ids.clear()

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> RespanLlamaIndexSpan | None:
        if not content_allowed(self.capture_content):
            self.veto(id_)
        active_span = self.open_spans.get(id_)
        if active_span is None:
            self._suppressed_ids.discard(id_)
            return None
        task = (
            getattr(result, "_result_task", None)
            if type(result).__module__.startswith("workflows.")
            else None
        )
        if isinstance(task, asyncio.Future):
            active_span.deferred = True

            def completed(done: Any) -> None:
                try:
                    response = done.result()
                except BaseException as error:  # noqa: BLE001 - observe original task outcome
                    self.finish_deferred(id_, error=error)
                else:
                    self.finish_deferred(id_, response=response)

            task.add_done_callback(completed)
            return None
        if inspect.isgenerator(result) or inspect.isasyncgen(result):
            active_span.deferred = True
            return None
        if active_span.capture and content_allowed(self.capture_content):
            active_span.otel_span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                safe_json(
                    _span_output_payload(
                        log_type=active_span.log_type,
                        result=result,
                    ),
                    complete=active_span.log_type == LOG_TYPE_TOOL,
                ),
            )
        if not active_span.capture or not content_allowed(self.capture_content):
            clear_content(active_span.otel_span)
        _end_otel_span(active_span=active_span)
        with self.lock:
            self.completed_spans = (self.completed_spans + [active_span])[-128:]
        return active_span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        err: BaseException | None = None,
        **kwargs: Any,
    ) -> RespanLlamaIndexSpan | None:
        if not content_allowed(self.capture_content):
            self.veto(id_)
        active_span = self.open_spans.get(id_)
        if active_span is None:
            return None
        if err is not None:
            if self._event_handler is not None:
                from types import SimpleNamespace

                self._event_handler._handle_exception(
                    event=SimpleNamespace(span_id=id_, exception=err)
                )
            _record_error(
                active_span.otel_span,
                err,
                active_span.capture and content_allowed(self.capture_content),
            )
        if not active_span.capture or not content_allowed(self.capture_content):
            clear_content(active_span.otel_span)
        _end_otel_span(active_span=active_span)
        with self.lock:
            self.dropped_spans = (self.dropped_spans + [active_span])[-128:]
        return active_span


class RespanLlamaIndexEventHandler(BaseEventHandler):
    """LlamaIndex event handler for LLM, embedding, and tool spans."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    capture_content: bool = True
    _span_handler: Any = PrivateAttr(default=None)
    _event_lock: Any = PrivateAttr(default_factory=RLock)
    _open_event_spans: dict[tuple[str | None, str], list[ActiveEventSpan]] = (
        PrivateAttr(default_factory=dict)
    )

    def __init__(
        self, *, capture_content: bool = True, span_handler: Any = None
    ) -> None:
        super().__init__(capture_content=capture_content)
        self._span_handler = span_handler

    @classmethod
    def class_name(cls) -> str:
        return "RespanLlamaIndexEventHandler"

    def handle(self, event: Any, **kwargs: Any) -> Any:
        try:
            with self._event_lock:
                self._handle(event, **kwargs)
        except Exception:  # noqa: BLE001 - observations cannot change SDK behavior
            logger.debug("Could not translate LlamaIndex event")

    def _handle(self, event: Any, **kwargs: Any) -> Any:
        if not content_allowed(self.capture_content):
            source_id = getattr(event, "span_id", None)
            if self._span_handler is not None:
                self._span_handler.veto(source_id)
            for (span_id, _), stack in self._open_event_spans.items():
                if span_id == source_id or (
                    self._span_handler is not None
                    and not self._span_handler.parent_capture(span_id)
                ):
                    for active in stack:
                        active.capture = False
                        active.last_response = None
                        clear_content(active.otel_span)
        event_name = event.class_name()
        if event_name == "LLMChatStartEvent":
            self._handle_chat_start(event=event)
        elif event_name == "LLMChatEndEvent":
            self._handle_chat_end(event=event)
        elif event_name == "LLMCompletionStartEvent":
            self._handle_completion_start(event=event)
        elif event_name == "LLMCompletionEndEvent":
            self._handle_completion_end(event=event)
        elif event_name in {"EmbeddingStartEvent", "SparseEmbeddingStartEvent"}:
            self._handle_embedding_start(event=event)
        elif event_name in {"EmbeddingEndEvent", "SparseEmbeddingEndEvent"}:
            self._handle_embedding_end(event=event)
        elif event_name in {"LLMChatInProgressEvent", "LLMCompletionInProgressEvent"}:
            self._handle_progress(event)
        elif event_name == "ExceptionEvent":
            self._handle_exception(event=event)

    def _handle_chat_start(self, *, event: Any) -> None:
        model_dict = getattr(event, "model_dict", None)
        attributes = _llm_base_attributes(
            entity_name=LLAMA_INDEX_CHAT_SPAN_NAME,
            log_type=LOG_TYPE_CHAT,
            request_type=LLMRequestTypeValues.CHAT.value,
            model_dict=model_dict,
        )
        active = self._push_event_span(
            span_id=getattr(event, "span_id", None),
            event_key=CHAT_EVENT_KEY,
            span_name=LLAMA_INDEX_CHAT_SPAN_NAME,
            attributes=attributes,
        )
        if active is None or not active.capture:
            return
        messages = chat_messages_to_dicts(getattr(event, "messages", []))
        _set_messages(attributes, SpanAttributes.LLM_PROMPTS, messages)
        attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(messages)
        tools = getattr(event, "additional_kwargs", {}).get("tools")
        if tools:
            attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(
                tools, complete=True
            )
        active.otel_span.set_attributes(_clean_attributes(attributes=attributes))

    def _handle_chat_end(self, *, event: Any) -> None:
        active_event_span = self._pop_event_span(
            span_id=getattr(event, "span_id", None),
            event_key=CHAT_EVENT_KEY,
        )
        if active_event_span is None:
            return
        response = getattr(event, "response", None) or active_event_span.last_response
        response_message = (
            chat_response_to_message_dict(response)
            if response is not None
            and active_event_span.capture
            and content_allowed(self.capture_content)
            else None
        )
        attributes: dict[str, Any] = {}
        if (
            active_event_span.capture
            and content_allowed(self.capture_content)
            and response_message is not None
        ):
            _set_messages(
                attributes, SpanAttributes.LLM_COMPLETIONS, [response_message]
            )
            attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(
                response_message
            )
        _set_usage_attributes(
            attributes=attributes,
            response=response,
        )
        _finish_event_span(
            active_event_span=active_event_span,
            attributes=attributes,
        )
        if self._span_handler is not None:
            self._span_handler.finish_deferred(
                getattr(event, "span_id", None), response=response
            )

    def _handle_completion_start(self, *, event: Any) -> None:
        model_dict = getattr(event, "model_dict", None)
        prompt = getattr(event, "prompt", "")
        attributes = _llm_base_attributes(
            entity_name=LLAMA_INDEX_COMPLETION_SPAN_NAME,
            log_type=LOG_TYPE_TEXT,
            request_type=LLMRequestTypeValues.CHAT.value,
            model_dict=model_dict,
        )
        active = self._push_event_span(
            span_id=getattr(event, "span_id", None),
            event_key=COMPLETION_EVENT_KEY,
            span_name=LLAMA_INDEX_COMPLETION_SPAN_NAME,
            attributes=attributes,
        )
        if active is not None and active.capture:
            active.otel_span.set_attributes(
                {
                    f"{SpanAttributes.LLM_PROMPTS}.0.role": MESSAGE_ROLE_USER,
                    f"{SpanAttributes.LLM_PROMPTS}.0.content": safe_text(
                        prompt, max_bytes=16_000
                    ),
                    SpanAttributes.TRACELOOP_ENTITY_INPUT: safe_json(
                        [{"role": MESSAGE_ROLE_USER, "content": prompt}]
                    ),
                }
            )

    def _handle_completion_end(self, *, event: Any) -> None:
        active_event_span = self._pop_event_span(
            span_id=getattr(event, "span_id", None),
            event_key=COMPLETION_EVENT_KEY,
        )
        if active_event_span is None:
            return
        response = getattr(event, "response", None) or active_event_span.last_response
        completion_text = (
            completion_response_to_text(response)
            if response is not None
            and active_event_span.capture
            and content_allowed(self.capture_content)
            else None
        )
        attributes: dict[str, Any] = {}
        if (
            active_event_span.capture
            and content_allowed(self.capture_content)
            and completion_text is not None
        ):
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = (
                MESSAGE_ROLE_ASSISTANT
            )
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = completion_text
            attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(
                {"role": MESSAGE_ROLE_ASSISTANT, "content": completion_text}
            )
        _set_usage_attributes(
            attributes=attributes,
            response=response,
        )
        _finish_event_span(
            active_event_span=active_event_span,
            attributes=attributes,
        )

        if self._span_handler is not None:
            self._span_handler.finish_deferred(
                getattr(event, "span_id", None), response=response
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
        if active_event_span.capture and content_allowed(self.capture_content):
            attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(chunks)
            attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(
                embeddings, complete=True
            )
            if chunks:
                attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] = (
                    _embedding_input(chunks=chunks)
                )
        _finish_event_span(
            active_event_span=active_event_span,
            attributes=attributes,
        )

    def _handle_progress(self, event: Any) -> None:
        event_key = (
            CHAT_EVENT_KEY
            if event.class_name() == "LLMChatInProgressEvent"
            else COMPLETION_EVENT_KEY
        )
        spans = self._open_event_spans.get(
            (getattr(event, "span_id", None), event_key), []
        )
        if spans:
            active = spans[-1]
            active.otel_span.set_attribute(SpanAttributes.GEN_AI_IS_STREAMING, True)
            if active.capture and content_allowed(self.capture_content):
                active.last_response = getattr(event, "response", None)
            elif not content_allowed(self.capture_content):
                active.capture = False
                active.last_response = None
                clear_content(active.otel_span)
            active.otel_span.set_attributes(
                usage_attributes(getattr(event, "response", None))
            )

    def _handle_exception(self, *, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        exception = getattr(event, "exception", None)
        if exception is None:
            return
        for key in [key for key in self._open_event_spans if key[0] == span_id]:
            for active in reversed(self._open_event_spans.pop(key)):
                _record_error(
                    active.otel_span,
                    exception,
                    active.capture and content_allowed(self.capture_content),
                )
                _finish_event_span(active_event_span=active, attributes={})
        if self._span_handler is not None:
            self._span_handler.finish_deferred(span_id, error=exception)

    def record_embedding_usage(self, response: Any) -> None:
        from llama_index_instrumentation.span import active_span_id

        if suppressed():
            return
        with self._event_lock:
            span_id = active_span_id.get()
            visited = set()
            stack = []
            while span_id and span_id not in visited:
                visited.add(span_id)
                stack = self._open_event_spans.get((span_id, EMBEDDING_EVENT_KEY), [])
                if stack or self._span_handler is None:
                    break
                native = self._span_handler.open_spans.get(span_id)
                span_id = native.parent_id if native is not None else None
            if stack:
                for key, value in usage_attributes(response).items():
                    if "output_tokens" not in key and "completion_tokens" not in key:
                        previous = (stack[-1].otel_span.attributes or {}).get(key, 0)
                        stack[-1].otel_span.set_attribute(key, previous + value)

    def close(self) -> None:
        for spans in self._open_event_spans.values():
            for active in spans:
                if not active.capture or not content_allowed(self.capture_content):
                    clear_content(active.otel_span)
                active.otel_span.end()
        self._open_event_spans.clear()

    def _push_event_span(
        self,
        *,
        span_id: str | None,
        event_key: str,
        span_name: str,
        attributes: dict[str, Any],
    ) -> ActiveEventSpan | None:
        if suppressed() or (
            self._span_handler is not None
            and span_id in self._span_handler._suppressed_ids
        ):
            return None
        parent = (
            self._span_handler.parent_context(span_id)
            if self._span_handler is not None
            else None
        )
        span, token, _ = _start_otel_span(
            span_name=span_name, attributes=attributes, parent_context=parent
        )
        capture = (
            span.is_recording()
            and content_allowed(self.capture_content)
            and (
                self._span_handler is None or self._span_handler.parent_capture(span_id)
            )
        )
        active = ActiveEventSpan(span, token, capture)
        source = span_id or ""
        if ".stream_" in source or ".astream_" in source:
            span.set_attribute(SpanAttributes.GEN_AI_IS_STREAMING, True)
        self._open_event_spans.setdefault((span_id, event_key), []).append(active)
        return active

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
    tracer = trace.get_tracer(__name__)
    otel_span = tracer.start_span(
        span_name,
        context=parent_context,
        attributes=_clean_attributes(attributes=attributes),
    )
    span_context = trace.set_span_in_context(
        otel_span,
        parent_context or context.get_current(),
    )
    return otel_span, None, span_context


def _start_detached_otel_span(
    *,
    span_name: str,
    attributes: dict[str, Any],
    parent_context: Any | None = None,
) -> Any:
    tracer = trace.get_tracer(__name__)
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
        if active_span.context_token is not None:
            context.detach(active_span.context_token)


def _finish_event_span(
    *,
    active_event_span: ActiveEventSpan,
    attributes: dict[str, Any],
) -> None:
    if not active_event_span.capture or not content_allowed():
        clear_content(active_event_span.otel_span)
        attributes = {
            key: value
            for key, value in attributes.items()
            if key
            not in (
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                SpanAttributes.LLM_REQUEST_FUNCTIONS,
            )
            and not key.startswith(
                (f"{SpanAttributes.LLM_PROMPTS}.", f"{SpanAttributes.LLM_COMPLETIONS}.")
            )
        }
    for key, value in _clean_attributes(attributes=attributes).items():
        active_event_span.otel_span.set_attribute(key, value)
    try:
        active_event_span.otel_span.end()
    finally:
        if active_event_span.context_token is not None:
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
        attributes[SpanAttributes.LLM_SYSTEM] = model_system
    return attributes


def _set_usage_attributes(*, attributes: dict[str, Any], response: Any) -> None:
    attributes.update(usage_attributes(response))


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


def _span_output_payload(*, log_type: str, result: Any) -> Any:
    if log_type == LOG_TYPE_TOOL:
        return _tool_output_payload(result=result)
    return result


def _span_input_payload(
    *,
    bound_args: inspect.BoundArguments,
    tags: dict[str, Any] | None,
    tool_name: str | None = None,
) -> Any:
    """Prefer public Workflows event summaries over internal runtime state.

    Current llama-index-workflows run spans bind a recursive broker state
    object. Serializing that object can fail before the root span is created.
    The SDK emits stable input summaries in instrumentation tags specifically
    for integrations, so consume those and keep the raw vendor tags off the
    exported span.
    """

    if tool_name is not None:
        return {
            "name": tool_name,
            "arguments": _tool_arguments(bound_args=bound_args),
        }

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
        "args": _public_arguments(bound_args.args),
        "kwargs": _public_arguments(bound_args.kwargs),
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
    if _is_executable_tool_span(entity_name=entity_name, instance=instance):
        return LOG_TYPE_TOOL
    if _is_tool_orchestration_span(entity_name=entity_name):
        return LOG_TYPE_TASK
    if "agent" in normalized_name or "agent" in instance_name:
        return LOG_TYPE_AGENT
    if parent_span_id is None:
        return LOG_TYPE_WORKFLOW
    return LOG_TYPE_TASK


def _is_executable_tool_span(*, entity_name: str, instance: Any | None) -> bool:
    """Return whether a native span wraps the tool implementation itself.

    Agent workflow methods such as ``call_tool`` and
    ``aggregate_tool_results`` coordinate a call but do not execute the
    application function. LlamaIndex's ``BaseTool.call`` / ``acall`` boundary
    is the one span that owns both the invocation arguments and result.
    """

    operation_name = entity_name.rsplit(".", maxsplit=1)[-1]
    return isinstance(instance, BaseTool) and operation_name in {"call", "acall"}


def _is_tool_orchestration_span(*, entity_name: str) -> bool:
    operation_name = entity_name.rsplit(".", maxsplit=1)[-1]
    return operation_name in {"_call_tool", "call_tool", "aggregate_tool_results"}


def _tool_name(*, tool: Any) -> str:
    metadata = getattr(tool, "metadata", None)
    if metadata is not None:
        for attr_name in ("name", "tool_name"):
            value = getattr(metadata, attr_name, None)
            if value:
                return str(value)
        get_name = getattr(metadata, "get_name", None)
        if callable(get_name):
            return str(get_name())
    for attr_name in ("name", "tool_name"):
        value = getattr(tool, attr_name, None)
        if value:
            return str(value)
    get_name = getattr(tool, "get_name", None)
    if callable(get_name):
        return str(get_name())
    return LLAMA_INDEX_DEFAULT_TOOL_NAME


def _tool_arguments(*, bound_args: inspect.BoundArguments) -> Any:
    positional = to_jsonable(getattr(bound_args, "args", ()), complete=True)
    keyword = to_jsonable(getattr(bound_args, "kwargs", {}), complete=True)
    if not positional:
        return keyword
    if not keyword:
        return positional
    return {"args": positional, "kwargs": keyword}


def _tool_output_payload(*, result: Any) -> Any:
    raw_output = getattr(result, "raw_output", None)
    if raw_output is not None:
        return raw_output
    content = getattr(result, "content", None)
    if content is not None:
        return content
    return result


def _record_error(span: Any, error: BaseException, capture: bool) -> None:
    message = None
    if capture:
        args = BaseException.args.__get__(error)
        message = safe_text(" ".join(value for value in args if isinstance(value, str)))
    span.add_event(
        "exception",
        {
            EXCEPTION_TYPE: type(error).__name__,
            **({EXCEPTION_MESSAGE: message} if message else {}),
        },
    )
    if not isinstance(error, (asyncio.CancelledError, GeneratorExit)):
        span.set_status(Status(StatusCode.ERROR, message))


def _set_messages(
    attrs: dict[str, Any], prefix: str, messages: list[dict[str, Any]]
) -> None:
    for index, message in enumerate(messages):
        base = f"{prefix}.{index}"
        attrs[f"{base}.role"] = message.get("role", "")
        if message.get("content") is not None:
            attrs[f"{base}.content"] = _content_attribute(value=message["content"])
        calls = message.get("tool_calls")
        if calls:
            attrs[f"{base}.tool_calls"] = safe_json(calls, complete=True)


def _native_tool_call_id(bound_args: Any) -> str | None:
    for value in getattr(bound_args, "arguments", {}).values():
        identifier = getattr(value, "tool_id", None) or getattr(
            value, "tool_call_id", None
        )
        if isinstance(identifier, str) and identifier:
            return identifier
    return None


def _public_arguments(value: Any) -> Any:
    from llama_index.core.llms import ChatMessage

    from respan_instrumentation_llama_index._serialization import (
        message_to_dict,
        normalize_message_sequence,
    )

    if isinstance(value, ChatMessage):
        return message_to_dict(value)
    if isinstance(value, (list, tuple)):
        values = [_public_arguments(item) for item in value]
        return (
            normalize_message_sequence(values)
            if all(isinstance(item, dict) and "role" in item for item in values)
            else values
        )
    if isinstance(value, dict):
        return {key: _public_arguments(item) for key, item in value.items()}
    return to_jsonable(value)
