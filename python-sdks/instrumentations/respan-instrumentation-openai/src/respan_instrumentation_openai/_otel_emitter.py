"""Emit OpenAI SDK calls as OTEL ReadableSpan objects.

Each ``emit_*`` builds a ``ReadableSpan`` carrying Respan's documented LLM
conventions (``llm.request.type``, ``gen_ai.system``, ``gen_ai.usage.*``,
``traceloop.entity.*``, ``respan.log_type``) and injects it into the single
OTEL pipeline via ``inject_span()``. The span nests under the current OTEL
span when one is active (e.g. a ``@workflow``/``@task`` decorator), otherwise
it is a root span (its own trace).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from opentelemetry import trace

from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_COMPLETION,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_RESPONSE,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_TOOL_CALLS,
)
from respan_sdk.utils.data_processing.id_processing import format_span_id, format_trace_id
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from respan_instrumentation_openai._constants import (
    ASSISTANT_ROLE,
    CHAT_SPAN_NAME,
    COMPLETION_SPAN_NAME,
    EMBEDDING_SPAN_NAME,
    LLM_COMPLETIONS,
    LLM_PROMPTS,
    LLM_REQUEST_FUNCTIONS,
    LLM_RESPONSE_ID,
    LLM_RESPONSE_MODEL,
    LLM_USAGE_TOTAL_TOKENS,
    OPENAI_SYSTEM,
    REQUEST_TYPE_CHAT,
    REQUEST_TYPE_COMPLETION,
    REQUEST_TYPE_EMBEDDING,
    RESPONSE_SPAN_NAME,
    SPAN_KIND_LLM,
    TRACELOOP_ENTITY_INPUT,
    TRACELOOP_ENTITY_NAME,
    TRACELOOP_ENTITY_OUTPUT,
    TRACELOOP_ENTITY_PATH,
    TRACELOOP_SPAN_KIND,
)
from respan_instrumentation_openai import _translator as tr

logger = logging.getLogger(__name__)

_PROMPT_PREFIX = f"{LLM_PROMPTS}."
_COMPLETION_PREFIX = f"{LLM_COMPLETIONS}."


# ---------------------------------------------------------------------------
# Trace context (nesting)
# ---------------------------------------------------------------------------


def _current_trace_parent_ids() -> tuple[str | None, str | None]:
    """Return (trace_id, parent_id) hex strings from the active OTEL span.

    When no live span is active the call is standalone → (None, None) → the
    emitted span becomes a root (its own trace).
    """
    try:
        ctx = trace.get_current_span().get_span_context()
    except Exception:
        return None, None
    tid = getattr(ctx, "trace_id", 0) or 0
    sid = getattr(ctx, "span_id", 0) or 0
    if not tid or not sid:
        return None, None
    return format_trace_id(trace_id=tid), format_span_id(span_id=sid)


# ---------------------------------------------------------------------------
# Attribute builders
# ---------------------------------------------------------------------------


def _base_attrs(*, span_name: str, log_type: str, request_type: str) -> dict[str, Any]:
    return {
        TRACELOOP_SPAN_KIND: SPAN_KIND_LLM,
        TRACELOOP_ENTITY_NAME: span_name,
        TRACELOOP_ENTITY_PATH: span_name,
        GEN_AI_SYSTEM: OPENAI_SYSTEM,
        LLM_REQUEST_TYPE: request_type,
        RESPAN_LOG_TYPE: log_type,
    }


def _set_model(attrs: dict[str, Any], request_kwargs: dict[str, Any], response: Any) -> None:
    model = tr.request_model(request_kwargs)
    if model:
        attrs[LLM_REQUEST_MODEL] = model
    resp_model = tr.response_model(response)
    if resp_model:
        attrs[LLM_RESPONSE_MODEL] = resp_model
    rid = tr.response_id(response)
    if rid:
        attrs[LLM_RESPONSE_ID] = rid


def _set_prompt_attrs(attrs: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        if role is not None:
            attrs[f"{_PROMPT_PREFIX}{i}.role"] = str(role)
        if content is not None:
            attrs[f"{_PROMPT_PREFIX}{i}.content"] = tr.to_attr_value(content)


def _set_completion_attrs(attrs: dict[str, Any], content: str) -> None:
    attrs[f"{_COMPLETION_PREFIX}0.role"] = ASSISTANT_ROLE
    attrs[f"{_COMPLETION_PREFIX}0.content"] = content


def _set_usage(attrs: dict[str, Any], response: Any) -> None:
    usage = tr.extract_usage(response)
    if "prompt" in usage:
        attrs[LLM_USAGE_PROMPT_TOKENS] = usage["prompt"]
    if "completion" in usage:
        attrs[LLM_USAGE_COMPLETION_TOKENS] = usage["completion"]
    if "total" in usage:
        attrs[LLM_USAGE_TOTAL_TOKENS] = usage["total"]


def build_chat_attrs(*, request_kwargs: dict[str, Any], response: Any = None) -> dict[str, Any]:
    attrs = _base_attrs(span_name=CHAT_SPAN_NAME, log_type=LOG_TYPE_CHAT, request_type=REQUEST_TYPE_CHAT)
    messages = tr.normalize_chat_messages(request_kwargs.get("messages"))
    attrs[TRACELOOP_ENTITY_INPUT] = tr.format_input_messages(messages)
    _set_prompt_attrs(attrs, messages)
    tools = tr.normalize_tools(request_kwargs.get("tools"))
    if tools:
        attrs[LLM_REQUEST_FUNCTIONS] = tr.safe_json(tools)
    _set_model(attrs, request_kwargs, response)
    if response is not None:
        attrs[TRACELOOP_ENTITY_OUTPUT] = tr.format_chat_output(response)
        _set_completion_attrs(attrs, tr.format_chat_output(response))
        tool_calls = tr.extract_chat_tool_calls(response)
        if tool_calls:
            attrs[RESPAN_SPAN_TOOL_CALLS] = tr.safe_json(tool_calls)
            attrs[f"{_COMPLETION_PREFIX}0.tool_calls"] = tr.safe_json(tool_calls)
        _set_usage(attrs, response)
    return attrs


def build_completion_attrs(*, request_kwargs: dict[str, Any], response: Any = None) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=COMPLETION_SPAN_NAME, log_type=LOG_TYPE_COMPLETION, request_type=REQUEST_TYPE_COMPLETION
    )
    messages = tr.normalize_text_prompts(request_kwargs.get("prompt"))
    attrs[TRACELOOP_ENTITY_INPUT] = tr.format_input_messages(messages)
    _set_prompt_attrs(attrs, messages)
    _set_model(attrs, request_kwargs, response)
    if response is not None:
        output = tr.format_completion_output(response)
        attrs[TRACELOOP_ENTITY_OUTPUT] = output
        _set_completion_attrs(attrs, output)
        _set_usage(attrs, response)
    return attrs


def build_response_attrs(*, request_kwargs: dict[str, Any], response: Any = None) -> dict[str, Any]:
    attrs = _base_attrs(span_name=RESPONSE_SPAN_NAME, log_type=LOG_TYPE_RESPONSE, request_type=REQUEST_TYPE_CHAT)
    messages = tr.normalize_responses_input(request_kwargs.get("input"))
    attrs[TRACELOOP_ENTITY_INPUT] = tr.format_input_messages(messages)
    _set_prompt_attrs(attrs, messages)
    tools = tr.normalize_tools(request_kwargs.get("tools"))
    if tools:
        attrs[LLM_REQUEST_FUNCTIONS] = tr.safe_json(tools)
    _set_model(attrs, request_kwargs, response)
    if response is not None:
        output = tr.format_responses_output(response)
        attrs[TRACELOOP_ENTITY_OUTPUT] = output
        _set_completion_attrs(attrs, output)
        _set_usage(attrs, response)
    return attrs


def build_embedding_attrs(*, request_kwargs: dict[str, Any], response: Any = None) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=EMBEDDING_SPAN_NAME, log_type=LOG_TYPE_EMBEDDING, request_type=REQUEST_TYPE_EMBEDDING
    )
    inputs = tr.normalize_embedding_inputs(request_kwargs.get("input"))
    attrs[TRACELOOP_ENTITY_INPUT] = tr.safe_json(inputs)
    attrs[f"{_PROMPT_PREFIX}0.content"] = tr.safe_json(inputs)
    _set_model(attrs, request_kwargs, response)
    if response is not None:
        summary = tr.embedding_summary(response)
        if summary:
            attrs[TRACELOOP_ENTITY_OUTPUT] = tr.safe_json(summary)
        _set_usage(attrs, response)
    return attrs


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


def emit_span(
    *,
    span_name: str,
    attrs: dict[str, Any],
    start_ns: int,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    try:
        trace_id, parent_id = _current_trace_parent_ids()
        span = build_readable_span(
            name=span_name,
            trace_id=trace_id,
            parent_id=parent_id,
            start_time_ns=start_ns,
            end_time_ns=time.time_ns(),
            attributes=attrs,
            error_message=error_message,
            status_code=status_code,
        )
        inject_span(span=span)
    except Exception:
        logger.debug("Failed to emit OpenAI span %r", span_name, exc_info=True)


def _emit(builder, span_name, *, request_kwargs, response, start_ns, error_message, status_code) -> None:
    try:
        attrs = builder(request_kwargs=request_kwargs, response=response)
    except Exception:
        logger.debug("Failed to build attrs for %r", span_name, exc_info=True)
        attrs = {}
    emit_span(
        span_name=span_name,
        attrs=attrs,
        start_ns=start_ns,
        error_message=error_message,
        status_code=status_code,
    )


def emit_chat_span(*, request_kwargs, start_ns, response=None, error_message=None, status_code=200) -> None:
    _emit(build_chat_attrs, CHAT_SPAN_NAME, request_kwargs=request_kwargs, response=response,
          start_ns=start_ns, error_message=error_message, status_code=status_code)


def emit_completion_span(*, request_kwargs, start_ns, response=None, error_message=None, status_code=200) -> None:
    _emit(build_completion_attrs, COMPLETION_SPAN_NAME, request_kwargs=request_kwargs, response=response,
          start_ns=start_ns, error_message=error_message, status_code=status_code)


def emit_response_span(*, request_kwargs, start_ns, response=None, error_message=None, status_code=200) -> None:
    _emit(build_response_attrs, RESPONSE_SPAN_NAME, request_kwargs=request_kwargs, response=response,
          start_ns=start_ns, error_message=error_message, status_code=status_code)


def emit_embedding_span(*, request_kwargs, start_ns, response=None, error_message=None, status_code=200) -> None:
    _emit(build_embedding_attrs, EMBEDDING_SPAN_NAME, request_kwargs=request_kwargs, response=response,
          start_ns=start_ns, error_message=error_message, status_code=status_code)
