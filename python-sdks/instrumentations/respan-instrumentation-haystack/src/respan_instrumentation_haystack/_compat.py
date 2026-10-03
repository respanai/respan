"""Released Haystack values missing from the upstream OpenInference adapter."""

from __future__ import annotations

import contextvars
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from openinference.instrumentation.config import REDACTED_VALUE
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceSpanKindValues,
    ToolCallAttributes,
)
from openinference.semconv.trace import (
    SpanAttributes as OIAttributes,
)
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv_ai import SpanAttributes
from respan_tracing.decorators.base import _should_send_prompts

from ._content import without_content

_ACTIVE_COMPONENTS: contextvars.ContextVar[tuple[int, ...]] = contextvars.ContextVar(
    "respan_haystack_active_components", default=()
)
_EMBEDDINGS: dict[tuple[int, int], dict[str, str]] = {}


def _json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _message(value: Any, role: str) -> dict[str, Any]:
    if isinstance(value, str):
        return {"role": role, "content": value}
    message = value.to_openai_dict_format(require_tool_call_ids=False)
    results = getattr(value, "tool_call_results", ())
    if results:
        result = results[0]
        message["content"] = _json(
            {
                "name": result.origin.tool_name,
                "result": result.result,
                "tool_call_id": result.origin.id,
            }
        )
    return message


def _messages(values: Any, prefix: str, role: str, config: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    if not isinstance(values, (list, tuple)):
        return attributes
    for index, value in enumerate(values):
        if not isinstance(value, str) and not hasattr(value, "to_openai_dict_format"):
            continue
        message = _message(value, role)
        base = f"{prefix}.{index}"
        attributes[f"{base}.{MessageAttributes.MESSAGE_ROLE}"] = message["role"]
        content = message.get("content")
        is_input = prefix == OIAttributes.LLM_INPUT_MESSAGES
        hide_text = config.hide_input_text if is_input else config.hide_output_text
        if isinstance(content, str) and hide_text:
            content = REDACTED_VALUE
        elif isinstance(content, list):
            content = [
                (
                    {**part, "text": REDACTED_VALUE}
                    if hide_text and part.get("type") == "text"
                    else part
                )
                for part in content
                if not (
                    is_input
                    and config.hide_input_images
                    and part.get("type") in {"image_url", "input_image"}
                )
            ]
        if content is not None:
            attributes[f"{base}.{MessageAttributes.MESSAGE_CONTENT}"] = (
                content if isinstance(content, str) else _json(content)
            )
        for call_index, call in enumerate(message.get("tool_calls", ())):
            call_base = f"{base}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{call_index}"
            if call.get("id") is not None:
                attributes[f"{call_base}.{ToolCallAttributes.TOOL_CALL_ID}"] = call[
                    "id"
                ]
            function = call.get("function", {})
            attributes[f"{call_base}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] = (
                function.get("name", "")
            )
            attributes[
                f"{call_base}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
            ] = function.get("arguments", "{}")
    return attributes


def _payload(value: Any) -> Any:
    from haystack.dataclasses import ChatMessage

    if isinstance(value, ChatMessage):
        return _message(value, value.role.value)
    if isinstance(value, dict):
        return {key: _payload(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_payload(child) for child in value]
    return value


def apply_embedding_capture(span: Any) -> None:
    context = span.get_span_context()
    if context is not None and hasattr(context, "trace_id"):
        captured = _EMBEDDINGS.pop((context.trace_id, context.span_id), None)
        if captured:
            span._attributes = {**dict(span.attributes or {}), **captured}


def install_compatibility(delegate: Any) -> list[tuple[Any, str, Any, Any]]:
    instrumentor = getattr(delegate, "_instrumentor", None)
    tracer = getattr(instrumentor, "_tracer", None)
    if tracer is None:
        return []
    wrappers = importlib.import_module(
        "openinference.instrumentation.haystack._wrappers"
    )
    config = tracer._self_config
    patches = []

    def patch(owner, name, replacement):
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        patches.append((owner, name, original, replacement))
        return original

    try:
        original_request = wrappers._set_component_runner_request_attributes
        original_response = wrappers._set_component_runner_response_attributes

        def request(bound_arguments, instance):
            attributes = original_request(bound_arguments, instance)
            arguments = bound_arguments.arguments
            from haystack.components.agents import Agent
            from haystack.components.builders import ChatPromptBuilder, PromptBuilder

            if isinstance(instance, Agent):
                attributes[OIAttributes.OPENINFERENCE_SPAN_KIND] = (
                    OpenInferenceSpanKindValues.AGENT.value
                )
            elif isinstance(instance, (PromptBuilder, ChatPromptBuilder)):
                attributes[OIAttributes.OPENINFERENCE_SPAN_KIND] = (
                    OpenInferenceSpanKindValues.CHAIN.value
                )
            if "messages" in arguments:
                attributes[OIAttributes.INPUT_VALUE] = _json(_payload(arguments))
                if (
                    attributes.get(OIAttributes.OPENINFERENCE_SPAN_KIND)
                    == OpenInferenceSpanKindValues.LLM.value
                ):
                    attributes.update(
                        _messages(
                            arguments["messages"],
                            OIAttributes.LLM_INPUT_MESSAGES,
                            "user",
                            config,
                        )
                    )
            if (
                config.hide_input_messages
                or config.hide_input_text
                or config.hide_input_images
            ):
                attributes[OIAttributes.INPUT_VALUE] = REDACTED_VALUE
            tools = arguments.get("tools")
            if isinstance(tools, (list, tuple)):
                attributes[OIAttributes.LLM_TOOLS] = _json(
                    [
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters,
                            },
                        }
                        for tool in tools
                        if hasattr(tool, "parameters")
                    ]
                )
            return attributes if _should_send_prompts() else without_content(attributes)

        def response(bound_arguments, component_type, result, instance):
            attributes = original_response(
                bound_arguments, component_type, result, instance
            )
            if isinstance(result, dict) and any(
                key in result for key in ("replies", "messages", "last_message")
            ):
                attributes[OIAttributes.OUTPUT_VALUE] = _json(_payload(result))
            if component_type is wrappers.ComponentType.GENERATOR:
                attributes.update(
                    _messages(
                        result.get("replies"),
                        OIAttributes.LLM_OUTPUT_MESSAGES,
                        "assistant",
                        config,
                    )
                )
            if config.hide_output_messages or config.hide_output_text:
                attributes[OIAttributes.OUTPUT_VALUE] = REDACTED_VALUE
            if component_type is wrappers.ComponentType.EMBEDDER:
                # The shared bridge bounds general collections. Embedding vectors are
                # numerical data and must retain every dimension after translation.
                captured = {}
                # Reuse upstream's validated extraction (including base64 vectors),
                # before OpenInference/Respan apply generic collection limits.
                prefix = OIAttributes.EMBEDDING_EMBEDDINGS + "."
                vectors_by_index = {}
                texts_by_index = {}
                for key, value in attributes.items():
                    if not key.startswith(prefix):
                        continue
                    index, _, field = key[len(prefix) :].partition(".")
                    if not index.isdigit():
                        continue
                    if field == EmbeddingAttributes.EMBEDDING_VECTOR:
                        vectors_by_index[int(index)] = list(value)
                    elif field == EmbeddingAttributes.EMBEDDING_TEXT:
                        texts_by_index[int(index)] = value
                vectors = [
                    vectors_by_index[index] for index in sorted(vectors_by_index)
                ]
                texts = [texts_by_index[index] for index in sorted(texts_by_index)]
                if "text" in bound_arguments.arguments and len(texts) == 1:
                    texts = texts[0]
                if not config.hide_inputs and not config.hide_embeddings_text and texts:
                    captured[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _json(texts)
                if (
                    not config.hide_outputs
                    and not config.hide_embeddings_vectors
                    and not config.hide_embedding_vectors
                    and vectors
                ):
                    captured[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json(vectors)
                current_span = trace.get_current_span()
                context = current_span.get_span_context()
                if current_span.is_recording() and captured and _should_send_prompts():
                    _EMBEDDINGS[(context.trace_id, context.span_id)] = captured
                metadata = result.get("meta")
                usage = metadata.get("usage", {}) if isinstance(metadata, dict) else {}
                if not isinstance(usage, dict):
                    usage = {}
                for field, key in (
                    ("prompt_tokens", OIAttributes.LLM_TOKEN_COUNT_PROMPT),
                    ("completion_tokens", OIAttributes.LLM_TOKEN_COUNT_COMPLETION),
                    ("total_tokens", OIAttributes.LLM_TOKEN_COUNT_TOTAL),
                ):
                    value = usage.get(field)
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 0
                    ):
                        attributes[key] = value
            return attributes if _should_send_prompts() else without_content(attributes)

        patch(wrappers, "_set_component_runner_request_attributes", request)
        patch(wrappers, "_set_component_runner_response_attributes", response)
        original_sync = wrappers._ComponentRunWrapper.__call__
        original_async = wrappers._AsyncComponentRunWrapper.__call__

        def sync_call(wrapper, wrapped, instance, args, kwargs):
            active = _ACTIVE_COMPONENTS.get()
            if id(instance) in active:
                return wrapped(*args, **kwargs)
            return original_sync(wrapper, wrapped, instance, args, kwargs)

        async def async_call(wrapper, wrapped, instance, args, kwargs):
            active = _ACTIVE_COMPONENTS.get()
            token = _ACTIVE_COMPONENTS.set((*active, id(instance)))
            try:
                return await original_async(wrapper, wrapped, instance, args, kwargs)
            finally:
                _ACTIVE_COMPONENTS.reset(token)

        patch(wrappers._ComponentRunWrapper, "__call__", sync_call)
        patch(wrappers._AsyncComponentRunWrapper, "__call__", async_call)

        # Copy context only for Haystack-owned retrieval pools; other applications'
        # executors and their scheduling behavior remain untouched.
        class ContextThreadPoolExecutor(ThreadPoolExecutor):
            def submit(self, fn, /, *args, **kwargs):
                context = contextvars.copy_context()
                return super().submit(context.run, fn, *args, **kwargs)

        for module_name in (
            "haystack.components.retrievers.multi_query_text_retriever",
            "haystack.components.retrievers.multi_query_embedding_retriever",
            "haystack.components.retrievers.multi_retriever",
        ):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            patch(module, "ThreadPoolExecutor", ContextThreadPoolExecutor)

        # Haystack 3 moved tool execution from ToolInvoker into the Agent. The
        # upstream component instrumentor cannot observe these individual calls.
        try:
            tool_calling = importlib.import_module(
                "haystack.components.agents.tool_calling"
            )
        except ImportError:
            return patches

        @contextmanager
        def tool_span(tool, tool_call, parent_span):
            attributes = {
                OIAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                OIAttributes.TOOL_NAME: tool_call.tool_name,
                OIAttributes.INPUT_VALUE: _json(
                    {"name": tool_call.tool_name, "arguments": tool_call.arguments}
                ),
            }
            if not _should_send_prompts():
                attributes = without_content(attributes)
            if tool_call.id is not None:
                attributes[gen_ai_attributes.GEN_AI_TOOL_CALL_ID] = tool_call.id
            with tracer.start_as_current_span(
                tool_call.tool_name, attributes=attributes
            ) as span:

                class ToolSpan:
                    def set_content_tag(self, key, value):
                        if key.endswith(".output"):
                            if _should_send_prompts():
                                span.set_attribute(
                                    OIAttributes.OUTPUT_VALUE, _json(value)
                                )
                            if isinstance(value, dict) and "error" in value:
                                span.set_status(
                                    trace.Status(
                                        trace.StatusCode.ERROR, str(value["error"])
                                    )
                                )

                yield ToolSpan()

        patch(tool_calling, "_create_tool_span", tool_span)
        return patches

    except Exception:
        restore_compatibility(patches)
        raise


def restore_compatibility(patches: list[tuple[Any, str, Any, Any]]) -> None:
    for owner, name, original, replacement in reversed(patches):
        if getattr(owner, name, None) is replacement:
            setattr(owner, name, original)
    _EMBEDDINGS.clear()
