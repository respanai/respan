"""Translate Guardrails native telemetry without retaining duplicate payloads."""

import ast
import json
from typing import Any

from openinference.semconv.trace import MessageAttributes
from openinference.semconv.trace import SpanAttributes as OIAttributes
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_GUARDRAIL
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


def parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return value


def _clean(value: Any, depth: int = 0) -> Any:
    if depth > 20:
        return "<depth limit>"
    if isinstance(value, str):
        parsed = parse(value)
        if isinstance(parsed, (dict, list)):
            return json.dumps(_clean(parsed, depth + 1), default=str)
        return value
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if any(
                    part in str(key).lower().replace("_", "")
                    for part in (
                        "apikey",
                        "authorization",
                        "password",
                        "accesstoken",
                        "secret",
                    )
                )
                else _clean(item, depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_clean(item, depth + 1) for item in value]
    return value


def serialize(value: Any) -> str:
    return json.dumps(_clean(parse(value)), default=str, ensure_ascii=False)


def _integer(value: Any) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        result = int(value)
        return result if result >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def guardrails_type(span: Any, attrs: dict[str, Any]) -> str | None:
    value = attrs.get("type")
    if isinstance(value, str) and value.startswith("guardrails/"):
        return value
    scope = getattr(getattr(span, "instrumentation_scope", None), "name", "")
    if scope == "guardrails-ai" or scope.startswith("guardrails.telemetry."):
        # Native streaming calls can end before Guardrails attaches its type.
        if span.name == "call":
            return "guardrails/guard/step/call"
        if span.name == "step":
            return "guardrails/guard/step"
        if span.name in {"guard", "stream_guard_span"}:
            return "guardrails/guard"
    return None


def _messages(attrs: dict[str, Any], source: str) -> list[dict[str, Any]]:
    messages: dict[int, dict[str, Any]] = {}
    for key, value in attrs.items():
        if not key.startswith(source + "."):
            continue
        parts = key[len(source) + 1 :].split(".", 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        fields = {
            MessageAttributes.MESSAGE_ROLE: "role",
            MessageAttributes.MESSAGE_CONTENT: "content",
            MessageAttributes.MESSAGE_TOOL_CALLS: "tool_calls",
        }
        if parts[1] in fields:
            messages.setdefault(int(parts[0]), {})[fields[parts[1]]] = value
    return [messages[index] for index in sorted(messages)]


def _write_messages(attrs: dict[str, Any], messages: list, prefix: str) -> None:
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        for field in ("role", "content", "tool_calls"):
            value = message.get(field)
            if value is not None:
                attrs[f"{prefix}.{index}.{field}"] = (
                    serialize(value)
                    if field == "tool_calls" or not isinstance(value, str)
                    else value
                )


def normalize(span: Any, include_content: bool, llm_calls: int | None) -> bool:
    original = getattr(span, "_attributes", None)
    if original is None:
        return False
    attrs = dict(original)
    kind = guardrails_type(span, attrs)
    if kind is None:
        return False
    operation = {
        "guardrails/guard": "guardrails.guard",
        "guardrails/guard/step": "guardrails.step",
        "guardrails/guard/step/call": "guardrails.call",
        "guardrails/guard/step/validator": "guardrails.validator",
    }.get(kind, f"guardrails.{span.name}")
    attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_NAME, operation)
    attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_PATH, "")
    attrs.pop(SpanAttributes.TRACELOOP_SPAN_KIND, None)
    invocation = parse(attrs.get(OIAttributes.LLM_INVOCATION_PARAMETERS))
    invocation = invocation if isinstance(invocation, dict) else {}
    input_messages = invocation.get("messages")
    if not isinstance(input_messages, list):
        input_messages = _messages(attrs, OIAttributes.LLM_INPUT_MESSAGES)
    output_messages = _messages(attrs, OIAttributes.LLM_OUTPUT_MESSAGES)
    is_llm = kind == "guardrails/guard/step/call" and bool(
        invocation
        or input_messages
        or output_messages
        or attrs.get(OIAttributes.LLM_MODEL_NAME)
        or any(
            attrs.get(key) is not None
            for key in (
                OIAttributes.LLM_TOKEN_COUNT_PROMPT,
                OIAttributes.LLM_TOKEN_COUNT_COMPLETION,
                OIAttributes.LLM_TOKEN_COUNT_TOTAL,
            )
        )
    )
    attrs[RESPAN_LOG_TYPE] = LOG_TYPE_CHAT if is_llm else LOG_TYPE_GUARDRAIL
    if include_content:
        for source, target in (
            (OIAttributes.INPUT_VALUE, SpanAttributes.TRACELOOP_ENTITY_INPUT),
            (OIAttributes.OUTPUT_VALUE, SpanAttributes.TRACELOOP_ENTITY_OUTPUT),
        ):
            if source in attrs:
                attrs[target] = serialize(attrs[source])
    if is_llm:
        attrs[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
        model = attrs.get(OIAttributes.LLM_MODEL_NAME) or invocation.get("model")
        if model:
            attrs[SpanAttributes.LLM_REQUEST_MODEL] = model
        provider = attrs.get(OIAttributes.LLM_PROVIDER)
        if provider:
            attrs[SpanAttributes.LLM_SYSTEM] = provider
        for field, target in (
            ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
            ("max_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
            ("top_p", SpanAttributes.LLM_REQUEST_TOP_P),
        ):
            if invocation.get(field) is not None:
                attrs[target] = invocation[field]
        prompt = _integer(attrs.get(OIAttributes.LLM_TOKEN_COUNT_PROMPT))
        completion = _integer(attrs.get(OIAttributes.LLM_TOKEN_COUNT_COMPLETION))
        total = _integer(attrs.get(OIAttributes.LLM_TOKEN_COUNT_TOTAL))
        for value, keys in (
            (
                prompt,
                (SpanAttributes.LLM_USAGE_PROMPT_TOKENS, GEN_AI_USAGE_INPUT_TOKENS),
            ),
            (
                completion,
                (
                    SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                    GEN_AI_USAGE_OUTPUT_TOKENS,
                ),
            ),
        ):
            if value is not None:
                for key in keys:
                    attrs[key] = value
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        if total is not None:
            attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total
        if include_content:
            _write_messages(attrs, input_messages, SpanAttributes.LLM_PROMPTS)
            _write_messages(attrs, output_messages, SpanAttributes.LLM_COMPLETIONS)
            if invocation.get("tools") is not None:
                attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = serialize(
                    invocation["tools"]
                )
    raw_output = parse(attrs.get(OIAttributes.OUTPUT_VALUE))
    if (
        is_llm
        and isinstance(raw_output, dict)
        and "output" in raw_output
        and not raw_output["output"]
        and not output_messages
    ):
        # Native stream descriptors (including 0.9's empty output envelope)
        # are not assistant responses.
        attrs.pop(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, None)
    if kind == "guardrails/guard":
        # Native Guardrails derives this from reasks, including local parses.
        # Count observed call spans, without treating absent usage as no call.
        if getattr(span, "links", ()):
            pass  # Linked stream summaries end after the original guard span.
        elif llm_calls is not None:
            if "number_of_llm_calls" in attrs:
                attrs["number_of_llm_calls"] = llm_calls
        else:
            attrs.pop("number_of_llm_calls", None)
    raw_keys = {
        OIAttributes.INPUT_VALUE,
        OIAttributes.OUTPUT_VALUE,
        OIAttributes.INPUT_MIME_TYPE,
        OIAttributes.OUTPUT_MIME_TYPE,
        OIAttributes.OPENINFERENCE_SPAN_KIND,
        OIAttributes.LLM_INVOCATION_PARAMETERS,
        OIAttributes.LLM_MODEL_NAME,
        OIAttributes.LLM_PROVIDER,
        OIAttributes.LLM_TOKEN_COUNT_PROMPT,
        OIAttributes.LLM_TOKEN_COUNT_COMPLETION,
        OIAttributes.LLM_TOKEN_COUNT_TOTAL,
        OIAttributes.LLM_FUNCTION_CALL,
        OIAttributes.LLM_PROMPT_TEMPLATE,
        OIAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
        OIAttributes.LLM_PROMPT_TEMPLATE_VERSION,
        "input",
        "args",
    }
    for key in list(attrs):
        if (
            key in raw_keys
            or key.startswith(
                (
                    OIAttributes.LLM_INPUT_MESSAGES + ".",
                    OIAttributes.LLM_OUTPUT_MESSAGES + ".",
                    "validator.validate.",
                    "validator.init.",
                )
            )
            or not include_content
            and (
                key
                in {
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    SpanAttributes.LLM_REQUEST_FUNCTIONS,
                }
                or key.startswith(
                    (
                        SpanAttributes.LLM_PROMPTS + ".",
                        SpanAttributes.LLM_COMPLETIONS + ".",
                    )
                )
            )
        ):
            del attrs[key]
    span._attributes = attrs
    return is_llm
