"""Translate OpenInference spans → OpenLLMetry/Traceloop format.

This SpanProcessor converts spans produced by OpenInference instrumentors
(Haystack, CrewAI, LangChain, Google ADK, etc.) into the Traceloop/OpenLLMetry
semantic conventions that the Respan backend expects.

The mapping is the exact reverse of Arize's ``openinference-instrumentation-openllmetry``
package (``_span_processor.py``) which converts OpenLLMetry → OpenInference.

Arize mapping (OpenLLMetry → OI):
    traceloop.span.kind          → openinference.span.kind
    traceloop.entity.input       → input.value + input.mime_type
    traceloop.entity.output      → output.value + output.mime_type
    gen_ai.prompt.N.*            → llm.input_messages.N.message.*
    gen_ai.completion.N.*        → llm.output_messages.N.message.*
    gen_ai.usage.input_tokens    → llm.token_count.prompt
    gen_ai.usage.output_tokens   → llm.token_count.completion
    llm.usage.total_tokens       → llm.token_count.total
    llm.usage.cache_read_input_tokens → llm.token_count.prompt_details.cache_read
    gen_ai.request.model         → llm.invocation_parameters.model
    gen_ai.request.temperature   → llm.invocation_parameters.temperature
    gen_ai.request.top_p         → llm.invocation_parameters.top_p
    llm.top_k                    → llm.invocation_parameters.top_k
    llm.chat.stop_sequences      → llm.invocation_parameters.stop_sequences
    llm.request.repetition_penalty → llm.invocation_parameters.repetition_penalty
    llm.frequency_penalty        → llm.invocation_parameters.frequency_penalty
    llm.presence_penalty         → llm.invocation_parameters.presence_penalty
    llm.request.functions        → llm.tools
    gen_ai.system                → llm.system
    gen_ai.provider.name         → llm.provider

This module reverses every mapping above.

References:
- Arize source: openinference-instrumentation-openllmetry/_span_processor.py
- OpenInference semconv: openinference.semconv.trace.SpanAttributes
- OpenLLMetry semconv: opentelemetry.semconv_ai.SpanAttributes
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List

from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes as TLSpanAttributes
from openinference.semconv.trace import SpanAttributes as OISpanAttributes

from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)

# Traceloop attributes (from upstream opentelemetry-semantic-conventions-ai)
TRACELOOP_SPAN_KIND = TLSpanAttributes.TRACELOOP_SPAN_KIND
TRACELOOP_ENTITY_NAME = TLSpanAttributes.TRACELOOP_ENTITY_NAME
TRACELOOP_ENTITY_INPUT = TLSpanAttributes.TRACELOOP_ENTITY_INPUT
TRACELOOP_ENTITY_OUTPUT = TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT
TRACELOOP_ENTITY_PATH = TLSpanAttributes.TRACELOOP_ENTITY_PATH
GEN_AI_PROMPT_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
GEN_AI_COMPLETION_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."
TL_LLM_REQUEST_TEMPERATURE = TLSpanAttributes.LLM_REQUEST_TEMPERATURE
TL_LLM_REQUEST_TOP_P = TLSpanAttributes.LLM_REQUEST_TOP_P
TL_LLM_REQUEST_MAX_TOKENS = TLSpanAttributes.LLM_REQUEST_MAX_TOKENS
TL_LLM_REQUEST_FUNCTIONS = TLSpanAttributes.LLM_REQUEST_FUNCTIONS
TL_LLM_REQUEST_REPETITION_PENALTY = TLSpanAttributes.LLM_REQUEST_REPETITION_PENALTY
TL_LLM_USAGE_TOTAL_TOKENS = TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS
TL_LLM_TOP_K = TLSpanAttributes.LLM_TOP_K
TL_LLM_CHAT_STOP_SEQUENCES = TLSpanAttributes.LLM_CHAT_STOP_SEQUENCES
TL_LLM_FREQUENCY_PENALTY = TLSpanAttributes.LLM_FREQUENCY_PENALTY
TL_LLM_PRESENCE_PENALTY = TLSpanAttributes.LLM_PRESENCE_PENALTY

# OpenInference attributes (from upstream openinference-semantic-conventions)
OI_INPUT_VALUE = OISpanAttributes.INPUT_VALUE
OI_SPAN_KIND = OISpanAttributes.OPENINFERENCE_SPAN_KIND
OI_INPUT_MIME_TYPE = OISpanAttributes.INPUT_MIME_TYPE
OI_OUTPUT_VALUE = OISpanAttributes.OUTPUT_VALUE
OI_OUTPUT_MIME_TYPE = OISpanAttributes.OUTPUT_MIME_TYPE
OI_LLM_MODEL_NAME = OISpanAttributes.LLM_MODEL_NAME
OI_LLM_PROVIDER = OISpanAttributes.LLM_PROVIDER
OI_LLM_SYSTEM = OISpanAttributes.LLM_SYSTEM
OI_LLM_INVOCATION_PARAMETERS = OISpanAttributes.LLM_INVOCATION_PARAMETERS
OI_LLM_TOKEN_COUNT_PROMPT = OISpanAttributes.LLM_TOKEN_COUNT_PROMPT
OI_LLM_TOKEN_COUNT_COMPLETION = OISpanAttributes.LLM_TOKEN_COUNT_COMPLETION
OI_LLM_TOKEN_COUNT_TOTAL = OISpanAttributes.LLM_TOKEN_COUNT_TOTAL
OI_LLM_TOKEN_COUNT_CACHE_READ = OISpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
OI_LLM_TOOLS = OISpanAttributes.LLM_TOOLS
OI_AGENT_NAME = OISpanAttributes.AGENT_NAME

_GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens"
_RESPAN_MODEL = "model"
_RESPAN_PROMPT_TOKENS = "prompt_tokens"
_RESPAN_COMPLETION_TOKENS = "completion_tokens"
_RESPAN_TOTAL_REQUEST_TOKENS = "total_request_tokens"
_RESPAN_TOOLS = "tools"
_RESPAN_TOOL_CALLS = "tool_calls"

_OI_INPUT_MESSAGES_PREFIX = "llm.input_messages."
_OI_OUTPUT_MESSAGES_PREFIX = "llm.output_messages."
_OI_TOKEN_COUNT_PREFIX = "llm.token_count."
_OI_TOOLS_PREFIX = "llm.tools."
_OI_MESSAGE_ROLE = "message.role"
_OI_MESSAGE_CONTENT = "message.content"
_OI_MESSAGE_CONTENT_PREFIX = "message.content."
_OI_MESSAGE_TOOL_CALLS_PREFIX = "message.tool_calls."
_OI_MESSAGE_FUNCTION_CALL_NAME = "message.function_call_name"
_OI_MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = "message.function_call_arguments_json"
_OI_MESSAGE_FINISH_REASON = "message.finish_reason"
_OI_TOOL_PREFIX = "tool."
_OI_TOOL_JSON_SCHEMA = "tool.json_schema"
_OI_TOOL_CALL_PREFIX = "tool_call."

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Span kind mapping: OpenInference → Traceloop
# Exact reverse of Arize's _SPAN_KIND_MAPPING
# ---------------------------------------------------------------------------
_OI_KIND_TO_TRACELOOP: Dict[str, str] = {
    "CHAIN": "workflow",
    "LLM": LOG_TYPE_CHAT,
    "TOOL": "tool",
    "AGENT": "agent",
    "RETRIEVER": "task",
    "EMBEDDING": LOG_TYPE_EMBEDDING,
    "RERANKER": "task",
    "GUARDRAIL": "task",
    "EVALUATOR": "task",
    "PROMPT": "task",
    "UNKNOWN": "task",
}

_OI_KIND_TO_LOG_TYPE: Dict[str, str] = {
    "CHAIN": LOG_TYPE_WORKFLOW,
    "LLM": LOG_TYPE_CHAT,
    "TOOL": LOG_TYPE_TOOL,
    "AGENT": LOG_TYPE_AGENT,
    "RETRIEVER": LOG_TYPE_TASK,
    "EMBEDDING": LOG_TYPE_EMBEDDING,
    "RERANKER": LOG_TYPE_TASK,
    "GUARDRAIL": LOG_TYPE_GUARDRAIL,
    "EVALUATOR": LOG_TYPE_TASK,
    "PROMPT": LOG_TYPE_TASK,
    "UNKNOWN": LOG_TYPE_TASK,
}

# OI span kinds that represent LLM calls
_LLM_KINDS = {"LLM", "EMBEDDING"}

# Invocation parameter key → OpenLLMetry target attribute
_INVOCATION_PARAM_MAP: Dict[str, str] = {
    "model": LLM_REQUEST_MODEL,
    "temperature": TL_LLM_REQUEST_TEMPERATURE,
    "top_p": TL_LLM_REQUEST_TOP_P,
    "max_tokens": TL_LLM_REQUEST_MAX_TOKENS,
    "max_output_tokens": TL_LLM_REQUEST_MAX_TOKENS,
    "top_k": TL_LLM_TOP_K,
    "stop_sequences": TL_LLM_CHAT_STOP_SEQUENCES,
    "stop": TL_LLM_CHAT_STOP_SEQUENCES,
    "repetition_penalty": TL_LLM_REQUEST_REPETITION_PENALTY,
    "frequency_penalty": TL_LLM_FREQUENCY_PENALTY,
    "presence_penalty": TL_LLM_PRESENCE_PENALTY,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_str(value: Any) -> str:
    """Ensure value is a JSON string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _parse_json(value: Any) -> Any:
    """Parse a JSON string, or return value as-is if not a string."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _collect_oi_message_buckets(
    attrs: Dict[str, Any],
    oi_prefix: str,
) -> Dict[int, Dict[str, Any]]:
    """Group indexed OI message attributes by message index."""
    buckets: Dict[int, Dict[str, Any]] = defaultdict(dict)

    for key, val in attrs.items():
        if not key.startswith(oi_prefix):
            continue
        rest = key[len(oi_prefix):]
        parts = rest.split(".", 1)
        if not parts[0].isdigit():
            continue
        idx = int(parts[0])
        field = parts[1] if len(parts) > 1 else ""
        buckets[idx][field] = val

    return buckets


def _normalize_structured_list(value: Any) -> List[Dict[str, Any]] | None:
    """Parse a JSON string or structured object into a list of dicts."""
    parsed = _parse_json(value)
    if isinstance(parsed, list):
        normalized = [item for item in parsed if isinstance(item, dict)]
        return normalized or None
    if isinstance(parsed, dict):
        return [parsed]
    return None


def _set_nested_value(target: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """Assign a nested dict field from a dotted path."""
    parts = dotted_path.split(".")
    cursor = target
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = value


def _canonicalize_for_signature(value: Any) -> Any:
    """Recursively canonicalize structured values for deterministic signatures."""
    if isinstance(value, dict):
        return {
            key: _canonicalize_for_signature(value[key])
            for key in sorted(value)
        }
    if isinstance(value, list):
        return [_canonicalize_for_signature(item) for item in value]
    return value


def _normalize_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool-call payloads to one canonical internal shape."""
    normalized: Dict[str, Any] = {}

    tool_call_id = tool_call.get("id")
    if tool_call_id is not None:
        normalized["id"] = tool_call_id

    function = tool_call.get("function")
    normalized_function: Dict[str, Any] = {}
    if isinstance(function, dict):
        function_name = function.get("name")
        if function_name is not None:
            normalized_function["name"] = function_name
        function_arguments = function.get("arguments")
        if function_arguments is not None:
            normalized_function["arguments"] = function_arguments

    tool_type = tool_call.get("type")
    if tool_type is not None:
        normalized["type"] = tool_type
    elif normalized_function:
        normalized["type"] = "function"

    if normalized_function:
        normalized["function"] = normalized_function

    return normalized


def _tool_call_signature(tool_call: Dict[str, Any]) -> str:
    """Return a deterministic semantic signature for a tool call."""
    normalized = _normalize_tool_call(tool_call)
    function = normalized.get("function")
    if isinstance(function, dict) and "arguments" in function:
        function["arguments"] = _parse_json(function["arguments"])
    return json.dumps(
        _canonicalize_for_signature(normalized),
        default=str,
        separators=(",", ":"),
    )


def _extract_tool_calls_from_buckets(
    buckets: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]] | None:
    """Rebuild direct tool_calls payloads from indexed OI message attrs."""
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for idx in sorted(buckets):
        raw = buckets[idx]
        tool_call_buckets: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for field_key, field_val in raw.items():
            if not field_key.startswith(_OI_MESSAGE_TOOL_CALLS_PREFIX):
                continue
            rest = field_key[len(_OI_MESSAGE_TOOL_CALLS_PREFIX):]
            parts = rest.split(".", 1)
            if not parts[0].isdigit() or len(parts) == 1:
                continue
            tc_idx = int(parts[0])
            tc_field = parts[1]
            if tc_field.startswith(_OI_TOOL_CALL_PREFIX):
                tc_field = tc_field[len(_OI_TOOL_CALL_PREFIX):]
            tool_call_buckets[tc_idx][tc_field] = field_val

        for tc_idx in sorted(tool_call_buckets):
            tool_call: Dict[str, Any] = {}
            for field_key, field_val in tool_call_buckets[tc_idx].items():
                _set_nested_value(tool_call, field_key, field_val)
            tool_call = _normalize_tool_call(tool_call)
            if not tool_call:
                continue
            signature = _tool_call_signature(tool_call)
            if signature not in seen:
                seen.add(signature)
                result.append(tool_call)

        func_name = raw.get(_OI_MESSAGE_FUNCTION_CALL_NAME)
        func_args = raw.get(_OI_MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON)
        if func_name is None and func_args is None:
            continue

        legacy_tool_call: Dict[str, Any] = {
            "type": "function",
            "function": {},
        }
        if func_name is not None:
            legacy_tool_call["function"]["name"] = func_name
        if func_args is not None:
            legacy_tool_call["function"]["arguments"] = func_args
        legacy_tool_call = _normalize_tool_call(legacy_tool_call)
        if not legacy_tool_call:
            continue
        signature = _tool_call_signature(legacy_tool_call)
        if signature not in seen:
            seen.add(signature)
            result.append(legacy_tool_call)

    return result or None


def _extract_message_content(raw: Dict[str, Any]) -> Any:
    """Rebuild message content from scalar or indexed OpenInference fields."""
    content = raw.get(_OI_MESSAGE_CONTENT)
    if content is not None:
        return content

    indexed_content: List[tuple[int, Any]] = []
    for field_key, field_val in raw.items():
        if not field_key.startswith(_OI_MESSAGE_CONTENT_PREFIX):
            continue
        idx_str = field_key[len(_OI_MESSAGE_CONTENT_PREFIX):]
        if not idx_str.isdigit():
            continue
        indexed_content.append((int(idx_str), field_val))

    if not indexed_content:
        return None

    ordered_values = [value for _, value in sorted(indexed_content)]
    if len(ordered_values) == 1:
        return ordered_values[0]
    if all(isinstance(value, str) for value in ordered_values):
        return "\n".join(ordered_values)
    return ordered_values


def _extract_tool_calls(
    attrs: Dict[str, Any],
    oi_prefixes: List[str],
) -> List[Dict[str, Any]] | None:
    """Collect unique tool calls across one or more indexed OI message groups."""
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for prefix in oi_prefixes:
        tool_calls = _extract_tool_calls_from_buckets(
            _collect_oi_message_buckets(attrs=attrs, oi_prefix=prefix)
        )
        if not tool_calls:
            continue
        for tool_call in tool_calls:
            signature = _tool_call_signature(tool_call)
            if signature in seen:
                continue
            seen.add(signature)
            result.append(tool_call)

    return result or None


def _extract_tools_from_indexed_attrs(
    attrs: Dict[str, Any],
) -> List[Dict[str, Any]] | None:
    """Rebuild tool definitions from indexed OI llm.tools.N.tool.* attributes."""
    buckets = _collect_oi_message_buckets(attrs=attrs, oi_prefix=_OI_TOOLS_PREFIX)
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for idx in sorted(buckets):
        raw = buckets[idx]
        tool: Dict[str, Any] = {}

        json_schema = raw.get(_OI_TOOL_JSON_SCHEMA)
        if json_schema is not None:
            parsed_json_schema = _parse_json(json_schema)
            if isinstance(parsed_json_schema, dict):
                tool.update(parsed_json_schema)
            else:
                tool["json_schema"] = parsed_json_schema

        for field_key, field_val in raw.items():
            if field_key == _OI_TOOL_JSON_SCHEMA:
                continue
            normalized_key = (
                field_key[len(_OI_TOOL_PREFIX):]
                if field_key.startswith(_OI_TOOL_PREFIX)
                else field_key
            )
            _set_nested_value(
                target=tool,
                dotted_path=normalized_key,
                value=_parse_json(field_val),
            )

        if "tool" in tool and len(tool) == 1 and isinstance(tool["tool"], dict):
            tool = tool["tool"]
        if not tool:
            continue

        signature = _safe_json_str(tool)
        if signature not in seen:
            seen.add(signature)
            result.append(tool)

    return result or None


def _has_equivalent_modern_tool_call(
    raw: Dict[str, Any],
    legacy_name: Any,
    legacy_arguments: Any,
) -> bool:
    """Return True when a legacy function_call matches an existing tool_call."""
    if legacy_name is None and legacy_arguments is None:
        return False

    legacy_tool_call: Dict[str, Any] = {
        "type": "function",
        "function": {},
    }
    if legacy_name is not None:
        legacy_tool_call["function"]["name"] = legacy_name
    if legacy_arguments is not None:
        legacy_tool_call["function"]["arguments"] = legacy_arguments

    legacy_tool_call = _normalize_tool_call(legacy_tool_call)
    if not legacy_tool_call:
        return False

    legacy_signature = _tool_call_signature(legacy_tool_call)
    modern_tool_call_buckets: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for field_key, field_val in raw.items():
        if not field_key.startswith(_OI_MESSAGE_TOOL_CALLS_PREFIX):
            continue
        rest = field_key[len(_OI_MESSAGE_TOOL_CALLS_PREFIX):]
        parts = rest.split(".", 1)
        if not parts[0].isdigit() or len(parts) == 1:
            continue
        tc_idx = int(parts[0])
        tc_field = parts[1]
        if tc_field.startswith(_OI_TOOL_CALL_PREFIX):
            tc_field = tc_field[len(_OI_TOOL_CALL_PREFIX):]
        modern_tool_call_buckets[tc_idx][tc_field] = field_val

    for tool_call_bucket in modern_tool_call_buckets.values():
        tool_call: Dict[str, Any] = {}
        for field_key, field_val in tool_call_bucket.items():
            _set_nested_value(tool_call, field_key, field_val)
        tool_call = _normalize_tool_call(tool_call)
        if not tool_call:
            continue
        if _tool_call_signature(tool_call) == legacy_signature:
            return True

    return False


def _oi_messages_to_openllmetry(
    attrs: Dict[str, Any],
    oi_prefix: str,
    gen_ai_prefix: str,
) -> None:
    """Convert OI indexed messages to OpenLLMetry gen_ai.prompt.N / gen_ai.completion.N format.

    Reverse of Arize's ``_collect_oi_messages()`` which reads gen_ai.prompt.N.*
    and builds OI Message objects stored as llm.input_messages.N.message.*.

    OI format (source):
        llm.input_messages.0.message.role = "user"
        llm.input_messages.0.message.content = "hello"
        llm.input_messages.0.message.tool_calls.0.tool_call.function.name = "get_weather"
        llm.input_messages.0.message.tool_calls.0.tool_call.function.arguments = '{"city":"NYC"}'
        llm.input_messages.0.message.function_call_name = "get_weather"
        llm.input_messages.0.message.function_call_arguments_json = '{"city":"NYC"}'

    OpenLLMetry format (target):
        gen_ai.prompt.0.role = "user"
        gen_ai.prompt.0.content = "hello"
        gen_ai.prompt.0.tool_calls.0.function.name = "get_weather"
        gen_ai.prompt.0.tool_calls.0.function.arguments = '{"city":"NYC"}'
    """
    buckets = _collect_oi_message_buckets(attrs=attrs, oi_prefix=oi_prefix)

    for idx in sorted(buckets):
        raw = buckets[idx]
        target = f"{gen_ai_prefix}.{idx}"

        # message.role → gen_ai.prompt.N.role
        role = raw.get(_OI_MESSAGE_ROLE)
        if role:
            attrs[f"{target}.role"] = role

        # message.content / message.content.K → gen_ai.prompt.N.content
        content = _extract_message_content(raw)
        if content is not None:
            attrs[f"{target}.content"] = content

        # message.tool_calls.M.tool_call.function.name → gen_ai.prompt.N.tool_calls.M.function.name
        for field_key, field_val in raw.items():
            if field_key.startswith(_OI_MESSAGE_TOOL_CALLS_PREFIX):
                rest = field_key[len(_OI_MESSAGE_TOOL_CALLS_PREFIX):]
                parts = rest.split(".", 1)
                if parts[0].isdigit() and len(parts) > 1:
                    tc_idx = parts[0]
                    tc_field = parts[1]
                    # Strip "tool_call." prefix (OI nests under tool_call.*)
                    if tc_field.startswith(_OI_TOOL_CALL_PREFIX):
                        tc_field = tc_field[len(_OI_TOOL_CALL_PREFIX):]
                    attrs[f"{target}.tool_calls.{tc_idx}.{tc_field}"] = field_val

        structured_tool_calls = _extract_tool_calls_from_buckets({idx: raw})
        if structured_tool_calls:
            attrs.setdefault(f"{target}.tool_calls", structured_tool_calls)

        # message.function_call_name → gen_ai.prompt.N.function_call.name
        func_name = raw.get(_OI_MESSAGE_FUNCTION_CALL_NAME)
        func_args = raw.get(_OI_MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON)
        has_matching_tool_call = _has_equivalent_modern_tool_call(raw, func_name, func_args)
        if func_name and not has_matching_tool_call:
            attrs[f"{target}.function_call.name"] = func_name

        # message.function_call_arguments_json → gen_ai.prompt.N.function_call.arguments
        if func_args and not has_matching_tool_call:
            attrs[f"{target}.function_call.arguments"] = func_args

        # message.finish_reason → gen_ai.completion.N.finish_reason (completions only)
        finish_reason = raw.get(_OI_MESSAGE_FINISH_REASON)
        if finish_reason:
            attrs[f"{target}.finish_reason"] = finish_reason


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class OpenInferenceTranslator(SpanProcessor):
    """SpanProcessor that translates OpenInference attributes to OpenLLMetry/Traceloop.

    Detects OI spans by the presence of ``openinference.span.kind`` and
    enriches them with the Traceloop attributes the Respan backend expects.

    All mappings are the exact reverse of Arize's openinference-instrumentation-openllmetry.
    After translation, redundant raw OpenInference attributes are removed so they
    do not leak into passthrough metadata / custom properties.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return
        attrs = dict(original_attrs)

        oi_kind = attrs.get(OI_SPAN_KIND)
        if not oi_kind:
            return

        oi_kind_upper = str(oi_kind).upper()
        is_llm_kind = oi_kind_upper in _LLM_KINDS
        is_llm_only = oi_kind_upper == "LLM"
        logger.debug("[OI→TL] Translating %s span: %s", oi_kind_upper, span.name)

        # --- Span kind (reverse of Arize _SPAN_KIND_MAPPING) ---
        traceloop_kind = _OI_KIND_TO_TRACELOOP.get(oi_kind_upper, "task")
        attrs.setdefault(TRACELOOP_SPAN_KIND, traceloop_kind)
        attrs.setdefault(RESPAN_LOG_TYPE, _OI_KIND_TO_LOG_TYPE.get(oi_kind_upper, "task"))

        # --- Entity name ---
        entity_name = attrs.get(OI_AGENT_NAME) or span.name
        attrs.setdefault(TRACELOOP_ENTITY_NAME, entity_name)

        # --- Entity path (empty = root candidate) ---
        attrs.setdefault(TRACELOOP_ENTITY_PATH, "")

        # --- Input / output (reverse of Arize _map_generic_span) ---
        input_val = attrs.get(OI_INPUT_VALUE)
        if input_val is not None:
            attrs.setdefault(TRACELOOP_ENTITY_INPUT, _safe_json_str(input_val))

        output_val = attrs.get(OI_OUTPUT_VALUE)
        if output_val is not None:
            attrs.setdefault(TRACELOOP_ENTITY_OUTPUT, _safe_json_str(output_val))

        # --- Model name (reverse: llm.model_name → gen_ai.request.model) ---
        model = attrs.get(OI_LLM_MODEL_NAME)
        if model:
            attrs.setdefault(LLM_REQUEST_MODEL, model)
            if is_llm_kind:
                attrs.setdefault(_RESPAN_MODEL, model)

        # --- System / provider (reverse of Arize _extract_llm_provider_and_system) ---
        system = attrs.get(OI_LLM_SYSTEM)
        if system:
            attrs.setdefault(GEN_AI_SYSTEM, str(system).lower())

        provider = attrs.get(OI_LLM_PROVIDER)
        if provider:
            attrs.setdefault(_GEN_AI_PROVIDER_NAME, str(provider).lower())
            # Also set gen_ai.system if not already set (provider is a good fallback)
            attrs.setdefault(GEN_AI_SYSTEM, str(provider).lower())

        # --- Token counts (reverse of Arize token_count extraction) ---
        prompt_tokens = attrs.get(OI_LLM_TOKEN_COUNT_PROMPT)
        if prompt_tokens is not None:
            attrs.setdefault(LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
            attrs.setdefault(_GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
            if is_llm_kind:
                attrs.setdefault(_RESPAN_PROMPT_TOKENS, prompt_tokens)

        completion_tokens = attrs.get(OI_LLM_TOKEN_COUNT_COMPLETION)
        if completion_tokens is not None:
            attrs.setdefault(LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
            attrs.setdefault(_GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
            if is_llm_kind:
                attrs.setdefault(_RESPAN_COMPLETION_TOKENS, completion_tokens)

        total_tokens = attrs.get(OI_LLM_TOKEN_COUNT_TOTAL)
        if total_tokens is not None:
            attrs.setdefault(TL_LLM_USAGE_TOTAL_TOKENS, total_tokens)
            if is_llm_kind:
                attrs.setdefault(_RESPAN_TOTAL_REQUEST_TOKENS, total_tokens)
        elif is_llm_kind and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            attrs.setdefault(
                _RESPAN_TOTAL_REQUEST_TOKENS,
                (prompt_tokens or 0) + (completion_tokens or 0),
            )

        cache_read = attrs.get(OI_LLM_TOKEN_COUNT_CACHE_READ)
        if cache_read is not None:
            attrs.setdefault(_LLM_USAGE_CACHE_READ_INPUT_TOKENS, cache_read)

        direct_tools = _normalize_structured_list(attrs.get(OI_LLM_TOOLS))
        if direct_tools is None:
            direct_tools = _extract_tools_from_indexed_attrs(attrs)
        if direct_tools is not None:
            attrs.setdefault(RESPAN_SPAN_TOOLS, _safe_json_str(direct_tools))
            if is_llm_only:
                attrs.setdefault(_RESPAN_TOOLS, direct_tools)

        direct_tool_calls = _extract_tool_calls(
            attrs=attrs,
            oi_prefixes=[_OI_OUTPUT_MESSAGES_PREFIX],
        )
        if direct_tool_calls is not None:
            attrs.setdefault(RESPAN_SPAN_TOOL_CALLS, _safe_json_str(direct_tool_calls))
            if is_llm_only:
                attrs.setdefault(_RESPAN_TOOL_CALLS, direct_tool_calls)

        # --- LLM-specific: messages, invocation params, tools ---
        if oi_kind_upper in _LLM_KINDS:
            self._translate_llm(attrs=attrs, oi_kind_upper=oi_kind_upper)

        self._remove_redundant_oi_attrs(attrs)
        span._attributes = attrs

    def _translate_llm(self, attrs: Dict[str, Any], oi_kind_upper: str) -> None:
        """Extra translation for LLM/EMBEDDING spans."""
        request_type = (
            LLMRequestTypeValues.EMBEDDING.value
            if oi_kind_upper == "EMBEDDING"
            else LLMRequestTypeValues.CHAT.value
        )
        attrs.setdefault(LLM_REQUEST_TYPE, request_type)

        # --- Messages (reverse of Arize _collect_oi_messages) ---
        _oi_messages_to_openllmetry(attrs, _OI_INPUT_MESSAGES_PREFIX, GEN_AI_PROMPT_PREFIX.rstrip("."))
        _oi_messages_to_openllmetry(attrs, _OI_OUTPUT_MESSAGES_PREFIX, GEN_AI_COMPLETION_PREFIX.rstrip("."))

        # --- Invocation parameters (reverse of Arize invocation_params extraction) ---
        # OI stores all params as a single JSON string; OpenLLMetry uses individual attributes
        inv_params_raw = attrs.get(OI_LLM_INVOCATION_PARAMETERS)
        if inv_params_raw:
            params = _parse_json(inv_params_raw)
            if isinstance(params, dict):
                for key, val in params.items():
                    target_attr = _INVOCATION_PARAM_MAP.get(key)
                    if target_attr:
                        attrs.setdefault(target_attr, val)

        # --- Tools (reverse of Arize _handle_tool_list) ---
        # OI: llm.tools = JSON string of tool definitions
        # OpenLLMetry: llm.request.functions = JSON string of tool definitions
        tools_raw = attrs.get(OI_LLM_TOOLS)
        if tools_raw:
            attrs.setdefault(TL_LLM_REQUEST_FUNCTIONS, tools_raw)
        elif attrs.get(RESPAN_SPAN_TOOLS):
            attrs.setdefault(TL_LLM_REQUEST_FUNCTIONS, attrs[RESPAN_SPAN_TOOLS])

    @staticmethod
    def _remove_redundant_oi_attrs(attrs: Dict[str, Any]) -> None:
        """Remove noisy raw OpenInference attrs while keeping promoted fields."""
        keys_to_remove = {
            OI_SPAN_KIND,
            OI_INPUT_VALUE,
            OI_INPUT_MIME_TYPE,
            OI_OUTPUT_VALUE,
            OI_OUTPUT_MIME_TYPE,
            OI_LLM_MODEL_NAME,
            OI_LLM_PROVIDER,
            OI_LLM_SYSTEM,
            OI_LLM_INVOCATION_PARAMETERS,
            OI_LLM_TOKEN_COUNT_PROMPT,
            OI_LLM_TOKEN_COUNT_COMPLETION,
            OI_LLM_TOKEN_COUNT_TOTAL,
            OI_LLM_TOKEN_COUNT_CACHE_READ,
            OI_LLM_TOOLS,
            OI_AGENT_NAME,
        }
        prefixes_to_remove = (
            _OI_INPUT_MESSAGES_PREFIX,
            _OI_OUTPUT_MESSAGES_PREFIX,
            _OI_TOKEN_COUNT_PREFIX,
            _OI_TOOLS_PREFIX,
        )

        for key in keys_to_remove:
            attrs.pop(key, None)

        for key in list(attrs.keys()):
            if any(key.startswith(prefix) for prefix in prefixes_to_remove):
                attrs.pop(key, None)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
