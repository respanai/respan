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
from typing import Any, Dict

from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from openinference.semconv.trace import SpanAttributes as OISpanAttributes

from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    OPENINFERENCE_SPAN_KIND,
    RESPAN_LOG_TYPE,
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

# OpenInference attributes (from upstream openinference-semantic-conventions)
OI_INPUT_VALUE = OISpanAttributes.INPUT_VALUE
OI_OUTPUT_VALUE = OISpanAttributes.OUTPUT_VALUE
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
OI_INPUT_MIME_TYPE = OISpanAttributes.INPUT_MIME_TYPE
OI_OUTPUT_MIME_TYPE = OISpanAttributes.OUTPUT_MIME_TYPE

# OI scalar attributes to remove after translation (their data has been
# mapped to Traceloop/OpenLLMetry equivalents).  openinference.span.kind is
# deliberately kept — the backend uses it for span-type identification.
_OI_KEYS_TO_REMOVE = frozenset({
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
})

# OI indexed-attribute prefixes to remove (llm.input_messages.*, llm.output_messages.*)
_OI_PREFIXES_TO_REMOVE = (
    "llm.input_messages.",
    "llm.output_messages.",
    "llm.tools.",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenLLMetry / Traceloop attribute keys not yet in respan_sdk.constants
# (wire-format keys used only for the reverse-Arize mapping)
# ---------------------------------------------------------------------------
# Token attributes (OpenLLMetry side)
_TL_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_TL_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_TL_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
_TL_USAGE_CACHE_READ = "llm.usage.cache_read_input_tokens"

# Invocation parameter attributes (OpenLLMetry side)
_TL_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
_TL_REQUEST_TOP_P = "gen_ai.request.top_p"
_TL_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
_TL_TOP_K = "llm.top_k"
_TL_STOP_SEQUENCES = "llm.chat.stop_sequences"
_TL_REPETITION_PENALTY = "llm.request.repetition_penalty"
_TL_FREQUENCY_PENALTY = "llm.frequency_penalty"
_TL_PRESENCE_PENALTY = "llm.presence_penalty"

# Provider (OpenLLMetry side)
_TL_PROVIDER_NAME = "gen_ai.provider.name"

# Tools (OpenLLMetry side)
_TL_REQUEST_FUNCTIONS = "llm.request.functions"

# ---------------------------------------------------------------------------
# Span kind mapping: OpenInference → Traceloop
# Exact reverse of Arize's _SPAN_KIND_MAPPING
# ---------------------------------------------------------------------------
_OI_KIND_TO_TRACELOOP: Dict[str, str] = {
    "CHAIN": "workflow",
    "LLM": "task",
    "TOOL": "tool",
    "AGENT": "agent",
    "RETRIEVER": "task",
    "EMBEDDING": "task",
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
    "temperature": _TL_REQUEST_TEMPERATURE,
    "top_p": _TL_REQUEST_TOP_P,
    "max_tokens": _TL_REQUEST_MAX_TOKENS,
    "max_output_tokens": _TL_REQUEST_MAX_TOKENS,
    "top_k": _TL_TOP_K,
    "stop_sequences": _TL_STOP_SEQUENCES,
    "stop": _TL_STOP_SEQUENCES,
    "repetition_penalty": _TL_REPETITION_PENALTY,
    "frequency_penalty": _TL_FREQUENCY_PENALTY,
    "presence_penalty": _TL_PRESENCE_PENALTY,
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

    for idx in sorted(buckets):
        raw = buckets[idx]
        target = f"{gen_ai_prefix}.{idx}"

        # message.role → gen_ai.prompt.N.role
        role = raw.get("message.role")
        if role:
            attrs[f"{target}.role"] = role

        # message.content → gen_ai.prompt.N.content
        content = raw.get("message.content")
        if content is not None:
            attrs[f"{target}.content"] = content

        # message.tool_calls.M.tool_call.function.name → gen_ai.prompt.N.tool_calls.M.function.name
        for field_key, field_val in raw.items():
            if field_key.startswith("message.tool_calls."):
                rest = field_key[len("message.tool_calls."):]
                parts = rest.split(".", 1)
                if parts[0].isdigit() and len(parts) > 1:
                    tc_idx = parts[0]
                    tc_field = parts[1]
                    # Strip "tool_call." prefix (OI nests under tool_call.*)
                    if tc_field.startswith("tool_call."):
                        tc_field = tc_field[len("tool_call."):]
                    attrs[f"{target}.tool_calls.{tc_idx}.{tc_field}"] = field_val

        # message.function_call_name → gen_ai.prompt.N.function_call.name
        func_name = raw.get("message.function_call_name")
        if func_name:
            attrs[f"{target}.function_call.name"] = func_name

        # message.function_call_arguments_json → gen_ai.prompt.N.function_call.arguments
        func_args = raw.get("message.function_call_arguments_json")
        if func_args:
            attrs[f"{target}.function_call.arguments"] = func_args

        # message.finish_reason → gen_ai.completion.N.finish_reason (completions only)
        finish_reason = raw.get("message.finish_reason")
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
    After translation the original OI attributes are removed so they don't
    leak into the backend's ``metadata`` blob as noise.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        attrs = getattr(span, "_attributes", None)
        if attrs is None:
            return

        oi_kind = attrs.get(OPENINFERENCE_SPAN_KIND)
        if not oi_kind:
            return

        oi_kind_upper = str(oi_kind).upper()
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

        # --- System / provider (reverse of Arize _extract_llm_provider_and_system) ---
        system = attrs.get(OI_LLM_SYSTEM)
        if system:
            attrs.setdefault(GEN_AI_SYSTEM, str(system).lower())

        provider = attrs.get(OI_LLM_PROVIDER)
        if provider:
            attrs.setdefault(_TL_PROVIDER_NAME, str(provider).lower())
            # Also set gen_ai.system if not already set (provider is a good fallback)
            attrs.setdefault(GEN_AI_SYSTEM, str(provider).lower())

        # --- Token counts (reverse of Arize token_count extraction) ---
        prompt_tokens = attrs.get(OI_LLM_TOKEN_COUNT_PROMPT)
        if prompt_tokens is not None:
            attrs.setdefault(LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
            attrs.setdefault(_TL_USAGE_INPUT_TOKENS, prompt_tokens)

        completion_tokens = attrs.get(OI_LLM_TOKEN_COUNT_COMPLETION)
        if completion_tokens is not None:
            attrs.setdefault(LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
            attrs.setdefault(_TL_USAGE_OUTPUT_TOKENS, completion_tokens)

        total_tokens = attrs.get(OI_LLM_TOKEN_COUNT_TOTAL)
        if total_tokens is not None:
            attrs.setdefault(_TL_USAGE_TOTAL_TOKENS, total_tokens)

        cache_read = attrs.get(OI_LLM_TOKEN_COUNT_CACHE_READ)
        if cache_read is not None:
            attrs.setdefault(_TL_USAGE_CACHE_READ, cache_read)

        # --- LLM-specific: messages, invocation params, tools ---
        if oi_kind_upper in _LLM_KINDS:
            self._translate_llm(attrs)

        # --- Remove translated OI attributes so they don't pollute metadata ---
        self._remove_oi_attrs(attrs)

    @staticmethod
    def _remove_oi_attrs(attrs: Dict[str, Any]) -> None:
        """Remove original OI attributes that have been translated."""
        keys_to_delete = [
            k for k in attrs
            if k in _OI_KEYS_TO_REMOVE
            or any(k.startswith(p) for p in _OI_PREFIXES_TO_REMOVE)
        ]
        for k in keys_to_delete:
            try:
                del attrs[k]
            except (KeyError, TypeError):
                pass

    def _translate_llm(self, attrs: Dict[str, Any]) -> None:
        """Extra translation for LLM/EMBEDDING spans."""
        # Mark as chat request type
        attrs.setdefault(LLM_REQUEST_TYPE, "chat")

        # --- Messages (reverse of Arize _collect_oi_messages) ---
        _oi_messages_to_openllmetry(attrs, "llm.input_messages.", "gen_ai.prompt")
        _oi_messages_to_openllmetry(attrs, "llm.output_messages.", "gen_ai.completion")

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
            attrs.setdefault(_TL_REQUEST_FUNCTIONS, tools_raw)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
