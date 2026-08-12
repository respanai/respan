from types import SimpleNamespace

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_NAME,
)
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_autogen._native_processor import (
    AUTOGEN_CORE_SCOPE_NAME,
    AUTOGEN_OPERATION_EXECUTE_TOOL,
    AUTOGEN_OPERATION_INVOKE_AGENT,
    AutoGenNativeSpanProcessor,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_AGENT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
)


def _make_span(attrs: dict, *, scope_name: str = AUTOGEN_CORE_SCOPE_NAME):
    return SimpleNamespace(
        name="test-span",
        _attributes=dict(attrs),
        instrumentation_scope=SimpleNamespace(name=scope_name),
    )


def test_processor_maps_native_invoke_agent_to_agent_log_type():
    span = _make_span({
        GEN_AI_OPERATION_NAME: AUTOGEN_OPERATION_INVOKE_AGENT,
        GEN_AI_SYSTEM: "autogen",
        GEN_AI_AGENT_NAME: "planner",
    })

    AutoGenNativeSpanProcessor().on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "planner"
    assert GEN_AI_SYSTEM not in span._attributes
    assert SpanAttributes.LLM_REQUEST_TYPE not in span._attributes


def test_processor_maps_native_execute_tool_to_tool_log_type():
    span = _make_span({
        GEN_AI_OPERATION_NAME: AUTOGEN_OPERATION_EXECUTE_TOOL,
        GEN_AI_SYSTEM: "autogen",
        GEN_AI_TOOL_NAME: "estimate_latency",
        SpanAttributes.LLM_REQUEST_TYPE: "chat",
    })

    AutoGenNativeSpanProcessor().on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert (
        span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME]
        == "estimate_latency"
    )
    assert GEN_AI_SYSTEM not in span._attributes
    assert SpanAttributes.LLM_REQUEST_TYPE not in span._attributes


def test_processor_handles_immutable_ended_span_attributes():
    """Regression: a real *ended* span exposes ``_attributes`` as a frozen
    ``BoundedAttributes``. Mutating it in place raises ``TypeError`` (which
    previously propagated out of the user's ``agent.run()``). The processor
    must copy + reassign instead."""
    from opentelemetry.attributes import BoundedAttributes

    frozen = BoundedAttributes(
        attributes={
            GEN_AI_OPERATION_NAME: AUTOGEN_OPERATION_INVOKE_AGENT,
            GEN_AI_SYSTEM: "autogen",
            GEN_AI_AGENT_NAME: "planner",
        },
        immutable=True,
    )
    span = SimpleNamespace(
        name="test-span",
        _attributes=frozen,
        instrumentation_scope=SimpleNamespace(name=AUTOGEN_CORE_SCOPE_NAME),
    )

    # Must not raise TypeError.
    AutoGenNativeSpanProcessor().on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "planner"
    assert GEN_AI_SYSTEM not in span._attributes


def test_processor_ignores_non_autogen_core_scope():
    span = _make_span(
        {
            GEN_AI_OPERATION_NAME: AUTOGEN_OPERATION_INVOKE_AGENT,
            GEN_AI_SYSTEM: "autogen",
        },
        scope_name="openinference.instrumentation.autogen_agentchat",
    )
    original_attrs = dict(span._attributes)

    AutoGenNativeSpanProcessor().on_end(span)

    assert span._attributes == original_attrs
