"""Contract tests for OpenInference-to-Respan translation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from opentelemetry.attributes import BoundedAttributes
from respan_instrumentation_openinference._serialization import (
    MAX_ATTRIBUTE_CHARS,
    MAX_LABEL_CHARS,
)
from respan_instrumentation_openinference._translator import OpenInferenceTranslator


def _make_span(attrs: dict, name: str = "test-span") -> SimpleNamespace:
    return SimpleNamespace(_attributes=dict(attrs), name=name)


@pytest.fixture
def translator() -> OpenInferenceTranslator:
    return OpenInferenceTranslator()


def test_non_openinference_span_is_untouched(translator):
    span = _make_span({"custom.keep": "value"})
    translator._translate(span, True)
    assert span._attributes == {"custom.keep": "value"}


@pytest.mark.parametrize(
    ("kind", "log_type"),
    [
        ("CHAIN", "workflow"),
        ("LLM", "chat"),
        ("TOOL", "tool"),
        ("AGENT", "agent"),
        ("RETRIEVER", "task"),
        ("EMBEDDING", "embedding"),
        ("GUARDRAIL", "guardrail"),
    ],
)
def test_kind_maps_only_to_auto_span_log_type(translator, kind, log_type):
    span = _make_span({"openinference.span.kind": kind})
    translator._translate(span, True)

    assert span._attributes["respan.entity.log_type"] == log_type
    assert "traceloop.span.kind" not in span._attributes
    assert "openinference.span.kind" not in span._attributes


def test_entity_input_output_are_valid_json_and_custom_attrs_survive(translator):
    span = _make_span(
        {
            "openinference.span.kind": "CHAIN",
            "input.value": "hello world",
            "output.value": {"answer": 42},
            "custom.request_id": "request-123",
        }
    )
    translator._translate(span, True)

    assert json.loads(span._attributes["traceloop.entity.input"]) == "hello world"
    assert json.loads(span._attributes["traceloop.entity.output"]) == {"answer": 42}
    assert span._attributes["custom.request_id"] == "request-123"


def test_model_provider_and_real_usage_are_canonical_only(translator):
    span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "llm.model_name": "gpt-4.1-mini",
            "llm.system": "OpenAI",
            "llm.provider": "Azure",
            "llm.token_count.prompt": 11,
            "llm.token_count.completion": 5,
            "llm.token_count.total": 16,
            "llm.token_count.prompt_details.cache_read": 3,
            "model": "forbidden",
            "prompt_tokens": 999,
        }
    )
    translator._translate(span, True)
    attrs = span._attributes

    assert attrs["gen_ai.request.model"] == "gpt-4.1-mini"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.provider.name"] == "azure"
    assert attrs["gen_ai.usage.prompt_tokens"] == 11
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.completion_tokens"] == 5
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["llm.usage.total_tokens"] == 16
    assert attrs["gen_ai.usage.cache_read_input_tokens"] == 3
    assert "llm.usage.cache_read_input_tokens" not in attrs
    assert "model" not in attrs
    assert "prompt_tokens" not in attrs
    assert "completion_tokens" not in attrs
    assert "total_request_tokens" not in attrs


def test_provider_and_system_fall_back_to_each_other(translator):
    provider_only = _make_span(
        {"openinference.span.kind": "LLM", "llm.provider": "Anthropic"}
    )
    translator._translate(provider_only, True)
    assert provider_only._attributes["gen_ai.system"] == "anthropic"
    assert provider_only._attributes["gen_ai.provider.name"] == "anthropic"

    system_only = _make_span({"openinference.span.kind": "LLM", "llm.system": "OpenAI"})
    translator._translate(system_only, True)
    assert system_only._attributes["gen_ai.system"] == "openai"
    assert system_only._attributes["gen_ai.provider.name"] == "openai"


def test_messages_reconstruct_canonical_content_and_entity_json(translator):
    span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "Hello",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content.0": "First",
            "llm.output_messages.0.message.content.1": "Second",
            "llm.output_messages.0.message.finish_reason": "stop",
        }
    )
    translator._translate(span, True)
    attrs = span._attributes

    assert attrs["gen_ai.prompt.0.role"] == "user"
    assert attrs["gen_ai.prompt.0.content"] == "Hello"
    assert attrs["gen_ai.completion.0.role"] == "assistant"
    assert attrs["gen_ai.completion.0.content"] == "First\nSecond"
    assert attrs["gen_ai.completion.0.finish_reason"] == "stop"
    assert json.loads(attrs["traceloop.entity.input"]) == {
        "messages": [{"content": "Hello", "role": "user"}]
    }
    assert json.loads(attrs["traceloop.entity.output"]) == {
        "messages": [
            {
                "content": "First\nSecond",
                "finish_reason": "stop",
                "role": "assistant",
            }
        ]
    }


def test_tools_and_current_turn_calls_are_json_and_alias_free(translator):
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "llm.tools": json.dumps([tool, tool]),
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call-1",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "lookup_weather",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"city":"Tokyo"}',
            "llm.output_messages.0.message.function_call_name": "lookup_weather",
            "llm.output_messages.0.message.function_call_arguments_json": '{"city":"Tokyo"}',
        }
    )
    translator._translate(span, True)
    attrs = span._attributes

    assert json.loads(attrs["llm.request.functions"]) == [tool]
    calls = json.loads(attrs["gen_ai.completion.0.tool_calls"])
    assert calls == [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"Tokyo"}',
            },
        }
    ]
    for alias in (
        "respan.span.tools",
        "respan.span.tool_calls",
        "tools",
        "tool_calls",
    ):
        assert alias not in attrs
    assert not any(key.startswith("llm.output_messages.") for key in attrs)


def test_history_calls_stay_on_prompt_and_not_current_turn(translator):
    span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "llm.input_messages.0.message.role": "assistant",
            "llm.input_messages.0.message.tool_calls.0.tool_call.id": "history-1",
            "llm.input_messages.0.message.tool_calls.0.tool_call.function.name": "lookup",
            "llm.input_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"q":"old"}',
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "No tool this turn",
        }
    )
    translator._translate(span, True)

    history = json.loads(span._attributes["gen_ai.prompt.0.tool_calls"])
    assert history[0]["id"] == "history-1"
    assert "gen_ai.completion.0.tool_calls" not in span._attributes


def test_indexed_tools_from_bounded_attributes_are_promoted(translator):
    tool = {
        "name": "lookup_weather",
        "description": "Get weather",
        "input_schema": {"type": "object"},
    }
    span = _make_span({})
    span._attributes = BoundedAttributes(
        maxlen=64,
        attributes={
            "openinference.span.kind": "LLM",
            "llm.tools.0.tool.json_schema": json.dumps(tool),
        },
        immutable=False,
    )
    translator._translate(span, True)

    assert isinstance(span._attributes, dict)
    assert json.loads(span._attributes["llm.request.functions"]) == [tool]
    assert "llm.tools.0.tool.json_schema" not in span._attributes


def test_tool_span_has_contract_input_and_output(translator):
    span = _make_span(
        {
            "openinference.span.kind": "TOOL",
            "tool.name": "lookup_weather",
            "input.value": '{"city":"Tokyo"}',
            "output.value": '{"temperature_c":22}',
        },
        name="tool-call",
    )
    translator._translate(span, True)

    assert span._attributes["traceloop.entity.name"] == "lookup_weather"
    assert json.loads(span._attributes["traceloop.entity.input"]) == {
        "name": "lookup_weather",
        "arguments": {"city": "Tokyo"},
    }
    assert json.loads(span._attributes["traceloop.entity.output"]) == {
        "temperature_c": 22
    }
    assert "tool.name" not in span._attributes


def test_embedding_indexed_fields_map_model_input_and_vector(translator):
    span = _make_span(
        {
            "openinference.span.kind": "EMBEDDING",
            "embedding.model_name": "text-embedding-3-small",
            "embedding.embeddings.0.embedding.text": "hello",
            "embedding.embeddings.0.embedding.vector": [0.1, 0.2, 0.3],
        }
    )
    translator._translate(span, True)

    assert span._attributes["llm.request.type"] == "embedding"
    assert span._attributes["gen_ai.request.model"] == "text-embedding-3-small"
    assert json.loads(span._attributes["traceloop.entity.input"]) == "hello"
    assert json.loads(span._attributes["traceloop.entity.output"]) == [0.1, 0.2, 0.3]
    assert not any(key.startswith("embedding.") for key in span._attributes)


def test_invocation_parameters_use_canonical_attributes(translator):
    span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "llm.invocation_parameters": json.dumps(
                {
                    "model": "claude-sonnet",
                    "temperature": 0.2,
                    "stop": ["END"],
                    "stream": True,
                }
            ),
        }
    )
    translator._translate(span, True)

    assert span._attributes["gen_ai.request.model"] == "claude-sonnet"
    assert span._attributes["gen_ai.request.temperature"] == 0.2
    assert span._attributes["llm.chat.stop_sequences"] == ("END",)
    assert span._attributes["llm.is_streaming"] is True
    assert "llm.invocation_parameters" not in span._attributes


def test_sensitive_nested_values_are_redacted_and_output_is_bounded(translator):
    span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "input.value": json.dumps(
                {
                    "authorization": "Bearer top-secret-token",
                    "nested": {"api_key": "sk-supersecret123", "prompt": "safe"},
                    "large": "x" * (MAX_ATTRIBUTE_CHARS * 2),
                }
            ),
            "output.value": "Bearer another-secret-token",
        }
    )
    translator._translate(span, True)

    entity_input = span._attributes["traceloop.entity.input"]
    entity_output = span._attributes["traceloop.entity.output"]
    assert len(entity_input) <= MAX_ATTRIBUTE_CHARS
    assert len(entity_output) <= MAX_ATTRIBUTE_CHARS
    assert "top-secret-token" not in entity_input
    assert "supersecret123" not in entity_input
    assert "another-secret-token" not in entity_output
    json.loads(entity_input)
    json.loads(entity_output)


def test_preexisting_canonical_content_is_also_bounded_and_redacted(translator):
    span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "traceloop.entity.input": json.dumps(
                {"password": "do-not-export", "body": "x" * 32_000}
            ),
            "gen_ai.completion.0.content": "Bearer hidden-credential-value",
            "tool.description": "raw OpenInference field",
        }
    )
    translator._translate(span, True)

    entity_input = span._attributes["traceloop.entity.input"]
    assert len(entity_input) <= MAX_ATTRIBUTE_CHARS
    assert "do-not-export" not in entity_input
    assert (
        "hidden-credential-value" not in span._attributes["gen_ai.completion.0.content"]
    )
    assert "tool.description" not in span._attributes


def test_preexisting_canonical_labels_are_bounded_and_redacted(translator):
    span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "traceloop.entity.name": (
                "Bearer entity-name-secret " + ("n" * MAX_LABEL_CHARS * 2)
            ),
            "traceloop.entity.path": (
                "password=cleartext-path-secret " + ("p" * MAX_LABEL_CHARS * 2)
            ),
            "traceloop.entity.input": "password=cleartext-body-secret",
            "gen_ai.request.model": (
                "sk-model-secret-12345678 " + ("m" * MAX_LABEL_CHARS * 2)
            ),
            "gen_ai.prompt.0.role": (
                "Bearer role-secret-value " + ("r" * MAX_LABEL_CHARS * 2)
            ),
        }
    )
    translator._translate(span, True)
    attrs = span._attributes

    for key in (
        "traceloop.entity.name",
        "traceloop.entity.path",
        "gen_ai.request.model",
        "gen_ai.prompt.0.role",
    ):
        assert len(attrs[key]) <= MAX_LABEL_CHARS
        assert "[TRUNCATED]" in attrs[key]
    exported = json.dumps(attrs, sort_keys=True)
    for secret in (
        "entity-name-secret",
        "cleartext-path-secret",
        "cleartext-body-secret",
        "model-secret-12345678",
        "role-secret-value",
    ):
        assert secret not in exported
    assert json.loads(attrs["traceloop.entity.input"]) == "password=[REDACTED]"


def test_serializer_never_calls_arbitrary_stringification(translator):
    class Explosive:
        def __str__(self):
            raise AssertionError("must not stringify arbitrary values")

        def __repr__(self):
            raise AssertionError("must not repr arbitrary values")

    span = _make_span(
        {
            "openinference.span.kind": "CHAIN",
            "input.value": {"value": Explosive()},
        }
    )
    translator._translate(span, True)

    assert json.loads(span._attributes["traceloop.entity.input"]) == {
        "value": "[UNSUPPORTED:Explosive]"
    }
