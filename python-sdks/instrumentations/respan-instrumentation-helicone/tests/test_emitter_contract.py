from __future__ import annotations

import json

import pytest
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_PROVIDER_NAME,
    GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_helicone import _emitter, _serialization
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.otlp_constants import (
    OTLP_ATTR_KEY,
    OTLP_ATTRIBUTES_KEY,
    OTLP_NAME_KEY,
    OTLP_PARENT_SPAN_ID_KEY,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_INTERNAL_SPAN_NAME_DETAIL,
    RESPAN_INTERNAL_SPAN_NAME_KIND,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_PROPERTIES,
    RESPAN_SESSION_ID,
)
from respan_tracing.exporters.respan import _span_to_otlp_json


@pytest.fixture
def captured(monkeypatch):
    spans = []
    monkeypatch.setattr(
        _emitter, "inject_span", lambda span: spans.append(span) or True
    )
    return spans


def emit(captured, *, request, response, provider=None, error=None, capture=True):
    assert _emitter.emit_helicone_log(
        provider=provider,
        request=request,
        response=response,
        options={"start_time": 10.0, "end_time": 11.0},
        capture_content=capture,
        error=error,
    )
    assert len(captured) == 1
    return captured[0]


def test_openai_chat_contract_tools_usage_and_no_aliases(captured):
    span = emit(
        captured,
        provider="openai",
        request={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "weather"}}],
            "api_key": "request-secret",
        },
        response={
            "model": "gpt-test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Tokyo"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
    )
    attrs = span.attributes

    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[RESPAN_LOG_METHOD] == LogMethodChoices.TRACING_INTEGRATION.value
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert attrs[GEN_AI_PROVIDER_NAME] == "openai"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-test"
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 10
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 5
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 15
    assert attrs[HTTP_RESPONSE_STATUS_CODE] == 200
    assert "status_code" not in attrs
    assert (
        json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"]["name"]
        == "weather"
    )
    calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert calls[0]["id"] == "call-1"
    assert "request-secret" not in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    for alias in (
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "respan.span.tools",
        "respan.span.tool_calls",
    ):
        assert alias not in attrs
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs


def test_request_and_response_models_remain_distinct(captured):
    span = emit(
        captured,
        provider="router",
        request={"model": "requested-model", "messages": []},
        response={"model": "resolved-model", "choices": []},
    )

    assert span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "requested-model"
    assert span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "resolved-model"


def test_anthropic_shape_maps_content_and_modern_usage(captured):
    span = emit(
        captured,
        provider="anthropic",
        request={
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hello"}],
        },
        response={
            "model": "claude-test",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello back"}],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        },
    )
    attrs = span.attributes

    assert attrs[SpanAttributes.LLM_SYSTEM] == "anthropic"
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 7
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    content = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"])
    assert content == [{"type": "text", "text": "hello back"}]


def test_roles_are_normalized_to_the_canonical_contract(captured):
    span = emit(
        captured,
        provider="custom",
        request={
            "model": "role-model",
            "messages": [
                {"role": "USER", "content": "one"},
                {"role": "human", "content": "two"},
                {"role": "ai", "content": "three"},
                {"role": "bot", "content": "four"},
                {"role": "function", "content": "five"},
                {"role": "developer", "content": "six"},
                {"role": "unknown-role", "content": "seven"},
                {"content": "eight"},
            ],
        },
        response={
            "choices": [{"message": {"role": "BOT", "content": "normalized response"}}]
        },
    )
    attrs = span.attributes

    assert [
        attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.role"] for index in range(8)
    ] == [
        "user",
        "user",
        "assistant",
        "assistant",
        "tool",
        "system",
        "user",
        "user",
    ]
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"


def test_primitive_custom_messages_receive_user_roles(captured):
    span = emit(
        captured,
        provider="custom",
        request={"model": "primitive-messages", "messages": ["hello", 42]},
        response={"choices": []},
    )
    attrs = span.attributes

    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "hello"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == "42"


def test_anthropic_tool_use_maps_current_turn_tool_calls(captured):
    span = emit(
        captured,
        provider="anthropic",
        request={
            "model": "claude-test",
            "messages": [
                {"role": "user", "content": "weather"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Previous lookup."},
                        {
                            "type": "tool_use",
                            "id": "toolu-history",
                            "name": "weather",
                            "input": {"city": "Paris"},
                        },
                    ],
                },
                {"role": "user", "content": "Now check Tokyo."},
            ],
            "tools": [{"name": "weather", "input_schema": {"type": "object"}}],
        },
        response={
            "model": "claude-test",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "toolu-1",
                    "name": "weather",
                    "input": {"city": "Tokyo"},
                },
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 4,
                "cache_read_input_tokens": 8,
            },
        },
    )
    attrs = span.attributes

    current_content = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"])
    assert current_content == [
        {"type": "text", "text": "Checking."},
        {
            "type": "tool_use",
            "id": "toolu-1",
            "name": "weather",
            "input": {"city": "Tokyo"},
        },
    ]
    historical_content = json.loads(attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"])
    assert historical_content[1]["id"] == "toolu-history"
    historical = json.loads(attrs[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls"])
    assert historical[0]["id"] == "toolu-history"
    assert historical[0]["function"]["arguments"] == '{"city":"Paris"}'
    calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert calls == [
        {
            "id": "toolu-1",
            "type": "function",
            "function": {
                "name": "weather",
                "arguments": '{"city":"Tokyo"}',
            },
        }
    ]
    assert calls[0]["id"] != historical[0]["id"]
    assert attrs[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 8
    assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 8


def test_anthropic_system_and_structured_blocks_are_lossless(captured):
    system_blocks = [
        {"type": "text", "text": "Use tools safely."},
        {"type": "text", "text": "Keep structure."},
    ]
    user_blocks = [
        {"type": "text", "text": "Inspect Tokyo."},
        {"type": "image", "source": {"type": "base64", "data": "safe-image"}},
    ]
    span = emit(
        captured,
        provider="anthropic",
        request={
            "model": "claude-structured",
            "system": system_blocks,
            "messages": [{"role": "user", "content": user_blocks}],
        },
        response={
            "model": "claude-structured-response",
            "role": "assistant",
            "content": [{"type": "text", "text": "Structured response."}],
        },
    )
    attrs = span.attributes

    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert json.loads(attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"]) == system_blocks
    assert json.loads(attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"]) == user_blocks
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "claude-structured"
    assert attrs[SpanAttributes.LLM_RESPONSE_MODEL] == "claude-structured-response"


def test_stream_lines_are_aggregated_with_terminal_usage(captured):
    chunks = [
        {
            "model": "stream-model",
            "choices": [{"delta": {"role": "assistant", "content": "Helicone "}}],
        },
        {
            "model": "stream-model",
            "choices": [{"delta": {"content": "streams."}}],
        },
        {
            "model": "stream-model",
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    ]
    span = emit(
        captured,
        request={
            "model": "stream-model",
            "messages": [{"role": "user", "content": "stream"}],
            "stream": True,
        },
        response="\n".join(json.dumps(chunk) for chunk in chunks),
    )
    attrs = span.attributes

    assert attrs[SpanAttributes.GEN_AI_IS_STREAMING] is True
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Helicone streams."
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 6


def test_documented_log_request_chunks_shape_and_ttft(captured):
    response = {
        "chunks": [
            {
                "model": "chunk-model",
                "choices": [{"delta": {"content": "Chunked "}}],
            },
            {
                "model": "chunk-model",
                "choices": [{"delta": {"content": "response."}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        ],
        "time_to_first_token_ms": 18.25,
    }
    span = emit(
        captured,
        provider="openai",
        request={"model": "chunk-model", "messages": []},
        response=response,
    )
    attrs = span.attributes

    assert attrs[SpanAttributes.GEN_AI_IS_STREAMING] is True
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == ("Chunked response.")
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 5
    assert (
        json.loads(attrs[RESPAN_METADATA])["helicone"]["time_to_first_token_ms"]
        == 18.25
    )
    assert attrs[GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] == 0.01825


def test_anthropic_streaming_events_preserve_blocks_tools_and_usage(captured):
    events = [
        {
            "type": "message_start",
            "message": {
                "model": "claude-stream-response",
                "usage": {"input_tokens": 9, "cache_read_input_tokens": 4},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Streaming."},
        },
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu-stream",
                "name": "lookup",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"topic":"tracing"}',
            },
        },
        {"type": "message_delta", "usage": {"output_tokens": 3}},
    ]
    span = emit(
        captured,
        provider="anthropic",
        request={"model": "claude-stream-request", "messages": [], "stream": True},
        response={"chunks": events, "time_to_first_token_ms": 7.5},
    )
    attrs = span.attributes

    blocks = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"])
    assert blocks[0] == {"type": "text", "text": "Streaming."}
    assert blocks[1]["input"] == {"topic": "tracing"}
    calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert calls[0]["id"] == "toolu-stream"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "claude-stream-request"
    assert attrs[SpanAttributes.LLM_RESPONSE_MODEL] == "claude-stream-response"
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 9
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 4


def test_google_custom_shape_contents_candidates_usage_and_models(captured):
    span = emit(
        captured,
        provider="google",
        request={
            "model": "gemini-request",
            "systemInstruction": {"parts": [{"text": "Be concise."}]},
            "contents": [
                {"role": "user", "parts": [{"text": "Use the weather tool."}]}
            ],
            "tools": [
                {"functionDeclarations": [{"name": "weather", "parameters": {}}]}
            ],
        },
        response={
            "modelVersion": "gemini-response",
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Checking."},
                            {
                                "functionCall": {
                                    "name": "weather",
                                    "args": {"city": "Tokyo"},
                                }
                            },
                        ],
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 6,
                "candidatesTokenCount": 2,
                "totalTokenCount": 8,
                "cachedContentTokenCount": 3,
            },
        },
    )
    attrs = span.attributes

    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-request"
    assert attrs[SpanAttributes.LLM_RESPONSE_MODEL] == "gemini-response"
    assert json.loads(attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"]) == [
        {"text": "Be concise."}
    ]
    assert json.loads(attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"]) == [
        {"text": "Use the weather tool."}
    ]
    output_parts = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"])
    assert output_parts[1]["functionCall"]["name"] == "weather"
    calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert calls[0]["function"]["arguments"] == '{"city":"Tokyo"}'
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 6
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 2
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 8
    assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3


def test_embedding_contract_preserves_vector(captured):
    vector = [float(index) / 100 for index in range(512)]
    span = emit(
        captured,
        provider="openai",
        request={"_type": "embedding", "model": "embed-test", "input": ["hello"]},
        response={
            "model": "embed-test",
            "data": [{"index": 0, "embedding": vector}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        },
    )

    assert span.attributes[RESPAN_LOG_TYPE] == "embedding"
    input_value = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert input_value == ["hello"]
    assert output == vector
    assert "llm.embeddings.0" not in span.attributes


def test_embedding_vector_larger_than_legacy_cap_is_not_truncated(captured):
    vector = [float(index) for index in range(20_000)]
    span = emit(
        captured,
        request={"_type": "embedding", "model": "large-vector", "input": "value"},
        response={"data": [{"embedding": vector}]},
    )

    output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert len(output) == 20_000
    assert output == vector
    assert "llm.embeddings.0" not in span.attributes


def test_embedding_vectors_preserve_non_finite_diagnostics_and_batches(captured):
    vector = [float(index) for index in range(300)]
    vector[150] = float("nan")
    batch = [vector, [float(index) for index in range(300, 600)]]
    span = emit(
        captured,
        request={
            "_type": "embedding",
            "model": "diagnostic-vector",
            "input": ["a", "b"],
        },
        response={"data": [{"embedding": item} for item in batch]},
    )

    output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert len(output) == 2
    assert len(output[0]) == 300
    assert output[0][150] is None
    assert output[1] == batch[1]


def test_embedding_vector_batch_larger_than_collection_cap_is_not_truncated(captured):
    batch = [[float(index), float(index + 1)] for index in range(200)]
    span = emit(
        captured,
        request={"_type": "embedding", "model": "batch-vector", "input": ["x"] * 200},
        response={"data": [{"embedding": item} for item in batch]},
    )

    output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output == batch


def test_indexed_prompt_attributes_are_bounded(captured):
    span = emit(
        captured,
        provider="custom",
        request={
            "model": "large-history",
            "messages": [
                {"role": "user", "content": f"message-{index}"}
                for index in range(10_000)
            ],
        },
        response={"choices": []},
    )

    role_keys = [
        key
        for key in span.attributes
        if key.startswith(f"{SpanAttributes.LLM_PROMPTS}.") and key.endswith(".role")
    ]
    assert len(role_keys) == _serialization.MAX_COLLECTION_ITEMS
    assert (
        f"{SpanAttributes.LLM_PROMPTS}.{_serialization.MAX_COLLECTION_ITEMS}.role"
        not in span.attributes
    )


def test_prompt_request_is_text_not_chat(captured):
    span = emit(
        captured,
        provider="custom-provider",
        request={"model": "completion-model", "prompt": "Complete this sentence"},
        response={"choices": [{"text": " with an observable result."}]},
    )

    assert span.attributes[RESPAN_LOG_TYPE] == "text"
    assert span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == (
        "Complete this sentence"
    )
    assert span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        " with an observable result."
    )


def test_missing_provider_uses_custom_identity_not_helicone(captured):
    span = emit(
        captured,
        request={"model": "private-model", "messages": []},
        response={"choices": []},
    )

    assert span.attributes[SpanAttributes.LLM_SYSTEM] == "custom"
    assert span.attributes[GEN_AI_PROVIDER_NAME] == "custom"


def test_model_and_tool_identity_suffixes_are_concise(captured):
    long_model = "model-" + ("m" * 1_000)
    llm_span = emit(
        captured,
        request={"model": long_model, "messages": []},
        response={"choices": []},
    )
    assert len(llm_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]) == 128
    assert len(llm_span.name) <= len("llm.") + 128

    captured.clear()
    tool_span = emit(
        captured,
        request={"_type": "tool", "toolName": "t" * 1_000, "input": {}},
        response={"ok": True},
    )
    assert len(tool_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME]) == 128
    assert len(tool_span.attributes[RESPAN_INTERNAL_SPAN_NAME_DETAIL]) == 128


@pytest.mark.parametrize(
    ("request_payload", "expected_log_type", "expected_name"),
    [
        (
            {"_type": "tool", "toolName": "lookup", "input": {"city": "Tokyo"}},
            "tool",
            "lookup",
        ),
        (
            {"_type": "vector_db", "operation": "search", "vector": [0.1, 0.2]},
            "task",
            "vector_db.search",
        ),
        (
            {"_type": "data", "name": "database_query", "query": "select 1"},
            "task",
            "database_query",
        ),
    ],
)
def test_custom_types_map_without_fake_tool_call_aliases(
    captured, request_payload, expected_log_type, expected_name
):
    span = emit(captured, request=request_payload, response={"status": "success"})
    attrs = span.attributes

    assert attrs[RESPAN_LOG_TYPE] == expected_log_type
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == expected_name
    assert "tool_calls" not in attrs
    assert "respan.span.tool_calls" not in attrs
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs


def test_tool_hints_drive_semantic_name_and_strip_in_both_styles(captured, monkeypatch):
    snapshot = _emitter.HeliconeEmissionContext(
        trace_id="1" * 32,
        parent_id="2" * 16,
        propagated_attributes={},
    )
    assert _emitter.emit_helicone_log(
        provider=None,
        request={
            "_type": "tool",
            "toolName": "get_weather",
            "input": {"city": "Tokyo"},
        },
        response={"temperature_f": 72},
        options={"start_time": 10.0, "end_time": 11.0},
        capture_content=True,
        context_snapshot=snapshot,
    )
    span = captured[0]
    assert span.attributes[RESPAN_INTERNAL_SPAN_NAME_KIND] == "tool"
    assert span.attributes[RESPAN_INTERNAL_SPAN_NAME_DETAIL] == "get_weather"

    for style, expected_name in (("semantic", "tool.get_weather"), ("legacy", "tool")):
        monkeypatch.setenv("RESPAN_SPAN_NAME_STYLE", style)
        exported = _span_to_otlp_json(span)
        assert exported[OTLP_NAME_KEY] == expected_name
        assert exported[OTLP_PARENT_SPAN_ID_KEY] == "2" * 16
        exported_keys = {
            attribute[OTLP_ATTR_KEY] for attribute in exported[OTLP_ATTRIBUTES_KEY]
        }
        assert RESPAN_INTERNAL_SPAN_NAME_KIND not in exported_keys
        assert RESPAN_INTERNAL_SPAN_NAME_DETAIL not in exported_keys


def test_capture_content_false_keeps_identity_usage_and_error(captured):
    span = emit(
        captured,
        provider="openai",
        request={
            "model": "private",
            "messages": [{"role": "user", "content": "secret"}],
        },
        response={
            "model": "private",
            "choices": [{"message": {"role": "assistant", "content": "private"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        error=RuntimeError("safe failure"),
        capture=False,
    )
    attrs = span.attributes

    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "private"
    assert attrs[ERROR_MESSAGE] == "safe failure"
    assert attrs[ERROR_TYPE] == "RuntimeError"
    assert attrs[HTTP_RESPONSE_STATUS_CODE] == 500
    assert "status_code" not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    assert span.status.is_ok is False


def test_error_payload_cannot_be_downgraded_by_success_status(captured):
    span = emit(
        captured,
        request={"model": "failed", "messages": []},
        response={
            "status_code": 200,
            "error": {"message": "provider failed"},
        },
    )

    assert span.status.is_ok is False
    assert span.attributes[ERROR_MESSAGE] == "provider failed"
    assert span.attributes[ERROR_TYPE] == "HeliconeError"
    assert span.attributes[HTTP_RESPONSE_STATUS_CODE] == 500
    assert "status_code" not in span.attributes


def test_response_error_type_uses_provider_status_type(captured):
    span = emit(
        captured,
        request={"model": "failed", "messages": []},
        response={
            "status": "error",
            "type": "quota_exceeded",
            "message": "provider quota exceeded",
        },
    )

    assert span.attributes[ERROR_MESSAGE] == "provider quota exceeded"
    assert span.attributes[ERROR_TYPE] == "quota_exceeded"
    assert span.attributes[HTTP_RESPONSE_STATUS_CODE] == 500


def test_non_finite_timings_fall_back_without_dropping_span(captured):
    assert _emitter.emit_helicone_log(
        provider="openai",
        request={"model": "timing-model", "messages": []},
        response={"choices": []},
        options={"start_time": float("inf"), "end_time": float("inf")},
        capture_content=True,
    )
    assert len(captured) == 1
    assert captured[0].start_time < 10**30
    assert captured[0].end_time < 10**30


def test_safe_helicone_headers_and_ttft_use_single_canonical_metadata(captured):
    assert _emitter.emit_helicone_log(
        provider="openai",
        request={"model": "model", "messages": []},
        response={"choices": []},
        options={
            "start_time": 10.0,
            "end_time": 11.0,
            "time_to_first_token_ms": 42.5,
            "additional_headers": {
                "Helicone-Session-Id": "session-1",
                "Helicone-User-Id": "user-1",
                "Helicone-Session-Name": "support-chat",
                "Helicone-Property-Environment": "test",
                "Helicone-Property-Api-Key": "must-not-appear",
                "Authorization": "Bearer transport-secret",
            },
        },
        constructor_headers={
            "Helicone-Session-Name": "constructor-session",
            "Helicone-Property-Tier": "pro",
            "Authorization": "Bearer constructor-secret",
        },
        capture_content=True,
    )
    attrs = captured[0].attributes

    assert attrs[RESPAN_SESSION_ID] == "session-1"
    assert attrs[RESPAN_CUSTOMER_PARAMS_ID] == "user-1"
    metadata = json.loads(attrs[RESPAN_METADATA])
    assert metadata == {
        "helicone": {
            "session_name": "support-chat",
            "properties": {"environment": "test", "tier": "pro"},
            "time_to_first_token_ms": 42.5,
        }
    }
    assert not any(key.startswith(f"{RESPAN_METADATA}.") for key in attrs)
    assert attrs[GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] == 0.0425
    combined = "\n".join(str(value) for value in attrs.values())
    assert "must-not-appear" not in combined
    assert "transport-secret" not in combined
    assert "constructor-secret" not in combined


def test_builder_context_snapshot_preserves_parent_and_propagated_attrs(captured):
    snapshot = _emitter.HeliconeEmissionContext(
        trace_id="1" * 32,
        parent_id="2" * 16,
        propagated_attributes={
            "respan.metadata.snapshot": "creation-context",
            RESPAN_PROPERTIES: {"nested": {"x": 1}},
            "example.primitive_array": [1, 2, 3],
        },
    )
    assert _emitter.emit_helicone_log(
        provider="openai",
        request={"model": "delayed", "messages": []},
        response={"choices": []},
        options={"start_time": 10.0, "end_time": 11.0},
        capture_content=True,
        is_streaming=True,
        context_snapshot=snapshot,
    )
    span = captured[0]

    assert f"{span.context.trace_id:032x}" == "1" * 32
    assert span.parent is not None
    assert f"{span.parent.span_id:016x}" == "2" * 16
    assert json.loads(span.attributes[RESPAN_METADATA]) == {
        "snapshot": "creation-context"
    }
    assert "respan.metadata.snapshot" not in span.attributes
    assert json.loads(span.attributes[RESPAN_PROPERTIES]) == {"nested": {"x": 1}}
    assert span.attributes["example.primitive_array"] == (1, 2, 3)
    assert span.attributes[SpanAttributes.GEN_AI_IS_STREAMING] is True


def test_capture_emission_context_freezes_mutable_propagation(captured, monkeypatch):
    source = {
        RESPAN_PROPERTIES: {"nested": {"x": 1}},
        "example.primitive_array": [1, 2, 3],
    }
    monkeypatch.setattr(_emitter, "_current_ids", lambda: ("1" * 32, "2" * 16))
    monkeypatch.setattr(_emitter, "read_propagated_attributes", lambda: source)
    snapshot = _emitter.capture_emission_context()

    source[RESPAN_PROPERTIES]["nested"]["x"] = 99
    source["example.primitive_array"].append(4)
    assert _emitter.emit_helicone_log(
        provider="openai",
        request={"model": "delayed", "messages": []},
        response={"choices": []},
        options={"start_time": 10.0, "end_time": 11.0},
        capture_content=True,
        context_snapshot=snapshot,
    )
    attrs = captured[0].attributes

    assert json.loads(attrs[RESPAN_PROPERTIES]) == {"nested": {"x": 1}}
    assert attrs["example.primitive_array"] == (1, 2, 3)


def test_hostile_objects_and_secret_strings_never_escape(captured):
    class Hostile:
        def __repr__(self):
            raise AssertionError("repr must not run")

        def __str__(self):
            raise AssertionError("str must not run")

    span = emit(
        captured,
        request={
            "model": "safe-model",
            "messages": [{"role": "user", "content": "token=visible-secret"}],
            "max_tokens": 100,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "token_count": 15,
            "tokenizer": "safe-tokenizer",
            "promptTokens": 11,
            "completionTokens": 6,
            "tokenCount": 17,
            "nested": {
                "api_key": "nested-api-secret",
                "refresh_token": "nested-refresh-secret",
                "accessToken": "nested-access-secret",
                "refreshToken": "nested-camel-refresh-secret",
                "idToken": "nested-id-secret",
                "authToken": "nested-auth-secret",
                "bearerToken": "nested-bearer-secret",
                "clientSecret": "nested-client-secret",
                "privateKey": "nested-private-secret",
                "sessionToken": "nested-session-secret",
                "setCookie": "nested-cookie-secret",
                "secret_key": "nested-secret-key",
                "aws_secret_access_key": "nested-aws-secret",
                "awsAccessKeyId": "nested-aws-access-id",
                "apiToken": "nested-api-token",
                "oauth_token": "nested-oauth-token",
                "github_token": "nested-github-token",
                "session_cookie": "nested-session-cookie",
                "aws_credentials": "nested-aws-credentials",
                "password_hash": "nested-password-hash",
                "max_tokens": 22,
            },
            "payload": Hostile(),
            "Authorization": "Bearer credential-secret",
        },
        response={"value": Hostile()},
    )
    combined = "\n".join(str(value) for value in span.attributes.values())

    assert "visible-secret" not in combined
    assert "credential-secret" not in combined
    assert "nested-api-secret" not in combined
    assert "nested-refresh-secret" not in combined
    assert "nested-access-secret" not in combined
    assert "nested-camel-refresh-secret" not in combined
    assert "nested-id-secret" not in combined
    assert "nested-auth-secret" not in combined
    assert "nested-bearer-secret" not in combined
    assert "nested-client-secret" not in combined
    assert "nested-private-secret" not in combined
    assert "nested-session-secret" not in combined
    assert "nested-cookie-secret" not in combined
    assert "nested-secret-key" not in combined
    assert "nested-aws-secret" not in combined
    assert "nested-aws-access-id" not in combined
    assert "nested-api-token" not in combined
    assert "nested-oauth-token" not in combined
    assert "nested-github-token" not in combined
    assert "nested-session-cookie" not in combined
    assert "nested-aws-credentials" not in combined
    assert "nested-password-hash" not in combined
    assert "Hostile" in combined
    input_payload = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert input_payload["max_tokens"] == 100
    assert input_payload["prompt_tokens"] == 10
    assert input_payload["completion_tokens"] == 5
    assert input_payload["token_count"] == 15
    assert input_payload["tokenizer"] == "safe-tokenizer"
    assert input_payload["promptTokens"] == 11
    assert input_payload["completionTokens"] == 6
    assert input_payload["tokenCount"] == 17
    assert input_payload["nested"]["max_tokens"] == 22
    assert input_payload["nested"]["api_key"] == "<redacted>"
    assert input_payload["nested"]["refresh_token"] == "<redacted>"
    for key in (
        "accessToken",
        "refreshToken",
        "idToken",
        "authToken",
        "bearerToken",
        "clientSecret",
        "privateKey",
        "sessionToken",
        "setCookie",
        "secret_key",
        "aws_secret_access_key",
        "awsAccessKeyId",
        "apiToken",
        "oauth_token",
        "github_token",
        "session_cookie",
        "aws_credentials",
        "password_hash",
    ):
        assert input_payload["nested"][key] == "<redacted>"


def test_error_json_secrets_are_redacted_without_hiding_token_counts(captured):
    span = emit(
        captured,
        request={"model": "error-model", "messages": []},
        response={},
        error=RuntimeError(
            'failure {"api_key":"error secret with spaces, and comma",'
            '"accessToken":"camel secret; with separator",'
            '"token_count":12,"promptTokens":13,"max_tokens":20} '
            "secret_key=bare-secret,remaining\n"
            "Authorization: Basic dXNlcjpwYXNz\n"
            "authorization=ApiKey actual-secret"
        ),
    )

    message = span.attributes[ERROR_MESSAGE]
    assert span.attributes[ERROR_TYPE] == "RuntimeError"
    assert "error secret" not in message
    assert "and comma" not in message
    assert "camel secret" not in message
    assert "with separator" not in message
    assert "bare-secret" not in message
    assert "remaining" not in message
    assert "dXNlcjpwYXNz" not in message
    assert "actual-secret" not in message
    assert '"token_count":12' in message
    assert '"promptTokens":13' in message
    assert '"max_tokens":20' in message


def test_large_vector_mode_bounds_unrelated_collections():
    vector = [float(index) for index in range(512)]
    encoded = _serialization.json_dumps(
        {
            "vector": vector,
            "unrelated_numeric": [float(index) for index in range(20_000)],
            "unrelated": ["x" * 1_000 for _ in range(500)],
        },
        preserve_large_vectors=True,
    )
    payload = json.loads(encoded)

    assert payload["vector"] == vector
    assert len(payload["unrelated"]) < 500
    assert payload["unrelated"][-1] == {"truncated": True}
    assert len(payload["unrelated_numeric"]) < 20_000
    assert payload["unrelated_numeric"][-1] == {"truncated": True}
    assert len(encoded.encode("utf-8")) <= _serialization.MAX_ATTRIBUTE_BYTES


def test_structured_bearer_redaction_is_valid_json_and_idempotent():
    encoded = _serialization.json_dumps(
        [
            {"type": "text", "text": "Bearer secret-token"},
            {"type": "text", "text": "keep this content"},
        ]
    )

    assert json.loads(encoded) == [
        {"text": "Bearer <redacted>", "type": "text"},
        {"text": "keep this content", "type": "text"},
    ]
    assert _serialization.safe_text(encoded) == encoded


def test_nested_json_and_authorization_strings_are_fully_redacted():
    cases = (
        '{"error":{"api_key":"nested-secret"}}',
        '{"text":"Authorization: Basic nested-secret"}',
        'failure={"context":{"aws_secret_access_key":"nested-aws"}}',
        '{"proxyAuthorization":"Basic proxy-secret"}',
        '{"x-authorization":"Basic x-secret"}',
        '"Authorization: Basic scalar-secret"',
    )

    for value in cases:
        redacted = _serialization.safe_text(value)
        assert "nested-secret" not in redacted
        assert "nested-aws" not in redacted
        assert "proxy-secret" not in redacted
        assert "x-secret" not in redacted
        assert "scalar-secret" not in redacted


def test_deep_json_redaction_is_bounded_and_never_raises():
    nested: object = {"api_key": "deep-secret"}
    for _ in range(_serialization.MAX_DEPTH * 3):
        nested = {"nested": nested}
    redacted = _serialization.safe_text(json.dumps(nested))
    assert "deep-secret" not in redacted

    deeply_encoded = ("[" * 2_000) + '{"api_key":"recursive-secret"}' + ("]" * 2_000)
    redacted_encoded = _serialization.safe_text(deeply_encoded)
    assert "recursive-secret" not in redacted_encoded

    large_json = json.dumps([{"api_key": f"secret-{index}"} for index in range(1_000)])
    bounded = json.loads(_serialization.safe_text(large_json))
    assert len(bounded) == _serialization.MAX_COLLECTION_ITEMS + 1
    assert bounded[-1] == {"truncated": True}
    assert "secret-999" not in json.dumps(bounded)


def test_sensitive_long_keys_are_classified_before_display_truncation():
    long_key = f"{'x' * _serialization.MAX_TEXT_BYTES}_api_key"
    for preserve_vectors in (False, True):
        encoded = _serialization.json_dumps(
            {long_key: "long-key-secret", "vector": [0.1, 0.2]},
            preserve_large_vectors=preserve_vectors,
        )
        assert "long-key-secret" not in encoded
