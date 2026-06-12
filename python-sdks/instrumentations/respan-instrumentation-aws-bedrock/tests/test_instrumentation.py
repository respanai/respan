from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_instrumentation_aws_bedrock import AWSBedrockInstrumentor
from respan_instrumentation_aws_bedrock import _instrumentation
from respan_instrumentation_aws_bedrock._otel_emitter import build_bedrock_attrs
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)


OFF_CONTRACT_ALIASES = {
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "tools",
    "tool_calls",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
    RESPAN_SPAN_TOOLS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_HANDOFFS,
}


class _OneShotBody:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._bytes = json.dumps(payload).encode("utf-8")
        self._read = False

    def read(self) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._bytes


@pytest.fixture(autouse=True)
def reset_instrumentation_globals() -> None:
    _instrumentation._original_make_api_call = None


@pytest.fixture()
def captured_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    spans: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_aws_bedrock._otel_emitter.inject_span",
        lambda span: spans.append(span),
    )
    return spans


@pytest.fixture()
def fake_botocore(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    class BaseClient:
        def __init__(self, service_name: str = "bedrock-runtime") -> None:
            self.meta = SimpleNamespace(
                service_model=SimpleNamespace(service_name=service_name)
            )

        def _make_api_call(
            self, operation_name: str, api_params: dict[str, Any]
        ) -> Any:
            if operation_name == "InvokeModel":
                return {
                    "body": _OneShotBody(
                        {
                            "id": "msg_123",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Hello from Bedrock"}],
                            "usage": {"input_tokens": 5, "output_tokens": 7},
                        }
                    ),
                    "contentType": "application/json",
                    "ResponseMetadata": {"HTTPStatusCode": 200},
                }
            if operation_name == "Converse":
                return {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "toolUse": {
                                        "toolUseId": "toolu_1",
                                        "name": "get_weather",
                                        "input": {"city": "Tokyo"},
                                    }
                                }
                            ],
                        }
                    },
                    "usage": {"inputTokens": 11, "outputTokens": 13, "totalTokens": 24},
                    "ResponseMetadata": {"HTTPStatusCode": 200},
                }
            if operation_name == "ConverseStream":
                return {
                    "stream": iter(
                        [
                            {"messageStart": {"role": "assistant"}},
                            {
                                "contentBlockDelta": {
                                    "delta": {"text": "Hello "},
                                }
                            },
                            {
                                "contentBlockDelta": {
                                    "delta": {"text": "stream"},
                                }
                            },
                            {
                                "metadata": {
                                    "usage": {
                                        "inputTokens": 2,
                                        "outputTokens": 3,
                                        "totalTokens": 5,
                                    }
                                }
                            },
                        ]
                    ),
                    "ResponseMetadata": {"HTTPStatusCode": 200},
                }
            return {"ok": True}

    botocore_module = ModuleType("botocore")
    client_module = ModuleType("botocore.client")
    setattr(client_module, "BaseClient", BaseClient)
    setattr(botocore_module, "client", client_module)
    monkeypatch.setitem(sys.modules, "botocore", botocore_module)
    monkeypatch.setitem(sys.modules, "botocore.client", client_module)
    return BaseClient


def _anthropic_request_body() -> str:
    return json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Say hello"}],
        }
    )


def test_invoke_model_emits_chat_span_and_preserves_response_body(
    fake_botocore: type[Any],
    captured_spans: list[Any],
) -> None:
    instrumentor = AWSBedrockInstrumentor()
    instrumentor.activate()

    response = fake_botocore()._make_api_call(
        "InvokeModel",
        {
            "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
            "body": _anthropic_request_body(),
            "contentType": "application/json",
        },
    )

    assert (
        json.loads(response["body"].read())["content"][0]["text"]
        == "Hello from Bedrock"
    )
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[TLSpanAttributes.LLM_SYSTEM] == "bedrock"
    assert (
        attrs[TLSpanAttributes.LLM_REQUEST_MODEL]
        == "anthropic.claude-3-5-haiku-20241022-v1:0"
    )
    assert attrs[TLSpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.content"] == "Say hello"
    assert attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert (
        attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hello from Bedrock"
    )
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 7
    assert attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 5
    assert attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 7
    assert attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 12
    assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] == "aws_bedrock.chat"
    assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_PATH] == "aws_bedrock.chat"
    assert TLSpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)

    instrumentor.deactivate()


def test_converse_promotes_tools_and_tool_calls_without_aliases(
    fake_botocore: type[Any],
    captured_spans: list[Any],
) -> None:
    instrumentor = AWSBedrockInstrumentor()
    instrumentor.activate()

    fake_botocore()._make_api_call(
        "Converse",
        {
            "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
            "messages": [
                {"role": "user", "content": [{"text": "What is the weather?"}]}
            ],
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "get_weather",
                            "description": "Get weather for a city.",
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                }
                            },
                        }
                    }
                ]
            },
        },
    )

    attrs = captured_spans[0]._attributes
    tools = json.loads(attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS])
    tool_calls = json.loads(attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert tools[0]["function"]["name"] == "get_weather"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert tool_calls[0]["function"]["arguments"] == '{"city": "Tokyo"}'
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)

    instrumentor.deactivate()


def test_stream_emits_span_after_stream_is_consumed(
    fake_botocore: type[Any],
    captured_spans: list[Any],
) -> None:
    instrumentor = AWSBedrockInstrumentor()
    instrumentor.activate()

    response = fake_botocore()._make_api_call(
        "ConverseStream",
        {
            "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
            "messages": [{"role": "user", "content": [{"text": "Stream hello"}]}],
        },
    )

    assert len(captured_spans) == 0
    assert len(list(response["stream"])) == 4
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hello stream"
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 2
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    assert attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 5
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)

    instrumentor.deactivate()


def test_non_bedrock_client_is_ignored(
    fake_botocore: type[Any],
    captured_spans: list[Any],
) -> None:
    instrumentor = AWSBedrockInstrumentor()
    instrumentor.activate()

    response = fake_botocore(service_name="s3")._make_api_call("ListBuckets", {})

    assert response == {"ok": True}
    assert captured_spans == []

    instrumentor.deactivate()


def test_active_workflow_name_is_attached_to_chat_span() -> None:
    token = context_api.attach(
        context_api.set_value(
            TLSpanAttributes.TRACELOOP_ENTITY_NAME,
            "aws_bedrock_invoke_model",
        )
    )
    try:
        attrs = build_bedrock_attrs(
            operation_name="InvokeModel",
            api_params={
                "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
                "body": _anthropic_request_body(),
            },
            response_payload={
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )
    finally:
        context_api.detach(token)

    assert attrs[TLSpanAttributes.TRACELOOP_WORKFLOW_NAME] == "aws_bedrock_invoke_model"
