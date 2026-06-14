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
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_instrumentation_sagemaker import SageMakerInstrumentor
from respan_instrumentation_sagemaker import _instrumentation
from respan_instrumentation_sagemaker._constants import (
    INVOKE_ENDPOINT_ASYNC_OPERATION,
    INVOKE_ENDPOINT_OPERATION,
    INVOKE_ENDPOINT_STREAM_OPERATION,
)
from respan_instrumentation_sagemaker._otel_emitter import build_sagemaker_attrs
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
    def __init__(self, payload: Any) -> None:
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
        "respan_instrumentation_sagemaker._otel_emitter.inject_span",
        lambda span: spans.append(span),
    )
    return spans


@pytest.fixture()
def fake_botocore(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    class BaseClient:
        def __init__(self, service_name: str = "sagemaker-runtime") -> None:
            self.meta = SimpleNamespace(
                service_model=SimpleNamespace(service_name=service_name)
            )

        def _make_api_call(
            self, operation_name: str, api_params: dict[str, Any]
        ) -> Any:
            if operation_name == INVOKE_ENDPOINT_OPERATION:
                body = json.loads(api_params.get("Body", b"{}"))
                if "messages" in body:
                    return {
                        "Body": _OneShotBody(
                            {
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": "",
                                            "tool_calls": [
                                                {
                                                    "id": "call_1",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "get_weather",
                                                        "arguments": '{"city": "Tokyo"}',
                                                    },
                                                }
                                            ],
                                        }
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 11,
                                    "completion_tokens": 7,
                                    "total_tokens": 18,
                                },
                            }
                        ),
                        "ContentType": "application/json",
                        "ResponseMetadata": {"HTTPStatusCode": 200},
                    }
                return {
                    "Body": _OneShotBody(
                        [
                            {
                                "generated_text": "Hello from SageMaker",
                                "details": {
                                    "input_tokens": 5,
                                    "generated_tokens": 4,
                                },
                            }
                        ]
                    ),
                    "ContentType": "application/json",
                    "ResponseMetadata": {"HTTPStatusCode": 200},
                }
            if operation_name == INVOKE_ENDPOINT_STREAM_OPERATION:
                return {
                    "Body": iter(
                        [
                            {
                                "PayloadPart": {
                                    "Bytes": json.dumps(
                                        {"token": {"text": "Hello "}}
                                    ).encode("utf-8")
                                }
                            },
                            {
                                "PayloadPart": {
                                    "Bytes": json.dumps(
                                        {
                                            "token": {"text": "stream"},
                                            "usage": {
                                                "input_tokens": 2,
                                                "generated_tokens": 3,
                                            },
                                        }
                                    ).encode("utf-8")
                                }
                            },
                        ]
                    ),
                    "ContentType": "application/json",
                    "ResponseMetadata": {"HTTPStatusCode": 200},
                }
            if operation_name == INVOKE_ENDPOINT_ASYNC_OPERATION:
                return {
                    "InferenceId": "inference-123",
                    "OutputLocation": "s3://bucket/output.json",
                    "ResponseMetadata": {"HTTPStatusCode": 202},
                }
            return {"ok": True}

    botocore_module = ModuleType("botocore")
    client_module = ModuleType("botocore.client")
    setattr(client_module, "BaseClient", BaseClient)
    setattr(botocore_module, "client", client_module)
    monkeypatch.setitem(sys.modules, "botocore", botocore_module)
    monkeypatch.setitem(sys.modules, "botocore.client", client_module)
    return BaseClient


def test_invoke_endpoint_emits_text_span_and_preserves_response_body(
    fake_botocore: type[Any],
    captured_spans: list[Any],
) -> None:
    instrumentor = SageMakerInstrumentor()
    instrumentor.activate()

    response = fake_botocore()._make_api_call(
        INVOKE_ENDPOINT_OPERATION,
        {
            "EndpointName": "jumpstart-text-endpoint",
            "Body": json.dumps(
                {
                    "inputs": "Say hello from a SageMaker endpoint.",
                    "parameters": {"max_new_tokens": 16},
                }
            ).encode("utf-8"),
            "ContentType": "application/json",
            "Accept": "application/json",
            "CustomAttributes": "respan_model=gpt-4o-mini",
        },
    )

    assert json.loads(response["Body"].read())[0]["generated_text"] == (
        "Hello from SageMaker"
    )
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[TLSpanAttributes.LLM_SYSTEM] == "sagemaker"
    assert attrs[TLSpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert (
        attrs[TLSpanAttributes.LLM_REQUEST_TYPE]
        == LLMRequestTypeValues.CHAT.value
    )
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert (
        attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.content"]
        == "Say hello from a SageMaker endpoint."
    )
    assert attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert (
        attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "Hello from SageMaker"
    )
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4
    assert attrs[GenAIAttributes.GEN_AI_USAGE_PROMPT_TOKENS] == 5
    assert attrs[GenAIAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] == 4
    assert attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 5
    assert attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
    assert attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 9
    assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] == "sagemaker.chat"
    assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_PATH] == "sagemaker.chat"
    assert TLSpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)

    instrumentor.deactivate()


def test_chat_tools_and_tool_calls_use_canonical_fields_without_aliases(
    fake_botocore: type[Any],
    captured_spans: list[Any],
) -> None:
    instrumentor = SageMakerInstrumentor()
    instrumentor.activate()

    fake_botocore()._make_api_call(
        INVOKE_ENDPOINT_OPERATION,
        {
            "EndpointName": "chat-endpoint",
            "Body": json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the weather in Tokyo?",
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather for a city.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                },
                            },
                        }
                    ],
                }
            ).encode("utf-8"),
            "ContentType": "application/json",
            "Accept": "application/json",
        },
    )

    attrs = captured_spans[0]._attributes
    tools = json.loads(attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS])
    tool_calls = json.loads(attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[TLSpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert tools[0]["function"]["name"] == "get_weather"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert tool_calls[0]["function"]["arguments"] == '{"city": "Tokyo"}'
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)

    instrumentor.deactivate()


def test_stream_emits_span_after_stream_is_consumed(
    fake_botocore: type[Any],
    captured_spans: list[Any],
) -> None:
    instrumentor = SageMakerInstrumentor()
    instrumentor.activate()

    response = fake_botocore()._make_api_call(
        INVOKE_ENDPOINT_STREAM_OPERATION,
        {
            "EndpointName": "stream-endpoint",
            "Body": json.dumps({"inputs": "Stream a greeting."}).encode("utf-8"),
            "ContentType": "application/json",
            "Accept": "application/json",
        },
    )

    assert len(captured_spans) == 0
    assert len(list(response["Body"])) == 2
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hello stream"
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 2
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    assert attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 5
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)

    instrumentor.deactivate()


def test_async_endpoint_emits_text_span_with_output_location(
    fake_botocore: type[Any],
    captured_spans: list[Any],
) -> None:
    instrumentor = SageMakerInstrumentor()
    instrumentor.activate()

    fake_botocore()._make_api_call(
        INVOKE_ENDPOINT_ASYNC_OPERATION,
        {
            "EndpointName": "async-endpoint",
            "InputLocation": "s3://bucket/input.json",
            "ContentType": "application/json",
            "Accept": "application/json",
            "CustomAttributes": "respan_model=gpt-4o-mini",
        },
    )

    attrs = captured_spans[0]._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[TLSpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert (
        attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.content"]
        == "s3://bucket/input.json"
    )
    assert (
        attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "s3://bucket/output.json"
    )
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)

    instrumentor.deactivate()


def test_non_sagemaker_client_is_ignored(
    fake_botocore: type[Any],
    captured_spans: list[Any],
) -> None:
    instrumentor = SageMakerInstrumentor()
    instrumentor.activate()

    response = fake_botocore(service_name="s3")._make_api_call("ListBuckets", {})

    assert response == {"ok": True}
    assert captured_spans == []

    instrumentor.deactivate()


def test_active_workflow_name_is_attached_to_span() -> None:
    token = context_api.attach(
        context_api.set_value(
            TLSpanAttributes.TRACELOOP_ENTITY_NAME,
            "sagemaker_invoke_endpoint",
        )
    )
    try:
        attrs = build_sagemaker_attrs(
            operation_name=INVOKE_ENDPOINT_OPERATION,
            api_params={
                "EndpointName": "jumpstart-text-endpoint",
                "Body": json.dumps({"inputs": "Say hello"}).encode("utf-8"),
            },
            response_payload=[
                {
                    "generated_text": "Hello",
                    "details": {"input_tokens": 1, "generated_tokens": 2},
                }
            ],
        )
    finally:
        context_api.detach(token)

    assert attrs[TLSpanAttributes.TRACELOOP_WORKFLOW_NAME] == "sagemaker_invoke_endpoint"
