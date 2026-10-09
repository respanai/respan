"""Controlled HTTP and AWS event-stream frames parsed by released botocore."""

import base64
import io
import json
import struct
import zlib

import boto3
from botocore.awsrequest import AWSResponse
from botocore.config import Config
from urllib3.response import HTTPResponse


def client(*, retries=0):
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="controlled",
        aws_secret_access_key="controlled",
        endpoint_url="https://bedrock-runtime.invalid",
        config=Config(retries={"max_attempts": retries}),
    )


def frame(event_type, payload, message_type="event"):
    headers = b""
    for key, data in (
        (":message-type", message_type),
        (":event-type", event_type),
        (":content-type", "application/json"),
    ):
        key, data = key.encode(), data.encode()
        headers += (
            bytes([len(key)]) + key + b"\x07" + struct.pack(">H", len(data)) + data
        )
    data = json.dumps(payload).encode()
    prelude = struct.pack(">II", 16 + len(headers) + len(data), len(headers))
    data = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + data
    return data + struct.pack(">I", zlib.crc32(data))


def transport(client, payload, *, status=200, event_stream=False):
    raw = payload if type(payload) is bytes else json.dumps(payload).encode()
    headers = {
        "content-type": "application/vnd.amazon.eventstream"
        if event_stream
        else "application/json",
        "content-length": str(len(raw)),
    }
    # AWSResponse and urllib3's native response exercise HTTP body and event
    # deserializers rather than Stubber's bypass of the response parser.
    response = AWSResponse(
        "https://bedrock-runtime.invalid",
        status,
        headers,
        HTTPResponse(body=io.BytesIO(raw), preload_content=False),
    )
    client._endpoint.http_session.send = lambda request: response
    return response


def converse_response():
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "native hello"},
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "controlled reasoning",
                                "signature": "signature-value",
                            }
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "call-1",
                            "name": "weather",
                            "input": {
                                "city": "Shanghai",
                                "zero": 0,
                                "false": False,
                                "empty": "",
                            },
                        }
                    },
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {
            "inputTokens": 0,
            "outputTokens": 7,
            "totalTokens": 7,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokens": 3,
        },
        "metrics": {"latencyMs": 1},
    }


def converse_frames():
    return b"".join(
        frame(kind, payload)
        for kind, payload in [
            ("messageStart", {"role": "assistant"}),
            (
                "contentBlockStart",
                {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "call-1", "name": "weather"}},
                },
            ),
            (
                "contentBlockDelta",
                {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '{"city":"Shanghai",'}},
                },
            ),
            (
                "contentBlockDelta",
                {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '"zero":0,"false":false}'}},
                },
            ),
            ("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"text": "hello"}}),
            (
                "contentBlockDelta",
                {
                    "contentBlockIndex": 2,
                    "delta": {"reasoningContent": {"text": "controlled reasoning"}},
                },
            ),
            ("messageStop", {"stopReason": "tool_use"}),
            (
                "metadata",
                {
                    "usage": {
                        "inputTokens": 0,
                        "outputTokens": 7,
                        "totalTokens": 7,
                        "cacheReadInputTokens": 0,
                    },
                    "metrics": {"latencyMs": 1},
                },
            ),
        ]
    )


def invoke_frames():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 0}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call-2",
                "name": "weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"city":"Shanghai"}',
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "hello"},
        },
        {"type": "message_delta", "usage": {"output_tokens": 7}},
    ]
    # InvokeModel's chunk payload is opaque bytes on the actual event wire.
    return b"".join(
        frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()})
        for event in events
    )
