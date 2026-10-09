"""Released boto3 HTTP parser and CRC-correct native event stream fixtures."""

import io
import json
import struct
import zlib

import boto3
from botocore.awsrequest import AWSResponse
from botocore.config import Config
from urllib3.response import HTTPResponse


def frame(payload, kind="PayloadPart", message_type="event"):
    headers = b""
    for key, value in (
        (":message-type", message_type),
        (":event-type" if message_type == "event" else ":exception-type", kind),
        (":content-type", "application/octet-stream"),
    ):
        k = key.encode()
        v = value.encode()
        headers += bytes([len(k)]) + k + b"\x07" + struct.pack(">H", len(v)) + v
    prelude = struct.pack(">II", 16 + len(headers) + len(payload), len(headers))
    data = (
        prelude
        + struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
        + headers
        + payload
    )
    return data + struct.pack(">I", zlib.crc32(data) & 0xFFFFFFFF)


def client(payload=None, *, events=None, status=200, headers=None):
    c = boto3.client(
        "sagemaker-runtime",
        region_name="us-east-1",
        aws_access_key_id="controlled",
        aws_secret_access_key="controlled",
        endpoint_url="https://fixture.invalid",
        config=Config(retries={"max_attempts": 0}),
    )
    raw_data = events if events is not None else json.dumps(payload).encode()
    calls = []
    raws = []

    def send(request):
        raw = HTTPResponse(
            body=io.BytesIO(raw_data), preload_content=False, decode_content=False
        )
        raws.append(raw)
        calls.append(request)
        return AWSResponse(
            request.url,
            status,
            {
                "content-type": "application/vnd.amazon.eventstream"
                if events is not None
                else "application/json",
                "content-length": str(len(raw_data)),
                "x-amzn-requestid": "controlled-request",
                **(headers or {}),
            },
            raw,
        )

    c._endpoint.http_session.send = send
    return c, calls, raws
