"""Controlled HTTP/SSE fixtures consumed by the released Aleph Alpha SDK."""

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, ClassVar


def model_name():
    return "controlled-aleph-model"


def _prompt_text(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt") or payload.get("input") or []
    if isinstance(prompt, list):
        parts = [item.get("data", "") for item in prompt if isinstance(item, dict)]
        return " ".join(part for part in parts if part)
    if isinstance(prompt, str):
        return prompt
    return "mock prompt"


class _MockAlephHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].strip("/")
        if path == "version":
            self._send_text("mock-aleph-alpha")
            return
        if path == "models_available":
            self._send_json([{"name": model_name()}])
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].strip("/")
        body = self._read_json_body()
        if body.get("stream"):
            self._send_sse(self._stream_events(path, body))
            return
        self._send_json(self._response_for(path, body))

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_text(self, text: str) -> None:
        encoded = text.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: Any) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_sse(self, events: list[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _response_for(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        model = body.get("model") or model_name()
        if path == "complete":
            return {
                "model_version": f"{model}-mock",
                "completions": [
                    {
                        "completion": f"Mock completion for: {_prompt_text(body)}",
                        "finish_reason": "stop",
                    }
                ],
                "num_tokens_prompt_total": 9,
                "num_tokens_generated": 7,
            }
        if path == "chat/completions":
            has_tools = bool(body.get("tools"))
            message: dict[str, Any] = {
                "role": "assistant",
                "content": "Mock Aleph Alpha chat response.",
            }
            finish_reason = "stop"
            if has_tools:
                finish_reason = "tool_calls"
                message["tool_calls"] = [
                    {
                        "id": "call_mock_lookup",
                        "type": "function",
                        "function": {
                            "name": "lookup_policy",
                            "arguments": '{"topic":"observability"}',
                        },
                    }
                ]
            return {"choices": [{"finish_reason": finish_reason, "message": message}]}
        if path == "embed":
            return {
                "model_version": f"{model}-mock",
                "embeddings": {"-1": {"mean": [0.1, 0.2, 0.3]}},
                "tokens": ["mock", "embedding"],
                "num_tokens_prompt_total": 5,
            }
        if path == "embeddings":
            items = body.get("input")
            count = (
                len(items)
                if isinstance(items, list) and items and isinstance(items[0], str)
                else 1
            )
            return {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2, 0.3],
                        "index": index,
                    }
                    for index in range(count)
                ],
                "model": model,
                "usage": {"prompt_tokens": 6, "total_tokens": 6},
            }
        if path in {"semantic_embed", "instructable_embed"}:
            return {
                "model_version": f"{model}-mock",
                "embedding": [0.4, 0.5, 0.6],
                "num_tokens_prompt_total": 6,
            }
        if path == "batch_semantic_embed":
            prompts = body.get("prompts") or []
            return {
                "model_version": f"{model}-mock",
                "embeddings": [[0.7, 0.8, 0.9] for _ in prompts],
                "num_tokens_prompt_total": max(1, len(prompts)) * 4,
            }
        if path == "evaluate":
            return {
                "model_version": f"{model}-mock",
                "message": None,
                "result": {"log_probability": -1.25},
                "num_tokens_prompt_total": 8,
            }
        if path == "explain":
            return {
                "model_version": f"{model}-mock",
                "explanations": [
                    {
                        "target": body.get("target", "mock target"),
                        "items": [
                            {
                                "type": "text",
                                "scores": [{"start": 0, "length": 4, "score": 0.72}],
                            },
                            {
                                "type": "target",
                                "scores": [{"start": 0, "length": 4, "score": 0.28}],
                            },
                        ],
                    }
                ],
            }
        return {"ok": True, "model_version": f"{model}-mock"}

    def _stream_events(self, path: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        model = body.get("model") or model_name()
        if path == "complete":
            return [
                {"type": "stream_chunk", "index": 0, "completion": "Mock stream "},
                {"type": "stream_chunk", "index": 0, "completion": "completion."},
                {
                    "type": "stream_summary",
                    "index": 0,
                    "model_version": f"{model}-mock",
                    "finish_reason": "stop",
                },
                {
                    "type": "completion_summary",
                    "num_tokens_prompt_total": 10,
                    "num_tokens_generated": 4,
                },
            ]
        if path == "chat/completions":
            return [
                {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"delta": {"content": "Mock streaming "}}]},
                {"choices": [{"delta": {"content": "chat."}}]},
                {
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 3,
                        "total_tokens": 12,
                    }
                },
                {"choices": [{"finish_reason": "stop"}]},
            ]
        return []


class Handler(_MockAlephHandler):
    requests: ClassVar[list] = []

    def do_POST(self):
        path = self.path.split("?", 1)[0].strip("/")
        body = self._read_json_body()
        type(self).requests.append((path, body))
        if "controlled-error" in json.dumps(body):
            encoded = b'Authorization: Bearer "PRIVATE-ERROR"'
            self.send_response(400)
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        elif body.get("stream"):
            self._send_sse(self._stream_events(path, body))
        else:
            self._send_json(self._response_for(path, body))

    def _response_for(self, path, body):
        if path == "complete" and "empty-candidates" in json.dumps(body):
            return {
                "model_version": "controlled-v1",
                "completions": [],
                "num_tokens_prompt_total": 0,
                "num_tokens_generated": 0,
            }
        result = super()._response_for(path, body)
        if path in {"semantic_embed", "instructable_embed"}:
            result["embedding"] = [float(i % 3) for i in range(5001)]
            result["num_tokens_prompt_total"] = 0
        if path == "embed":
            result["embeddings"]["-1"]["mean"] = [float(i % 3) for i in range(5001)]
        if path == "embeddings":
            for entry in result["data"]:
                entry["embedding"] = [float(i % 3) for i in range(5001)]
        if path == "batch_semantic_embed":
            result["embeddings"] = [
                [float(i % 3) for i in range(5001)] for _ in body["prompts"]
            ]
        return result

    def _stream_events(self, path, body):
        if "split-secret" in json.dumps(body):
            if path == "chat/completions":
                return [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "role": "assistant",
                                    "content": "Authorization: Bear",
                                },
                                "finish_reason": None,
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {"content": 'er "PRIVATE-FRAGMENT"'},
                                "finish_reason": None,
                            }
                        ]
                    },
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                ]
            return [
                {
                    "completion": "Authorization: Bear",
                    "type": "stream_chunk",
                    "index": 0,
                },
                {
                    "completion": 'er "PRIVATE-FRAGMENT"',
                    "type": "stream_chunk",
                    "index": 0,
                },
            ]
        return super()._stream_events(path, body)
