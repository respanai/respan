"""Released Writer clients, typed models and actual HTTPX/SSE bytes."""

import json

import httpx
from writerai import AsyncWriter, Writer


class Body(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, frames):
        self.frames = frames
        self.reads = 0
        self.closes = 0

    def __iter__(self):
        for frame in self.frames:
            self.reads += 1
            yield frame

    async def __aiter__(self):
        for frame in self:
            yield frame

    def close(self):
        self.closes += 1

    async def aclose(self):
        self.close()


class NativeRuntime:
    def __init__(
        self,
        *,
        chunks=300,
        error=False,
        retry=False,
        stream_error=False,
        empty=False,
        vector=False,
        secret_fragments=False,
        callback=None,
    ):
        self.requests = []
        self.responses = []
        self.bodies = []
        self.chunks = chunks
        self.error = error
        self.retry = retry
        self.stream_error = stream_error
        self.empty = empty
        self.vector = vector
        self.secret_fragments = secret_fragments
        self.callback = callback
        self.response_callbacks = 0

    def payload(self, path, body):
        if path.endswith("/chat"):
            content = (
                '{"summary":"native structured","sentiment":"positive","value":0}'
                if body.get("response_format")
                else "native response"
            )
            message = {
                "role": "assistant",
                "content": content,
                "reasoning": "actual native reasoning",
                "zero": 0,
                "false": False,
            }
            if body.get("tools") and not any(
                m.get("role") == "tool" for m in body.get("messages", [])
            ):
                message["content"] = ""
                message["tool_calls"] = [
                    {
                        "id": "call-native-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Tokyo"}',
                        },
                    }
                ]
            result = {
                "id": "chat-native",
                "object": "chat.completion",
                "created": 0,
                "model": "reported-writer",
                "choices": []
                if self.empty
                else [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 3,
                    "total_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "feedback": {"blocked": self.empty, "zero": 0, "false": False},
            }
            if self.vector:
                result["native_vector"] = [float(i) / 5001 for i in range(5001)]
            return result
        if path.endswith("/completions"):
            return {
                "model": "reported-completion",
                "choices": [{"text": "native completion"}],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 3,
                    "total_tokens": 3,
                },
                "native_extra": {"zero": 0, "false": False},
            }
        if path.endswith("/graphs/question"):
            return {
                "answer": "native graph answer",
                "question": body.get("question"),
                "sources": [
                    {"file_id": "native-file", "snippet": "complete native citation"}
                ],
                "native_extra": {"zero": 0, "false": False},
            }
        if "/applications/" in path:
            return {
                "title": "native application",
                "suggestion": "native application generation",
                "native_extra": {"zero": 0, "false": False},
            }
        if path.endswith("/vision"):
            return {
                "data": "native vision response",
                "native_extra": {"zero": 0, "false": False},
            }
        if path.endswith("/translation"):
            return {
                "data": "native translated response",
                "native_extra": {"zero": 0, "false": False},
            }
        if path.endswith("/web-search"):
            return {
                "query": body.get("query"),
                "answer": "native web answer",
                "sources": [
                    {
                        "url": "https://example.invalid/path",
                        "raw_content": "native source",
                    }
                ],
                "native_extra": {"zero": 0, "false": False},
            }
        if "/pdf-parser/" in path:
            return {
                "content": "native parsed PDF",
                "native_extra": {"zero": 0, "false": False},
            }
        return {"actual_native_empty": True}

    def stream_frames(self, path):
        frames = []
        for index in range(self.chunks):
            if self.stream_error and index == 2:
                frames.append(
                    {
                        "error": {
                            "message": "controlled native stream failure",
                            "type": "server_error",
                        }
                    }
                )
                break
            text = str(index) + ","
            if self.secret_fragments:
                text = (
                    ['Authorization: Bearer "controlled-', 'secret"'][index]
                    if index < 2
                    else ""
                )
            if path.endswith("/chat"):
                delta = {"role": "assistant", "content": text}
                if self.secret_fragments:
                    delta["tool_calls"] = [
                        {
                            "index": 0,
                            "id": "call-native-1",
                            "type": "function",
                            "function": {
                                "name": "get_weather" if index == 0 else None,
                                "arguments": [
                                    '{"private_key":"controlled-',
                                    'secret","zero":0,"flag":false}',
                                ][index]
                                if index < 2
                                else "",
                            },
                        }
                    ]
                frame = {
                    "id": "stream-native",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "reported-writer",
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": "stop"
                            if index == self.chunks - 1
                            else None,
                        }
                    ],
                }
                if index == self.chunks - 1:
                    frame["usage"] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 3,
                        "total_tokens": 3,
                    }
            elif path.endswith("/completions"):
                frame = {"value": text, "model": "reported-completion"}
            elif path.endswith("/graphs/question"):
                frame = {"answer": text, "sources": []}
            else:
                frame = {"delta": {"content": text}}
            frames.append(frame)
        return [
            (
                ("event: error\n" if "error" in frame else "")
                + "data: "
                + json.dumps(frame)
                + "\n\n"
            ).encode()
            for frame in frames
        ] + [b"data: [DONE]\n\n"]

    def respond(self, request):
        body = json.loads(request.content) if request.content else {}
        self.requests.append(body)
        if self.callback:
            self.callback(request)
        if (
            self.error
            or (self.retry and len(self.requests) == 1)
            or any(
                message.get("content") == "RESPAN_EXPECTED_WRITER_ERROR"
                for message in body.get("messages", [])
                if type(message) is dict
            )
        ):
            response = httpx.Response(
                429,
                json={"error": {"message": "controlled native failure"}},
                headers={"retry-after-ms": "0"},
                request=request,
            )
        elif body.get("stream") is True:
            stream = Body(self.stream_frames(request.url.path))
            self.bodies.append(stream)
            response = httpx.Response(
                200,
                stream=stream,
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        else:
            response = httpx.Response(
                200, json=self.payload(request.url.path, body), request=request
            )
        self.responses.append(response)
        return response

    def response_hook(self, response):
        self.response_callbacks += 1

    async def async_response_hook(self, response):
        self.response_hook(response)

    def client(self, *, max_retries=0):
        return Writer(
            api_key="controlled-fixture",
            max_retries=max_retries,
            http_client=httpx.Client(
                transport=httpx.MockTransport(self.respond),
                event_hooks={"response": [self.response_hook]},
            ),
        )

    def async_client(self):
        return AsyncWriter(
            api_key="controlled-fixture",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(self.respond),
                event_hooks={"response": [self.async_response_hook]},
            ),
        )


def invoke(client, operation, **extra):
    if operation == "chat":
        return client.chat.chat(
            model="native-writer",
            messages=[{"role": "user", "content": "native input"}],
            **extra,
        )
    if operation == "completion":
        return client.completions.create(
            model="native-writer", prompt="native input", **extra
        )
    if operation == "graph":
        return client.graphs.question(
            graph_ids=["native-graph"], question="native input", **extra
        )
    if operation == "application":
        return client.applications.generate_content(
            "native-application",
            inputs=[{"id": "input", "value": ["native input"]}],
            **extra,
        )
    if operation == "vision":
        return client.vision.analyze(
            model="palmyra-vision",
            prompt="native input",
            variables=[{"name": "image", "file_id": "native-file"}],
            **extra,
        )
    if operation == "translation":
        return client.translation.translate(
            model="palmyra-translate",
            text="native input",
            source_language_code="en",
            target_language_code="fr",
            formality=False,
            length_control=False,
            mask_profanity=False,
            **extra,
        )
    if operation == "web_search":
        return client.tools.web_search(query="native input", **extra)
    return client.tools.parse_pdf("native-file", format="markdown", **extra)
