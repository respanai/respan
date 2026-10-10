import json

import httpx
import replicate

BASE = "https://fixture.invalid"


def prediction(
    *, output=None, status="succeeded", input=None, ident="controlled", metrics=None
):
    return {
        "id": ident,
        "model": "owner/model",
        "version": "a" * 64,
        "status": status,
        "input": input or {},
        "output": output,
        "error": "controlled model failure" if status == "failed" else None,
        "logs": "controlled log",
        "metrics": metrics
        or {"input_token_count": 0, "output_token_count": 3, "predict_time": 0.01},
        "urls": {
            "get": BASE + "/v1/predictions/" + ident,
            "cancel": BASE + "/v1/predictions/" + ident + "/cancel",
            "stream": BASE + "/stream/" + ident,
        },
    }


class Native:
    def __init__(
        self,
        output=None,
        *,
        status="succeeded",
        chunks=250,
        http_status=201,
        output_chunks=None,
    ):
        self.output = output
        self.status = status
        self.chunks = chunks
        self.output_chunks = output_chunks
        self.http_status = http_status
        self.calls = []
        self.responses = []
        self.statuses = None
        self._input = {}

        def handler(r):
            self.calls.append((r.method, r.url.path))
            if r.url.path.startswith("/stream/"):
                data = (
                    "".join(
                        "event: output\nid: "
                        + str(i)
                        + "\ndata: "
                        + (
                            self.output_chunks[i]
                            if self.output_chunks is not None
                            else "chunk-" + str(i)
                        )
                        + "\n\n"
                        for i in range(self.chunks)
                    )
                    + "event: done\nid: done\ndata: {}\n\n"
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=httpx.ByteStream(data.encode()),
                )
            if r.url.path.endswith("/output/file"):
                return httpx.Response(200, content=b"native")
            if r.method == "POST" and r.url.path.endswith("/predictions"):
                if self.http_status >= 400:
                    return httpx.Response(
                        self.http_status, json={"detail": "controlled provider error"}
                    )
                body = json.loads(r.content)
                self._input = body.get("input", {})
                status = (
                    self.statuses.pop(0)
                    if self.statuses and len(self.statuses) > 1
                    else self.statuses[0]
                    if self.statuses
                    else self.status
                )
                return httpx.Response(
                    self.http_status,
                    json=prediction(
                        output=self.output, status=status, input=self._input
                    ),
                )
            if r.url.path.endswith("/predictions/controlled"):
                status = (
                    self.statuses.pop(0)
                    if self.statuses and len(self.statuses) > 1
                    else self.statuses[0]
                    if self.statuses
                    else self.status
                )
                return httpx.Response(
                    200,
                    json=prediction(
                        output=self.output, status=status, input=self._input
                    ),
                )
            if r.url.path == "/v1/predictions":
                return httpx.Response(
                    200,
                    json={
                        "results": [prediction(output=self.output)],
                        "next": None,
                        "previous": None,
                    },
                )
            if "/versions/" in r.url.path:
                return httpx.Response(
                    200,
                    json={
                        "id": "a" * 64,
                        "created_at": "2026-10-05T00:00:00Z",
                        "cog_version": "0.1",
                        "openapi_schema": {},
                    },
                )
            return httpx.Response(
                200, json=prediction(output=self.output, status=self.status)
            )

        def observe(r):
            response = handler(r)
            self.responses.append(response)
            return response

        self.client = replicate.Client(
            api_token="controlled",
            base_url=BASE,
            transport=httpx.MockTransport(observe),
        )
        self.client.poll_interval = 0

    def close(self):
        self.client._client.close()
