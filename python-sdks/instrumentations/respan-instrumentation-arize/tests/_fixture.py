"""Native Arize REST and requests-futures transports with controlled data."""

from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from urllib.parse import urlparse

import requests
import urllib3
from arize import ArizeClient
from openinference.semconv.trace import EmbeddingAttributes
from openinference.semconv.trace import SpanAttributes as OI
from requests_futures.sessions import FuturesSession

PROJECT_ID = base64.b64encode(b"Project:fixture").decode()
SPACE_ID = base64.b64encode(b"Space:fixture").decode()
DATASET_ID = base64.b64encode(b"Dataset:fixture").decode()
TIME = "2026-09-29T12:00:00Z"
PAGINATION = {"has_more": False, "next_cursor": None}


class FixtureTransport:
    def __init__(self, *, fail=False, delayed=False):
        self.fail = fail
        self.delayed = delayed
        self.entered = Event()
        self.release = Event()
        if not delayed:
            self.release.set()
        self.requests = []
        self.responses = []
        self.before_rest = None
        self.client = ArizeClient(
            api_key="fixture-key",
            api_host="https://fixture.invalid",
            enable_caching=False,
        )
        generated = self.client.datasets._api.api_client
        generated.rest_client.pool_manager.request = self.rest
        session = requests.Session()
        session.send = self.send
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.client.ml._session = FuturesSession(
            session=session, executor=self.executor
        )

    def span(self):
        return {
            "name": "controlled historical embedding",
            "context": {"trace_id": "0" * 31 + "1", "span_id": "0" * 15 + "1"},
            "kind": "EMBEDDING",
            "start_time": TIME,
            "end_time": TIME,
            "attributes": {
                f"{OI.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}": [
                    float(i) for i in range(3072)
                ],
                "tool_calls": [
                    {
                        "id": "historical-call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
        }

    def dataset(self):
        return {
            "id": DATASET_ID,
            "name": "fixture-dataset",
            "space_id": SPACE_ID,
            "created_at": TIME,
            "updated_at": TIME,
        }

    def response(self, method, path, body):
        tail = path.rstrip("/").split("/")[-1]
        if "examples" in path and "datasets" in path:
            if method == "GET":
                return {
                    "examples": [
                        {
                            "id": "example-fixture",
                            "created_at": TIME,
                            "updated_at": TIME,
                            "input": {"question": "controlled question"},
                            "output": "controlled expected",
                        }
                    ],
                    "pagination": PAGINATION,
                }
            if method == "DELETE":
                return {
                    "completed": True,
                    "deleted_example_ids": ["example-fixture"],
                    "not_deleted_example_ids": [],
                }
            return {
                **self.dataset(),
                "dataset_version_id": "version-fixture",
                "example_ids": ["example-fixture"],
            }
        if path.endswith("/datasets"):
            if method == "GET":
                return {"datasets": [], "pagination": PAGINATION}
            return self.dataset()
        if "/datasets/" in path:
            return self.dataset()
        if tail == "spans":
            return {"spans": [self.span()], "pagination": PAGINATION}
        if tail == "traces":
            return {
                "traces": [
                    {
                        "trace_id": "0" * 31 + "1",
                        "root_span_id": "0" * 15 + "1",
                        "spans_truncated": False,
                        "spans": [self.span()],
                    }
                ],
                "pagination": PAGINATION,
            }
        if "audit" in tail:
            return {"logs": [], "pagination": PAGINATION}
        for resource, key in (
            ("projects", "projects"),
            ("prompts", "prompts"),
            ("experiments", "experiments"),
            ("evaluators", "evaluators"),
            ("integrations", "integrations"),
            ("webhooks", "webhooks"),
            ("annotation-configs", "annotation_configs"),
            ("annotation_configs", "annotation_configs"),
            ("roles", "roles"),
            ("resource-restrictions", "resource_restrictions"),
            ("resource_restrictions", "resource_restrictions"),
        ):
            if tail == resource:
                return {key: [], "pagination": PAGINATION}
        raise AssertionError(
            "Unconfigured controlled REST route: " + method + " " + path
        )

    def rest(self, method, url, **kwargs):
        path = urlparse(url).path
        self.requests.append({"transport": "rest", "method": method, "path": path})
        self.entered.set()
        if self.before_rest is not None:
            self.before_rest()
        body = (
            {"code": "NOT_FOUND", "message": "controlled native missing dataset"}
            if self.fail
            else self.response(method, path, kwargs.get("body"))
        )
        return urllib3.HTTPResponse(
            body=json.dumps(body).encode(),
            status=404 if self.fail else 200,
            headers={"content-type": "application/json"},
        )

    def send(self, request, **kwargs):
        self.requests.append(
            {
                "transport": "future",
                "method": request.method,
                "path": urlparse(request.url).path,
            }
        )
        self.entered.set()
        if not self.release.wait(5):
            raise requests.Timeout("Controlled future transport not released")
        if self.fail:
            raise requests.ConnectionError("controlled native upload failure")
        response = requests.Response()
        response.status_code = 202
        response._content = b'{"accepted":true}'
        response.url = "https://fixture.invalid/records"
        self.responses.append(response)
        return response

    def close(self):
        self.release.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
