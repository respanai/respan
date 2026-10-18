"""A local stand-in for Respan's session-recording endpoint, for tests.

Implements the contract: ``POST <base>/browser-sessions/recordings/reserve/``
(JSON, returns the replay's id and link before anything is recorded) and
``POST <base>/browser-sessions/recordings/`` (gzipped JSON, Bearer key; a
``recording_id`` must be one this stub reserved). Every request is recorded, so
a test can assert on exactly what an instrumentation would send without
touching a real backend.
"""

from __future__ import annotations

import gzip
import json
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from respan_tracing.utils.session_recording import RECORDINGS_PATH, RESERVE_PATH


@dataclass
class RecordedUpload:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class RecordingStub:
    base_url: str
    uploads: list[RecordedUpload] = field(default_factory=list)
    reservations: list[RecordedUpload] = field(default_factory=list)
    # Set to e.g. 500 to make every upload fail.
    status: int = 201
    # Set to 404 to act like a backend without the reserve endpoint.
    reserve_status: int = 201

    def view_url(self, recording_id: str) -> str:
        return f"{self.base_url}/replays/{recording_id}.html"


@contextmanager
def recording_stub_server(status: int = 201, reserve_status: int = 201) -> Iterator[RecordingStub]:
    stub: Optional[RecordingStub] = None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path.endswith(RESERVE_PATH):
                self._reserve(json.loads(raw or b"{}"))
                return
            if not self.path.endswith(RECORDINGS_PATH):
                self._reply(404, {})  # e.g. the span exporter's own POSTs
                return
            if self.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            body = json.loads(raw or b"{}")
            stub.uploads.append(RecordedUpload(self.path, dict(self.headers), body))
            if stub.status != 201:
                self._reply(stub.status, {"detail": "stub failure"})
                return
            if not body.get("events"):
                self._reply(400, {"detail": "events must be non-empty"})
                return
            recording_id = body.get("recording_id")
            if recording_id is not None and recording_id not in {
                r.body["recording_id"] for r in stub.reservations
            }:
                self._reply(400, {"detail": "unknown recording_id"})
                return
            self._reply(201, self._links(recording_id or str(uuid.uuid4())))

        def _reserve(self, body: dict) -> None:
            if stub.reserve_status != 201:
                self._reply(stub.reserve_status, {"detail": "stub reserve failure"})
                return
            links = self._links(str(uuid.uuid4()))
            stub.reservations.append(
                RecordedUpload(self.path, dict(self.headers), {**body, **links})
            )
            self._reply(201, links)

        def _links(self, recording_id: str) -> dict:
            return {
                "recording_id": recording_id,
                "view_url": stub.view_url(recording_id),
                "events_url": f"{stub.base_url}/replays/{recording_id}.json.gz",
                "expires_at": "2099-01-01T00:00:00Z",
            }

        def _reply(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    stub = RecordingStub(
        base_url=f"http://127.0.0.1:{server.server_address[1]}/api",
        status=status,
        reserve_status=reserve_status,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        server.shutdown()
        server.server_close()
