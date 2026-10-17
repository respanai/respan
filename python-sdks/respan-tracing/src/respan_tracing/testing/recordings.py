"""A local stand-in for Respan's session-recording endpoint, for tests.

Implements the upload half of the contract (``POST <base>/browser-sessions/recordings/``,
gzipped JSON, Bearer key) and records every request, so a test can assert on
exactly what an instrumentation would send without touching a real backend.
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

from respan_tracing.utils.session_recording import RECORDINGS_PATH


@dataclass
class RecordedUpload:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class RecordingStub:
    base_url: str
    uploads: list[RecordedUpload] = field(default_factory=list)
    # Set to e.g. 500 to make every upload fail.
    status: int = 201

    def view_url(self, recording_id: str) -> str:
        return f"{self.base_url}/replays/{recording_id}.html"


@contextmanager
def recording_stub_server(status: int = 201) -> Iterator[RecordingStub]:
    stub: Optional[RecordingStub] = None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
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
            recording_id = str(uuid.uuid4())
            self._reply(
                201,
                {
                    "recording_id": recording_id,
                    "view_url": stub.view_url(recording_id),
                    "events_url": f"{stub.base_url}/replays/{recording_id}.json.gz",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )

        def _reply(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    stub = RecordingStub(base_url=f"http://127.0.0.1:{server.server_address[1]}/api", status=status)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        server.shutdown()
        server.server_close()
