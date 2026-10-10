"""Actual Elasticsearch urllib3/aiohttp HTTP fixture; no vendor client replacements."""

import gzip
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@contextmanager
def server(
    payload=None,
    *,
    status=200,
    sequence=None,
    content_type="application/json",
    product=True,
):
    requests = []
    queue = list(sequence or [])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            decoded = (
                gzip.decompress(raw)
                if self.headers.get("Content-Encoding") == "gzip"
                else raw
            )
            requests.append(
                {
                    "method": self.command,
                    "target": self.path,
                    "body": decoded,
                    "raw_body": raw,
                }
            )
            code, value = queue.pop(0) if queue else (status, payload)
            if value is None:
                value = {
                    "took": 0,
                    "timed_out": False,
                    "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
                    "custom": {"zero": 0, "false": False, "empty": ""},
                }
            body = (
                value
                if type(value) is bytes
                else (
                    value.encode()
                    if type(value) is str and content_type != "application/json"
                    else json.dumps(value).encode()
                )
            )
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if product:
                self.send_header("X-Elastic-Product", "Elasticsearch")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = respond

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
