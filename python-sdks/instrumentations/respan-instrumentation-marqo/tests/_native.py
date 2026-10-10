import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


@contextmanager
def server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def serve(self):
            path = urlsplit(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else None
            requests.append({"method": self.command, "path": self.path, "body": body})
            status = 503 if path.endswith("/health") else 200
            if path == "/":
                payload = {"version": "3.18.2"}
            elif status == 503:
                payload = {
                    "message": 'controlled error Bearer "PRIVATE SPACE"',
                    "code": "service_unavailable",
                    "type": "service_unavailable",
                    "link": "",
                }
            elif path.endswith("/embed"):
                payload = {
                    "embeddings": [
                        {
                            "content": "native",
                            "embedding": [float(i) for i in range(5001)],
                            "flag": False,
                        }
                    ],
                    "processingTimeMs": 0,
                    "model": "actual-controlled-model",
                    "empty": "",
                }
            elif path.endswith("/search"):
                payload = {
                    "hits": [
                        {"_id": str(i), "_score": 0, "flag": False} for i in range(75)
                    ],
                    "processingTimeMs": 0,
                }
            elif path.endswith("/recommend"):
                payload = {"hits": [], "processingTimeMs": 0}
            elif path.endswith("/documents") and self.command in ["POST", "PUT"]:
                payload = {
                    "errors": False,
                    "items": [
                        {"_id": str(i), "status": 200}
                        for i in range(len(body.get("documents", [])))
                    ],
                    "processingTimeMs": 0,
                }
            elif path.endswith("/documents/delete-batch"):
                payload = {"deletedDocuments": len(body)}
            elif "/documents/" in path:
                payload = {"_id": "doc", "vector": [0.0] * 5001, "flag": False}
            elif path.endswith("/documents"):
                payload = {
                    "results": [
                        {"_id": str(i), "vector": [0.0] * 5001}
                        for i in range(len(body or []))
                    ]
                }
            elif path.endswith("/settings"):
                payload = {
                    "model": "actual-controlled-model",
                    "normalizeEmbeddings": False,
                }
            elif path.endswith("/stats"):
                payload = {"numberOfDocuments": 0, "numberOfVectors": 0}
            else:
                payload = {
                    "acknowledged": True,
                    "results": [],
                    "status": "ready",
                    "models": [],
                }
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = serve

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join()
