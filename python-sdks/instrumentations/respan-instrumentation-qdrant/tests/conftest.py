import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient, models
from respan_instrumentation_qdrant import QdrantInstrumentor


@pytest.fixture
def engine():
    client = QdrantClient(":memory:")
    client.create_collection(
        "native",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT),
    )
    client.upsert(
        "native",
        points=[
            models.PointStruct(
                id=i,
                vector=[1.0, 0.0, 0.0],
                payload={"body": "PRIVATE", "flag": False, "zero": 0, "empty": ""},
            )
            for i in range(2)
        ],
    )
    yield client
    client.close()


@pytest.fixture
def runtime():
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    owner = QdrantInstrumentor(tracer_provider=provider)
    owner.activate()
    yield provider, memory, owner
    owner.deactivate()
    provider.shutdown()


@pytest.fixture
def native_http():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    controls = {
        "pending": threading.Event(),
        "release": threading.Event(),
        "delay": False,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.reply(
                200, {"title": "qdrant", "version": "1.19.1", "commit": "controlled"}
            )

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "missing" in self.path:
                self.reply(
                    400,
                    {
                        "status": {"error": 'native error Bearer "PRIVATE SPACE"'},
                        "time": 0.0,
                    },
                )
                return
            ids = payload.get("ids", [0])
            if controls["delay"] and ids == [0]:
                controls["pending"].set()
                if not controls["release"].wait(10):
                    self.reply(
                        500, {"status": {"error": "controlled timeout"}, "time": 0.0}
                    )
                    return
            self.reply(
                200,
                {
                    "result": [
                        {
                            "id": i,
                            "payload": {
                                "flag": False,
                                "zero": 0,
                                "empty": "",
                                "body": "PRIVATE",
                            },
                            "vector": [1.0, 0.0, 0.0],
                        }
                        for i in ids
                    ],
                    "status": "ok",
                    "time": 0.0,
                },
            )

        def reply(self, code, data):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}", controls
    controls["release"].set()
    server.shutdown()
    server.server_close()
    worker.join(2)
