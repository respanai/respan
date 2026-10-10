import importlib.metadata
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import chromadb
import httpx
import pytest
from chromadb.config import Settings
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_chroma import ChromaInstrumentor


@pytest.fixture
def engine(tmp_path):
    client = chromadb.PersistentClient(
        path=str(tmp_path),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    yield client
    if hasattr(client, "close"):
        client.close()
    else:
        client._system.stop()


@pytest.fixture
def runtime():
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    owner = ChromaInstrumentor(tracer_provider=provider)
    owner.activate()
    yield provider, memory, owner
    owner.deactivate()
    provider.shutdown()


@pytest.fixture(scope="session")
def native_server(tmp_path_factory):
    path = tmp_path_factory.mktemp("native-http")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    current = not importlib.metadata.version("chromadb").startswith("0.5.")
    version = "v2" if current else "v1"
    log = (path / "server.log").open("w")
    process = subprocess.Popen(
        [
            str(Path(sys.executable).parent / "chroma"),
            "run",
            "--path",
            str(path / "db"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=dict(os.environ, ANONYMIZED_TELEMETRY="False"),
        stdout=log,
        stderr=log,
    )
    try:
        for _ in range(150):
            if process.poll() is not None:
                raise AssertionError((path / "server.log").read_text())
            try:
                result = httpx.get(
                    f"http://127.0.0.1:{port}/api/{version}/heartbeat", timeout=1
                )
                if result.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise AssertionError("Native released Chroma server did not start")
        yield port
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
