import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import grpc
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pymilvus import DataType, MilvusClient
from respan_instrumentation_milvus import MilvusInstrumentor


@pytest.fixture(scope="session")
def native_server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("milvus-engine")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                os.getenv(
                    "RESPAN_MILVUS_LITE_EXECUTABLE",
                    str(Path(sys.executable).parent / "milvus-lite"),
                ),
                "server",
                "--data-dir",
                str(directory / "db"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=log,
            env=dict(os.environ),
        )
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        try:
            for _ in range(150):
                if process.poll() is not None:
                    raise AssertionError((directory / "server.log").read_text())
                try:
                    grpc.channel_ready_future(channel).result(timeout=0.1)
                    break
                except grpc.FutureTimeoutError:
                    time.sleep(0.1)
            else:
                raise AssertionError("Native Lite server unavailable")
            yield f"http://127.0.0.1:{port}"
        finally:
            channel.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.fixture
def engine(native_server):
    client = MilvusClient(uri=native_server)
    for name in client.list_collections():
        client.drop_collection(name)
    yield client
    client.close()


@pytest.fixture
def runtime():
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    owner = MilvusInstrumentor(tracer_provider=provider)
    owner.activate()
    yield provider, memory, owner
    owner.deactivate()
    provider.shutdown()


def populated(client, name="native_docs", count=3, dimension=4):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
    client.create_collection(name, schema=schema)
    rows = [
        {
            "id": i,
            "vector": [float(i % 3)] * dimension,
            "text": f"native row {i}",
            "flag": False,
        }
        for i in range(count)
    ]
    client.insert(name, rows)
    return name, rows
