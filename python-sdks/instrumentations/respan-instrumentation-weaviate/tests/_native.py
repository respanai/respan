"""Actual released Weaviate HTTP/gRPC protocol fixture, no SDK patching."""

import json
import struct
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import grpc
import weaviate
from weaviate.connect import ConnectionParams
from weaviate.proto.v1 import (
    aggregate_pb2,
    batch_pb2,
    properties_pb2,
    search_get_pb2,
    tenants_pb2,
    weaviate_pb2_grpc,
)


class Protocol:
    def __init__(self):
        self.requests = []
        self.grpc_requests = []
        self.count = 3
        self.dim = 4
        self.grpc_error = None
        self.schemas = {}
        owner = self

        class HTTP(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def handle_request(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length)) if length else None
                owner.requests.append((self.command, self.path, body))
                path = self.path.split("?")[0]
                code = 200
                result = {}
                if path == "/v1/meta":
                    result = {"version": "1.39.0", "modules": {}}
                elif path == "/v1/schema" and self.command == "POST":
                    schema = {
                        "vectorizer": "none",
                        "vectorIndexType": "flat",
                        "vectorIndexConfig": {
                            "distance": "cosine",
                            "vectorCacheMaxObjects": 1000,
                        },
                        "invertedIndexConfig": {
                            "bm25": {"b": 0.75, "k1": 1.2},
                            "cleanupIntervalSeconds": 60,
                            "stopwords": {
                                "preset": "en",
                                "additions": [],
                                "removals": [],
                            },
                        },
                        "replicationConfig": {"factor": 1},
                        "shardingConfig": {
                            "virtualPerPhysical": 128,
                            "desiredCount": 1,
                            "actualCount": 1,
                            "desiredVirtualCount": 128,
                            "actualVirtualCount": 128,
                            "key": "_id",
                            "strategy": "hash",
                            "function": "murmur3",
                        },
                    }
                    schema.update(body)
                    for prop in schema.get("properties", []):
                        prop.setdefault("indexFilterable", True)
                        prop.setdefault("indexSearchable", True)
                    schema.setdefault("vectorConfig", {})
                    for config in schema["vectorConfig"].values():
                        config.update(
                            {
                                "vectorIndexType": "flat",
                                "vectorIndexConfig": {
                                    "distance": "cosine",
                                    "vectorCacheMaxObjects": 1000,
                                },
                            }
                        )
                    owner.schemas[body["class"]] = schema
                    result = schema
                elif path == "/v1/nodes":
                    result = {"nodes": [{"name": "controlled", "status": "HEALTHY"}]}
                elif "/shards" in path:
                    result = [
                        {"name": "shard", "status": "READY", "vectorQueueSize": 0}
                    ]
                elif path == "/v1/schema":
                    result = {"classes": list(owner.schemas.values())}
                elif path.startswith("/v1/schema/"):
                    name = path.split("/")[3]
                    if self.command == "DELETE":
                        owner.schemas.pop(name, None)
                    elif name in owner.schemas:
                        result = owner.schemas[name]
                    else:
                        code = 404
                        result = {
                            "error": [{"message": "controlled missing collection"}]
                        }
                elif path == "/v1/objects" and self.command == "POST":
                    result = body
                    code = 200
                elif path.startswith("/v1/objects/"):
                    if self.command == "HEAD" or self.command == "DELETE":
                        code = 204
                    else:
                        result = body or {
                            "id": str(uuid.UUID(int=1)),
                            "class": "Docs",
                            "properties": {
                                "text": "native HTTP",
                                "zero": 0,
                                "flag": False,
                            },
                        }
                elif path.startswith("/v1/batch/objects"):
                    result = [
                        dict(x, result={"status": "SUCCESS"}) for x in (body or [])
                    ]
                elif "/tenants" in path:
                    result = body or [{"name": "tenant", "activityStatus": "HOT"}]
                elif "/shards" in path:
                    result = [{"name": "shard", "status": "READY"}]
                payload = json.dumps(result).encode() if code != 204 else b""
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = handle_request

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), HTTP)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

        class RPC(weaviate_pb2_grpc.WeaviateServicer):
            def Search(self, request, context):
                owner.grpc_requests.append(("Search", request))
                if owner.grpc_error:
                    context.abort(owner.grpc_error, "controlled native grpc error")
                rows = []
                for n in range(0 if request.after else owner.count):
                    props = properties_pb2.Properties(
                        fields={
                            "text": properties_pb2.Value(text_value=f"native {n}"),
                            "flag": properties_pb2.Value(bool_value=False),
                            "zero": properties_pb2.Value(int_value=0),
                        }
                    )
                    meta = search_get_pb2.MetadataResult(
                        id=str(uuid.UUID(int=n + 1)),
                        vector_bytes=struct.pack(
                            f"<{owner.dim}f", *([0.25] * owner.dim)
                        ),
                        distance=0.0,
                        distance_present=True,
                    )
                    rows.append(
                        search_get_pb2.SearchResult(
                            properties=search_get_pb2.PropertiesResult(
                                non_ref_props=props,
                                target_collection=request.collection,
                            ),
                            metadata=meta,
                        )
                    )
                return search_get_pb2.SearchReply(results=rows, took=0.0)

            def Aggregate(self, request, context):
                owner.grpc_requests.append(("Aggregate", request))
                return aggregate_pb2.AggregateReply(
                    single_result=aggregate_pb2.AggregateReply.Single(
                        objects_count=owner.count
                    )
                )

            def BatchObjects(self, request, context):
                owner.grpc_requests.append(("BatchObjects", request))
                return batch_pb2.BatchObjectsReply()

            def TenantsGet(self, request, context):
                owner.grpc_requests.append(("TenantsGet", request))
                return tenants_pb2.TenantsGetReply(
                    tenants=[tenants_pb2.Tenant(name="tenant", activity_status=1)]
                )

            def BatchStream(self, requests, context):
                for request in requests:
                    owner.grpc_requests.append(("BatchStream", request))
                    if request.HasField("start"):
                        yield batch_pb2.BatchStreamReply(
                            started=batch_pb2.BatchStreamReply.Started()
                        )
                    elif request.HasField("data"):
                        ids = [obj.uuid for obj in request.data.objects.values]
                        yield batch_pb2.BatchStreamReply(
                            results=batch_pb2.BatchStreamReply.Results(
                                successes=[
                                    batch_pb2.BatchStreamReply.Results.Success(
                                        uuid=value
                                    )
                                    for value in ids
                                ]
                            )
                        )
                        yield batch_pb2.BatchStreamReply(
                            acks=batch_pb2.BatchStreamReply.Acks(uuids=ids)
                        )
                    elif request.HasField("stop"):
                        return

        self.grpc = grpc.server(ThreadPoolExecutor(max_workers=4))
        weaviate_pb2_grpc.add_WeaviateServicer_to_server(RPC(), self.grpc)
        self.grpc_port = self.grpc.add_insecure_port("127.0.0.1:0")

    def __enter__(self):
        self.thread.start()
        self.grpc.start()
        return self

    def __exit__(self, *args):
        self.grpc.stop(0).wait()
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()

    def client(self):
        return weaviate.connect_to_custom(
            http_host="127.0.0.1",
            http_port=self.http.server_port,
            http_secure=False,
            grpc_host="127.0.0.1",
            grpc_port=self.grpc_port,
            grpc_secure=False,
            skip_init_checks=True,
        )

    def async_client(self):
        return weaviate.WeaviateAsyncClient(
            connection_params=ConnectionParams.from_params(
                http_host="127.0.0.1",
                http_port=self.http.server_port,
                http_secure=False,
                grpc_host="127.0.0.1",
                grpc_port=self.grpc_port,
                grpc_secure=False,
            ),
            skip_init_checks=True,
        )


if __name__ == "__main__":
    with Protocol() as p, p.client() as client:
        col = client.collections.use("Docs")
        out = col.query.fetch_objects(limit=75, include_vector=True)
        print(
            json.dumps(
                {
                    "type": type(out).__name__,
                    "rows": len(out.objects),
                    "vectors": len(out.objects[0].vector["default"]),
                    "http": len(p.requests),
                    "grpc": len(p.grpc_requests),
                }
            )
        )
