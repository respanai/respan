"""Local real-GRPC fixture for released Vertex AI SDK clients and wrappers."""

import concurrent.futures
import warnings

import grpc
import vertexai
from google.auth.credentials import AnonymousCredentials
from google.cloud.aiplatform_v1.services.prediction_service import (
    PredictionServiceAsyncClient,
    PredictionServiceClient,
)
from google.cloud.aiplatform_v1.services.prediction_service.transports.grpc import (
    PredictionServiceGrpcTransport,
)
from google.cloud.aiplatform_v1.services.prediction_service.transports.grpc_asyncio import (
    PredictionServiceGrpcAsyncIOTransport,
)
from google.cloud.aiplatform_v1.types import (
    GenerateContentRequest,
    GenerateContentResponse,
    PredictRequest,
    PredictResponse,
)
from google.protobuf.json_format import ParseDict
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel


class NativeRuntime:
    def __init__(self):
        self.requests = []
        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
        handlers = {
            "GenerateContent": grpc.unary_unary_rpc_method_handler(
                self.generate,
                request_deserializer=GenerateContentRequest.deserialize,
                response_serializer=GenerateContentResponse.serialize,
            ),
            "StreamGenerateContent": grpc.unary_stream_rpc_method_handler(
                self.stream,
                request_deserializer=GenerateContentRequest.deserialize,
                response_serializer=GenerateContentResponse.serialize,
            ),
            "Predict": grpc.unary_unary_rpc_method_handler(
                self.predict,
                request_deserializer=PredictRequest.deserialize,
                response_serializer=PredictResponse.serialize,
            ),
        }
        self.server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "google.cloud.aiplatform.v1.PredictionService", handlers
                ),
            )
        )
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{self.port}")
        self.client = PredictionServiceClient(
            transport=PredictionServiceGrpcTransport(
                channel=self.channel, credentials=AnonymousCredentials()
            )
        )
        vertexai.init(
            project="native-fixture",
            location="us-central1",
            credentials=AnonymousCredentials(),
        )

    def text(self, request):
        return " ".join(p.text for c in request.contents for p in c.parts)

    def response(self, text="native response", *, usage=True, tool=False):
        parts = (
            [{"function_call": {"name": "get_weather", "args": {"city": "Tokyo"}}}]
            if tool
            else [{"text": text}]
        )
        value = {
            "candidates": [
                {"content": {"role": "model", "parts": parts}, "finish_reason": "STOP"}
            ]
        }
        if usage:
            value["usage_metadata"] = {
                "prompt_token_count": 7,
                "candidates_token_count": 3,
                "total_token_count": 10,
            }
        if (
            usage
            and "cached_content_token_count"
            in GenerateContentResponse.UsageMetadata.meta.fields
        ):
            value["usage_metadata"]["cached_content_token_count"] = 2
        return GenerateContentResponse(value)

    def generate(self, request, ctx):
        self.requests.append(request)
        text = self.text(request)
        if "failure" in text:
            ctx.abort(grpc.StatusCode.UNAVAILABLE, "controlled provider failure")
        if "blocked" in text:
            return GenerateContentResponse(
                {"prompt_feedback": {"block_reason": "SAFETY"}}
            )
        return self.response(
            tool="weather" in text
            and not any(
                p.function_response.name for c in request.contents for p in c.parts
            )
        )

    def stream(self, request, ctx):
        self.requests.append(request)
        for i in range(70):
            yield self.response(str(i) + ",", usage=i == 69)

    def predict(self, request, ctx):
        self.requests.append(request)
        return PredictResponse.wrap(
            ParseDict(
                {
                    "predictions": [
                        {
                            "embeddings": {
                                "values": [float(i) for i in range(5001)],
                                "statistics": {"token_count": 5, "truncated": False},
                            }
                        }
                        for _ in request.instances
                    ]
                },
                PredictResponse.pb()(),
            )
        )

    def model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GenerativeModel("gemini-2.5-flash")
        model.__dict__["_prediction_client"] = self.client
        model.__dict__["_prediction_client_value"] = self.client
        assert model._prediction_client is self.client, (
            "Native client cache did not select loopback transport"
        )
        return model

    def embedding(self):
        model = TextEmbeddingModel(
            "text-embedding-005",
            endpoint_name="projects/native-fixture/locations/us-central1/publishers/google/models/text-embedding-005",
        )
        model._endpoint._prediction_client_value = self.client
        assert model._endpoint._prediction_client is self.client, (
            "Native embedding client did not select loopback transport"
        )
        return model

    async def async_model(self):
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{self.port}")
        client = PredictionServiceAsyncClient(
            transport=PredictionServiceGrpcAsyncIOTransport(
                channel=channel, credentials=AnonymousCredentials()
            )
        )
        model = self.model()
        model.__dict__["_prediction_async_client"] = client
        model.__dict__["_prediction_async_client_value"] = client
        assert model._prediction_async_client is client, (
            "Native async client cache did not select loopback transport"
        )
        return model, channel

    async def async_embedding(self):
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{self.port}")
        client = PredictionServiceAsyncClient(
            transport=PredictionServiceGrpcAsyncIOTransport(
                channel=channel, credentials=AnonymousCredentials()
            )
        )
        model = self.embedding()
        model._endpoint._prediction_async_client_value = client
        assert model._endpoint._prediction_async_client is client, (
            "Native async embedding client did not select loopback transport"
        )
        return model, channel

    def close(self):
        self.client.transport.close()
        self.server.stop(0).wait()
