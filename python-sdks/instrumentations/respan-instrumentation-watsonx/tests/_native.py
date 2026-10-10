"""Real SDK construction and HTTPX JSON/SSE fixture responses."""

import json
import warnings

import httpx
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference


class Body(httpx.SyncByteStream):
    def __init__(self, frames):
        self.frames = frames
        self.reads = 0
        self.closed = 0

    def __iter__(self):
        for frame in self.frames:
            self.reads += 1
            yield (
                ("data: " + json.dumps(frame) + "\n\n").encode()
                if type(frame) is dict
                else frame
            )

    def close(self):
        self.closed += 1


class ABody(httpx.AsyncByteStream):
    def __init__(self, frames):
        self.body = Body(frames)

    async def __aiter__(self):
        for data in self.body:
            yield data

    async def aclose(self):
        self.body.close()


def runtime(payload=None, *, frames=None, status=200):
    requests = []
    responses = []
    b = Body(frames) if frames is not None else None
    ab = ABody(frames) if frames is not None else None
    default = {
        "model_id": "reported",
        "results": [
            {
                "generated_text": "native",
                "input_token_count": 0,
                "generated_token_count": 0,
                "stop_reason": "eos_token",
            }
        ],
    }

    def handle(request, async_mode=False):
        requests.append(request)
        response = (
            httpx.Response(status, stream=ab if async_mode else b)
            if frames is not None
            else httpx.Response(
                status, json=payload if payload is not None else default
            )
        )
        responses.append(response)
        return response

    async def ahandle(request):
        return handle(request, True)

    h = httpx.Client(transport=httpx.MockTransport(handle))
    a = httpx.AsyncClient(transport=httpx.MockTransport(ahandle))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c = APIClient(
            Credentials(
                url="https://us-south.ml.cloud.ibm.com", token="controlled-token"
            ),
            project_id="controlled-project",
            httpx_client=h,
            async_httpx_client=a,
            scope_validation=False,
        )
        model = ModelInference(
            api_client=c, model_id="ibm/granite", validate=False, max_retries=0
        )
        emb = Embeddings(
            api_client=c, model_id="ibm/slate", batch_size=2, max_retries=0
        )
    return c, model, emb, requests, responses, b, ab
