import asyncio
import json
import os
import threading
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Sequence, TypeVar, Union

import requests
from dify_client import models


DEFAULT_RESPAN_BASE_URL = "https://api.respan.ai/api"
DEFAULT_RESPAN_MODEL = "gpt-4o-mini"
_ASYNC_STREAM_SENTINEL = object()
_T = TypeVar("_T")


def _to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _stringify_for_prompt(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
                continue
            if item.get("type") == "output_text" and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if text is not None:
            return str(text)
    return ""


def _build_usage(*, usage: Dict[str, Any], latency: float) -> Dict[str, Any]:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_unit_price": str(usage.get("prompt_unit_price") or "0"),
        "prompt_price_unit": str(usage.get("prompt_price_unit") or "0"),
        "prompt_price": str(usage.get("prompt_price") or "0"),
        "completion_unit_price": str(usage.get("completion_unit_price") or "0"),
        "completion_price_unit": str(usage.get("completion_price_unit") or "0"),
        "completion_price": str(usage.get("completion_price") or "0"),
        "total_price": str(usage.get("total_price") or "0"),
        "currency": str(usage.get("currency") or "USD"),
        "latency": latency,
    }


def _build_metadata(*, usage: Dict[str, Any], latency: float) -> Dict[str, Any]:
    return {
        "usage": _build_usage(usage=usage, latency=latency),
        "retriever_resources": [],
    }


def _coerce_created_at(value: Any) -> int:
    if isinstance(value, int):
        return value
    return int(time.time())


def _prepare_messages(*, text: str, inputs: Dict[str, Any], files: Sequence[Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if inputs:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Use these application inputs as context when answering.\n\n"
                    f"{_stringify_for_prompt(inputs)}"
                ),
            }
        )

    if not files:
        messages.append({"role": "user", "content": text})
        return messages

    content: List[Dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})

    for file in files:
        file_data = _to_dict(file)
        transfer_method = file_data.get("transfer_method")
        file_type = file_data.get("type")
        url = file_data.get("url")
        if transfer_method != "remote_url":
            raise ValueError("Gateway mode only supports remote_url files.")
        if file_type != "image":
            raise ValueError("Gateway mode only supports image files.")
        if not url:
            raise ValueError("Gateway mode requires file URLs for remote_url inputs.")
        content.append({"type": "image_url", "image_url": {"url": url}})

    if len(content) == 1 and content[0]["type"] == "text":
        messages.append({"role": "user", "content": text})
    else:
        messages.append({"role": "user", "content": content})
    return messages


def _extract_chat_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return _extract_text(message.get("content"))


def _extract_stream_delta(chunk_json: Dict[str, Any]) -> str:
    choices = chunk_json.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return _extract_text(delta.get("content"))


class _BaseRespanGatewayClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.api_base = (base_url or os.getenv("RESPAN_BASE_URL") or DEFAULT_RESPAN_BASE_URL).rstrip("/")
        self.model = model or os.getenv("RESPAN_MODEL") or DEFAULT_RESPAN_MODEL
        self.timeout = timeout

    def _chat_completions_url(self) -> str:
        return f"{self.api_base}/chat/completions"

    def _headers(self, headers: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        merged = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if headers:
            merged.update(headers)
        return merged

    def _request_timeout(self, **kwargs: Any) -> Any:
        return kwargs.get("timeout", self.timeout)

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(f"Respan gateway request failed: {detail}") from exc

    def _chat_payload(self, req: models.ChatRequest) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": _prepare_messages(
                text=req.query,
                inputs=_to_dict(req.inputs),
                files=req.files,
            ),
            "disable_log": True,
        }

    def _completion_payload(self, req: models.CompletionRequest) -> Dict[str, Any]:
        inputs = _to_dict(req.inputs)
        query = str(inputs.pop("query", ""))
        return {
            "model": self.model,
            "messages": _prepare_messages(
                text=query,
                inputs=inputs,
                files=req.files,
            ),
            "disable_log": True,
        }

    def _workflow_payload(self, req: models.WorkflowsRunRequest) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are emulating a Dify workflow run over the Respan gateway. "
                        "Produce a helpful result from the provided workflow inputs."
                    ),
                },
                {
                    "role": "user",
                    "content": _stringify_for_prompt(_to_dict(req.inputs)),
                },
            ],
            "disable_log": True,
        }

    def _blocking_completion_response(
        self,
        *,
        response_json: Dict[str, Any],
        conversation_id: str,
        mode: str,
        latency: float,
        response_cls: type[_T],
    ) -> _T:
        return response_cls(
            message_id=response_json.get("id") or str(uuid.uuid4()),
            conversation_id=conversation_id,
            mode=mode,
            answer=_extract_chat_text(response_json),
            metadata=_build_metadata(
                usage=response_json.get("usage") or {},
                latency=latency,
            ),
            created_at=_coerce_created_at(response_json.get("created")),
        )

    def _workflow_response(
        self,
        *,
        response_json: Dict[str, Any],
        latency: float,
    ) -> models.WorkflowsRunResponse:
        workflow_run_id = response_json.get("id") or str(uuid.uuid4())
        created_at = _coerce_created_at(response_json.get("created"))
        usage = response_json.get("usage") or {}
        usage_model = _build_usage(usage=usage, latency=latency)
        data = models.WorkflowFinishedData(
            id=workflow_run_id,
            workflow_id="respan-gateway",
            sequence_number=1,
            status=models.WorkflowStatus.SUCCEEDED,
            outputs={"text": _extract_chat_text(response_json)},
            error=None,
            elapsed_time=latency,
            total_tokens=int(usage_model["total_tokens"]),
            total_steps=1,
            created_at=created_at,
            finished_at=int(time.time()),
            created_by={},
            files=[],
        )
        return models.WorkflowsRunResponse(
            log_id=workflow_run_id,
            task_id=workflow_run_id,
            data=data,
        )

    def _iter_sse_json(
        self,
        *,
        payload: Dict[str, Any],
        headers: Optional[Mapping[str, str]] = None,
        timeout: Any = None,
    ) -> Iterator[Dict[str, Any]]:
        with requests.post(
            self._chat_completions_url(),
            headers=self._headers(headers),
            json=payload,
            timeout=timeout or self.timeout,
            stream=True,
        ) as response:
            self._raise_for_status(response)
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    continue
                yield json.loads(data)

    def _stream_completion_events(
        self,
        *,
        payload: Dict[str, Any],
        conversation_id: str,
        builder: Any,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Any = None,
    ) -> Iterator[Any]:
        start_time = time.time()
        message_id = str(uuid.uuid4())
        last_created_at = int(start_time)
        final_usage: Dict[str, Any] = {}

        for chunk_json in self._iter_sse_json(payload=payload, headers=headers, timeout=timeout):
            message_id = chunk_json.get("id") or message_id
            last_created_at = _coerce_created_at(chunk_json.get("created"))
            chunk_usage = chunk_json.get("usage")
            if isinstance(chunk_usage, dict):
                final_usage = chunk_usage

            delta_text = _extract_stream_delta(chunk_json)
            if not delta_text:
                continue

            yield builder(
                {
                    "event": models.StreamEvent.MESSAGE.value,
                    "task_id": message_id,
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "answer": delta_text,
                    "created_at": last_created_at,
                }
            )

        yield builder(
            {
                "event": models.StreamEvent.MESSAGE_END.value,
                "task_id": message_id,
                "message_id": message_id,
                "conversation_id": conversation_id,
                "created_at": int(time.time()),
                "metadata": _build_metadata(
                    usage=final_usage,
                    latency=time.time() - start_time,
                ),
            }
        )

    def _workflow_stream_events(
        self,
        *,
        req: models.WorkflowsRunRequest,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Any = None,
    ) -> Iterator[Any]:
        start_time = int(time.time())
        workflow_run_id = str(uuid.uuid4())
        started_event = models.build_workflows_stream_response(
            {
                "event": models.StreamEvent.WORKFLOW_STARTED.value,
                "task_id": workflow_run_id,
                "workflow_run_id": workflow_run_id,
                "data": {
                    "id": workflow_run_id,
                    "workflow_id": "respan-gateway",
                    "sequence_number": 1,
                    "inputs": _to_dict(req.inputs),
                    "created_at": start_time,
                },
            }
        )
        yield started_event

        node_started = models.build_workflows_stream_response(
            {
                "event": models.StreamEvent.NODE_STARTED.value,
                "task_id": workflow_run_id,
                "workflow_run_id": workflow_run_id,
                "data": {
                    "id": workflow_run_id,
                    "node_id": "gateway-chat-completion",
                    "node_type": "llm",
                    "title": "Respan Gateway",
                    "index": 0,
                    "inputs": _to_dict(req.inputs),
                    "created_at": start_time,
                    "extras": {},
                },
            }
        )
        yield node_started

        payload = self._workflow_payload(req)
        start = time.time()
        response_json = self._response_json_from_stream(
            payload=payload,
            headers=headers,
            timeout=timeout,
        )
        latency = time.time() - start
        answer = _extract_chat_text(response_json)
        usage = response_json.get("usage") or {}
        finished_at = int(time.time())

        node_finished = models.build_workflows_stream_response(
            {
                "event": models.StreamEvent.NODE_FINISHED.value,
                "task_id": workflow_run_id,
                "workflow_run_id": workflow_run_id,
                "data": {
                    "id": workflow_run_id,
                    "node_id": "gateway-chat-completion",
                    "node_type": "llm",
                    "title": "Respan Gateway",
                    "index": 0,
                    "inputs": _to_dict(req.inputs),
                    "process_data": None,
                    "outputs": {"text": answer},
                    "status": models.WorkflowStatus.SUCCEEDED.value,
                    "error": None,
                    "elapsed_time": latency,
                    "execution_metadata": {
                        "total_tokens": int(usage.get("total_tokens") or 0),
                        "total_price": str(usage.get("total_price") or "0"),
                        "currency": str(usage.get("currency") or "USD"),
                    },
                    "created_at": start_time,
                    "finished_at": finished_at,
                    "files": [],
                },
            }
        )
        yield node_finished

        workflow_finished = models.build_workflows_stream_response(
            {
                "event": models.StreamEvent.WORKFLOW_FINISHED.value,
                "task_id": workflow_run_id,
                "workflow_run_id": workflow_run_id,
                "data": {
                    "id": workflow_run_id,
                    "workflow_id": "respan-gateway",
                    "sequence_number": 1,
                    "status": models.WorkflowStatus.SUCCEEDED.value,
                    "outputs": {"text": answer},
                    "error": None,
                    "elapsed_time": latency,
                    "total_tokens": int(usage.get("total_tokens") or 0),
                    "total_steps": 1,
                    "created_at": start_time,
                    "finished_at": finished_at,
                    "created_by": {},
                    "files": [],
                },
            }
        )
        yield workflow_finished

    def _request_json(
        self,
        *,
        payload: Dict[str, Any],
        headers: Optional[Mapping[str, str]] = None,
        timeout: Any = None,
    ) -> Dict[str, Any]:
        response = requests.post(
            self._chat_completions_url(),
            headers=self._headers(headers),
            json=payload,
            timeout=timeout or self.timeout,
        )
        self._raise_for_status(response)
        return response.json()

    def _response_json_from_stream(
        self,
        *,
        payload: Dict[str, Any],
        headers: Optional[Mapping[str, str]] = None,
        timeout: Any = None,
    ) -> Dict[str, Any]:
        stream_payload = dict(payload)
        stream_payload["stream"] = True

        response_id = str(uuid.uuid4())
        created_at = int(time.time())
        model_name = self.model
        text_parts: List[str] = []
        final_usage: Dict[str, Any] = {}

        for chunk_json in self._iter_sse_json(payload=stream_payload, headers=headers, timeout=timeout):
            response_id = chunk_json.get("id") or response_id
            if chunk_json.get("created") is not None:
                created_at = _coerce_created_at(chunk_json.get("created"))
            if chunk_json.get("model"):
                model_name = chunk_json["model"]

            chunk_usage = chunk_json.get("usage")
            if isinstance(chunk_usage, dict):
                final_usage = chunk_usage

            delta_text = _extract_stream_delta(chunk_json)
            if delta_text:
                text_parts.append(delta_text)

        choices = []
        if text_parts:
            choices = [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "".join(text_parts),
                    },
                }
            ]

        return {
            "id": response_id,
            "created": created_at,
            "model": model_name,
            "object": "chat.completion",
            "choices": choices,
            "usage": final_usage,
        }

    def _chat_messages_blocking(self, req: models.ChatRequest, **kwargs: Any) -> models.ChatResponse:
        conversation_id = req.conversation_id or str(uuid.uuid4())
        payload = self._chat_payload(req)
        start = time.time()
        # Aggregate the streaming path because the blocking gateway response
        # can return usage metadata with empty choices.
        response_json = self._response_json_from_stream(
            payload=payload,
            headers=kwargs.get("headers"),
            timeout=self._request_timeout(**kwargs),
        )
        latency = time.time() - start
        return self._blocking_completion_response(
            response_json=response_json,
            conversation_id=conversation_id,
            mode="chat",
            latency=latency,
            response_cls=models.ChatResponse,
        )

    def _completion_messages_blocking(self, req: models.CompletionRequest, **kwargs: Any) -> models.CompletionResponse:
        conversation_id = req.conversation_id or ""
        payload = self._completion_payload(req)
        start = time.time()
        response_json = self._response_json_from_stream(
            payload=payload,
            headers=kwargs.get("headers"),
            timeout=self._request_timeout(**kwargs),
        )
        latency = time.time() - start
        return self._blocking_completion_response(
            response_json=response_json,
            conversation_id=conversation_id,
            mode="completion",
            latency=latency,
            response_cls=models.CompletionResponse,
        )

    def _run_workflows_blocking(self, req: models.WorkflowsRunRequest, **kwargs: Any) -> models.WorkflowsRunResponse:
        payload = self._workflow_payload(req)
        start = time.time()
        response_json = self._response_json_from_stream(
            payload=payload,
            headers=kwargs.get("headers"),
            timeout=self._request_timeout(**kwargs),
        )
        latency = time.time() - start
        return self._workflow_response(response_json=response_json, latency=latency)

    def _chat_messages_stream(self, req: models.ChatRequest, **kwargs: Any) -> Iterator[Any]:
        conversation_id = req.conversation_id or str(uuid.uuid4())
        payload = self._chat_payload(req)
        payload["stream"] = True
        yield from self._stream_completion_events(
            payload=payload,
            conversation_id=conversation_id,
            builder=models.build_chat_stream_response,
            headers=kwargs.get("headers"),
            timeout=self._request_timeout(**kwargs),
        )

    def _completion_messages_stream(self, req: models.CompletionRequest, **kwargs: Any) -> Iterator[Any]:
        conversation_id = req.conversation_id or ""
        payload = self._completion_payload(req)
        payload["stream"] = True
        yield from self._stream_completion_events(
            payload=payload,
            conversation_id=conversation_id,
            builder=models.build_completion_stream_response,
            headers=kwargs.get("headers"),
            timeout=self._request_timeout(**kwargs),
        )


class RespanGatewayClient(_BaseRespanGatewayClient):
    def chat_messages(
        self,
        req: models.ChatRequest,
        **kwargs: Any,
    ) -> Union[models.ChatResponse, Iterator[models.ChatStreamResponse]]:
        if req.response_mode == models.ResponseMode.BLOCKING:
            return self._chat_messages_blocking(req, **kwargs)
        if req.response_mode == models.ResponseMode.STREAMING:
            return self._chat_messages_stream(req, **kwargs)
        raise ValueError(f"Invalid request_mode: {req.response_mode}")

    def completion_messages(
        self,
        req: models.CompletionRequest,
        **kwargs: Any,
    ) -> Union[models.CompletionResponse, Iterator[models.CompletionStreamResponse]]:
        if req.response_mode == models.ResponseMode.BLOCKING:
            return self._completion_messages_blocking(req, **kwargs)
        if req.response_mode == models.ResponseMode.STREAMING:
            return self._completion_messages_stream(req, **kwargs)
        raise ValueError(f"Invalid request_mode: {req.response_mode}")

    def run_workflows(
        self,
        req: models.WorkflowsRunRequest,
        **kwargs: Any,
    ) -> Union[models.WorkflowsRunResponse, Iterator[models.WorkflowsRunStreamResponse]]:
        if req.response_mode == models.ResponseMode.BLOCKING:
            return self._run_workflows_blocking(req, **kwargs)
        if req.response_mode == models.ResponseMode.STREAMING:
            return self._workflow_stream_events(
                req=req,
                headers=kwargs.get("headers"),
                timeout=self._request_timeout(**kwargs),
            )
        raise ValueError(f"Invalid request_mode: {req.response_mode}")


class RespanAsyncGatewayClient(_BaseRespanGatewayClient):
    async def _run_in_thread(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _sync_iter_to_async(self, iterator_factory: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()

        def worker() -> None:
            try:
                for item in iterator_factory(*args, **kwargs):
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_ASYNC_STREAM_SENTINEL), loop).result()

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is _ASYNC_STREAM_SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def achat_messages(
        self,
        req: models.ChatRequest,
        **kwargs: Any,
    ) -> Union[models.ChatResponse, AsyncIterator[models.ChatStreamResponse]]:
        if req.response_mode == models.ResponseMode.BLOCKING:
            return await self._run_in_thread(self._chat_messages_blocking, req, **kwargs)
        if req.response_mode == models.ResponseMode.STREAMING:
            return self._sync_iter_to_async(self._chat_messages_stream, req, **kwargs)
        raise ValueError(f"Invalid request_mode: {req.response_mode}")

    async def acompletion_messages(
        self,
        req: models.CompletionRequest,
        **kwargs: Any,
    ) -> Union[models.CompletionResponse, AsyncIterator[models.CompletionStreamResponse]]:
        if req.response_mode == models.ResponseMode.BLOCKING:
            return await self._run_in_thread(self._completion_messages_blocking, req, **kwargs)
        if req.response_mode == models.ResponseMode.STREAMING:
            return self._sync_iter_to_async(self._completion_messages_stream, req, **kwargs)
        raise ValueError(f"Invalid request_mode: {req.response_mode}")

    async def arun_workflows(
        self,
        req: models.WorkflowsRunRequest,
        **kwargs: Any,
    ) -> Union[models.WorkflowsRunResponse, AsyncIterator[models.WorkflowsRunStreamResponse]]:
        if req.response_mode == models.ResponseMode.BLOCKING:
            return await self._run_in_thread(self._run_workflows_blocking, req, **kwargs)
        if req.response_mode == models.ResponseMode.STREAMING:
            return self._sync_iter_to_async(
                self._workflow_stream_events,
                req,
                headers=kwargs.get("headers"),
                timeout=self._request_timeout(**kwargs),
            )
        raise ValueError(f"Invalid request_mode: {req.response_mode}")
