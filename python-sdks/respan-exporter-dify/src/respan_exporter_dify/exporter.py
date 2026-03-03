import logging
import os
from datetime import datetime
from typing import Any, Optional

from dify_client import AsyncClient, Client
from dify_client.models import ResponseMode
from respan_sdk.constants import RESPAN_TRACING_INGEST_ENDPOINT
from respan_sdk.respan_types import RespanParams
from respan_sdk.utils.time import now_utc
from respan_exporter_dify.utils import export_dify_call


logger = logging.getLogger(__name__)


class _BaseDifyClient:
    def __init__(
        self,
        client_cls: type,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 10,
        client: Optional[Any] = None,
        dify_api_key: Optional[str] = None,
        dify_api_base: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("RESPAN_API_KEY")
        self.endpoint = endpoint or os.getenv("RESPAN_ENDPOINT") or RESPAN_TRACING_INGEST_ENDPOINT
        self.timeout = timeout

        if client is not None:
            self._client = client
        elif dify_api_key is not None:
            self._client = client_cls(api_key=dify_api_key, api_base=dify_api_base or "https://api.dify.ai/v1")
        else:
            raise RuntimeError(f"Must provide a {client_cls.__name__} or dify_api_key")

    def _get_export_func(self, method_name: str, start_time: datetime, kwargs: Any, params: RespanParams):
        # Capture only the minimal kwargs needed for export (payload building uses "req").
        # Avoids holding other keys from the caller's kwargs in the closure until export runs.
        export_kwargs = {k: kwargs[k] for k in ("req",) if k in kwargs}

        def _export(
            *,
            end_time: datetime,
            status: str,
            result: Any,
            error_message: Optional[str],
        ) -> None:
            export_dify_call(
                api_key=self.api_key,
                endpoint=self.endpoint,
                timeout=self.timeout,
                method_name=method_name,
                start_time=start_time,
                end_time=end_time,
                status=status,
                kwargs=export_kwargs,
                result=result,
                error_message=error_message,
                params=params,
            )
        return _export


class RespanDifyClient(_BaseDifyClient):
    """
    Wrapper for Dify Python SDK (`dify-client-python`) that exports call logs to Respan.

    **Streaming:** For streaming responses, export runs in the generator's ``finally``
    block and therefore only when the stream is consumed (iterated to completion or
    closed). If the caller never iterates the returned stream, no export is sent.
    This is acceptable when callers always consume or explicitly close streams.

    **Streaming memory:** All stream chunks are accumulated in memory and passed to
    export when the stream ends. For long streams this can use significant memory;
    an alternative would be to export only a final aggregated result (e.g. summary
    or last chunk) instead of the full list of chunks.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 10,
        client: Optional["Client"] = None,
        dify_api_key: Optional[str] = None,
        dify_api_base: Optional[str] = None,
    ) -> None:
        super().__init__(
            client_cls=Client,
            api_key=api_key,
            endpoint=endpoint,
            timeout=timeout,
            client=client,
            dify_api_key=dify_api_key,
            dify_api_base=dify_api_base,
        )

    def _call_and_export(
        self,
        *,
        method_name: str,
        respan_params: Optional[RespanParams] = None,
        **kwargs: Any,
    ) -> Any:
        params = RespanParams.model_validate(respan_params) if respan_params else RespanParams()
        method = getattr(self._client, method_name)

        if params.disable_log is True:
            return method(**kwargs)

        req = kwargs.get("req")
        is_stream = req and getattr(req, "response_mode", None) == ResponseMode.STREAMING

        start_time = now_utc()
        _export = self._get_export_func(method_name, start_time, kwargs, params)

        try:
            result = method(**kwargs)

            if is_stream:
                def stream_generator():
                    # Accumulate all chunks for export in finally; long streams use more memory.
                    events = []
                    status = "success"
                    error_msg = None
                    try:
                        for chunk in result:
                            events.append(chunk)
                            yield chunk
                    except Exception as exc:
                        status = "error"
                        error_msg = str(exc)
                        raise
                    finally:
                        _export(
                            end_time=now_utc(),
                            status=status,
                            result=events,
                            error_message=error_msg,
                        )

                return stream_generator()
            else:
                _export(
                    end_time=now_utc(),
                    status="success",
                    result=result,
                    error_message=None,
                )
                return result

        except Exception as exc:
            _export(
                end_time=now_utc(),
                status="error",
                result=None,
                error_message=str(exc),
            )
            raise

    def chat_messages(self, req, *, respan_params: Optional[RespanParams] = None, **kwargs: Any) -> Any:
        return self._call_and_export(method_name="chat_messages", respan_params=respan_params, req=req, **kwargs)

    def completion_messages(self, req, *, respan_params: Optional[RespanParams] = None, **kwargs: Any) -> Any:
        return self._call_and_export(method_name="completion_messages", respan_params=respan_params, req=req, **kwargs)

    def run_workflows(self, req, *, respan_params: Optional[RespanParams] = None, **kwargs: Any) -> Any:
        return self._call_and_export(method_name="run_workflows", respan_params=respan_params, req=req, **kwargs)


class RespanAsyncDifyClient(_BaseDifyClient):
    """
    Wrapper for Dify Python SDK (`dify-client-python`) AsyncClient that exports call logs to Respan.

    **Streaming:** For streaming responses, export runs in the async generator's
    ``finally`` block and therefore only when the stream is consumed (iterated to
    completion or closed). If the caller never iterates the returned stream, no
    export is sent. This is acceptable when callers always consume or explicitly
    close streams.

    **Streaming memory:** All stream chunks are accumulated in memory and passed to
    export when the stream ends. For long streams this can use significant memory;
    an alternative would be to export only a final aggregated result (e.g. summary
    or last chunk) instead of the full list of chunks.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 10,
        client: Optional["AsyncClient"] = None,
        dify_api_key: Optional[str] = None,
        dify_api_base: Optional[str] = None,
    ) -> None:
        super().__init__(
            client_cls=AsyncClient,
            api_key=api_key,
            endpoint=endpoint,
            timeout=timeout,
            client=client,
            dify_api_key=dify_api_key,
            dify_api_base=dify_api_base,
        )

    async def _call_and_export(
        self,
        *,
        method_name: str,
        respan_params: Optional[RespanParams] = None,
        **kwargs: Any,
    ) -> Any:
        params = RespanParams.model_validate(respan_params) if respan_params else RespanParams()
        method = getattr(self._client, method_name)

        if params.disable_log is True:
            return await method(**kwargs)

        req = kwargs.get("req")
        is_stream = req and getattr(req, "response_mode", None) == ResponseMode.STREAMING

        start_time = now_utc()
        _export = self._get_export_func(method_name, start_time, kwargs, params)

        try:
            result = await method(**kwargs)

            if is_stream:
                async def stream_generator():
                    # Accumulate all chunks for export in finally; long streams use more memory.
                    events = []
                    status = "success"
                    error_msg = None
                    try:
                        async for chunk in result:
                            events.append(chunk)
                            yield chunk
                    except Exception as exc:
                        status = "error"
                        error_msg = str(exc)
                        raise
                    finally:
                        _export(
                            end_time=now_utc(),
                            status=status,
                            result=events,
                            error_message=error_msg,
                        )

                return stream_generator()
            else:
                _export(
                    end_time=now_utc(),
                    status="success",
                    result=result,
                    error_message=None,
                )
                return result

        except Exception as exc:
            _export(
                end_time=now_utc(),
                status="error",
                result=None,
                error_message=str(exc),
            )
            raise

    async def achat_messages(self, req, *, respan_params: Optional[RespanParams] = None, **kwargs: Any) -> Any:
        return await self._call_and_export(method_name="achat_messages", respan_params=respan_params, req=req, **kwargs)

    async def acompletion_messages(self, req, *, respan_params: Optional[RespanParams] = None, **kwargs: Any) -> Any:
        return await self._call_and_export(method_name="acompletion_messages", respan_params=respan_params, req=req, **kwargs)

    async def arun_workflows(self, req, *, respan_params: Optional[RespanParams] = None, **kwargs: Any) -> Any:
        return await self._call_and_export(method_name="arun_workflows", respan_params=respan_params, req=req, **kwargs)


def create_client(
    *,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    timeout: int = 10,
    client: Optional["Client"] = None,
    dify_api_key: Optional[str] = None,
    dify_api_base: Optional[str] = None,
) -> RespanDifyClient:
    """
    Create a Respan-exporting Dify Client.
    """
    return RespanDifyClient(
        api_key=api_key,
        endpoint=endpoint,
        timeout=timeout,
        client=client,
        dify_api_key=dify_api_key,
        dify_api_base=dify_api_base,
    )


def create_async_client(
    *,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    timeout: int = 10,
    client: Optional["AsyncClient"] = None,
    dify_api_key: Optional[str] = None,
    dify_api_base: Optional[str] = None,
) -> RespanAsyncDifyClient:
    """
    Create a Respan-exporting Dify AsyncClient.
    """
    return RespanAsyncDifyClient(
        api_key=api_key,
        endpoint=endpoint,
        timeout=timeout,
        client=client,
        dify_api_key=dify_api_key,
        dify_api_base=dify_api_base,
    )
