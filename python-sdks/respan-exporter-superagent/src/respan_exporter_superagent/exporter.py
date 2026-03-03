import logging
import os
from typing import Any, Optional

from respan_sdk.constants import RESPAN_TRACING_INGEST_ENDPOINT
from respan_sdk.respan_types import RespanParams
from respan_sdk.utils.export import send_payloads, validate_payload
from respan_sdk.utils.time import now_utc
from respan_exporter_superagent.utils import build_payload

try:
    from safety_agent import create_client as superagent_create_client
except Exception:  # pragma: no cover
    superagent_create_client = None


logger = logging.getLogger(__name__)


class RespanSuperagentClient:
    """
    Wrapper for Superagent Python SDK (`safety-agent`) that exports call logs to Respan.

    Methods supported:
    - guard
    - redact
    - scan
    - test
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 10,
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("RESPAN_API_KEY")
        self.endpoint = (
            endpoint
            or os.getenv("RESPAN_ENDPOINT")
            or RESPAN_TRACING_INGEST_ENDPOINT
        )
        self.timeout = timeout

        if client is not None:
            self._client = client
        else:
            if superagent_create_client is None:
                raise RuntimeError("safety-agent must be installed to create a Superagent client")
            self._client = superagent_create_client()

    async def guard(self, *, respan_params: Optional[RespanParams] = None, **kwargs: Any) -> Any:
        return await self._call_and_export(method_name="guard", respan_params=respan_params, **kwargs)

    async def redact(self, *, respan_params: Optional[RespanParams] = None, **kwargs: Any) -> Any:
        return await self._call_and_export(method_name="redact", respan_params=respan_params, **kwargs)

    async def scan(self, *, respan_params: Optional[RespanParams] = None, **kwargs: Any) -> Any:
        return await self._call_and_export(method_name="scan", respan_params=respan_params, **kwargs)

    async def test(self, *, respan_params: Optional[RespanParams] = None, **kwargs: Any) -> Any:
        return await self._call_and_export(method_name="test", respan_params=respan_params, **kwargs)

    async def _call_and_export(
        self,
        *,
        method_name: str,
        respan_params: Optional[RespanParams],
        **kwargs: Any,
    ) -> Any:
        params = (
            RespanParams.model_validate(respan_params)
            if respan_params
            else RespanParams()
        )
        if params.disable_log is True:
            method = getattr(self._client, method_name)
            return await method(**kwargs)

        start_time = now_utc()
        end_time = now_utc()
        error_message: Optional[str] = None
        status = "success"

        def _export(result: Any) -> None:
            if not self.api_key:
                return

            input_value = None
            if "input" in kwargs:
                input_value = kwargs.get("input")
            elif "text" in kwargs:
                input_value = kwargs.get("text")
            elif "repo" in kwargs:
                input_value = kwargs.get("repo")

            try:
                payload = build_payload(
                    method_name=method_name,
                    start_time=start_time,
                    end_time=end_time,
                    status=status,
                    input_value=input_value,
                    output_value=result,
                    error_message=error_message,
                    export_params=params,
                )
                validated_payload = validate_payload(payload)
            except Exception as exc:
                logger.exception("Failed to validate Respan payload: %s", exc)
                return

            send_payloads(
                api_key=self.api_key,
                endpoint=self.endpoint,
                timeout=self.timeout,
                payloads=[validated_payload],
                context="respan superagent ingest",
            )

        try:
            method = getattr(self._client, method_name)
            result = await method(**kwargs)
            end_time = now_utc()
            _export(result)
            return result
        except Exception as exc:
            end_time = now_utc()
            status = "error"
            error_message = str(exc)
            _export(None)
            raise


def create_client(
    *,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    timeout: int = 10,
    client: Optional[Any] = None,
) -> RespanSuperagentClient:
    """
    Create a Respan-exporting Superagent client.

    This intentionally mirrors `safety_agent.create_client()` while adding:
    - `RESPAN_API_KEY` / `RESPAN_ENDPOINT` forwarding
    - automatic export of `guard`, `redact`, `scan`, `test` call logs
    """
    return RespanSuperagentClient(api_key=api_key, endpoint=endpoint, timeout=timeout, client=client)

