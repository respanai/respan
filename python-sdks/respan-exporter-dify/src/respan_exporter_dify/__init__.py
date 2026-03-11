"""
Respan exporter for Dify Python SDK.

Import client classes directly from submodules:
    from respan_exporter_dify.exporter import RespanDifyClient, RespanAsyncDifyClient
    from respan_exporter_dify.gateway import RespanGatewayClient, RespanAsyncGatewayClient
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from dify_client import AsyncClient, Client

import respan_exporter_dify.exporter
import respan_exporter_dify.sdk_compat

if TYPE_CHECKING:
    from respan_exporter_dify.exporter import RespanAsyncDifyClient, RespanDifyClient

__version__ = "0.1.0"


def create_client(
    *,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    timeout: int = 60,
    client: Optional[Client] = None,
    dify_api_key: Optional[str] = None,
    dify_api_base: Optional[str] = None,
    gateway_base_url: Optional[str] = None,
    gateway_model: Optional[str] = None,
) -> "RespanDifyClient":
    return respan_exporter_dify.exporter.create_client(
        api_key=api_key,
        endpoint=endpoint,
        timeout=timeout,
        client=client,
        dify_api_key=dify_api_key,
        dify_api_base=dify_api_base,
        gateway_base_url=gateway_base_url,
        gateway_model=gateway_model,
    )


def create_async_client(
    *,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    timeout: int = 60,
    client: Optional[AsyncClient] = None,
    dify_api_key: Optional[str] = None,
    dify_api_base: Optional[str] = None,
    gateway_base_url: Optional[str] = None,
    gateway_model: Optional[str] = None,
) -> "RespanAsyncDifyClient":
    return respan_exporter_dify.exporter.create_async_client(
        api_key=api_key,
        endpoint=endpoint,
        timeout=timeout,
        client=client,
        dify_api_key=dify_api_key,
        dify_api_base=dify_api_base,
        gateway_base_url=gateway_base_url,
        gateway_model=gateway_model,
    )


def flush_pending_exports(*, timeout: Optional[float] = None) -> None:
    respan_exporter_dify.sdk_compat.flush_export_threads(timeout=timeout)


__all__ = ["create_client", "create_async_client", "flush_pending_exports"]
