"""Patching lifecycle and fail-open behaviour (no browser needed: httpx.MockTransport)."""

from __future__ import annotations

import json

import httpx
import kernel
import pytest
from kernel import Kernel
from kernel.resources.browsers.computer import AsyncComputerResource, ComputerResource
from respan_instrumentation_kernel import KernelInstrumentor, _instrumentation
from respan_instrumentation_kernel._serialization import call_arguments, decode_b64

BROWSER = {
    "session_id": "s1",
    "cdp_ws_url": "wss://vm.example/cdp?jwt=t0ken",
    "webdriver_ws_url": "wss://vm.example/wd?jwt=t0ken",
    "created_at": "2026-10-16T00:00:00Z",
    "headless": True,
    "stealth": False,
    "region": "us-east",
    "timeout_seconds": 60,
    "memory": {"total_mb": 1024},
}


def mock_client() -> Kernel:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/browsers":
            return httpx.Response(200, json=BROWSER)
        return httpx.Response(200)

    return Kernel(
        api_key="k",
        base_url="http://kernel.test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def test_activate_patches_sync_and_async_and_deactivate_restores():
    original_sync = ComputerResource.__dict__["click_mouse"]
    original_async = AsyncComputerResource.__dict__["click_mouse"]
    original_action = kernel.KernelApp.__dict__["action"]
    instrumentor = KernelInstrumentor()
    instrumentor.activate()
    try:
        assert ComputerResource.__dict__["click_mouse"] is not original_sync
        assert AsyncComputerResource.__dict__["click_mouse"] is not original_async
        assert kernel.KernelApp.__dict__["action"] is not original_action
    finally:
        instrumentor.deactivate()
    assert ComputerResource.__dict__["click_mouse"] is original_sync
    assert AsyncComputerResource.__dict__["click_mouse"] is original_async
    assert kernel.KernelApp.__dict__["action"] is original_action


def test_second_instrumentor_shares_the_patches_and_rejects_other_settings():
    original = ComputerResource.__dict__["type_text"]
    first, second = KernelInstrumentor(), KernelInstrumentor()
    first.activate()
    second.activate()
    try:
        with pytest.raises(ValueError):
            KernelInstrumentor(capture_content=False).activate()
        first.deactivate()
        assert ComputerResource.__dict__["type_text"] is not original
    finally:
        second.deactivate()
    assert ComputerResource.__dict__["type_text"] is original


def test_calls_still_work_when_telemetry_fails(monkeypatch):
    instrumentor = KernelInstrumentor()
    instrumentor.activate()
    try:

        def broken(self, result):
            raise RuntimeError("telemetry bug")

        monkeypatch.setattr(_instrumentation._Call, "finish", broken)
        assert mock_client().browsers.create().session_id == "s1"
    finally:
        instrumentor.deactivate()


def test_raw_response_wrappers_go_through_the_patch():
    instrumentor = KernelInstrumentor()
    instrumentor.activate()
    try:
        raw = mock_client().browsers.with_raw_response.create()
        assert json.loads(raw.http_response.content)["session_id"] == "s1"
    finally:
        instrumentor.deactivate()


def test_handlers_registered_while_active_pass_through_after_deactivate():
    instrumentor = KernelInstrumentor()
    instrumentor.activate()
    app = kernel.App("lifecycle")

    @app.action("echo")
    def echo(ctx: kernel.KernelContext, payload: dict) -> dict:
        return payload

    instrumentor.deactivate()
    handler = app.get_action("echo").handler
    assert handler(kernel.KernelContext(invocation_id="i"), {"a": 1}) == {"a": 1}


def test_call_arguments_drop_unset_and_transport_options():
    def method(
        self, id_or_name, *, text, delay=kernel.omit, extra_headers=None, timeout=kernel.not_given
    ):
        pass

    assert call_arguments(method, ("s1",), {"text": "hi", "extra_headers": {"x": "y"}}) == {
        "id_or_name": "s1",
        "text": "hi",
    }


def test_decode_b64_keeps_non_base64_text():
    assert decode_b64("aGkK") == "hi\n"
    assert decode_b64("not base64!") == "not base64!"
