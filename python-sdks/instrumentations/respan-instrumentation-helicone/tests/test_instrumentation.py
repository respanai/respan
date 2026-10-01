from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import tomllib
from respan_instrumentation_helicone import HeliconeInstrumentor, _instrumentation
from respan_tracing.core.tracer import RespanTracer


def test_manifest_pins_tested_helicone_helpers_minor_line():
    manifest = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["tool"]["poetry"]["dependencies"]["helicone-helpers"] == (
        ">=1.2.1,<1.3.0"
    )
    assert manifest["tool"]["poetry"]["dependencies"]["respan-sdk"] == (
        ">=2.7.4,<3.0.0"
    )
    assert (
        manifest["tool"]["poetry"]["dependencies"]["opentelemetry-semantic-conventions"]
        == ">=0.65b0,<0.66"
    )


class FakeLogger:
    send_calls: ClassVar[list[tuple]] = []

    def __init__(self, headers=None):
        self.headers = dict(headers or {})

    def send_log(self, provider, request, response, options):
        self.__class__.send_calls.append((provider, request, response, options))
        return "sent"

    def log_request(
        self,
        request,
        operation,
        additional_headers=None,
        provider=None,
    ):
        additional_headers = additional_headers or {}
        recorder = SimpleNamespace(
            request=request,
            results={},
            append_results=lambda value: recorder.results.update(value),
            get_results=lambda: recorder.results,
        )
        result = operation(recorder)
        self.send_log(
            provider,
            request,
            recorder.get_results(),
            {
                "start_time": 1.0,
                "end_time": 2.0,
                "additional_headers": additional_headers,
            },
        )
        return result

    def log_builder(self, request, additional_headers=None):
        builder = FakeBuilder(self)
        builder.request = dict(request)
        return builder


class FakeBuilder:
    def __init__(self, logger):
        self.logger = logger
        self.request = {"model": "builder-model", "messages": []}
        self.response_body = "builder response"
        self.error = None
        self.status = 200
        self.was_cancelled = False
        self.stream_chunks = []

    async def send_log(self):
        return self.logger.send_log(
            None,
            self.request,
            self.response_body,
            {"start_time": 1.0, "end_time": 2.0},
        )


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    if _instrumentation._GENERATION is not None:
        _instrumentation._GENERATION.enabled = False
    _instrumentation._restore_all()
    _instrumentation._REFCOUNT = 0
    _instrumentation._CONFIG = None
    _instrumentation._GENERATION = None
    FakeLogger.send_calls.clear()
    RespanTracer.reset_instance()
    monkeypatch.setattr(
        _instrumentation,
        "_load_classes",
        lambda: (FakeLogger, FakeBuilder),
    )
    yield
    if _instrumentation._GENERATION is not None:
        _instrumentation._GENERATION.enabled = False
    _instrumentation._restore_all()
    _instrumentation._REFCOUNT = 0
    _instrumentation._CONFIG = None
    _instrumentation._GENERATION = None
    RespanTracer.reset_instance()


def test_activate_patches_all_canonical_surfaces_and_deactivate_restores():
    originals = (
        FakeLogger.send_log,
        FakeLogger.log_request,
        FakeLogger.log_builder,
        FakeBuilder.send_log,
    )
    instrumentor = HeliconeInstrumentor()

    instrumentor.activate()

    assert instrumentor._is_instrumented is True
    assert FakeLogger.send_log.__respan_helicone_wrapper__ is True
    assert FakeLogger.log_request.__respan_helicone_wrapper__ is True
    assert FakeLogger.log_builder.__respan_helicone_wrapper__ is True
    assert FakeBuilder.send_log.__respan_helicone_wrapper__ is True

    instrumentor.deactivate()

    assert instrumentor._is_instrumented is False
    assert (
        FakeLogger.send_log,
        FakeLogger.log_request,
        FakeLogger.log_builder,
        FakeBuilder.send_log,
    ) == originals


def test_activate_is_idempotent_and_instances_share_one_generation():
    first = HeliconeInstrumentor(capture_content=False)
    second = HeliconeInstrumentor(capture_content=False)

    first.activate()
    first.activate()
    installed = FakeLogger.send_log
    second.activate()

    assert FakeLogger.send_log is installed
    assert _instrumentation._REFCOUNT == 2
    first.deactivate()
    assert FakeLogger.send_log is installed
    second.deactivate()
    assert _instrumentation._REFCOUNT == 0
    assert not hasattr(FakeLogger.send_log, "__respan_helicone_wrapper__")


def test_mismatched_capture_policy_is_rejected(caplog):
    first = HeliconeInstrumentor(capture_content=True)
    second = HeliconeInstrumentor(capture_content=False)
    first.activate()

    with caplog.at_level(logging.ERROR):
        second.activate()

    assert first._is_instrumented is True
    assert second._is_instrumented is False
    assert _instrumentation._REFCOUNT == 1
    assert "different capture_content" in caplog.text


def test_activation_skips_when_respan_tracing_is_disabled(caplog):
    RespanTracer(is_enabled=False)
    instrumentor = HeliconeInstrumentor()

    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert not hasattr(FakeLogger.send_log, "__respan_helicone_wrapper__")
    assert "tracing is disabled" in caplog.text


def test_activation_handles_missing_sdk(monkeypatch, caplog):
    monkeypatch.setattr(
        _instrumentation,
        "_load_classes",
        lambda: (_ for _ in ()).throw(ImportError("helicone_helpers")),
    )
    instrumentor = HeliconeInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert "missing dependency" in caplog.text


def test_send_log_preserves_return_and_emits_once(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_helicone_log",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = HeliconeInstrumentor(capture_content=False)
    instrumentor.activate()
    logger = FakeLogger(
        headers={"Helicone-Property-Constructor": "safe", "Authorization": "secret"}
    )

    result = logger.send_log(
        "openai",
        {"model": "gpt-test"},
        {"choices": []},
        {
            "start_time": 1.0,
            "end_time": 2.0,
            "additional_headers": {"Authorization": "Bearer secret"},
        },
    )

    assert result == "sent"
    assert len(FakeLogger.send_calls) == 1
    assert len(emitted) == 1
    assert emitted[0]["provider"] == "openai"
    assert emitted[0]["capture_content"] is False
    assert emitted[0]["constructor_headers"] == logger.headers


def test_log_request_success_flows_through_one_sink(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_helicone_log",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    logger = FakeLogger()

    def operation(recorder):
        recorder.append_results({"answer": "ok"})
        return "application-result"

    result = logger.log_request(
        {"model": "local", "messages": [{"role": "user", "content": "hi"}]},
        operation,
        provider="openai",
    )

    assert result == "application-result"
    assert len(FakeLogger.send_calls) == 1
    assert len(emitted) == 1
    assert emitted[0]["error"] is None


def test_log_request_operation_error_gets_outer_fallback(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_helicone_log",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    logger = FakeLogger(headers={"Helicone-Session-Name": "constructor-session"})

    def operation(_recorder):
        raise ValueError("deterministic failure")

    with pytest.raises(ValueError, match="deterministic failure"):
        logger.log_request(
            {"model": "broken"},
            operation,
            additional_headers={"Helicone-Property-Path": "outer-error"},
            provider="anthropic",
        )

    assert FakeLogger.send_calls == []
    assert len(emitted) == 1
    assert isinstance(emitted[0]["error"], ValueError)
    assert emitted[0]["status_code"] == 500
    assert emitted[0]["provider"] == "anthropic"
    assert emitted[0]["options"]["additional_headers"] == {
        "Helicone-Property-Path": "outer-error"
    }
    assert emitted[0]["constructor_headers"] == logger.headers


def test_log_request_cancellation_gets_outer_fallback(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_helicone_log",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    logger = FakeLogger()

    def operation(_recorder):
        raise asyncio.CancelledError("deterministic cancellation")

    with pytest.raises(asyncio.CancelledError, match="deterministic cancellation"):
        logger.log_request(
            {"model": "cancelled"},
            operation,
            additional_headers={"Helicone-Property-Path": "cancelled"},
        )

    assert FakeLogger.send_calls == []
    assert len(emitted) == 1
    assert isinstance(emitted[0]["error"], asyncio.CancelledError)
    assert emitted[0]["status_code"] == 500


def test_nested_direct_send_then_outer_failure_emits_both(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_helicone_log",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    logger = FakeLogger()

    def operation(_recorder):
        logger.send_log(
            "openai",
            {"model": "nested-direct", "messages": []},
            {"choices": []},
            {"start_time": 1.0, "end_time": 1.5},
        )
        raise RuntimeError("outer callback failed")

    with pytest.raises(RuntimeError, match="outer callback failed"):
        logger.log_request(
            {"model": "outer", "messages": []}, operation, provider="openai"
        )

    assert len(FakeLogger.send_calls) == 1
    assert len(emitted) == 2
    assert emitted[0]["request"]["model"] == "nested-direct"
    assert emitted[0]["error"] is None
    assert emitted[1]["request"]["model"] == "outer"
    assert isinstance(emitted[1]["error"], RuntimeError)


def test_nested_direct_success_keeps_one_helper_terminal_sink(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_helicone_log",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    logger = FakeLogger()

    def operation(recorder):
        logger.send_log(
            "openai",
            {"model": "nested"},
            {"model": "nested-response"},
            {"start_time": 1.0, "end_time": 1.2},
        )
        recorder.append_results({"model": "outer-response"})
        return "ok"

    assert logger.log_request({"model": "outer"}, operation, provider="openai") == "ok"

    assert len(FakeLogger.send_calls) == 2
    assert [item["request"]["model"] for item in emitted] == ["nested", "outer"]


def test_builder_async_error_context_reaches_sink(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_helicone_log",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    builder = FakeBuilder(FakeLogger())
    builder.error = RuntimeError("builder failed")
    builder.status = 500

    assert asyncio.run(builder.send_log()) == "sent"

    assert len(emitted) == 1
    assert isinstance(emitted[0]["error"], RuntimeError)
    assert emitted[0]["status_code"] == 500


def test_empty_builder_is_streaming_and_uses_creation_context(monkeypatch):
    emitted = []
    snapshot = object()
    monkeypatch.setattr(
        _instrumentation,
        "capture_emission_context",
        lambda: snapshot,
    )
    monkeypatch.setattr(
        _instrumentation,
        "emit_helicone_log",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    builder = FakeLogger().log_builder(
        {"model": "empty-stream", "messages": [], "stream": True}
    )
    builder.response_body = ""

    asyncio.run(builder.send_log())

    assert emitted[0]["is_streaming"] is True
    assert emitted[0]["context_snapshot"] is snapshot


def test_cancelled_builder_maps_to_499(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_helicone_log",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    builder = FakeBuilder(FakeLogger())
    builder.was_cancelled = True

    asyncio.run(builder.send_log())

    assert emitted[0]["status_code"] == 499


def test_foreign_post_patch_is_not_clobbered_and_old_generation_disables(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_helicone_log",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    installed = FakeLogger.send_log

    def foreign(instance, *args, **kwargs):
        return installed(instance, *args, **kwargs)

    FakeLogger.send_log = foreign
    instrumentor.deactivate()

    assert FakeLogger.send_log is foreign
    FakeLogger().send_log(None, {}, {}, {"start_time": 1, "end_time": 2})
    assert emitted == []


def test_explicit_helicone_instrumentation_does_not_patch_provider_sdks():
    class ProviderSDK:
        def create(self):
            return "provider-result"

    original = ProviderSDK.create
    instrumentor = HeliconeInstrumentor()

    instrumentor.activate()

    assert ProviderSDK.create is original
    assert ProviderSDK().create() == "provider-result"
