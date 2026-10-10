from __future__ import annotations

import asyncio

import pytest
from respan_instrumentation_openlit._openai_hooks import (
    _AsyncStreamProxy,
    _SyncStreamProxy,
)


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.status = None
        self.end_calls = 0
        self._recording = True

    def is_recording(self) -> bool:
        return self._recording

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status) -> None:
        self.status = status

    def end(self) -> None:
        self.end_calls += 1
        self._recording = False


class BlockingAsyncStream:
    def __init__(self) -> None:
        self._span = RecordingSpan()
        self._llmresponse = "partial async output"
        self.started = asyncio.Event()
        self.close_calls = 0

    async def __anext__(self):
        self.started.set()
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.close_calls += 1


class ProviderFailure(RuntimeError):
    pass


class CloseFailure(RuntimeError):
    pass


class EnterFailure(RuntimeError):
    pass


class ExitFailure(RuntimeError):
    pass


class FailingSyncStream:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self._span = RecordingSpan()
        self._llmresponse = "partial"
        self.close_calls = 0

    def __next__(self):
        raise ProviderFailure("provider iteration failed")

    def __enter__(self):
        if self.mode == "enter":
            raise EnterFailure("enter failed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        if self.mode == "exit":
            raise ExitFailure("exit failed")
        return False

    def close(self) -> None:
        self.close_calls += 1
        if self.mode in {"close", "iteration"}:
            raise CloseFailure("close failed")


class FailingAsyncStream(FailingSyncStream):
    async def __anext__(self):
        raise ProviderFailure("provider iteration failed")

    async def __aenter__(self):
        if self.mode == "enter":
            raise EnterFailure("enter failed")
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        if self.mode == "exit":
            raise ExitFailure("exit failed")
        return False

    async def close(self) -> None:
        self.close_calls += 1
        if self.mode in {"close", "iteration"}:
            raise CloseFailure("close failed")


def test_async_stream_cancellation_closes_source_and_ends_exactly_once() -> None:
    async def run() -> None:
        source = BlockingAsyncStream()
        proxy = _AsyncStreamProxy(source, capture_content=True)
        task = asyncio.create_task(proxy.__anext__())
        await source.started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert source.close_calls == 1
        assert source._span.end_calls == 1
        assert source._span.status.is_ok is False
        assert "error.message" not in source._span.attributes
        assert source._span.status.description == "CancelledError"

        await proxy.close()
        assert source.close_calls == 1
        assert source._span.end_calls == 1

    asyncio.run(run())


def test_sync_stream_failures_finish_once_and_preserve_provider_error() -> None:
    iteration = FailingSyncStream("iteration")
    proxy = _SyncStreamProxy(iteration, capture_content=True)
    with pytest.raises(ProviderFailure):
        next(proxy)
    assert iteration._span.end_calls == 1
    assert "error.message" not in iteration._span.attributes
    assert iteration._span.status.description == "ProviderFailure"

    entering = FailingSyncStream("enter")
    with pytest.raises(EnterFailure):
        _SyncStreamProxy(entering, capture_content=True).__enter__()
    assert entering._span.end_calls == 1
    assert entering.close_calls == 1

    exiting = FailingSyncStream("exit")
    with pytest.raises(ExitFailure):
        _SyncStreamProxy(exiting, capture_content=True).__exit__(None, None, None)
    assert exiting._span.end_calls == 1
    assert "error.message" not in exiting._span.attributes
    assert exiting._span.status.description == "ExitFailure"

    closing = FailingSyncStream("close")
    with pytest.raises(CloseFailure):
        _SyncStreamProxy(closing, capture_content=True).close()
    assert closing._span.end_calls == 1


def test_async_stream_failures_finish_once_and_preserve_provider_error() -> None:
    async def run() -> None:
        iteration = FailingAsyncStream("iteration")
        proxy = _AsyncStreamProxy(iteration, capture_content=True)
        with pytest.raises(ProviderFailure):
            await proxy.__anext__()
        assert iteration._span.end_calls == 1
        assert "error.message" not in iteration._span.attributes
        assert iteration._span.status.description == "ProviderFailure"

        entering = FailingAsyncStream("enter")
        with pytest.raises(EnterFailure):
            await _AsyncStreamProxy(entering, capture_content=True).__aenter__()
        assert entering._span.end_calls == 1
        assert entering.close_calls == 1

        exiting = FailingAsyncStream("exit")
        with pytest.raises(ExitFailure):
            await _AsyncStreamProxy(exiting, capture_content=True).__aexit__(
                None, None, None
            )
        assert exiting._span.end_calls == 1
        assert "error.message" not in exiting._span.attributes
        assert exiting._span.status.description == "ExitFailure"

        closing = FailingAsyncStream("close")
        with pytest.raises(CloseFailure):
            await _AsyncStreamProxy(closing, capture_content=True).close()
        assert closing._span.end_calls == 1

    asyncio.run(run())
