"""Close upstream Cohere streaming spans on errors and early termination."""

import asyncio
import functools

from opentelemetry.trace import Status, StatusCode


def _record_error(span, error):
    if span.is_recording():
        span.set_status(Status(StatusCode.ERROR, str(error) or type(error).__name__))
        span.record_exception(error)


def _guard_sync(processor):
    @functools.wraps(processor)
    def guarded(span, event_logger, request_type, response):
        def guarded_response():
            try:
                yield from response
            except Exception as error:
                # The upstream sync processor ends in its finally block.
                # Mark the failure before that block exports the span.
                _record_error(span, error)
                raise

        source = guarded_response()
        iterator = processor(span, event_logger, request_type, source)
        primary_error = None
        try:
            yield from iterator
        except Exception as error:
            primary_error = error
            _record_error(span, error)
            raise
        finally:
            cleanup_error = None
            try:
                for value in (response, iterator, source):
                    try:
                        close = getattr(value, "close", None)
                        if close:
                            close()
                    except Exception as error:  # noqa: BLE001 - re-raised unless preserving a primary error
                        cleanup_error = cleanup_error or error
                        if primary_error is None:
                            _record_error(span, error)
                if primary_error is None and cleanup_error is not None:
                    raise cleanup_error
            finally:
                if span.is_recording():
                    span.end()

    return guarded


def _guard_async(processor):
    @functools.wraps(processor)
    async def guarded(span, event_logger, request_type, response):
        iterator = processor(span, event_logger, request_type, response)
        primary_error = None
        try:
            async for item in iterator:
                yield item
        except (Exception, asyncio.CancelledError) as error:
            primary_error = error
            _record_error(span, error)
            raise
        finally:
            cleanup_error = None
            try:
                for value in (response, iterator):
                    try:
                        close = getattr(value, "aclose", None)
                        if close:
                            await close()
                    except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001 - preserve primary error
                        cleanup_error = cleanup_error or error
                        if primary_error is None:
                            _record_error(span, error)
                if primary_error is None and cleanup_error is not None:
                    raise cleanup_error
            finally:
                if span.is_recording():
                    span.end()

    return guarded


def patch_stream_processors(module):
    originals = []
    for target in getattr(module, "WRAPPED_METHODS", []):
        processor = target.get("stream_process_func")
        if processor is None:
            continue
        wrapped = (
            _guard_async
            if target.get("object", "").startswith("Async")
            else _guard_sync
        )(processor)
        target["stream_process_func"] = wrapped
        originals.append((target, processor, wrapped))
    return originals


def restore_stream_processors(originals):
    for target, original, wrapped in originals:
        if target.get("stream_process_func") is wrapped:
            target["stream_process_func"] = original
