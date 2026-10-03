"""Groq inference resources not covered by OpenInference's chat instrumentor."""

from __future__ import annotations

import base64
import importlib
import inspect
import json
import logging
import threading
from pathlib import Path
from typing import Any

from openinference.instrumentation import TraceConfig
from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
)
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_SPEECH,
    LOG_TYPE_TRANSCRIPTION,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_INTERNAL_SPAN_NAME_KIND,
    RESPAN_LOG_TYPE,
)
from wrapt import FunctionWrapper

from ._privacy import content_enabled
from ._streaming import _capture_response, _request_parameters

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_OWNERS: set[object] = set()
_PATCHES: list[tuple[Any, Any, Any, dict]] = []
_AUDIO_LIMIT = 65536
_RESOURCES = (
    ("embeddings", "Embeddings", LOG_TYPE_EMBEDDING, "embedding"),
    ("audio.transcriptions", "Transcriptions", LOG_TYPE_TRANSCRIPTION, "transcribe"),
    ("audio.translations", "Translations", LOG_TYPE_TRANSCRIPTION, "transcribe"),
    ("audio.speech", "Speech", LOG_TYPE_SPEECH, "speech"),
)


def _dump(value: Any) -> str:
    return json.dumps(
        value,
        separators=(",", ":"),
        default=lambda obj: (
            obj.model_dump() if hasattr(obj, "model_dump") else type(obj).__name__
        ),
    )


def _input(parameters: dict) -> Any:
    result = dict(parameters)
    file = result.get("file")
    if file is not None:
        # Never read/seek caller-owned uploads or include an absolute local path.
        name = file[0] if isinstance(file, tuple) else getattr(file, "name", None)
        if isinstance(file, str | Path):
            name = file
        result["file"] = {"name": Path(str(name)).name if name else "upload"}
    return result


def _speech_response(response: Any, finish: Any) -> Any:
    """Observe native binary reads without pre-consuming streaming responses."""
    if response.http_response.is_stream_consumed:
        finish(response.http_response.content, None)
        return response
    read, close, iter_bytes = response.read, response.close, response.iter_bytes
    data = bytearray()
    truncated = False

    def add(chunk):
        nonlocal truncated
        remaining = _AUDIO_LIMIT - len(data)
        data.extend(chunk[:remaining])
        truncated = truncated or len(chunk) > remaining

    def final(error=None):
        finish(
            {
                "encoding": "base64",
                "data": base64.b64encode(data).decode(),
                "truncated": truncated,
            },
            error,
        )

    if inspect.iscoroutinefunction(read):

        async def traced_read():
            try:
                result = await read()
            except BaseException as exc:
                final(exc)
                raise
            finish(result, None)
            return result

        async def traced_close():
            try:
                return await close()
            except BaseException as exc:
                final(exc)
                raise
            finally:
                final()

        async def traced_iter(*args, **kwargs):
            try:
                async for chunk in iter_bytes(*args, **kwargs):
                    add(chunk)
                    yield chunk
            except GeneratorExit:
                raise
            except BaseException as exc:
                final(exc)
                raise
            finally:
                final()
    else:

        def traced_read():
            try:
                result = read()
            except BaseException as exc:
                final(exc)
                raise
            finish(result, None)
            return result

        def traced_close():
            try:
                return close()
            except BaseException as exc:
                final(exc)
                raise
            finally:
                final()

        def traced_iter(*args, **kwargs):
            try:
                for chunk in iter_bytes(*args, **kwargs):
                    add(chunk)
                    yield chunk
            except GeneratorExit:
                raise
            except BaseException as exc:
                final(exc)
                raise
            finally:
                final()

    response.read, response.close, response.iter_bytes = (
        traced_read,
        traced_close,
        traced_iter,
    )
    return response


def _start(parameters: dict, log_type: str, operation: str, config: TraceConfig):
    attrs = {
        RESPAN_LOG_TYPE: log_type,
        RESPAN_INTERNAL_SPAN_NAME_KIND: operation,
        SpanAttributes.LLM_SYSTEM: "groq",
        SpanAttributes.LLM_REQUEST_TYPE: operation,
        SpanAttributes.TRACELOOP_ENTITY_NAME: operation,
    }
    if parameters.get("model"):
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = str(parameters["model"])
    capture_content = content_enabled()
    hidden_input = (
        not capture_content
        or config.hide_inputs
        or config.hide_input_text
        or config.hide_input_messages
        or config.hide_prompts
    )
    if log_type == LOG_TYPE_EMBEDDING:
        hidden_input = hidden_input or config.hide_embeddings_text
    if not hidden_input:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _dump(
            parameters.get("input")
            if log_type == LOG_TYPE_EMBEDDING
            else _input(parameters)
        )
    span = trace.get_tracer(__name__).start_span(operation, attributes=attrs)
    finished = False

    def finish(value, error):
        nonlocal finished
        if finished:
            return
        finished = True
        try:
            if error is not None:
                span.record_exception(error)
                span.set_status(
                    trace.Status(
                        trace.StatusCode.ERROR, f"{type(error).__name__}: {error}"
                    )
                )
            else:
                span.set_status(trace.Status(trace.StatusCode.OK))
            hidden_output = (
                not capture_content
                or config.hide_outputs
                or config.hide_output_text
                or config.hide_output_messages
                or config.hide_choices
            )
            if log_type == LOG_TYPE_EMBEDDING:
                hidden_output = (
                    hidden_output
                    or config.hide_embedding_vectors
                    or config.hide_embeddings_vectors
                )
                output = (
                    [item.embedding for item in value.data]
                    if value is not None
                    else None
                )
                usage = getattr(value, "usage", None)
                count = getattr(usage, "prompt_tokens", None)
                if isinstance(count, int):
                    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, count)
                    span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, count)
                total = getattr(usage, "total_tokens", None)
                if isinstance(total, int):
                    span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total)
            elif log_type == LOG_TYPE_SPEECH:
                output = (
                    {
                        "encoding": "base64",
                        "data": base64.b64encode(value[:_AUDIO_LIMIT]).decode(),
                        "truncated": len(value) > _AUDIO_LIMIT,
                    }
                    if isinstance(value, bytes)
                    else value
                )
            else:
                output = getattr(value, "text", value)
            if output is not None and not hidden_output:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT, _dump(output)
                )
        except Exception:
            logger.debug("Could not capture Groq inference response", exc_info=True)
        finally:
            span.end()

    return span, finish


def _wrapper(state, log_type, operation, config, async_mode):
    def enabled():
        return state["active"] and not context.get_value(
            context._SUPPRESS_INSTRUMENTATION_KEY
        )

    def capture(response, finish):
        if log_type == LOG_TYPE_SPEECH and hasattr(response, "http_response"):
            return _speech_response(response, finish)
        return _capture_response(response, finish, async_stream=async_mode)

    if async_mode:

        async def wrapped_call(wrapped, instance, args, kwargs):
            if not enabled():
                return await wrapped(*args, **kwargs)
            span, finish = _start(
                _request_parameters(wrapped, args, kwargs), log_type, operation, config
            )
            with trace.use_span(
                span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ):
                try:
                    return capture(await wrapped(*args, **kwargs), finish)
                except BaseException as exc:
                    finish(None, exc)
                    raise
    else:

        def wrapped_call(wrapped, instance, args, kwargs):
            if not enabled():
                return wrapped(*args, **kwargs)
            span, finish = _start(
                _request_parameters(wrapped, args, kwargs), log_type, operation, config
            )
            with trace.use_span(
                span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ):
                try:
                    return capture(wrapped(*args, **kwargs), finish)
                except BaseException as exc:
                    finish(None, exc)
                    raise

    return wrapped_call


def patch_inference_resources(config: TraceConfig | None = None) -> object:
    """Patch available released SDK methods, sharing ownership across adapters."""
    token = object()
    with _LOCK:
        if _OWNERS:
            _OWNERS.add(token)
            return token
        config = config or TraceConfig()
        try:
            for module_name, class_name, log_type, operation in _RESOURCES:
                try:
                    module = importlib.import_module(f"groq.resources.{module_name}")
                except ImportError:
                    continue  # Methods absent from older supported SDK releases.
                for async_mode in (False, True):
                    cls = getattr(
                        module, f"Async{class_name}" if async_mode else class_name, None
                    )
                    if cls is None:
                        continue
                    original = inspect.getattr_static(cls, "create")
                    state = {"active": True}
                    owned = FunctionWrapper(
                        original,
                        _wrapper(state, log_type, operation, config, async_mode),
                    )
                    cls.create = owned
                    _PATCHES.append((cls, original, owned, state))
        except BaseException:
            restore_inference_resources(None)
            raise
        _OWNERS.add(token)
        return token


def restore_inference_resources(token: object | None) -> None:
    with _LOCK:
        _OWNERS.discard(token)
        if _OWNERS:
            return
        for cls, original, owned, state in reversed(_PATCHES):
            state["active"] = False
            if inspect.getattr_static(cls, "create") is owned:
                cls.create = original
        _PATCHES.clear()
