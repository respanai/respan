"""Native OpenAI SDK instrumentation for Respan.

Monkey-patches the ``create`` method of the OpenAI SDK's chat-completions,
completions, responses, and embeddings resources (sync + async) and emits a
Respan span per call via ``_otel_emitter``. No Traceloop dependency: spans are
built directly with Respan's documented LLM conventions, so the backend
classifies them as real LLM calls (``llm.request.type``) with token/cost
roll-up — not generic "task" spans.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any, Callable

from respan_instrumentation_openai._constants import (
    ASYNC_CHAT_CLASS,
    ASYNC_COMPLETIONS_CLASS,
    ASYNC_EMBEDDINGS_CLASS,
    ASYNC_RESPONSES_CLASS,
    CHAT_MODULE,
    COMPLETIONS_MODULE,
    CREATE_METHOD,
    EMBEDDINGS_MODULE,
    RESPONSES_MODULE,
    SYNC_CHAT_CLASS,
    SYNC_COMPLETIONS_CLASS,
    SYNC_EMBEDDINGS_CLASS,
    SYNC_RESPONSES_CLASS,
)
from respan_instrumentation_openai import _otel_emitter as emitter

logger = logging.getLogger(__name__)

_original_methods: dict[tuple[type[Any], str], Any] = {}


# ---------------------------------------------------------------------------
# Stream aggregation (best-effort; never breaks the caller's stream)
# ---------------------------------------------------------------------------


def _aggregate_chat(chunks: list[Any]) -> dict[str, Any]:
    parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage = model = rid = None
    for ch in chunks:
        model = getattr(ch, "model", None) or model
        rid = getattr(ch, "id", None) or rid
        u = getattr(ch, "usage", None)
        if u is not None:
            usage = u
        choices = getattr(ch, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if content:
            parts.append(content)
        # Accumulate streamed tool-call deltas by index (name once, args in fragments).
        for tcd in getattr(delta, "tool_calls", None) or []:
            idx = getattr(tcd, "index", 0) or 0
            slot = tool_calls.setdefault(
                idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if getattr(tcd, "id", None):
                slot["id"] = tcd.id
            fn = getattr(tcd, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["function"]["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["function"]["arguments"] += fn.arguments
    message: dict[str, Any] = {"role": "assistant", "content": "".join(parts) or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return {"model": model, "id": rid, "usage": usage, "choices": [{"message": message}]}


def _aggregate_completion(chunks: list[Any]) -> dict[str, Any]:
    parts: list[str] = []
    usage = model = rid = None
    for ch in chunks:
        model = getattr(ch, "model", None) or model
        rid = getattr(ch, "id", None) or rid
        u = getattr(ch, "usage", None)
        if u is not None:
            usage = u
        choices = getattr(ch, "choices", None) or []
        if choices:
            text = getattr(choices[0], "text", None)
            if text:
                parts.append(text)
    return {"model": model, "id": rid, "usage": usage, "choices": [{"text": "".join(parts)}]}


def _aggregate_response(events: list[Any]) -> Any:
    """Responses-API streaming: the final event carries the full ``response``."""
    final = None
    for ev in events:
        r = getattr(ev, "response", None)
        if r is not None:
            final = r
    return final


# kind -> (emit_fn, aggregator | None)
_KINDS: dict[str, tuple[Callable[..., None], Callable[[list[Any]], Any] | None]] = {
    "chat": (emitter.emit_chat_span, _aggregate_chat),
    "completion": (emitter.emit_completion_span, _aggregate_completion),
    "response": (emitter.emit_response_span, _aggregate_response),
    "embedding": (emitter.emit_embedding_span, None),
}


def _is_stream(obj: Any) -> bool:
    """Detect an OpenAI streaming response.

    Note: Pydantic v2 models define ``__iter__``, so a plain ``hasattr`` check
    would misfire on every non-streaming response. We match OpenAI's real
    ``Stream``/``AsyncStream`` types, plus ``__aiter__`` (which pydantic models
    do *not* have) as a fallback for async streams.
    """
    try:
        from openai import AsyncStream, Stream

        if isinstance(obj, (Stream, AsyncStream)):
            return True
    except Exception:
        pass
    return hasattr(obj, "__aiter__")


# ---------------------------------------------------------------------------
# Iterator wrappers
# ---------------------------------------------------------------------------


def _wrap_sync_stream(iterator, *, kind, request_kwargs, start_ns):
    emit_fn, aggregate = _KINDS[kind]
    chunks: list[Any] = []
    try:
        for chunk in iterator:
            chunks.append(chunk)
            yield chunk
    except Exception as exc:
        response = aggregate(chunks) if aggregate else None
        emit_fn(request_kwargs=request_kwargs, start_ns=start_ns, response=response,
                error_message=str(exc), status_code=500)
        raise
    else:
        response = aggregate(chunks) if aggregate else None
        emit_fn(request_kwargs=request_kwargs, start_ns=start_ns, response=response)


async def _wrap_async_stream(aiterator, *, kind, request_kwargs, start_ns):
    emit_fn, aggregate = _KINDS[kind]
    chunks: list[Any] = []
    try:
        async for chunk in aiterator:
            chunks.append(chunk)
            yield chunk
    except Exception as exc:
        response = aggregate(chunks) if aggregate else None
        emit_fn(request_kwargs=request_kwargs, start_ns=start_ns, response=response,
                error_message=str(exc), status_code=500)
        raise
    else:
        response = aggregate(chunks) if aggregate else None
        emit_fn(request_kwargs=request_kwargs, start_ns=start_ns, response=response)


# ---------------------------------------------------------------------------
# Method wrappers
# ---------------------------------------------------------------------------


def _make_sync_wrapper(original: Any, *, kind: str) -> Any:
    emit_fn, _ = _KINDS[kind]

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            emit_fn(request_kwargs=kwargs, start_ns=start_ns, error_message=str(exc), status_code=500)
            raise
        if kind != "embedding" and _is_stream(response):
            return _wrap_sync_stream(response, kind=kind, request_kwargs=kwargs, start_ns=start_ns)
        emit_fn(request_kwargs=kwargs, start_ns=start_ns, response=response)
        return response

    return wrapper


def _make_async_wrapper(original: Any, *, kind: str) -> Any:
    emit_fn, _ = _KINDS[kind]

    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            pending = original(self, *args, **kwargs)
            response = pending if hasattr(pending, "__aiter__") else await pending
        except Exception as exc:
            emit_fn(request_kwargs=kwargs, start_ns=start_ns, error_message=str(exc), status_code=500)
            raise
        if kind != "embedding" and _is_stream(response):
            return _wrap_async_stream(response, kind=kind, request_kwargs=kwargs, start_ns=start_ns)
        emit_fn(request_kwargs=kwargs, start_ns=start_ns, response=response)
        return response

    return wrapper


# ---------------------------------------------------------------------------
# Patch plumbing
# ---------------------------------------------------------------------------


def _load_class(module_path: str, class_name: str) -> type[Any] | None:
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        logger.debug("OpenAI module %s unavailable: %s", module_path, exc)
        return None
    return getattr(module, class_name, None)


def _patch(target_class: type[Any] | None, *, kind: str, is_async: bool) -> bool:
    """Patch one target class's create method.

    Returns True when the target's API surface is present (patched now or already
    patched), False when the class or method is missing. The caller treats an
    all-False result as an installed-but-incompatible openai SDK.
    """
    if target_class is None:
        return False
    original = getattr(target_class, CREATE_METHOD, None)
    if original is None:
        return False
    key = (target_class, CREATE_METHOD)
    if key in _original_methods:
        return True
    _original_methods[key] = original
    factory = _make_async_wrapper if is_async else _make_sync_wrapper
    setattr(target_class, CREATE_METHOD, factory(original, kind=kind))
    return True


# (module, class, kind, is_async)
_TARGETS = [
    (CHAT_MODULE, SYNC_CHAT_CLASS, "chat", False),
    (CHAT_MODULE, ASYNC_CHAT_CLASS, "chat", True),
    (COMPLETIONS_MODULE, SYNC_COMPLETIONS_CLASS, "completion", False),
    (COMPLETIONS_MODULE, ASYNC_COMPLETIONS_CLASS, "completion", True),
    (RESPONSES_MODULE, SYNC_RESPONSES_CLASS, "response", False),
    (RESPONSES_MODULE, ASYNC_RESPONSES_CLASS, "response", True),
    (EMBEDDINGS_MODULE, SYNC_EMBEDDINGS_CLASS, "embedding", False),
    (EMBEDDINGS_MODULE, ASYNC_EMBEDDINGS_CLASS, "embedding", True),
]


class OpenAIInstrumentor:
    """Respan instrumentor for direct OpenAI SDK usage.

    Usage::

        from respan import Respan
        from respan_instrumentation_openai import OpenAIInstrumentor

        respan = Respan(instrumentations=[OpenAIInstrumentor()])
    """

    name = "openai"

    def __init__(self) -> None:
        self._is_instrumented = False

    def activate(self) -> None:
        if self._is_instrumented:
            return
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            # SDK genuinely absent — expected when the app doesn't use OpenAI
            # (the openai SDK is an optional extra).
            logger.debug("OpenAI instrumentation inactive — openai not installed: %s", exc)
            return
        try:
            patched_any = False
            for module_path, class_name, kind, is_async in _TARGETS:
                if _patch(
                    _load_class(module_path, class_name), kind=kind, is_async=is_async
                ):
                    patched_any = True
            if not patched_any:
                # openai imports but none of its known API surfaces are present:
                # installed but incompatible — the silent-failure class this bundling
                # is meant to fix, so warn instead of leaving it quietly untraced.
                logger.warning(
                    "openai is installed but no known API surface could be "
                    "instrumented — incompatible version? OpenAI instrumentation inactive."
                )
        except Exception as exc:
            logger.warning("Failed to activate OpenAI instrumentation: %s", exc)
            self.deactivate()
            return
        self._is_instrumented = True
        logger.info("OpenAI SDK instrumentation activated")

    def deactivate(self) -> None:
        for (target_class, method_name), original in list(_original_methods.items()):
            try:
                setattr(target_class, method_name, original)
            except Exception:
                logger.debug("Failed to restore %s.%s", target_class, method_name, exc_info=True)
            finally:
                _original_methods.pop((target_class, method_name), None)
        self._is_instrumented = False
        logger.info("OpenAI SDK instrumentation deactivated")
