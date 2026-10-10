"""Observe native SDK operations with recording spans and transparent streams."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import math
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_STREAM,
    GEN_AI_RESPONSE_ID,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TEXT,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

from respan_instrumentation_openrouter._processor import _is_sensitive_key, _redact_text

logger = logging.getLogger(__name__)

_SCOPE = "respan.instrumentation.openrouter"
_SOURCE_CALL: ContextVar[Any] = ContextVar(
    "respan_openrouter_source_call", default=None
)
_BUSY: ContextVar[bool] = ContextVar("respan_openrouter_observing", default=False)


def structured(
    value: Any,
    depth: int = 0,
    *,
    schema: bool = False,
    path: tuple = (),
    secret_schema: bool = False,
) -> Any:
    """Only serialize known DTOs and built-ins; never stringify clients or secrets."""
    if depth > 40:
        return None
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _redact_text(value, limit=len(value.encode("utf-8")) + 100)
    if isinstance(value, BaseModel):
        try:
            value = value.model_dump(mode="json", by_alias=True, exclude_none=True)
        except Exception:  # noqa: BLE001 - provider DTO serialization must not mask SDK results
            return None
    if type(value) is dict:
        result = {}
        for k, v in value.items():
            if not isinstance(k, str):
                continue
            property_node = (
                schema
                and path
                and path[-1] == "properties"
                and (type(v) is dict or type(v) is bool)
            )
            sensitive = _is_sensitive_key(k)
            if secret_schema and k in {
                "default",
                "const",
                "enum",
                "example",
                "examples",
            }:
                result[k] = (
                    ["[REDACTED]" for _ in v] if type(v) is list else "[REDACTED]"
                )
            elif sensitive and not property_node:
                result[k] = "[REDACTED]"
            else:
                result[k] = structured(
                    v,
                    depth + 1,
                    schema=schema,
                    path=(*path, k),
                    secret_schema=property_node and sensitive,
                )
        return result
    if type(value) in (list, tuple):
        return [
            structured(
                item, depth + 1, schema=schema, path=path, secret_schema=secret_schema
            )
            for item in value
        ]
    return None


def encoded(value: Any, *, schema: bool = False) -> str:
    return json.dumps(
        structured(value, schema=schema),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    try:
        return getattr(value, name, default)
    except BaseException:  # noqa: BLE001 - hostile provider properties
        return default


def _counter(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _status(error: BaseException) -> int | None:
    for obj in (error, _field(error, "response"), _field(error, "raw_response")):
        code = _field(obj, "status_code")
        if type(code) is int and 100 <= code <= 599:
            return code
    return None


class Call:
    def __init__(
        self,
        kind: str,
        kwargs: dict[str, Any],
        provider: Any,
        policy: Any,
        runtime: Any = None,
    ):
        self.kind, self.kwargs, self.policy = kind, kwargs, policy
        self.runtime = runtime
        self.context = context.get_current()
        self.span = provider.get_tracer(_SCOPE).start_span(
            "openrouter." + kind, context=self.context
        )
        self.ctx = trace.set_span_in_context(self.span, self.context)
        self.done = False
        self.stream = bool(kwargs.get("stream"))
        self.response: Any = None
        self.chunks: dict[int, dict[str, Any]] = {}
        self.outputs: dict[int, dict[str, Any]] = {}
        self.meta: dict[str, Any] = {}
        self.source_seen = False
        self.source_usage: dict[str, Any] = {}
        self.event_error: dict[str, Any] | None = None
        self.allowed = policy.allowed(self.span)
        self.span.set_attributes(
            {
                SpanAttributes.TRACELOOP_ENTITY_NAME: "openrouter." + kind,
                SpanAttributes.TRACELOOP_ENTITY_PATH: "",
                RESPAN_LOG_TYPE: LOG_TYPE_EMBEDDING
                if kind == "embedding"
                else LOG_TYPE_TEXT
                if kind == "completion"
                else LOG_TYPE_CHAT,
                SpanAttributes.LLM_SYSTEM: "openrouter",
                GEN_AI_PROVIDER_NAME: "openrouter",
                SpanAttributes.LLM_REQUEST_TYPE: "embedding"
                if kind == "embedding"
                else "chat",
                SpanAttributes.GEN_AI_IS_STREAMING: self.stream,
                GEN_AI_REQUEST_STREAM: self.stream,
            }
        )
        if isinstance(kwargs.get("model"), str):
            self.span.set_attribute(
                SpanAttributes.LLM_REQUEST_MODEL,
                _redact_text(kwargs["model"], limit=4000),
            )

        if runtime is not None:
            runtime.calls.add(self)

    def restrict(self):
        self.allowed = False
        self.policy.deny(self.span)
        self.kwargs = {k: v for k, v in self.kwargs.items() if k in {"model", "stream"}}
        self.chunks.clear()
        self.outputs.clear()
        self.response = None
        self.event_error = None

    @contextmanager
    def active(self):
        token = context.attach(self.ctx)
        busy = _BUSY.set(True)
        source_token = _SOURCE_CALL.set(self)
        try:
            yield
        finally:
            self.allowed = self.allowed and self.policy.permitted_now(self.span)
            if not self.allowed:
                self.restrict()
            _SOURCE_CALL.reset(source_token)
            _BUSY.reset(busy)
            context.detach(token)

    def add(self, item: Any) -> None:
        if not self.span.is_recording():
            return
        self.allowed = self.allowed and self.policy.permitted_now(self.span)
        data = structured(item)
        if not isinstance(data, dict):
            return
        # SDKs may retain fields containing explicit nulls; keep only real usage.
        for key in ("id", "model", "usage"):
            if data.get(key) is not None:
                self.meta[key] = data[key]
        event = data.get("type", "")
        if self.kind == "response":
            response = data.get("response")
            if isinstance(response, dict):
                self.meta.update(
                    {
                        k: response[k]
                        for k in ("id", "model", "usage")
                        if response.get(k) is not None
                    }
                )
                if event in {
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                }:
                    self.response = response
                if event == "response.failed" and isinstance(
                    response.get("error"), dict
                ):
                    self.event_error = response["error"]
            if not self.allowed:
                return
            idx = data.get("output_index", 0)
            if event in {
                "response.output_item.added",
                "response.output_item.done",
            } and isinstance(data.get("item"), dict):
                self.outputs[idx] = data["item"]
            elif event == "response.function_call_arguments.delta":
                slot = self.outputs.setdefault(
                    idx, {"type": "function_call", "arguments": ""}
                )
                slot["arguments"] = slot.get("arguments", "") + (
                    data.get("delta") or ""
                )
            elif event == "response.output_text.delta":
                slot = self.outputs.setdefault(
                    idx, {"type": "message", "role": "assistant", "content": []}
                )
                ci = data.get("content_index", 0)
                content = slot.setdefault("content", [])
                while len(content) <= ci:
                    content.append({"type": "output_text", "text": ""})
                content[ci]["text"] += data.get("delta") or ""
            elif event == "response.output_text.done":
                # Final text replaces the incremental text, it does not append it.
                slot = self.outputs.setdefault(
                    idx, {"type": "message", "role": "assistant", "content": []}
                )
                ci = data.get("content_index", 0)
                content = slot.setdefault("content", [])
                while len(content) <= ci:
                    content.append({"type": "output_text", "text": ""})
                content[ci] = {"type": "output_text", "text": data.get("text", "")}
            return
        if not self.allowed:
            return
        for choice in data.get("choices") or []:
            idx = choice.get("index", 0)
            slot = self.chunks.setdefault(
                idx,
                {
                    "index": idx,
                    "message": {"role": "assistant", "content": ""},
                    "tools": {},
                },
            )
            if choice.get("finish_reason") is not None:
                slot["finish_reason"] = choice["finish_reason"]
            if self.kind == "completion":
                slot["text"] = slot.get("text", "") + (choice.get("text") or "")
                continue
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                slot["message"]["content"] += delta["content"]
            for field in ("reasoning", "refusal"):
                if isinstance(delta.get(field), str):
                    slot["message"][field] = (
                        slot["message"].get(field, "") + delta[field]
                    )
            for tc in delta.get("tool_calls") or []:
                ti = tc.get("index", 0)
                tool = slot["tools"].setdefault(
                    ti, {"type": "function", "function": {"name": "", "arguments": ""}}
                )
                if tc.get("id"):
                    tool["id"] = tc["id"]
                for key in ("name", "arguments"):
                    value = (tc.get("function") or {}).get(key)
                    if isinstance(value, str):
                        tool["function"][key] += value

    def aggregate(self) -> Any:
        if self.response is not None:
            return self.response
        data = dict(self.meta)
        if self.kind == "response":
            data["output"] = [self.outputs[i] for i in sorted(self.outputs)]
        else:
            data["choices"] = []
            for idx in sorted(self.chunks):
                slot = self.chunks[idx]
                if slot["tools"]:
                    slot["message"]["tool_calls"] = [
                        slot["tools"][i] for i in sorted(slot["tools"])
                    ]
                data["choices"].append({k: v for k, v in slot.items() if k != "tools"})
        return data

    def finish(self, response: Any = None, error: BaseException | None = None) -> None:
        if self.done:
            return
        self.done = True
        try:
            if not self.span.is_recording():
                return
            self.allowed = self.allowed and self.policy.permitted_now(self.span)
            data = structured(response if response is not None else self.aggregate())
            if not isinstance(data, dict):
                data = {}
            for field, attr in (
                ("model", SpanAttributes.LLM_RESPONSE_MODEL),
                ("id", GEN_AI_RESPONSE_ID),
            ):
                if isinstance(data.get(field), str):
                    self.span.set_attribute(attr, _redact_text(data[field], limit=4000))
            if data.get("status") == "failed" and isinstance(data.get("error"), dict):
                self.event_error = data["error"]
            usage = self.source_usage if self.source_seen else data.get("usage") or {}
            inp = _counter(usage.get("input_tokens", usage.get("prompt_tokens")))
            out = _counter(usage.get("output_tokens", usage.get("completion_tokens")))
            for n, keys in (
                (
                    inp,
                    (GEN_AI_USAGE_INPUT_TOKENS, SpanAttributes.LLM_USAGE_PROMPT_TOKENS),
                ),
                (
                    out,
                    (
                        GEN_AI_USAGE_OUTPUT_TOKENS,
                        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                    ),
                ),
                (
                    _counter(usage.get("total_tokens")),
                    (SpanAttributes.LLM_USAGE_TOTAL_TOKENS,),
                ),
            ):
                if n is not None:
                    for key in keys:
                        self.span.set_attribute(key, n)
            detail = (
                usage.get("input_tokens_details")
                or usage.get("prompt_tokens_details")
                or {}
            )
            cached = _counter(detail.get("cached_tokens"))
            reasoning = _counter(
                (
                    usage.get("output_tokens_details")
                    or usage.get("completion_tokens_details")
                    or {}
                ).get("reasoning_tokens")
            )
            if cached is not None:
                self.span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cached)
                self.span.set_attribute(
                    SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cached
                )
            if reasoning is not None:
                self.span.set_attribute(GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, reasoning)
            if self.allowed:
                self._content(
                    data,
                    response is not None
                    or bool(self.chunks)
                    or bool(self.outputs)
                    or self.response is not None,
                )
            if error is not None or self.event_error:
                self.span.set_status(Status(StatusCode.ERROR))
                self.span.set_attribute(
                    ERROR_TYPE,
                    type(error).__name__
                    if error
                    else str(self.event_error.get("code") or "response.failed"),
                )
                if error is not None and (code := _status(error)) is not None:
                    self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, code)
                if self.allowed:
                    message = (
                        next(
                            (a for a in error.args if isinstance(a, str)),
                            type(error).__name__,
                        )
                        if error
                        else self.event_error.get("message")
                    )
                    if isinstance(message, str):
                        self.span.set_attribute(
                            ERROR_MESSAGE, _redact_text(message, limit=4000)
                        )
        except Exception:
            logger.debug("OpenRouter serialization failed", exc_info=True)
        finally:
            if not self.allowed:
                self.policy.deny(self.span)
            try:
                self.span.end()
            except Exception:
                logger.debug("OpenRouter span completion failed", exc_info=True)
            if self.runtime is not None:
                self.runtime.calls.discard(self)
            self.kwargs = {}
            self.chunks.clear()
            self.outputs.clear()
            self.response = None
            self.meta.clear()
            self.source_usage.clear()
            self.event_error = None

    def _content(self, data: dict[str, Any], has_output: bool) -> None:
        if self.kind in {"response", "embedding", "completion"}:
            inp = structured(
                self.kwargs.get("input")
                if self.kind != "completion"
                else self.kwargs.get("prompt")
            )
        else:
            inp = structured(self.kwargs.get("messages")) or []
        self.span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, encoded(inp))
        messages = (
            inp
            if self.kind == "chat"
            else [{"role": "user", "content": inp}]
            if self.kind != "embedding"
            else []
        )
        for index, message in enumerate(messages[:8]):
            if not isinstance(message, dict):
                continue
            for key in ("role", "content", "tool_calls"):
                value = message.get(key)
                if value is not None:
                    self.span.set_attribute(
                        f"{SpanAttributes.LLM_PROMPTS}.{index}.{key}",
                        value if isinstance(value, str) else encoded(value),
                    )
        tools = structured(self.kwargs.get("tools"), schema=True)
        if tools:
            self.span.set_attribute(
                SpanAttributes.LLM_REQUEST_FUNCTIONS, encoded(tools, schema=True)
            )
        if not has_output:
            return
        if self.kind == "embedding":
            # Preserve float arrays and provider-returned encoded embeddings in full.
            vectors = [item.get("embedding") for item in data.get("data", [])]
            self.span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, encoded(vectors)
            )
            return
        if self.kind == "response":
            output = data.get("output") or []
            self.span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, encoded(output)
            )
            text = "".join(
                part.get("text", "")
                for item in output
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
            calls = [
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                    },
                }
                for item in output
                if item.get("type") == "function_call"
            ]
        else:
            choices = data.get("choices") or []
            output = [
                c.get("message") if self.kind == "chat" else {"text": c.get("text")}
                for c in choices
            ]
            self.span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, encoded(output)
            )
            for index, choice in enumerate(choices[:8]):
                message = choice.get("message") or {}
                text = (
                    choice.get("text")
                    if self.kind == "completion"
                    else message.get("content") or ""
                )
                self.span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role", "assistant"
                )
                self.span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content",
                    text if isinstance(text, str) else encoded(text),
                )
                if message.get("tool_calls"):
                    self.span.set_attribute(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{index}.tool_calls",
                        encoded(message["tool_calls"]),
                    )
            return
        self.span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
        self.span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", text)
        if calls:
            self.span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls", encoded(calls)
            )


class SyncStream:
    def __init__(self, stream: Any, call: Call):
        self._stream, self._iterator, self._call = stream, iter(stream), call

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __iter__(self):
        return self

    def _advance(self, method: str, *args: Any):
        with self._call.active():
            try:
                result = getattr(self._iterator, method)(*args)
            except StopIteration:
                self._call.finish()
                raise
            except BaseException as exc:
                self._call.finish(error=exc)
                raise
            try:
                self._call.add(result)
            except Exception:
                logger.debug("OpenRouter stream serialization failed", exc_info=True)
            return result

    def __next__(self):
        return self._advance("__next__")

    def send(self, value: Any):
        return self._advance("send", value)

    def throw(self, *args: Any):
        return self._advance("throw", *args)

    def close(self):
        with self._call.active():
            try:
                return self._stream.close()
            except BaseException as exc:
                self._call.finish(error=exc)
                raise
            finally:
                self._call.finish()

    def __enter__(self):
        with self._call.active():
            try:
                entered = self._stream.__enter__()
            except BaseException as exc:
                self._call.finish(error=exc)
                raise
            if entered is not self._stream:
                self._stream, self._iterator = entered, iter(entered)
            return self

    def __exit__(self, *args: object):
        with self._call.active():
            try:
                result = self._stream.__exit__(*args)
            except BaseException as exc:
                self._call.finish(error=exc)
                raise
            self._call.finish(error=args[1] if not result else None)
            return result


class AsyncStream:
    def __init__(self, stream: Any, call: Call):
        self._stream, self._iterator, self._call = stream, stream.__aiter__(), call

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __aiter__(self):
        return self

    async def _advance(self, method: str, *args: Any):
        with self._call.active():
            try:
                result = await getattr(self._iterator, method)(*args)
            except StopAsyncIteration:
                self._call.finish()
                raise
            except BaseException as exc:
                self._call.finish(error=exc)
                raise
            try:
                self._call.add(result)
            except Exception:
                logger.debug("OpenRouter stream serialization failed", exc_info=True)
            return result

    async def __anext__(self):
        return await self._advance("__anext__")

    async def asend(self, value: Any):
        return await self._advance("asend", value)

    async def athrow(self, *args: Any):
        return await self._advance("athrow", *args)

    async def close(self):
        with self._call.active():
            try:
                fn = getattr(self._stream, "aclose", None) or self._stream.close
                result = fn()
                return await result if inspect.isawaitable(result) else result
            except BaseException as exc:
                self._call.finish(error=exc)
                raise
            finally:
                self._call.finish()

    async def aclose(self):
        return await self.close()

    async def __aenter__(self):
        with self._call.active():
            try:
                entered = await self._stream.__aenter__()
            except BaseException as exc:
                self._call.finish(error=exc)
                raise
            if entered is not self._stream:
                self._stream, self._iterator = entered, entered.__aiter__()
            return self

    async def __aexit__(self, *args: object):
        with self._call.active():
            try:
                result = await self._stream.__aexit__(*args)
            except BaseException as exc:
                self._call.finish(error=exc)
                raise
            self._call.finish(error=args[1] if not result else None)
            return result


def wrapper(
    original: Any,
    *,
    kind: str,
    runtime: Any,
    is_async: bool,
    native: bool = False,
    fallback: Any = None,
):
    def begin(resource, args, kwargs):
        if not runtime.active:
            return None
        if (
            _BUSY.get()
            or context.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
            or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
        ):
            return None
        from respan_instrumentation_openrouter._instrumentation import (
            _delegate_instrumentation_suppressed,
        )

        if _delegate_instrumentation_suppressed():
            return None
        if not native and not runtime.normalize_all_openai_spans:
            from urllib.parse import urlsplit

            client = _field(resource, "_client")
            url = _field(client, "base_url")
            host = urlsplit(str(url)).hostname if url is not None else None
            if host != "openrouter.ai" and not (
                host and host.endswith(".openrouter.ai")
            ):
                return False
        try:
            bound = inspect.signature(original).bind_partial(resource, *args, **kwargs)
            values = {
                k: v for k, v in bound.arguments.items() if k not in {"self", "cls"}
            }
            values.update(values.pop("kwargs", {}))
        except (TypeError, ValueError):
            values = kwargs
        try:
            return Call(
                kind,
                values,
                runtime.tracer_provider,
                runtime.processor._policy,
                runtime,
            )
        except Exception:
            logger.debug("OpenRouter span startup failed", exc_info=True)
            return None

    def observed(result, call):
        if call.kind != "embedding" and call.stream:
            if hasattr(result, "__aiter__"):
                return AsyncStream(result, call)
            if hasattr(result, "__iter__"):
                return SyncStream(result, call)
        with call.active():
            call.finish(response=result)
        return result

    @wraps(original)
    def sync(resource, *args, **kwargs):
        call = begin(resource, args, kwargs)
        if call is False:
            return (fallback or original)(resource, *args, **kwargs)
        if call is None:
            return original(resource, *args, **kwargs)
        with call.active():
            try:
                result = original(resource, *args, **kwargs)
            except BaseException as exc:
                call.finish(error=exc)
                raise
        return observed(result, call)

    @wraps(original)
    async def asynchronous(resource, *args, **kwargs):
        call = begin(resource, args, kwargs)
        if call is False:
            return await (fallback or original)(resource, *args, **kwargs)
        if call is None:
            return await original(resource, *args, **kwargs)
        with call.active():
            try:
                result = await original(resource, *args, **kwargs)
            except BaseException as exc:
                call.finish(error=exc)
                raise
        return observed(result, call)

    installed = asynchronous if is_async else sync
    installed.__respan_openrouter_wrapper__ = True
    return installed


def install_native(runtime: Any):
    for module, cls, method, kind in (
        ("openrouter.chat", "Chat", "send", "chat"),
        ("openrouter.responses", "Responses", "send", "response"),
        ("openrouter.beta_responses", "BetaResponses", "send", "response"),
        ("openrouter.embeddings", "Embeddings", "generate", "embedding"),
    ):
        try:
            target = getattr(importlib.import_module(module), cls)
        except (ImportError, AttributeError):
            continue
        for name, asynchronous in ((method, False), (method + "_async", True)):
            if not hasattr(target, name):
                continue
            previous = inspect.getattr_static(target, name)
            installed = wrapper(
                previous, kind=kind, runtime=runtime, is_async=asynchronous, native=True
            )
            setattr(target, name, installed)
            runtime.native_patches.append((target, name, previous, installed))


def restore_native(runtime: Any):
    for target, name, previous, installed in reversed(runtime.native_patches):
        if inspect.getattr_static(target, name) is installed:
            setattr(target, name, previous)
    runtime.native_patches.clear()


def _observe_source(value: Any) -> None:
    call = _SOURCE_CALL.get()
    if call is None or not call.span.is_recording() or not isinstance(value, dict):
        return
    # Native SSE parsers expose their data inside a ServerEvent envelope.
    if isinstance(value.get("data"), dict):
        value = value["data"]
    if isinstance(value.get("response"), dict):
        value = value["response"]
    if not any(key in value for key in ("choices", "output", "usage", "data")):
        return
    call.source_seen = True
    usage = value.get("usage")
    if isinstance(usage, dict):
        scalar_keys = {
            "prompt_tokens",
            "completion_tokens",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        }
        detail_keys = {
            "prompt_tokens_details",
            "completion_tokens_details",
            "input_tokens_details",
            "output_tokens_details",
        }
        clean = {
            key: item
            for key, item in usage.items()
            if key in scalar_keys and _counter(item) is not None
        }
        for key in detail_keys:
            detail = usage.get(key)
            if isinstance(detail, dict):
                clean[key] = {
                    name: item
                    for name, item in detail.items()
                    if name in {"cached_tokens", "reasoning_tokens"}
                    and _counter(item) is not None
                }
        call.source_usage = clean


def install_source_observers(runtime: Any) -> None:
    # Observe the SDK's already-read JSON before its DTO validation can coerce
    # boolean usage into integers. Never read/consume an HTTP stream here.
    for name in (
        "openrouter.chat",
        "openrouter.responses",
        "openrouter.beta_responses",
        "openrouter.embeddings",
    ):
        try:
            module = importlib.import_module(name)
            previous = module.unmarshal_json_response
        except (ImportError, AttributeError):
            continue

        @wraps(previous)
        def parsed(typ, response, body=None, _original=previous):
            if _SOURCE_CALL.get() is not None:
                try:
                    _observe_source(json.loads(response.text if body is None else body))
                except Exception:
                    logger.debug(
                        "OpenRouter source usage observation failed", exc_info=True
                    )
            return _original(typ, response, body)

        module.unmarshal_json_response = parsed
        runtime.native_patches.append(
            (module, "unmarshal_json_response", previous, parsed)
        )
    module = importlib.import_module("openai._streaming")
    owner = module.ServerSentEvent
    previous = owner.json

    @wraps(previous)
    def event_json(event, _original=previous):
        result = _original(event)
        _observe_source(result)
        return result

    owner.json = event_json
    runtime.native_patches.append((owner, "json", previous, event_json))
    for module_name, cls, asynchronous in (
        ("openai._response", "APIResponse", False),
        ("openai._response", "AsyncAPIResponse", True),
        ("openai._legacy_response", "LegacyAPIResponse", False),
    ):
        module = importlib.import_module(module_name)
        owner = getattr(module, cls)
        previous = owner.parse

        def observe(response):
            raw = response.http_response
            if raw.is_stream_consumed:
                try:
                    _observe_source(raw.json())
                except Exception:
                    logger.debug(
                        "OpenAI-compatible source usage observation failed",
                        exc_info=True,
                    )

        if asynchronous:

            @wraps(previous)
            async def parse_async(response, *args, _original=previous, **kwargs):
                result = await _original(response, *args, **kwargs)
                observe(response)
                return result

            installed = parse_async
        else:

            @wraps(previous)
            def parse(response, *args, _original=previous, **kwargs):
                result = _original(response, *args, **kwargs)
                observe(response)
                return result

            installed = parse
        owner.parse = installed
        runtime.native_patches.append((owner, "parse", previous, installed))
    # OTel use_span restores context before Span.end. Check tracked parents at
    # that boundary so a delayed child cannot regain content after a private end.
    previous = context.detach

    @wraps(previous)
    def detach(token, _original=previous):
        if runtime.active:
            try:
                from respan_instrumentation_openrouter._policy import key, permitted

                current = trace.get_current_span()
                if key(current) in runtime.processor._policy.active and not permitted():
                    runtime.processor._policy.deny(current)
                    call = _SOURCE_CALL.get()
                    if call is not None and call.runtime is runtime:
                        call.restrict()
            except Exception:  # noqa: BLE001 - preserve the original detach operation
                logger.debug("OpenRouter privacy checkpoint failed")
        return _original(token)

    context.detach = detach
    runtime.native_patches.append((context, "detach", previous, detach))
