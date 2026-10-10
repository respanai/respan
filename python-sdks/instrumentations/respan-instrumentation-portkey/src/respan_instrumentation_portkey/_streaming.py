"""Scoped native delegation, iterator protocols and actual stream aggregation."""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from opentelemetry import context, trace

from respan_instrumentation_portkey._policy import (
    AncestorPolicy,
    content_allowed,
    suppressed,
)
from respan_instrumentation_portkey._processor import Operation
from respan_instrumentation_portkey._serialization import fields

_CURRENT: ContextVar[Operation | None] = ContextVar(
    "respan_portkey_operation", default=None
)
logger = logging.getLogger(__name__)


@contextmanager
def bypass_scope(suppress):
    token = (
        context.attach(context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True))
        if suppress
        else None
    )
    try:
        yield
    finally:
        if token is not None:
            context.detach(token)


def current_request():
    operation = _CURRENT.get()
    return operation.request if operation is not None else None


def observe_usage(value):
    operation = _CURRENT.get()
    if (
        operation is None
        or operation.done
        or not operation.span.is_recording()
        or not isinstance(value, dict)
    ):
        return
    value = value.get("response") if isinstance(value.get("response"), dict) else value
    if not any(key in value for key in ("choices", "output", "data", "usage")):
        return
    operation.source_seen = True
    usage = value.get("usage")
    if isinstance(usage, dict):
        operation.source_usage = {
            name: item
            for name, item in usage.items()
            if name
            in {
                "prompt_tokens",
                "completion_tokens",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "prompt_tokens_details",
                "completion_tokens_details",
                "input_tokens_details",
                "output_tokens_details",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "cache_write_tokens",
            }
        }


def native_snapshot(stream):
    from portkey_ai._vendor.openai.lib.streaming.chat import (
        AsyncChatCompletionStream,
        ChatCompletionStream,
    )
    from portkey_ai._vendor.openai.lib.streaming.responses import (
        AsyncResponseStream,
        ResponseStream,
    )

    if isinstance(stream, (ChatCompletionStream, AsyncChatCompletionStream)):
        return stream.current_completion_snapshot
    if isinstance(stream, (ResponseStream, AsyncResponseStream)):
        return stream._state._completed_response
    return None


class NativeSpan(trace.Span):
    """Let upstream own its scope; the outer operation owns final canonical data."""

    def __init__(self, operation):
        self.operation = operation

    def get_span_context(self):
        return self.operation.span.get_span_context()

    def is_recording(self):
        return self.operation.span.is_recording()

    def set_attribute(self, key, value):
        pass

    def set_attributes(self, attributes):
        pass

    def add_event(self, *args, **kwargs):
        pass

    def record_exception(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass

    def update_name(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        self.operation.checkpoint()


class NativeTracer:
    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)

    def start_span(self, *args, **kwargs):
        current = _CURRENT.get()
        return (
            NativeSpan(current)
            if current is not None and not current.done
            else self.native.start_span(*args, **kwargs)
        )


class SourceExtractor:
    """Canonical data is taken from native values, not upstream str/dump fallbacks."""

    def __getattr__(self, name):
        if name.startswith("get_"):
            return lambda *args, **kwargs: iter(())
        raise AttributeError(name)


class NativeJSON:
    """Observe module-local SDK response decoding without patching stdlib JSON."""

    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def loads(self, *args, **kwargs):
        result = self.original.loads(*args, **kwargs)
        operation = _CURRENT.get()
        if operation is not None and not operation.source_seen:
            try:
                observe_usage(result)
            except Exception:  # noqa: BLE001 - retain the native decoder value/error
                logger.debug("Could not observe Portkey response JSON")
        return result


class StreamAccumulator:
    def __init__(self, operation):
        self.operation = operation
        self.content = ""
        self.calls = {}
        self.response = None
        self.usage = None
        self.model = None
        self.saw_content = False
        self.failure = None

    def clear_content(self):
        self.content = ""
        self.calls = {}
        self.response = None
        self.saw_content = False
        if self.failure is not None:
            self.failure.pop("message", None)

    def add(self, item):
        data = fields(item)
        if data is None:
            return
        allowed = self.operation.checkpoint()
        if not self.operation.span.is_recording():
            return
        if isinstance(data.get("model"), str):
            self.model = data["model"]
        if fields(data.get("usage")) is not None:
            self.usage = data["usage"]
        event = data.get("type")
        if event == "chunk" and fields(data.get("chunk")) is not None:
            self.add(data["chunk"])
            return
        if event in {"response.completed", "response.done", "response.failed"}:
            response = fields(data.get("response"))
            if response is not None:
                if fields(response.get("usage")) is not None:
                    self.usage = response["usage"]
                if isinstance(response.get("model"), str):
                    self.model = response["model"]
                if (
                    event == "response.failed"
                    and fields(response.get("error")) is not None
                ):
                    failure = fields(response["error"])
                    self.failure = {
                        key: failure[key]
                        for key in ("code", "type", "message")
                        if key in failure and (allowed or key != "message")
                    }
                if allowed:
                    self.response = data["response"]
            return
        if not allowed:
            return
        if event == "response.output_text.delta" and isinstance(data.get("delta"), str):
            self.content += data["delta"]
            self.saw_content = True
        choices = data.get("choices")
        if not isinstance(choices, list):
            return
        for raw in choices:
            choice = fields(raw) or {}
            if choice.get("index", 0) != 0:
                continue
            delta = fields(choice.get("delta")) or {}
            text = delta.get("content", choice.get("text"))
            if isinstance(text, str):
                self.content += text
                self.saw_content = True
            calls = delta.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for raw_call in calls:
                call = fields(raw_call) or {}
                index = call.get("index")
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    continue
                slot = self.calls.setdefault(index, {"function": {}})
                for name in ["id", "type"]:
                    if isinstance(call.get(name), str):
                        slot[name] = call[name]
                function = fields(call.get("function")) or {}
                for name in ["name", "arguments"]:
                    if isinstance(function.get(name), str):
                        slot["function"][name] = (
                            slot["function"].get(name, "") + function[name]
                        )

    def result(self):
        if self.response is not None:
            return self.response
        result = {}
        if self.model is not None:
            result["model"] = self.model
        if self.usage is not None:
            result["usage"] = self.usage
        if self.failure is not None:
            result.update(status="failed", error=self.failure)
        if self.saw_content or self.calls:
            message = {}
            if self.saw_content:
                message.update(role="assistant", content=self.content)
            if self.calls:
                message["tool_calls"] = [self.calls[k] for k in sorted(self.calls)]
            result["choices"] = [{"index": 0, "message": message}]
        return result or None


@contextmanager
def operation_scope(operation):
    if operation.done:
        yield
        return
    token = _CURRENT.set(operation)
    try:
        with trace.use_span(
            operation.span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ):
            try:
                yield
            finally:
                operation.checkpoint()
    finally:
        _CURRENT.reset(token)


def finish(operation, response=None, error=None):
    try:
        operation.finish(response, error)
    except Exception:  # noqa: BLE001 - telemetry never replaces native result/error
        operation.done = True
        operation.request = {}
        operation.parent = None
        operation.stream_accumulator = None
        try:
            operation.span.end()
        except Exception:  # noqa: BLE001 - even an ending failure must preserve SDK results
            logger.debug("Could not end Portkey telemetry span")
        if operation.on_finish:
            operation.on_finish(operation)
        operation.source_usage = None


class SyncStreamProxy:
    def __init__(self, stream, operation):
        self._stream = stream
        self._iterator = iter(stream)
        self._operation = operation
        self._accumulator = StreamAccumulator(operation)
        operation.stream_accumulator = self._accumulator

    def __getattr__(self, name):
        value = getattr(self._stream, name)
        if name in {
            "get_final_completion",
            "get_final_response",
            "until_done",
        } and callable(value):

            def scoped(*args, **kwargs):
                try:
                    with operation_scope(self._operation):
                        result = value(*args, **kwargs)
                except BaseException as exc:
                    finish(self._operation, error=exc)
                    raise
                if fields(result) is not None and self._operation.checkpoint():
                    self._accumulator.response = result
                elif self._operation.checkpoint():
                    try:
                        self._accumulator.response = native_snapshot(self._stream)
                    except Exception:  # noqa: BLE001 - empty native snapshots stay absent
                        logger.debug("No completed Portkey stream snapshot")
                return result

            return scoped
        return value

    def __iter__(self):
        return self

    def _step(self, method, *args):
        try:
            with operation_scope(self._operation):
                item = method(*args)
        except StopIteration:
            finish(self._operation, self._accumulator.result())
            raise
        except BaseException as exc:
            finish(self._operation, error=exc)
            raise
        try:
            self._accumulator.add(item)
        except Exception:  # noqa: BLE001 - native chunk identity is preserved
            logger.debug("Could not normalize Portkey stream chunk")
        return item

    def __next__(self):
        return self._step(self._iterator.__next__)

    def send(self, value):
        return self._step(self._iterator.send, value)

    def throw(self, *args):
        return self._step(self._iterator.throw, *args)

    def close(self):
        error = None
        try:
            with operation_scope(self._operation):
                return self._stream.close()
        except BaseException as exc:
            error = exc
            raise
        finally:
            finish(self._operation, self._accumulator.result(), error)

    def __enter__(self):
        try:
            with operation_scope(self._operation):
                result = self._stream.__enter__()
        except BaseException as exc:
            finish(self._operation, error=exc)
            raise
        return self if result is self._stream else result

    def __exit__(self, *args):
        error = args[1]
        try:
            with operation_scope(self._operation):
                return self._stream.__exit__(*args)
        except BaseException as exc:
            error = exc
            raise
        finally:
            finish(self._operation, self._accumulator.result(), error)


class AsyncStreamProxy:
    def __init__(self, stream, operation):
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._operation = operation
        self._accumulator = StreamAccumulator(operation)
        operation.stream_accumulator = self._accumulator

    def __getattr__(self, name):
        value = getattr(self._stream, name)
        if name in {
            "get_final_completion",
            "get_final_response",
            "until_done",
        } and callable(value):

            async def scoped(*args, **kwargs):
                try:
                    with operation_scope(self._operation):
                        result = await value(*args, **kwargs)
                except BaseException as exc:
                    finish(self._operation, error=exc)
                    raise
                if fields(result) is not None and self._operation.checkpoint():
                    self._accumulator.response = result
                elif self._operation.checkpoint():
                    try:
                        self._accumulator.response = native_snapshot(self._stream)
                    except Exception:  # noqa: BLE001 - empty native snapshots stay absent
                        logger.debug("No completed Portkey async stream snapshot")
                return result

            return scoped
        return value

    def __aiter__(self):
        return self

    async def _step(self, method, *args):
        try:
            with operation_scope(self._operation):
                item = await method(*args)
        except StopAsyncIteration:
            finish(self._operation, self._accumulator.result())
            raise
        except BaseException as exc:
            finish(self._operation, error=exc)
            raise
        try:
            self._accumulator.add(item)
        except Exception:  # noqa: BLE001 - native chunk identity is preserved
            logger.debug("Could not normalize Portkey async stream chunk")
        return item

    async def __anext__(self):
        return await self._step(self._iterator.__anext__)

    async def asend(self, value):
        return await self._step(self._iterator.asend, value)

    async def athrow(self, *args):
        return await self._step(self._iterator.athrow, *args)

    async def aclose(self):
        error = None
        try:
            with operation_scope(self._operation):
                method = getattr(self._stream, "aclose", None) or self._stream.close
                result = method()
                return await result if inspect.isawaitable(result) else result
        except BaseException as exc:
            error = exc
            raise
        finally:
            finish(self._operation, self._accumulator.result(), error)

    async def close(self):
        return await self.aclose()

    async def __aenter__(self):
        try:
            with operation_scope(self._operation):
                result = await self._stream.__aenter__()
        except BaseException as exc:
            finish(self._operation, error=exc)
            raise
        return self if result is self._stream else result

    async def __aexit__(self, *args):
        error = args[1]
        try:
            with operation_scope(self._operation):
                return await self._stream.__aexit__(*args)
        except BaseException as exc:
            error = exc
            raise
        finally:
            finish(self._operation, self._accumulator.result(), error)


class StreamManagerProxy:
    def __init__(self, manager, operation):
        self._manager = manager
        self._operation = operation
        self._stream = None

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def __enter__(self):
        try:
            with operation_scope(self._operation):
                result = self._manager.__enter__()
        except BaseException as exc:
            finish(self._operation, error=exc)
            raise
        self._stream = SyncStreamProxy(result, self._operation)
        return self._stream

    def __exit__(self, *args):
        error = args[1]
        try:
            with operation_scope(self._operation):
                return self._manager.__exit__(*args)
        except BaseException as exc:
            error = exc
            raise
        finally:
            finish(
                self._operation,
                self._stream._accumulator.result() if self._stream else None,
                error,
            )

    async def __aenter__(self):
        try:
            with operation_scope(self._operation):
                result = await self._manager.__aenter__()
        except BaseException as exc:
            finish(self._operation, error=exc)
            raise
        self._stream = AsyncStreamProxy(result, self._operation)
        return self._stream

    async def __aexit__(self, *args):
        error = args[1]
        try:
            with operation_scope(self._operation):
                return await self._manager.__aexit__(*args)
        except BaseException as exc:
            error = exc
            raise
        finally:
            finish(
                self._operation,
                self._stream._accumulator.result() if self._stream else None,
                error,
            )


class Runtime:
    def __init__(self, provider, capture=True, config=None):
        self.provider = provider
        self.capture = capture
        self.config = config
        self.active = True
        self.hooks = []
        self.properties = []
        self.open = {}
        self.ended = {}
        self.lock = threading.RLock()
        self.policy = AncestorPolicy(capture)
        self.tracer = provider.get_tracer("openinference.instrumentation.portkey")

    def begin(self, kwargs, kind, stream):
        parent = _CURRENT.get()
        allowed = content_allowed(self.capture)
        if parent is None:
            parent_id = trace.get_current_span().get_span_context().span_id
            parent = self.open.get(parent_id)
            allowed = allowed and self.ended.get(parent_id, True)
        if parent is not None:
            parent.checkpoint()
            allowed = allowed and parent.capture
        span = self.tracer.start_span("portkey." + kind)
        allowed = allowed and self.policy.allowed(span)
        attrs = {k: kwargs.get(k) for k in ["model"] if isinstance(kwargs.get(k), str)}
        attrs["stream"] = stream
        operation = Operation(
            span,
            dict(kwargs) if allowed and span.is_recording() else {},
            kind,
            allowed,
            self.capture,
            parent=parent,
            metadata=attrs,
            on_finish=self.finished,
            policy=self.policy,
        )
        config = self.config
        if config is not None:
            operation.hide_inputs = bool(
                config.hide_inputs
                or config.hide_input_messages
                or config.hide_input_text
                or getattr(config, "hide_prompts", False)
            )
            operation.hide_outputs = bool(
                config.hide_outputs
                or config.hide_output_messages
                or config.hide_output_text
                or getattr(config, "hide_choices", False)
            )
            operation.hide_tools = bool(config.hide_llm_tools)
            operation.hide_vectors = bool(
                config.hide_embedding_vectors
                or getattr(config, "hide_embeddings_vectors", False)
            )
        with self.lock:
            self.open[span.get_span_context().span_id] = operation
        return operation

    def finished(self, operation):
        span_id = operation.span.get_span_context().span_id
        with self.lock:
            self.open.pop(span_id, None)
            self.ended[span_id] = operation.capture
            if len(self.ended) > 2048:
                self.ended.pop(next(iter(self.ended)))

    def wrap_result(self, result, operation, stream, manager):
        if manager:
            return StreamManagerProxy(result, operation)
        if stream and hasattr(result, "__aiter__"):
            return AsyncStreamProxy(result, operation)
        if stream and hasattr(result, "__iter__"):
            return SyncStreamProxy(result, operation)
        finish(operation, result)
        return result

    def wrap(self, original, kind, method):
        runtime = self
        asynchronous = inspect.iscoroutinefunction(original)

        def begin(instance, kwargs):
            operation = runtime.begin(
                kwargs, kind, method == "stream" or kwargs.get("stream") is True
            )
            operation.metadata["source_module"] = getattr(
                original, "__module__", ""
            ).rsplit(".", 1)[-1]
            try:
                host = instance._client.openai_client.base_url.host
                if operation.metadata["source_module"] == "generation":
                    host = instance._client._client.base_url.host or host
                operation.metadata["source_host"] = host
            except Exception:  # noqa: BLE001 - client metadata cannot replace the SDK call
                operation.metadata["source_host"] = None
            return operation

        if asynchronous:

            @wraps(original)
            async def wrapped(instance, *args, **kwargs):
                if not runtime.active or suppressed():
                    with bypass_scope(
                        suppressed() or getattr(runtime, "native_owned", False)
                    ):
                        return await original(instance, *args, **kwargs)
                try:
                    operation = begin(instance, kwargs)
                except Exception:  # noqa: BLE001 - telemetry setup failure preserves SDK call
                    return await original(instance, *args, **kwargs)
                try:
                    with operation_scope(operation):
                        result = await original(instance, *args, **kwargs)
                except BaseException as exc:
                    finish(operation, error=exc)
                    raise
                return runtime.wrap_result(
                    result,
                    operation,
                    operation.metadata["stream"],
                    method == "stream" and hasattr(result, "__aenter__"),
                )
        else:

            @wraps(original)
            def wrapped(instance, *args, **kwargs):
                if not runtime.active or suppressed():
                    with bypass_scope(
                        suppressed() or getattr(runtime, "native_owned", False)
                    ):
                        return original(instance, *args, **kwargs)
                try:
                    operation = begin(instance, kwargs)
                except Exception:  # noqa: BLE001 - telemetry setup failure preserves SDK call
                    return original(instance, *args, **kwargs)
                try:
                    with operation_scope(operation):
                        result = original(instance, *args, **kwargs)
                except BaseException as exc:
                    finish(operation, error=exc)
                    raise
                return runtime.wrap_result(
                    result,
                    operation,
                    operation.metadata["stream"],
                    method == "stream"
                    and (hasattr(result, "__enter__") or hasattr(result, "__aenter__")),
                )

        return wrapped

    def install(self):
        self.provider.add_span_processor(self.policy)
        original_detach = context.detach

        @wraps(original_detach)
        def detached(token):
            try:
                if self.active:
                    current = trace.get_current_span()
                    if self.policy.knows(current):
                        self.policy.observe(current)
                        with self.lock:
                            pending = list(self.open.values())
                        for operation in pending:
                            if not self.policy.allowed(operation.span):
                                operation.capture = False
                                operation.request = {}
                                if operation.stream_accumulator is not None:
                                    operation.stream_accumulator.clear_content()
            except Exception:  # noqa: BLE001 - policy observation cannot replace native detach
                logger.debug("Could not checkpoint Portkey ancestor policy")
            return original_detach(token)

        self.hooks.append((context, "detach", original_detach, detached))
        context.detach = detached
        specs = [
            (
                "chat_complete",
                ["Completions", "AsyncCompletions"],
                "chat",
                ["create", "parse", "stream"],
            ),
            ("generation", ["Completions", "AsyncCompletions"], "chat", ["create"]),
            ("embeddings", ["Embeddings", "AsyncEmbeddings"], "embedding", ["create"]),
            (
                "responses",
                ["Responses", "AsyncResponses"],
                "chat",
                ["create", "parse", "stream"],
            ),
            ("complete", ["Completion", "AsyncCompletion"], "text", ["create"]),
            (
                "beta_chat",
                ["BetaCompletions", "AsyncBetaCompletions"],
                "chat",
                ["parse", "stream"],
            ),
        ]
        for module, classes, kind, methods in specs:
            try:
                source = importlib.import_module(
                    "portkey_ai.api_resources.apis." + module
                )
            except ImportError:
                continue
            for name in classes:
                cls = getattr(source, name, None)
                if cls is None:
                    continue
                for method in methods:
                    original = inspect.getattr_static(cls, method, None)
                    if not callable(original):
                        continue
                    native = getattr(original, "_self_wrapper", None)
                    if (
                        native is not None
                        and type(native).__module__
                        == "openinference.instrumentation.portkey._wrappers"
                    ):
                        for attribute in [
                            "_tracer",
                            "_request_extractor",
                            "_response_extractor",
                        ]:
                            before = getattr(native, attribute)
                            replacement = (
                                NativeTracer(before)
                                if attribute == "_tracer"
                                else SourceExtractor()
                            )
                            self.properties.append(
                                (native, attribute, before, replacement)
                            )
                            setattr(native, attribute, replacement)
                    wrapper = self.wrap(original, kind, method)
                    self.hooks.append((cls, method, original, wrapper))
                    setattr(cls, method, wrapper)
        # Observe the JSON that the SDK already parses, before DTO construction
        # can turn bool/missing token counts into plausible provider usage.
        import httpx
        from portkey_ai._vendor.openai._streaming import ServerSentEvent

        for cls in (httpx.Response, ServerSentEvent):
            original = inspect.getattr_static(cls, "json")

            @wraps(original)
            def parsed(instance, *args, _original=original, **kwargs):
                result = _original(instance, *args, **kwargs)
                try:
                    operation = _CURRENT.get()
                    if isinstance(instance, httpx.Response) and operation is not None:
                        url = instance.request.url
                        module = operation.metadata.get("source_module")
                        if url.host == operation.metadata.get("source_host") and (
                            (
                                module in {"chat_complete", "beta_chat"}
                                and url.path.endswith("/chat/completions")
                            )
                            or (
                                module == "embeddings"
                                and url.path.endswith("/embeddings")
                            )
                            or (
                                module == "responses"
                                and url.path.endswith("/responses")
                            )
                            or (
                                module == "complete"
                                and url.path.endswith("/completions")
                                and not url.path.endswith("/chat/completions")
                            )
                            or (
                                module == "generation"
                                and "/prompts/" in url.path
                                and url.path.endswith("/completions")
                            )
                        ):
                            observe_usage(result)
                    elif not isinstance(instance, httpx.Response):
                        observe_usage(result)
                except Exception:  # noqa: BLE001 - source observation cannot alter parsing
                    logger.debug("Could not observe native Portkey usage")
                return result

            self.hooks.append((cls, "json", original, parsed))
            cls.json = parsed
        for module in ("chat_complete", "complete", "embeddings"):
            source = importlib.import_module("portkey_ai.api_resources.apis." + module)
            if hasattr(source, "json"):
                original = source.json
                replacement = NativeJSON(original)
                self.properties.append((source, "json", original, replacement))
                source.json = replacement

    def close(self):
        self.active = False
        with self.lock:
            pending = list(self.open.values())
        for operation in reversed(pending):
            finish(operation)
        for cls, method, original, wrapper in reversed(self.hooks):
            if inspect.getattr_static(cls, method) is wrapper:
                setattr(cls, method, original)
        for obj, attribute, before, replacement in reversed(self.properties):
            if getattr(obj, attribute, None) is replacement:
                setattr(obj, attribute, before)
        self.hooks = []
        self.properties = []
        self.open = {}
        self.ended = {}
        active = getattr(self.provider, "_active_span_processor", None)
        processors = getattr(active, "_span_processors", None)
        if processors is not None:
            active._span_processors = tuple(
                p for p in processors if p is not self.policy
            )
        self.policy.clear()
