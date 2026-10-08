"""AutoGen AgentChat instrumentation plugin for Respan."""

import logging
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Any

from opentelemetry import trace
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_autogen._native_processor import (
    AutoGenNativeSpanProcessor,
)

logger = logging.getLogger(__name__)

AUTOGEN_INSTRUMENTATION_NAME = "autogen"
TRACER_PROVIDER_KWARG = "tracer_provider"


def _respan_get_llm_tool_attributes(tools: Any) -> dict[str, Any]:
    """Capture both AutoGen Tool objects and current ToolSchema mappings.

    OpenInference AutoGen 0.1.11 accepts ``Tool | ToolSchema`` at the model
    boundary but serializes only ``Tool`` instances. AutoGen AgentChat 0.7.5
    normally passes the mapping form returned by a workbench, so its complete
    name/description/parameter schema otherwise disappears before the span is
    started.
    """

    if not isinstance(tools, Sequence) or isinstance(tools, str | bytes):
        return {}

    from openinference.instrumentation import safe_json_dumps
    from openinference.semconv.trace import SpanAttributes, ToolAttributes

    attributes: dict[str, Any] = {}
    for tool_index, tool in enumerate(tools):
        if isinstance(tool, Mapping):
            tool_schema: Any = dict(tool)
        else:
            tool_schema = getattr(tool, "schema", None)
        if tool_schema is None:
            continue

        if not isinstance(tool_schema, str):
            tool_schema = safe_json_dumps(tool_schema)
        attributes[
            f"{SpanAttributes.LLM_TOOLS}.{tool_index}.{ToolAttributes.TOOL_JSON_SCHEMA}"
        ] = tool_schema
    return attributes


_OWNERS_LOCK = Lock()
_PROCESSOR = None
_PRIVACY_PROCESSOR = None
_PROVIDER = None
_OWNERS = 0
_MODERN_OWNERS = 0
_SETTINGS = {}


def _register(provider, processor, privacy):
    from respan_instrumentation_openinference._translator import OpenInferenceTranslator

    active = getattr(provider, "_active_span_processor", None)
    processors = getattr(active, "_span_processors", None)
    if processors is None:
        raise RuntimeError(
            "AutoGen requires an SDK TracerProvider with an ordered processor chain"
        )
    remaining = tuple(p for p in processors if p not in (processor, privacy))
    translators = tuple(p for p in remaining if isinstance(p, OpenInferenceTranslator))
    others = tuple(p for p in remaining if p not in translators)
    active._span_processors = (processor, *translators, privacy, *others)


class AutoGenInstrumentor:
    """Trace modern AgentChat or explicitly selected legacy autogen APIs."""

    name = AUTOGEN_INSTRUMENTATION_NAME

    def __init__(self, *, api="agentchat", **instrumentor_kwargs):
        if api not in ("agentchat", "legacy"):
            raise ValueError("api must be 'agentchat' or 'legacy'")
        instrumentor_kwargs.pop(TRACER_PROVIDER_KWARG, None)
        self._api = api
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._is_instrumented = False

    def activate(self):
        global _PROCESSOR, _PRIVACY_PROCESSOR, _PROVIDER, _OWNERS, _MODERN_OWNERS
        if self._is_instrumented:
            return
        runtime = getattr(RespanTracer, "_instance", None)
        if runtime is not None and not getattr(runtime, "is_enabled", True):
            return
        from respan_instrumentation_autogen._native_processor import (
            AutoGenPrivacyProcessor,
        )

        if self._api == "legacy":
            from respan_instrumentation_autogen._legacy import LegacyAutoGenInstrumentor

            delegate_class = LegacyAutoGenInstrumentor
        else:
            from respan_instrumentation_autogen._modern import ModernAutoGenInstrumentor

            delegate_class = ModernAutoGenInstrumentor
        provider = trace.get_tracer_provider()
        with _OWNERS_LOCK:
            if _PROVIDER is not None and _PROVIDER is not provider:
                raise RuntimeError(
                    "AutoGen is already active on another tracer provider"
                )
            if (
                self._api in _SETTINGS
                and _SETTINGS[self._api][0] != self._instrumentor_kwargs
            ):
                raise ValueError(
                    "Active AutoGen owners must use matching instrumentation settings"
                )
            processor = _PROCESSOR or AutoGenNativeSpanProcessor()
            privacy = _PRIVACY_PROCESSOR or AutoGenPrivacyProcessor()
            delegate = OpenInferenceInstrumentor(
                delegate_class, **self._instrumentor_kwargs
            )
            try:
                delegate.activate()
                if not delegate_class().is_instrumented_by_opentelemetry:
                    raise RuntimeError("Install a supported AutoGen SDK version")
                _register(provider, processor, privacy)
            except BaseException:
                delegate.deactivate()
                raise
            _PROCESSOR, _PRIVACY_PROCESSOR, _PROVIDER = processor, privacy, provider
            previous_count = _SETTINGS.get(self._api, ({}, 0))[1]
            _SETTINGS[self._api] = (dict(self._instrumentor_kwargs), previous_count + 1)
            _OWNERS += 1
            _MODERN_OWNERS += self._api == "agentchat"
            processor.enabled = _MODERN_OWNERS > 0
            self._delegate = delegate
            self._is_instrumented = True

    def deactivate(self):
        global _PROCESSOR, _PRIVACY_PROCESSOR, _PROVIDER, _OWNERS, _MODERN_OWNERS
        if not self._is_instrumented:
            return
        with _OWNERS_LOCK:
            self._delegate.deactivate()
            self._delegate = None
            self._is_instrumented = False
            settings, count = _SETTINGS[self._api]
            if count == 1:
                _SETTINGS.pop(self._api)
            else:
                _SETTINGS[self._api] = (settings, count - 1)
            _OWNERS -= 1
            _MODERN_OWNERS -= self._api == "agentchat"
            _PROCESSOR.enabled = _MODERN_OWNERS > 0
            if _OWNERS:
                _register(_PROVIDER, _PROCESSOR, _PRIVACY_PROCESSOR)
                return
            active = _PROVIDER._active_span_processor
            active._span_processors = tuple(
                p
                for p in active._span_processors
                if p not in (_PROCESSOR, _PRIVACY_PROCESSOR)
            )
            _PROCESSOR.shutdown()
            _PRIVACY_PROCESSOR.shutdown()
            _PROCESSOR = _PRIVACY_PROCESSOR = _PROVIDER = None
