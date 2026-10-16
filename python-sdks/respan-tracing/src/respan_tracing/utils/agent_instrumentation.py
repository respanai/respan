"""Building blocks for instrumenting agent frameworks (browser agents first).

Lifted from respan-instrumentation-browser-use, where each piece is covered by
its baseline wire-contract tests:

- ``AgentScope``: the running agent's session id (default thread / session /
  trace group of every span) and its secrets (masked out of every span);
- ``agent_span``: start a Respan-exportable span with grouping, readable error
  status and secret masking applied;
- ``set_error`` / ``record_error``: an error message Respan can show;
- ``MethodPatcher``: patch framework methods once, restore them exactly;
- ``compact_image_data_url``: shrink inline screenshots before export;
- ``jsonable`` / ``json_string``: framework objects as the JSON Respan renders;
- ``suppress_provider_llm_spans``: keep provider SDK instrumentations from
  logging an LLM call the agent span already carries.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import enum
import io
import json
import logging
import math
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import date, datetime
from functools import cache, partial
from typing import Any, Optional
from urllib.parse import quote, quote_plus

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_MESSAGE,
    EXCEPTION_STACKTRACE,
    EXCEPTION_TYPE,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    TraceloopSpanKindValues,
)
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
)
from respan_sdk.constants.otlp_constants import OTEL_EXCEPTION_EVENT_NAME
from respan_sdk.constants.span_attributes import (
    RESPAN_INTERNAL_SPAN_NAME_DETAIL,
    RESPAN_LOG_TYPE,
    RESPAN_SESSION_ID,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)

logger = logging.getLogger(__name__)

ERROR_MESSAGE_LIMIT = 500
SCREENSHOT_MAX_WIDTH = 1280
SCREENSHOT_JPEG_QUALITY = 70
DATA_URL_PREFIX = "data:"
PATCH_MARKER = "__respan_patched__"
JSON_MAX_DEPTH = 32

_GROUPING_ATTRIBUTES = (RESPAN_THREADS_ID, RESPAN_SESSION_ID, RESPAN_TRACE_GROUP_ID)
# Chat spans carry llm.request.type instead of a span kind.
_SPAN_KINDS = {
    LOG_TYPE_AGENT: TraceloopSpanKindValues.AGENT.value,
    LOG_TYPE_TASK: TraceloopSpanKindValues.TASK.value,
    LOG_TYPE_TOOL: TraceloopSpanKindValues.TOOL.value,
}

Replacements = tuple[tuple[str, str], ...]


def safely(operation: Callable[[], Any]) -> None:
    """Run telemetry code that must never change the agent's behaviour."""
    try:
        operation()
    except Exception:
        logger.debug("agent telemetry failed open", exc_info=True)


def secret_replacements(sensitive_data: Any) -> Replacements:
    """(secret, "<secret>name</secret>") pairs, longest secret first.

    Accepts ``{name: value}`` or ``{scope: {name: value}}`` (browser-use's
    per-domain form). Percent-encoded forms are included because a secret typed
    into a URL appears encoded there.
    """
    replacements: dict[str, str] = {}
    for key, value in (sensitive_data or {}).items():
        items = value.items() if isinstance(value, Mapping) else [(key, value)]
        for name, secret in items:
            if not isinstance(secret, str) or not secret:
                continue
            placeholder = f"<secret>{name}</secret>"
            for variant in (secret, quote(secret, safe=""), quote_plus(secret)):
                replacements[variant] = placeholder
    return tuple(sorted(replacements.items(), key=lambda pair: -len(pair[0])))


class AgentScope:
    """Per-run state every span of an agent run inherits through contextvars."""

    def __init__(self, name: str) -> None:
        self._session: ContextVar[Optional[str]] = ContextVar(f"{name}_session", default=None)
        self._secrets: ContextVar[Replacements] = ContextVar(f"{name}_secrets", default=())

    @contextmanager
    def run(self, session_id: Any, sensitive_data: Any = None) -> Iterator[None]:
        session = self._session.set(str(session_id) if session_id is not None else None)
        secrets = self._secrets.set(secret_replacements(sensitive_data))
        try:
            yield
        finally:
            self._secrets.reset(secrets)
            self._session.reset(session)

    def mask(self, text: str) -> str:
        for secret, placeholder in self._secrets.get():
            text = text.replace(secret, placeholder)
        return text

    def mask_span(self, span: Any) -> None:
        if not self._secrets.get():
            return
        for key, value in list((span.attributes or {}).items()):
            if isinstance(value, str) and (masked := self.mask(value)) != value:
                span.set_attribute(key, masked)

    def apply_grouping(self, span: Any) -> None:
        # Respan's span processor already copied the customer's propagated
        # thread/session/trace group onto the span in start_span; those win.
        if (session := self._session.get()) is None:
            return
        for key in _GROUPING_ATTRIBUTES:
            if not span.attributes.get(key):
                span.set_attribute(key, session)

    @property
    def has_secrets(self) -> bool:
        return bool(self._secrets.get())


def set_error(
    span: Any,
    summary: str,
    detail: Any = None,
    *,
    with_message: bool,
    mask: Callable[[str], str] = str,
) -> None:
    """ERROR status whose description is what Respan shows as the error message."""
    if with_message and (message := mask(str(detail or "")).strip()):
        summary = f"{summary}: {message[:ERROR_MESSAGE_LIMIT]}"
    span.set_status(Status(StatusCode.ERROR, summary))


def record_error(
    span: Any, error: BaseException, *, with_message: bool, scope: Optional[AgentScope] = None
) -> None:
    mask = scope.mask if scope else str
    set_error(span, type(error).__name__, error, with_message=with_message, mask=mask)
    if not with_message:
        # Provider and validation errors echo prompts and raw model output.
        span.add_event(OTEL_EXCEPTION_EVENT_NAME, {EXCEPTION_TYPE: type(error).__name__})
    elif scope and scope.has_secrets:
        span.add_event(
            OTEL_EXCEPTION_EVENT_NAME,
            {
                EXCEPTION_TYPE: type(error).__name__,
                EXCEPTION_MESSAGE: mask(str(error)),
                EXCEPTION_STACKTRACE: mask("".join(traceback.format_exception(error))),
            },
        )
    else:
        span.record_exception(error)


@contextmanager
def agent_span(
    tracer: trace.Tracer,
    name: str,
    log_type: str,
    *,
    scope: AgentScope,
    content_allowed: Callable[[], bool],
    detail: Optional[str] = None,
) -> Iterator[Any]:
    """A span Respan exports (log type + span kind), grouped, masked, error-aware.

    ``detail`` is the display hint: Respan names agent/tool spans
    ``<kind>.<detail>``.
    """
    attributes: dict[str, Any] = {
        RESPAN_LOG_TYPE: log_type,
        SpanAttributes.TRACELOOP_ENTITY_NAME: name,
    }
    if detail:
        attributes[RESPAN_INTERNAL_SPAN_NAME_DETAIL] = detail
    if span_kind := _SPAN_KINDS.get(log_type):
        # respan-tracing exports only spans with a span kind or an LLM request
        # type; without it agent/step/action spans never leave the process.
        attributes[SpanAttributes.TRACELOOP_SPAN_KIND] = span_kind
    span = tracer.start_span(name, attributes=attributes)
    safely(partial(scope.apply_grouping, span))
    token = context_api.attach(trace.set_span_in_context(span))
    try:
        yield span
    except BaseException as error:
        safely(partial(record_error, span, error, with_message=content_allowed(), scope=scope))
        raise
    finally:
        context_api.detach(token)
        safely(partial(scope.mask_span, span))
        span.end()


class MethodPatcher:
    """Patch methods on framework classes once; restore exactly what was there."""

    def __init__(self) -> None:
        self._patches: list[tuple[type, str, Any]] = []

    def patch(self, owner: type, name: str, factory: Callable[[Any], Any]) -> None:
        original = owner.__dict__.get(name)
        if original is None or getattr(original, PATCH_MARKER, False):
            return
        wrapper = factory(original)
        setattr(wrapper, PATCH_MARKER, True)
        setattr(owner, name, wrapper)
        self._patches.append((owner, name, original))

    def restore(self) -> None:
        for owner, name, original in reversed(self._patches):
            if getattr(owner.__dict__.get(name), PATCH_MARKER, False):
                setattr(owner, name, original)
        self._patches.clear()


def compact_image_data_url(
    data_url: str,
    *,
    max_width: int = SCREENSHOT_MAX_WIDTH,
    quality: int = SCREENSHOT_JPEG_QUALITY,
) -> str:
    """Re-encode an inline image as a smaller JPEG data URL.

    Full-viewport PNG screenshots are ~0.5 MB of base64; large export batches
    time out. Returns the input unchanged when Pillow is missing, the data is
    unreadable, or the JPEG would not be smaller.
    """
    header, _, payload = data_url.partition(",")
    if not header.startswith(DATA_URL_PREFIX) or not header.endswith(";base64"):
        return data_url
    try:
        # Optional: Pillow ships with the frameworks that take screenshots.
        from PIL import Image

        image = Image.open(io.BytesIO(base64.b64decode(payload, validate=True))).convert("RGB")
        if image.width > max_width:
            image = image.resize((max_width, round(image.height * max_width / image.width)))
        out = io.BytesIO()
        image.save(out, "JPEG", quality=quality, optimize=True)
    except (ImportError, OSError, ValueError, binascii.Error):
        return data_url
    compact = f"data:image/jpeg;base64,{base64.b64encode(out.getvalue()).decode()}"
    return compact if len(compact) < len(data_url) else data_url


def jsonable(value: Any, *, depth: int = 0) -> Any:
    """A JSON-safe copy: pydantic models and dataclasses are dumped, bytes summarised.

    Unknown objects become ``{"type": <class name>}`` instead of ``str()`` or
    ``__dict__``: framework objects (browser sessions, clients) are large and
    can hold credentials.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, enum.Enum):
        return jsonable(value.value, depth=depth)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "length": len(value)}
    if depth >= JSON_MAX_DEPTH:
        return {"type": type(value).__name__, "truncated": True}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump(mode="json", exclude_none=True), depth=depth + 1)
        except Exception:  # telemetry must fail open
            return {"type": type(value).__name__, "serialization_error": True}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        return jsonable({k: v for k, v in fields.items() if v is not None}, depth=depth + 1)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [jsonable(v, depth=depth + 1) for v in value]
    return {"type": type(value).__name__}


def json_string(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, default=str)


@cache
def openai_suppressor() -> Callable[[], Any]:
    """Respan's OpenAI instrumentation switch, or a no-op when it isn't installed.

    Resolved once: Python does not cache a failed import.
    """
    try:
        from respan_instrumentation_openai._instrumentation import (
            suppress_openai_instrumentation,
        )
    except ImportError:
        return nullcontext
    return suppress_openai_instrumentation


@contextmanager
def suppress_provider_llm_spans() -> Iterator[None]:
    """Provider SDK instrumentations would log the call the agent's LLM span carries.

    Most honour OTel's language-model suppression key; Respan's OpenAI
    instrumentation has its own switch.
    """
    token = context_api.attach(
        context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
    )
    try:
        with openai_suppressor()():
            yield
    finally:
        context_api.detach(token)
