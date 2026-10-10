"""Content policy and builtin-only serialization for native Ollama payloads."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import threading
from collections import OrderedDict
from contextvars import ContextVar
from urllib.parse import urlsplit, urlunsplit

from opentelemetry import context, trace
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

_STARTING = ContextVar("respan_ollama_starting", default=None)

_TRACELOOP_CONTENT_KEY = "override_enable_content_tracing"
_POLICY_ATTR = "traceloop.enable_content_tracing"
_SECRET = re.compile(
    r"(?:api[_-]?key|access[_-]?key|secret[_-]?key|private[_-]?key|secret|password|passwd|authorization|credentials?|token|cookies?)$",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    r"""(?i)(["']?(?:api[_-]?key|access[_-]?key|secret[_-]?key|private[_-]?key|secret|password|authorization|credentials?|cookies?|token)["']?\s*[:=]\s*)("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\[REDACTED\]|[^\s,;\]}]+)"""
)
_AUTH = re.compile(
    r"""(?i)\b(Bearer|Basic)\s+(?:"[^"\r\n]*"|'[^'\r\n]*'|[A-Za-z0-9_+/.=:-]+)"""
)
_URL = re.compile(r"https?://[^\s\"'<>]+")


def sensitive_key(key: str) -> bool:
    return bool(_SECRET.search(key))


def text(value: str) -> str:
    value = _AUTH.sub(r"\1 [REDACTED]", value)

    def assignment(match):
        raw = match.group(2)
        quote = raw[0] if raw[:1] in ('"', "'") else ""
        return match.group(1) + quote + "[REDACTED]" + quote

    value = _ASSIGNMENT.sub(assignment, value)

    def url(match):
        try:
            parts = urlsplit(match.group())
        except ValueError:
            return "[REDACTED]"
        if parts.username is None and not parts.query:
            return match.group()
        host = parts.netloc.rsplit("@", 1)[-1]
        query = re.sub(
            r"(?i)([?&]?(?:token|api_key|access_token|signature|credential|password)=)[^&]*",
            r"\1[REDACTED]",
            parts.query,
        )
        return urlunsplit((parts.scheme, host, parts.path, query, parts.fragment))

    return _URL.sub(url, value)


def value(data, *, schema: bool = False, secret: bool = False, seen=None):
    """Never call arbitrary mapping, iterator, repr, or object serialization hooks."""
    if secret:
        return "[REDACTED]"
    if type(data) is float:
        return data if math.isfinite(data) else None
    if data is None or type(data) in (bool, int):
        return data
    if type(data) is str:
        # Native JSON strings and tool arguments are JSON strings. Parse
        # those structurally so schema property identifiers remain intact.
        if data.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(data)
            except ValueError:
                decoded = None
            if type(decoded) in (dict, list):
                return json.dumps(
                    value(decoded, schema=schema, seen=seen), ensure_ascii=False
                )
        return text(data)
    if type(data) in (bytes, bytearray):
        raw = bytes(data)
        try:
            return value(json.loads(raw), seen=seen)
        except (ValueError, UnicodeDecodeError):
            return {"base64": base64.b64encode(raw).decode("ascii")}
    if type(data) not in (dict, list, tuple):
        return None
    seen = set() if seen is None else seen
    if id(data) in seen:
        return None
    seen.add(id(data))
    try:
        if type(data) in (list, tuple):
            return [value(item, schema=schema, seen=seen) for item in data]
        result = {}
        for key, item in data.items():
            if type(key) is not str:
                continue
            # JSON Schema property names are identifiers, not credentials. Their
            # defaults/examples/const/enum still carry sensitive data.
            if schema and key == "properties" and type(item) is dict:
                result[key] = {
                    name: value(spec, schema=True, secret=False, seen=seen)
                    if not sensitive_key(name)
                    else _secret_schema(spec, seen)
                    for name, spec in item.items()
                    if type(name) is str
                }
            else:
                result[key] = value(
                    item,
                    schema=schema
                    or key in ("inputSchema", "input_schema", "parameters", "format"),
                    secret=sensitive_key(key)
                    and key
                    not in (
                        "input_tokens",
                        "output_tokens",
                        "prompt_tokens",
                        "completion_tokens",
                    ),
                    seen=seen,
                )
        return result
    finally:
        seen.remove(id(data))


def _secret_schema(spec, seen):
    if type(spec) is not dict:
        return value(spec, schema=True, seen=seen)
    if id(spec) in seen:
        return None
    seen.add(id(spec))
    try:
        return {
            key: "[REDACTED]"
            if key in ("default", "const", "enum", "example", "examples")
            else value(item, schema=True, seen=seen)
            for key, item in spec.items()
            if type(key) is str
        }
    finally:
        seen.remove(id(spec))


def json_text(data) -> str:
    return json.dumps(value(data), ensure_ascii=False, allow_nan=False)


def _flag(data):
    return (
        bool(data)
        if type(data) in (str, bool, int, float, dict, list, tuple)
        else data is not None
    )


def suppressed(ctx=None) -> bool:
    return any(
        _flag(context.get_value(key, candidate))
        for candidate in (ctx, context.get_current())
        for key in (
            _SUPPRESS_INSTRUMENTATION_KEY,
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
        )
    )


def content_allowed(ctx=None) -> bool:
    """Latch explicit content policy; SDK export suppression is transient."""
    return all(
        context.get_value(ENABLE_CONTENT_TRACING_KEY, candidate) is not False
        and context.get_value(_TRACELOOP_CONTENT_KEY, candidate) is not False
        for candidate in (ctx, context.get_current())
    ) and all(
        os.getenv(name, "true").strip().lower() not in ("false", "0", "no", "off")
        for name in ("TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT")
    )


class PolicyObserver(SpanProcessor):
    """Remember observed ancestors, including ended carriers, conservatively."""

    def __init__(self):
        self.records = OrderedDict()
        self.lock = threading.RLock()

    def on_start(self, span, parent_context=None):
        starting = _STARTING.get()
        if (
            starting is not None
            and starting.span is None
            and span.name == "ollama." + starting.mode
        ):
            starting.span = span
        parent = trace.get_current_span(parent_context).get_span_context()
        record = [
            span,
            content_allowed(parent_context),
            (parent.trace_id, parent.span_id)
            if parent.is_valid and not parent.is_remote
            else None,
        ]
        with self.lock:
            sc = span.get_span_context()
            sid = (sc.trace_id, sc.span_id)
            self.records[sid] = record
            if not record[1]:
                self._deny_chain(sid)
            while len(self.records) > 4096:
                self.records.popitem(last=False)

    def on_end(self, span):
        with self.lock:
            record = self.records.get((span.context.trace_id, span.context.span_id))
            if record:
                record[1] = (
                    record[1]
                    and content_allowed()
                    and (span.attributes or {}).get(_POLICY_ATTR) is not False
                    and (span.attributes or {}).get(ENABLE_CONTENT_TRACING_KEY)
                    is not False
                )
                if not record[1]:
                    self._deny_chain((span.context.trace_id, span.context.span_id))
                record[0] = None

    def _deny_chain(self, sid):
        seen = set()
        while sid is not None and sid not in seen:
            seen.add(sid)
            record = self.records.get(sid)
            if record is None:
                break
            record[1] = False
            sid = record[2]

    def deny(self, span):
        with self.lock:
            sc = span.get_span_context()
            self._deny_chain((sc.trace_id, sc.span_id))

    def allowed(self, carrier):
        sc = carrier.get_span_context()
        if not sc.is_valid or sc.is_remote:
            return True
        sid = (sc.trace_id, sc.span_id)
        observed_sid = sid
        with self.lock:
            seen = set()
            while sid is not None and sid not in seen:
                seen.add(sid)
                record = self.records.get(sid)
                if record is None:
                    self._deny_chain(observed_sid)
                    return False
                observed, allowed, sid = record
                if not allowed or (
                    observed is not None
                    and (
                        (observed.attributes or {}).get(_POLICY_ATTR) is False
                        or (observed.attributes or {}).get(ENABLE_CONTENT_TRACING_KEY)
                        is False
                    )
                ):
                    self._deny_chain(observed_sid)
                    return False
        return True

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True
