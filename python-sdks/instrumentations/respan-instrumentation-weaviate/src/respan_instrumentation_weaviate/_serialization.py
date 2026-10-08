"""Redact builtin JSON without invoking methods on unknown customer objects."""

from __future__ import annotations

import base64
import datetime
import enum
import json
import math
import re
import sys
import types
import uuid
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from weaviate.util import _WeaviateUUIDInt

MAX_DEPTH = 64
MAX_STRING_BYTES = 4_000
REDACTED = "[REDACTED]"
_SENSITIVE_SUFFIXES = (
    "apikey",
    "accesskey",
    "secretkey",
    "privatekey",
    "credentials",
    "authorization",
    "credential",
    "cookie",
    "cookies",
    "password",
    "secret",
    "sessiontoken",
    "token",
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:(?:api|access|secret|private)[_-]?key|credentials?|authorization|cookies?|password|(?:client[_-]?)?secret|session[_-]?token|token)[\"']?)"
    r"(\s*[:=]\s*)(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\[REDACTED\]|[^\s,;}\]]+)"
)
_AUTHORIZATION = re.compile(
    r"(?i)\b(?:bearer|basic)\s+(?:(?P<quote>\\*[\"'])(?:\\.|[^\r\n])*?(?P=quote)|[a-z0-9._~+/=-]+)"
)
_URL = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>]+")
_SCHEMA_KEYS = frozenset(
    {"type", "$ref", "properties", "items", "anyOf", "allOf", "oneOf"}
)


def _type_name(value):
    return type.__dict__["__name__"].__get__(type(value))


def _truncate_utf8(value, limit=MAX_STRING_BYTES):
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    suffix = "...[truncated]"
    return (
        encoded[: max(0, limit - len(suffix))].decode("utf-8", errors="ignore") + suffix
    )


def sanitize_url(value):
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if not parsed.scheme or not hostname:
            return "<redacted-endpoint>"
        netloc = f"[{hostname}]" if ":" in hostname else hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        query = urlencode(
            [
                (key, REDACTED if sensitive_key(key) else item)
                for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            ]
        )
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))
    except ValueError:
        return "<redacted-endpoint>"


def _redact_assignment(match):
    value = match.group(3)
    quote = value[0] if value[:1] in {'"', "'"} else ""
    return f"{match.group(1)}{match.group(2)}{quote}{REDACTED}{quote}"


def safe_text(value: Any, *, default="", truncate=False, max_bytes=None):
    if type(value) is str:
        # A JSON-encoded argument remains valid JSON after redaction. Repeated
        # sanitation is idempotent, including escaped quoted credential values.
        stripped = value.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except (ValueError, RecursionError):
                pass
            else:
                sanitized = to_jsonable(decoded)
                if sanitized != decoded:
                    value = json.dumps(
                        sanitized, ensure_ascii=False, separators=(",", ":")
                    )
                return _truncate_utf8(value) if truncate else value
        value = _URL.sub(lambda match: sanitize_url(match.group()), value)
        value = _AUTHORIZATION.sub(REDACTED, value)
        value = _SECRET_ASSIGNMENT.sub(_redact_assignment, value)
        return _truncate_utf8(value) if truncate else value
    if value is None:
        return default
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return str(value)
    return f"<{_type_name(value)}>"


def safe_exception_message(exc):
    try:
        arguments = BaseException.args.__get__(exc)
    except Exception:  # noqa: BLE001 - diagnostics never alter native errors.
        arguments = ()
    if type(arguments) is tuple:
        for argument in arguments:
            if any(
                type(argument) is item
                for item in (
                    str,
                    bool,
                    int,
                    float,
                )
            ):
                return safe_text(argument)
    return None


def sensitive_key(value):
    if type(value) is not str:
        return False
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    return any(normalized.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


_NATIVE_TYPES = ()
_UUID_INT = uuid.UUID.__dict__["int"]


def register_native_types():
    global _NATIVE_TYPES
    kinds = []
    for name, module in list(sys.modules.items()):
        if (
            name.startswith("weaviate.collections.")
            and type(module) is types.ModuleType
        ):
            for value in vars(module).values():
                if isinstance(value, type):
                    namespace = type.__dict__["__dict__"].__get__(value)
                    home = namespace.get("__module__")
                    if (
                        type(home) is str
                        and home.startswith("weaviate.collections.")
                        and not any(value is kind for kind in kinds)
                    ):
                        kinds.append(value)
    _NATIVE_TYPES = tuple(kinds)


def native_storage(value):
    if not any(type(value) is kind for kind in _NATIVE_TYPES):
        return None
    for owner in type.__dict__["__mro__"].__get__(type(value)):
        descriptor = type.__dict__["__dict__"].__get__(owner).get("__dict__")
        if type(descriptor) is types.GetSetDescriptorType:
            return descriptor.__get__(value, type(value))
    return None


def to_jsonable(
    value, *, depth=0, seen=None, schema_properties=False, schema_secret=False
):
    kind = type(value)
    if kind is uuid.UUID:
        return uuid.UUID.__str__(value)
    if kind is _WeaviateUUIDInt:
        lineage = type.__dict__["__mro__"].__get__(kind)
        if (
            len(lineage) == 3
            and lineage[0] is _WeaviateUUIDInt
            and lineage[1] is uuid.UUID
            and lineage[2] is object
        ):
            raw = _UUID_INT.__get__(value, uuid.UUID)
            if type(raw) is int and 0 <= raw < 1 << 128:
                return uuid.UUID.__str__(uuid.UUID(int=raw))
        return {"type": _type_name(value)}
    if kind is datetime.datetime:
        return datetime.datetime.isoformat(value)
    if kind is datetime.date:
        return datetime.date.isoformat(value)
    if any(kind is native for native in _NATIVE_TYPES):
        if isinstance(value, enum.Enum):
            return to_jsonable(native_storage(value).get("_value_"), seen=seen)
        raw = native_storage(value)
        if _type_name(value) in {"Collection", "CollectionAsync"}:
            return {"collection": safe_text(raw.get("name")), "type": _type_name(value)}
        if type(raw) is dict and type.__dict__["__dict__"].__get__(kind).get(
            "__module__", ""
        ).startswith("weaviate.collections.classes."):
            active = seen if seen is not None else set()
            identity = id(value)
            if identity in active:
                return "<cycle>"
            active.add(identity)
            try:
                return to_jsonable(
                    {key: item for key, item in raw.items() if type(key) is str},
                    seen=active,
                )
            finally:
                active.discard(identity)
        fields = type.__dict__["__dict__"].__get__(kind).get("_fields")
        if type(fields) is tuple and issubclass(kind, tuple):
            return to_jsonable(dict(zip(fields, tuple.__iter__(value))), seen=seen)
        return {"type": _type_name(value)}
    if value is None or any(
        kind is item
        for item in (
            bool,
            int,
        )
    ):
        return value
    if kind is str:
        return safe_text(value, truncate=False)
    if kind is float:
        return value if math.isfinite(value) else None
    if any(
        kind is item
        for item in (
            bytes,
            bytearray,
            memoryview,
        )
    ):
        return {"base64": base64.b64encode(value).decode("ascii"), "type": "bytes"}
    if not any(
        kind is item
        for item in (
            dict,
            list,
            tuple,
        )
    ):
        return {"type": _type_name(value)}
    active = seen if seen is not None else set()
    identity = id(value)
    if identity in active:
        return "<cycle>"
    active.add(identity)
    try:
        if kind is dict:
            result = {}
            for key, item in value.items():
                name = safe_text(key, truncate=False)
                secret = sensitive_key(key)
                definition = (
                    schema_properties
                    and type(item) is dict
                    and any(type(k) is str and k in _SCHEMA_KEYS for k in item)
                )
                if (secret and not definition) or (
                    schema_secret
                    and name in {"default", "examples", "example", "const", "enum"}
                ):
                    result[name] = REDACTED
                else:
                    result[name] = to_jsonable(
                        item,
                        depth=depth + 1,
                        seen=active,
                        schema_properties=name == "properties",
                        schema_secret=secret if definition else schema_secret,
                    )
            return result
        return [to_jsonable(item, depth=depth + 1, seen=active) for item in value]
    finally:
        active.discard(identity)


def json_dumps(value, *, max_bytes=None):
    return json.dumps(
        to_jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def safe_type_name(value):
    return _type_name(value)
