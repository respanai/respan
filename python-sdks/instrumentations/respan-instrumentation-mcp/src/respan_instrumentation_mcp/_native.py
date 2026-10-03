"""Avoid duplicate MCP 2 protocol spans inside an owned client operation."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes.mcp_attributes import MCP_METHOD_NAME

_METHODS = {
    "initialize": "initialize",
    "list_tools": "tools/list",
    "call_tool": "tools/call",
    "list_resources": "resources/list",
    "read_resource": "resources/read",
    "list_resource_templates": "resources/templates/list",
    "list_prompts": "prompts/list",
    "get_prompt": "prompts/get",
}
_OWNED_OPERATION: ContextVar[tuple[str, int, int, tuple[str, str]] | None] = ContextVar(
    "respan_mcp_owned_operation", default=None
)
_TARGET_MODULES = (
    "mcp.shared._otel",
    "mcp.shared.jsonrpc_dispatcher",
    "mcp.server._otel",
)


@contextmanager
def owned_operation(method_name: str, entity_name: str):
    context = trace.get_current_span().get_span_context()
    method = _METHODS[method_name]
    target = f" {entity_name}" if method_name in {"call_tool", "get_prompt"} else ""
    names = (f"MCP send {method}{target}", f"{method}{target}")
    token = _OWNED_OPERATION.set(
        (method, context.trace_id, context.span_id, names) if context.is_valid else None
    )
    try:
        yield
    finally:
        _OWNED_OPERATION.reset(token)


def _guard(original):
    @wraps(original)
    @contextmanager
    def guarded(*args, **kwargs):
        owned = _OWNED_OPERATION.get()
        current = trace.get_current_span().get_span_context()
        attributes = kwargs.get("attributes") or {}
        source_context = kwargs.get("context")
        incoming = (
            trace.get_current_span(source_context).get_span_context()
            if source_context is not None
            else current
        )
        if (
            owned is not None
            and (attributes.get(MCP_METHOD_NAME), current.trace_id, current.span_id)
            == owned[:3]
            and (incoming.trace_id, incoming.span_id) == owned[1:3]
            and (kwargs.get("name") or (args[0] if args else None)) in owned[3]
        ):
            # Keep the owning span's identity available to SDK context injection,
            # while avoiding mutations from the redundant protocol wrapper.
            yield trace.NonRecordingSpan(current)
            return
        with original(*args, **kwargs) as span:
            yield span

    return guarded


def patch_native_spans() -> list[tuple[Any, Any, Any]]:
    # Import first so late modules do not bind one of our temporary wrappers.
    modules = []
    for module_name in _TARGET_MODULES:
        try:
            modules.append(importlib.import_module(module_name))
        except ModuleNotFoundError as exc:
            if exc.name in {module_name, module_name.rpartition(".")[0]}:
                continue  # MCP 1.x does not have these native tracing helpers.
            raise
    patches = []
    try:
        for module in modules:
            original = getattr(module, "otel_span", None)
            if not callable(original):
                continue
            wrapped = _guard(original)
            module.otel_span = wrapped
            patches.append((module, original, wrapped))
    except Exception:
        restore_native_spans(patches)
        raise
    return patches


def restore_native_spans(patches) -> None:
    for module, original, wrapped in reversed(patches):
        if module.otel_span is wrapped:
            module.otel_span = original
