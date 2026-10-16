"""Irreversible policy bounds for native manual-log lifetimes."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.context import is_content_tracing_enabled

_ALLOW_TRACE_CONTENT_KEY = "allow_trace_content"
_CONTENT_ATTRIBUTE_KEYS = (_ALLOW_TRACE_CONTENT_KEY, ENABLE_CONTENT_TRACING_KEY)


@dataclass
class CapturePolicy:
    capture_content: bool
    enabled: bool
    parent: CapturePolicy | None = None
    owner: Any = None
    ancestor: Any = None


CURRENT_POLICY: ContextVar[CapturePolicy | None] = ContextVar(
    "respan_helicone_capture_policy", default=None
)


def runtime_enabled() -> bool:
    runtime = getattr(RespanTracer, "_instance", None)
    return runtime is None or bool(getattr(runtime, "is_enabled", True))


def _content_enabled() -> bool:
    return (
        context.get_value(_ALLOW_TRACE_CONTENT_KEY) is not False
        and is_content_tracing_enabled()
    )


def observe_policy(policy: CapturePolicy) -> None:
    """A veto cannot be undone after its context has detached."""
    enabled = runtime_enabled() and not context.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
    current_attributes = getattr(trace.get_current_span(), "attributes", None) or {}
    content = _content_enabled() and all(
        current_attributes.get(key) is not False for key in _CONTENT_ATTRIBUTE_KEYS
    )
    lineage = []
    current = policy
    while current is not None:
        lineage.append(current)
        enabled = (
            enabled
            and current.enabled
            and (
                current.owner is None or bool(getattr(current.owner, "enabled", False))
            )
        )
        attributes = getattr(current.ancestor, "attributes", None) or {}
        content = (
            content
            and current.capture_content
            and all(attributes.get(key) is not False for key in _CONTENT_ATTRIBUTE_KEYS)
        )
        current = current.parent
    for item in lineage:
        item.enabled = bool(item.enabled and enabled)
        item.capture_content = bool(item.capture_content and content)


def capture_policy(capture_content: bool = True, *, owner: Any = None) -> CapturePolicy:
    policy = CapturePolicy(
        capture_content=bool(capture_content),
        enabled=True,
        parent=CURRENT_POLICY.get(),
        owner=owner,
        ancestor=trace.get_current_span(),
    )
    observe_policy(policy)
    return policy
