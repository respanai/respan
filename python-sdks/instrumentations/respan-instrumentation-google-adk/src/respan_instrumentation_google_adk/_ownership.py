"""Restore only owned ADK bindings, including upstream partial activations."""

import importlib
import inspect

from wrapt import FunctionWrapper

_TARGETS = {
    "google.adk.runners": ("tracer", "Runner.run_async"),
    "google.adk.agents.base_agent": ("tracer", "BaseAgent.run_async"),
    "google.adk.flows.llm_flows.base_llm_flow": (
        "tracer",
        "trace_call_llm",
        "BaseLlmFlow._call_llm_async",
    ),
    "google.adk.flows.llm_flows.functions": ("tracer", "trace_tool_call"),
    "google.adk.flows.llm_flows.core._model_call": ("tracer", "trace_call_llm"),
    "google.adk.flows.llm_flows.tools._batch_executor": ("tracer",),
    "google.adk.telemetry.tracing": (
        "tracer",
        "trace_tool_call",
        "_build_compaction_attributes",
        "_build_compaction_result_attributes",
    ),
    "google.adk.apps.compaction": (
        "tracer",
        "_build_compaction_attributes",
        "_build_compaction_result_attributes",
    ),
    "google.adk.workflow._base_node": ("BaseNode.run",),
    "google.adk.telemetry.node_tracing": ("tracer",),
    "google.adk.tools.model_consult._advisor": ("_record_telemetry", "call_advisor"),
    "google.adk.tools.model_consult._model_consult_tool": ("call_advisor",),
}


class PatchTransaction:
    def __init__(self):
        self.bindings = []
        self.active = True
        for module_name, paths in _TARGETS.items():
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            for path in paths:
                owner = module
                parts = path.split(".")
                for part in parts[:-1]:
                    owner = getattr(owner, part, None)
                if owner is not None and hasattr(owner, parts[-1]):
                    value = inspect.getattr_static(owner, parts[-1])
                    self.bindings.append([owner, parts[-1], value, value])

    def guard(self):
        for record in self.bindings:
            owner, name, original, _ = record
            current = inspect.getattr_static(owner, name)
            if current is not original and callable(current):

                def wrap(original):
                    def wrapper(wrapped, instance, args, kwargs):
                        if self.active:
                            return wrapped(*args, **kwargs)
                        bound = (
                            original.__get__(instance, type(instance))
                            if instance is not None and hasattr(original, "__get__")
                            else original
                        )
                        return bound(*args, **kwargs)

                    return wrapper

                current = FunctionWrapper(current, wrap(original))
                setattr(owner, name, current)
            record[3] = current

    def restore(self, uninstrument=None, *, partial=False):
        self.active = False
        foreign = []
        for owner, name, original, owned in self.bindings:
            current = inspect.getattr_static(owner, name)
            foreign.append(
                (owner, name, original if partial or current is owned else current)
            )
        try:
            if uninstrument is not None:
                uninstrument()
        finally:
            for owner, name, value in reversed(foreign):
                setattr(owner, name, value)
            self.bindings.clear()
