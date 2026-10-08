"""First-party CrewAI event listener that emits canonical Respan spans."""

from __future__ import annotations

import functools
import json
import logging
import threading
import weakref
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from crewai.events.event_bus import CrewAIEventsBus, crewai_event_bus
from crewai.events.types import flow_events
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.flow_events import (
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from opentelemetry import trace
from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE

from respan_instrumentation_crewai._constants import (
    CREWAI_INSTRUMENTATION_NAME,
    MAX_BUFFERED_ENTRIES,
)
from respan_instrumentation_crewai._event_assembler import (
    CrewAIEventAssembler,
    SpanEndSpec,
    SpanStartSpec,
)
from respan_instrumentation_crewai._policy import (
    CONTENT_POLICY_ATTRIBUTE,
    ENABLED_POLICY_ATTRIBUTE,
    EVENT_CONTENT,
    TOOL_CALL_ID_ATTRIBUTE,
    content_allowed,
)
from respan_instrumentation_crewai._serialization import (
    json_attribute,
    normalize_provider,
    normalize_token_usage,
    normalize_tool_definitions,
    set_llm_message_attributes,
)

logger = logging.getLogger(__name__)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _task_from_event(event: Any) -> Any:
    return getattr(event, "task", None)


def _agent_from_event(event: Any) -> Any:
    return getattr(event, "agent", None)


def _task_id(event: Any) -> str | None:
    task = _task_from_event(event)
    return _text(getattr(task, "id", None) or getattr(event, "task_id", None))


def _agent_id(event: Any) -> str | None:
    agent = _agent_from_event(event)
    agent_info = getattr(event, "agent_info", None) or {}
    return _text(
        getattr(agent, "id", None)
        or getattr(event, "agent_id", None)
        or _value(agent_info, "id")
    )


def _agent_key(event: Any) -> str | None:
    agent = _agent_from_event(event)
    agent_info = getattr(event, "agent_info", None) or {}
    return _text(
        getattr(agent, "key", None)
        or getattr(event, "agent_key", None)
        or _value(agent_info, "key")
    )


def _agent_role(event: Any) -> str | None:
    agent = _agent_from_event(event)
    agent_info = getattr(event, "agent_info", None) or {}
    return _text(
        getattr(agent, "role", None)
        or getattr(event, "agent_role", None)
        or _value(agent_info, "role")
    )


def _task_hint(task_id: str | None) -> str | None:
    return f"task:{task_id}" if task_id else None


def _agent_hint_keys(event: Any) -> tuple[str, ...]:
    values = (
        ("agent", _agent_id(event)),
        ("agent-key", _agent_key(event)),
        ("agent-role", _agent_role(event)),
        ("agent-task", _task_id(event)),
    )
    return tuple(f"{prefix}:{value}" for prefix, value in values if value)


def _parent_hint_keys(event: Any) -> tuple[str, ...]:
    task_hint = _task_hint(_task_id(event))
    return (*_agent_hint_keys(event), *((task_hint,) if task_hint else ()))


def _crew_name(source: Any, event: Any) -> str | None:
    crew = getattr(event, "crew", None)
    if crew is None:
        task = _task_from_event(event)
        agent = _agent_from_event(event) or getattr(task, "agent", None)
        crew = getattr(agent, "crew", None)
    return _text(
        getattr(event, "crew_name", None)
        or getattr(crew, "name", None)
        or getattr(source, "name", None)
    )


def _base_attributes(
    *,
    log_type: str,
    entity_name: str,
    entity_path: str,
    input_value: Any = None,
    workflow_name: str | None = None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: log_type,
        SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: entity_path,
    }
    if input_value is not None:
        attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_attribute(input_value)
    if workflow_name:
        attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    return attributes


def _span_name(value: str, suffix: str) -> str:
    compact = " ".join(value.split())
    return f"{compact[:100]}.{suffix}"


def _crew_start_spec(source: Any, event: CrewKickoffStartedEvent) -> SpanStartSpec:
    crew_name = _crew_name(source, event) or "Crew"
    attributes = _base_attributes(
        log_type=LOG_TYPE_WORKFLOW,
        entity_name=crew_name,
        entity_path="",
        input_value=getattr(event, "inputs", None),
        workflow_name=crew_name,
    )
    return SpanStartSpec(name=_span_name(crew_name, "workflow"), attributes=attributes)


def _task_start_spec(source: Any, event: TaskStartedEvent) -> SpanStartSpec:
    task = _task_from_event(event) or source
    # The SDK fills event.task_name from description for unnamed tasks. Keep
    # that prompt in content fields instead of making it a span identifier.
    task_name = _text(getattr(task, "name", None)) or "Task"
    task_id = _task_id(event) or _text(getattr(task, "id", None))
    input_value = {
        "description": getattr(task, "description", None),
        "expected_output": getattr(task, "expected_output", None),
        "context": getattr(event, "context", None),
    }
    attributes = _base_attributes(
        log_type=LOG_TYPE_TASK,
        entity_name=task_name,
        entity_path=f"task.{task_name}",
        input_value=input_value,
        workflow_name=_crew_name(source, event),
    )
    remember_hint = _task_hint(task_id)
    return SpanStartSpec(
        name=_span_name(task_name, "task"),
        attributes=attributes,
        remember_hint_keys=((remember_hint,) if remember_hint else ()),
    )


def _agent_start_spec(event: AgentExecutionStartedEvent) -> SpanStartSpec:
    agent = _agent_from_event(event)
    role = _agent_role(event) or "Agent"
    task_id = _task_id(event)
    attributes = _base_attributes(
        log_type=LOG_TYPE_AGENT,
        entity_name=role,
        entity_path=f"agent.{role}",
        input_value=getattr(event, "task_prompt", None),
        workflow_name=_crew_name(agent, event),
    )
    task_hint = _task_hint(task_id)
    return SpanStartSpec(
        name=_span_name(role, "agent"),
        attributes=attributes,
        parent_hint_keys=((task_hint,) if task_hint else ()),
        remember_hint_keys=_agent_hint_keys(event),
    )


def _lite_agent_start_spec(event: LiteAgentExecutionStartedEvent) -> SpanStartSpec:
    role = _agent_role(event) or "Agent"
    attributes = _base_attributes(
        log_type=LOG_TYPE_AGENT,
        entity_name=role,
        entity_path=f"agent.{role}",
        input_value=getattr(event, "messages", None),
    )
    return SpanStartSpec(
        name=_span_name(role, "agent"),
        attributes=attributes,
        remember_hint_keys=_agent_hint_keys(event),
    )


def _tool_call_id(source: Any, event: Any) -> str | None:
    explicit = _text(getattr(event, "tool_call_id", None))
    if explicit:
        return explicit
    # CrewAgentExecutor exposes its current assistant message. The native tool
    # event omits the ID, so correlate only an unambiguous exact argument match.
    messages = getattr(source, "messages", None)
    if not isinstance(messages, list) or not messages:
        return None
    message = next(
        (
            item
            for item in reversed(messages)
            if isinstance(item, Mapping) and item.get("role") == "assistant"
        ),
        None,
    )
    if message is None:
        return None
    matches = []
    for call in message.get("tool_calls", ()) or ():
        if not isinstance(call, Mapping):
            continue
        function = call.get("function") or {}
        arguments = function.get("arguments")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (ValueError, TypeError):
            continue
        if (
            function.get("name") == event.tool_name
            and parsed == event.tool_args
            and call.get("id")
        ):
            matches.append(str(call["id"]))
    return matches[0] if len(matches) == 1 else None


def _tool_start_spec(source: Any, event: ToolUsageStartedEvent) -> SpanStartSpec:
    tool_name = _text(getattr(event, "tool_name", None)) or "Tool"
    tool_args = getattr(event, "tool_args", None)
    attributes = _base_attributes(
        log_type=LOG_TYPE_TOOL,
        entity_name=tool_name,
        entity_path=f"tool.{tool_name}",
        input_value={"name": tool_name, "arguments": tool_args},
    )
    call_id = (
        getattr(event, TOOL_CALL_ID_ATTRIBUTE)
        if hasattr(event, TOOL_CALL_ID_ATTRIBUTE)
        else _tool_call_id(source, event)
    )
    if call_id:
        attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ID] = call_id
    return SpanStartSpec(
        name=_span_name(tool_name, "tool"),
        attributes=attributes,
        parent_hint_keys=_parent_hint_keys(event),
    )


def _llm_correlation_key(call_id: Any) -> str | None:
    normalized = _text(call_id)
    return f"llm:{normalized}" if normalized else None


def _llm_start_spec(source: Any, event: LLMCallStartedEvent) -> SpanStartSpec:
    model = _text(getattr(event, "model", None) or getattr(source, "model", None))
    model_name = model or "unknown"
    messages = getattr(event, "messages", None)
    provider = normalize_provider(getattr(source, "provider", None), model)
    attributes = _base_attributes(
        log_type=LOG_TYPE_CHAT,
        entity_name=model_name,
        entity_path=f"llm.{model_name}",
        input_value=messages,
    )
    attributes[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
    if model:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model
    if provider:
        attributes[GenAIAttributes.GEN_AI_SYSTEM] = provider
    set_llm_message_attributes(attributes, messages=messages)

    tools = getattr(event, "tools", None)
    available_functions = getattr(event, "available_functions", None)
    if tools:
        attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_attribute(
            normalize_tool_definitions(tools)
        )
    elif available_functions:
        attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_attribute(
            normalize_tool_definitions(available_functions)
        )

    request_values = (
        (SpanAttributes.LLM_REQUEST_TEMPERATURE, getattr(event, "temperature", None)),
        (SpanAttributes.LLM_REQUEST_TOP_P, getattr(event, "top_p", None)),
        (SpanAttributes.LLM_REQUEST_MAX_TOKENS, getattr(event, "max_tokens", None)),
        (SpanAttributes.LLM_IS_STREAMING, getattr(event, "stream", None)),
    )
    for key, value in request_values:
        if value is not None:
            attributes[key] = value

    call_id = _text(getattr(event, "call_id", None))
    correlation_key = _llm_correlation_key(call_id)
    return SpanStartSpec(
        name=_span_name(model_name, "chat"),
        attributes=attributes,
        parent_hint_keys=_parent_hint_keys(event),
        correlation_keys=((correlation_key,) if correlation_key else ()),
    )


def _usage_attributes(usage: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_token_usage(usage)
    attributes: dict[str, Any] = {}
    prompt_tokens = normalized.get("prompt_tokens")
    completion_tokens = normalized.get("completion_tokens")
    total_tokens = normalized.get("total_tokens")
    cached_tokens = normalized.get("cached_tokens")
    reasoning_tokens = normalized.get("reasoning_tokens")
    cache_creation_tokens = normalized.get("cache_creation_tokens")

    if prompt_tokens is not None:
        attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
        attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
        attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
    if total_tokens is not None:
        attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens
    if cached_tokens is not None:
        attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = cached_tokens
    if reasoning_tokens is not None:
        attributes[SpanAttributes.LLM_USAGE_REASONING_TOKENS] = reasoning_tokens
    if cache_creation_tokens is not None:
        attributes[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] = (
            cache_creation_tokens
        )
    return attributes


class CrewAIEventListener:
    """Own CrewAI event subscriptions and translate them directly to Respan."""

    _token_patch_lock = threading.RLock()
    _token_patch_originals: ClassVar[dict[type[Any], Any]] = {}
    _token_patch_wrappers: ClassVar[dict[type[Any], Any]] = {}
    _token_patch_states: ClassVar[dict[type[Any], dict[str, bool]]] = {}
    _token_patch_listeners: weakref.WeakSet[CrewAIEventListener] = weakref.WeakSet()

    def __init__(self, tracer_provider: trace.TracerProvider | None = None) -> None:
        self._lifecycle_lock = threading.RLock()
        provider = tracer_provider or trace.get_tracer_provider()
        tracer = trace.get_tracer(
            CREWAI_INSTRUMENTATION_NAME,
            tracer_provider=provider,
        )
        self._assembler = CrewAIEventAssembler(tracer)
        self._event_bus: CrewAIEventsBus = crewai_event_bus
        self._handlers: list[tuple[type[Any], Any]] = []
        self._usage_by_call_id: OrderedDict[str, dict[str, int]] = OrderedDict()
        self._usage_lock = threading.RLock()
        self._is_shutdown = False
        self._emit_original = None
        self._emit_wrapper = None
        try:
            self._install_emission_policy()
            self._setup_listeners()
            self._install_token_usage_patch()
        except Exception:
            self.shutdown()
            raise

    def _install_emission_policy(self) -> None:
        original = self._event_bus.emit
        self._emit_original = original

        @functools.wraps(original)
        def emit(source, event, *args, **kwargs):
            if not self._is_shutdown:
                try:
                    object.__setattr__(
                        event, CONTENT_POLICY_ATTRIBUTE, content_allowed()
                    )
                    object.__setattr__(
                        event, ENABLED_POLICY_ATTRIBUTE, is_instrumentation_enabled()
                    )
                    if isinstance(event, ToolUsageStartedEvent):
                        object.__setattr__(
                            event, TOOL_CALL_ID_ATTRIBUTE, _tool_call_id(source, event)
                        )
                except Exception:
                    logger.debug(
                        "Could not capture CrewAI emission policy", exc_info=True
                    )
            return original(source, event, *args, **kwargs)

        self._emit_wrapper = emit
        self._event_bus.emit = emit

    def _register(self, event_type: type[Any], handler: Any) -> None:
        @functools.wraps(handler)
        def safe_handler(source: Any, event: Any) -> None:
            with self._lifecycle_lock:
                if self._is_shutdown:
                    return
                token = EVENT_CONTENT.set(
                    getattr(event, CONTENT_POLICY_ATTRIBUTE, content_allowed())
                )
                try:
                    handler(source, event)
                except Exception:
                    logger.exception(
                        "Failed to translate CrewAI event %s",
                        getattr(event, "type", type(event).__name__),
                    )
                finally:
                    EVENT_CONTENT.reset(token)

        registered = self._event_bus.on(event_type)(safe_handler)
        self._handlers.append((event_type, registered))

    def _setup_listeners(self) -> None:
        registrations = (
            (CrewKickoffStartedEvent, self._on_crew_started),
            (CrewKickoffCompletedEvent, self._on_crew_completed),
            (CrewKickoffFailedEvent, self._on_crew_failed),
            (TaskStartedEvent, self._on_task_started),
            (TaskCompletedEvent, self._on_task_completed),
            (TaskFailedEvent, self._on_task_failed),
            (AgentExecutionStartedEvent, self._on_agent_started),
            (AgentExecutionCompletedEvent, self._on_agent_completed),
            (AgentExecutionErrorEvent, self._on_agent_failed),
            (LiteAgentExecutionStartedEvent, self._on_lite_agent_started),
            (LiteAgentExecutionCompletedEvent, self._on_lite_agent_completed),
            (LiteAgentExecutionErrorEvent, self._on_lite_agent_failed),
            (ToolUsageStartedEvent, self._on_tool_started),
            (ToolUsageFinishedEvent, self._on_tool_completed),
            (ToolUsageErrorEvent, self._on_tool_failed),
            (LLMCallStartedEvent, self._on_llm_started),
            (LLMCallCompletedEvent, self._on_llm_completed),
            (LLMCallFailedEvent, self._on_llm_failed),
            (FlowStartedEvent, self._on_flow_started),
            (FlowFinishedEvent, self._on_flow_completed),
            (MethodExecutionStartedEvent, self._on_method_started),
            (MethodExecutionFinishedEvent, self._on_flow_completed),
            (MethodExecutionFailedEvent, self._on_flow_failed),
        )
        for event_type, handler in registrations:
            self._register(event_type, handler)
        for name, handler in (
            ("FlowFailedEvent", self._on_flow_failed),
            ("FlowPausedEvent", self._on_flow_completed),
            ("MethodExecutionPausedEvent", self._on_flow_completed),
        ):
            event_type = getattr(flow_events, name, None)
            if event_type is not None:
                self._register(event_type, handler)

    @staticmethod
    def _token_usage_patch_targets() -> tuple[type[Any], ...]:
        from crewai.llms.base_llm import BaseLLM

        targets: list[type[Any]] = [BaseLLM]
        try:
            from crewai.llms.providers.bedrock.completion import BedrockCompletion
        except ImportError:
            pass
        else:
            if BedrockCompletion is not BaseLLM:
                targets.append(BedrockCompletion)
        return tuple(targets)

    @classmethod
    def _notify_token_usage(cls, usage_data: Mapping[str, Any]) -> None:
        from crewai.llms import base_llm as base_llm_module

        try:
            call_id = base_llm_module.get_current_call_id()
        except Exception:
            logger.debug(
                "Failed to read CrewAI current LLM call ID",
                exc_info=True,
            )
            return
        with cls._token_patch_lock:
            listeners = list(cls._token_patch_listeners)
        for active_listener in listeners:
            active_listener._record_token_usage(call_id, usage_data)

    @classmethod
    def _install_token_usage_patch_for_listener(
        cls,
        listener: CrewAIEventListener,
    ) -> None:
        with cls._token_patch_lock:
            cls._token_patch_listeners.add(listener)
            for target_class in cls._token_usage_patch_targets():
                if target_class in cls._token_patch_originals:
                    continue
                original = target_class.__dict__.get("_track_token_usage_internal")
                if original is None:
                    continue

                state = {"active": True}
                cls._token_patch_states[target_class] = state

                @functools.wraps(original)
                def patched(
                    instance: Any,
                    usage_data: Mapping[str, Any],
                    *args: Any,
                    __original: Any = original,
                    __state: dict[str, bool] = state,
                    **kwargs: Any,
                ) -> Any:
                    result = __original(instance, usage_data, *args, **kwargs)
                    if __state["active"] and is_instrumentation_enabled():
                        try:
                            cls._notify_token_usage(usage_data)
                        except Exception:
                            logger.debug(
                                "Could not capture CrewAI usage", exc_info=True
                            )
                    return result

                cls._token_patch_originals[target_class] = original
                cls._token_patch_wrappers[target_class] = patched
                target_class._track_token_usage_internal = patched

    def _install_token_usage_patch(self) -> None:
        type(self)._install_token_usage_patch_for_listener(self)

    def _restore_token_usage_tracking(self) -> None:
        cls = type(self)
        with cls._token_patch_lock:
            cls._token_patch_listeners.discard(self)
            if cls._token_patch_listeners:
                return
            for state in cls._token_patch_states.values():
                state["active"] = False
            cls._token_patch_states.clear()
            for target_class, original in list(cls._token_patch_originals.items()):
                wrapper = cls._token_patch_wrappers.get(target_class)
                if (
                    wrapper is not None
                    and getattr(
                        target_class,
                        "_track_token_usage_internal",
                        None,
                    )
                    is wrapper
                ):
                    target_class._track_token_usage_internal = original
            cls._token_patch_originals.clear()
            cls._token_patch_wrappers.clear()

    def _record_token_usage(
        self,
        call_id: str,
        usage_data: Mapping[str, Any],
    ) -> None:
        with self._lifecycle_lock:
            if self._is_shutdown:
                return
            normalized_call_id = _text(call_id)
            if not normalized_call_id:
                return
            usage = normalize_token_usage(usage_data)
            if not usage:
                return
            with self._usage_lock:
                aggregate = self._usage_by_call_id.pop(normalized_call_id, {})
                for key, value in usage.items():
                    aggregate[key] = aggregate.get(key, 0) + value
                self._usage_by_call_id[normalized_call_id] = aggregate
                while len(self._usage_by_call_id) > MAX_BUFFERED_ENTRIES:
                    evicted_call_id, _ = self._usage_by_call_id.popitem(last=False)
                    logger.warning(
                        "Evicting oldest CrewAI LLM usage entry for %s",
                        evicted_call_id,
                    )

    def _consume_token_usage(
        self,
        call_id: Any,
        event_usage: Any = None,
    ) -> dict[str, int]:
        normalized_call_id = _text(call_id)
        with self._usage_lock:
            tracked = (
                self._usage_by_call_id.pop(normalized_call_id, {})
                if normalized_call_id
                else {}
            )
        if isinstance(event_usage, Mapping):
            normalized_event_usage = normalize_token_usage(event_usage)
            if normalized_event_usage:
                return normalized_event_usage
        return tracked

    def _on_crew_started(self, source: Any, event: CrewKickoffStartedEvent) -> None:
        self._assembler.start_span(event, _crew_start_spec(source, event))

    def _on_crew_completed(
        self,
        source: Any,
        event: CrewKickoffCompletedEvent,
    ) -> None:
        self._assembler.end_span(
            event,
            SpanEndSpec(output=getattr(event, "output", None)),
        )

    def _on_crew_failed(self, source: Any, event: CrewKickoffFailedEvent) -> None:
        self._assembler.end_span(
            event,
            SpanEndSpec(error=_text(getattr(event, "error", None)) or "Crew failed"),
        )

    def _on_task_started(self, source: Any, event: TaskStartedEvent) -> None:
        self._assembler.start_span(event, _task_start_spec(source, event))

    def _on_task_completed(self, source: Any, event: TaskCompletedEvent) -> None:
        self._assembler.end_span(
            event,
            SpanEndSpec(output=getattr(event, "output", None)),
        )

    def _on_task_failed(self, source: Any, event: TaskFailedEvent) -> None:
        self._assembler.end_span(
            event,
            SpanEndSpec(error=_text(getattr(event, "error", None)) or "Task failed"),
        )

    def _on_agent_started(
        self,
        source: Any,
        event: AgentExecutionStartedEvent,
    ) -> None:
        self._assembler.start_span(event, _agent_start_spec(event))

    def _on_agent_completed(
        self,
        source: Any,
        event: AgentExecutionCompletedEvent,
    ) -> None:
        self._assembler.end_span(
            event,
            SpanEndSpec(output=getattr(event, "output", None)),
        )

    def _on_agent_failed(
        self,
        source: Any,
        event: AgentExecutionErrorEvent,
    ) -> None:
        self._assembler.end_span(
            event,
            SpanEndSpec(error=_text(getattr(event, "error", None)) or "Agent failed"),
        )

    def _on_lite_agent_started(
        self,
        source: Any,
        event: LiteAgentExecutionStartedEvent,
    ) -> None:
        self._assembler.start_span(event, _lite_agent_start_spec(event))

    def _on_lite_agent_completed(
        self,
        source: Any,
        event: LiteAgentExecutionCompletedEvent,
    ) -> None:
        self._assembler.end_span(
            event,
            SpanEndSpec(output=getattr(event, "output", None)),
        )

    def _on_lite_agent_failed(
        self,
        source: Any,
        event: LiteAgentExecutionErrorEvent,
    ) -> None:
        self._assembler.end_span(
            event,
            SpanEndSpec(
                error=_text(getattr(event, "error", None)) or "LiteAgent failed"
            ),
        )

    def _on_tool_started(self, source: Any, event: ToolUsageStartedEvent) -> None:
        self._assembler.start_span(event, _tool_start_spec(source, event))

    def _on_tool_completed(
        self,
        source: Any,
        event: ToolUsageFinishedEvent,
    ) -> None:
        self._assembler.end_span(
            event,
            SpanEndSpec(
                output=getattr(event, "output", None),
                error=_text(getattr(getattr(event, "failure", None), "message", None)),
            ),
        )

    def _on_tool_failed(self, source: Any, event: ToolUsageErrorEvent) -> None:
        self._assembler.end_span(
            event,
            SpanEndSpec(error=_text(getattr(event, "error", None)) or "Tool failed"),
        )

    def _on_llm_started(self, source: Any, event: LLMCallStartedEvent) -> None:
        self._assembler.start_span(event, _llm_start_spec(source, event))

    def _on_llm_completed(self, source: Any, event: LLMCallCompletedEvent) -> None:
        call_id = getattr(event, "call_id", None)
        response = getattr(event, "response", None)
        attributes: dict[str, Any] = {}
        call_type = getattr(getattr(event, "call_type", None), "value", None)
        tool_call_response = (
            call_type == "tool_call"
            and isinstance(response, Sequence)
            and not isinstance(response, (str, bytes, bytearray))
        )
        set_llm_message_attributes(
            attributes,
            messages=getattr(event, "messages", None),
            response=response,
            tool_call_response=tool_call_response,
        )
        attributes.update(
            _usage_attributes(
                self._consume_token_usage(call_id, getattr(event, "usage", None))
            )
        )
        response_id = _text(getattr(event, "response_id", None))
        if response_id:
            attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] = response_id
        finish_reason = _text(getattr(event, "finish_reason", None))
        if finish_reason:
            attributes[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = finish_reason
        correlation_key = _llm_correlation_key(call_id)
        self._assembler.end_span(
            event,
            SpanEndSpec(output=response, attributes=attributes),
            correlation_keys=((correlation_key,) if correlation_key else ()),
        )

    def _on_llm_failed(self, source: Any, event: LLMCallFailedEvent) -> None:
        call_id = getattr(event, "call_id", None)
        self._consume_token_usage(call_id)
        correlation_key = _llm_correlation_key(call_id)
        self._assembler.end_span(
            event,
            SpanEndSpec(error=_text(getattr(event, "error", None)) or "LLM failed"),
            correlation_keys=((correlation_key,) if correlation_key else ()),
        )

    @staticmethod
    def _is_user_flow(source: Any) -> bool:
        return not type(source).__module__.startswith(("crewai.", "crewai_core."))

    def _on_flow_started(self, source: Any, event: Any) -> None:
        if not self._is_user_flow(source):
            self._assembler.open_scope(event)
            return
        name = _text(event.flow_name) or "Flow"
        self._assembler.start_span(
            event,
            SpanStartSpec(
                name=_span_name(name, "workflow"),
                attributes=_base_attributes(
                    log_type=LOG_TYPE_WORKFLOW,
                    entity_name=name,
                    entity_path="",
                    input_value=getattr(event, "inputs", None),
                    workflow_name=name,
                ),
            ),
        )

    def _on_method_started(self, source: Any, event: Any) -> None:
        if not self._is_user_flow(source):
            self._assembler.open_scope(event)
            return
        name = _text(event.method_name) or "method"
        self._assembler.start_span(
            event,
            SpanStartSpec(
                name=_span_name(name, "task"),
                attributes=_base_attributes(
                    log_type=LOG_TYPE_TASK,
                    entity_name=name,
                    entity_path=f"flow.{name}",
                    input_value={
                        "params": getattr(event, "params", None),
                        "state": getattr(event, "state", None),
                    },
                ),
            ),
        )

    def _on_flow_completed(self, source: Any, event: Any) -> None:
        if not self._is_user_flow(source):
            self._assembler.close_scope(event)
            return
        self._assembler.end_span(
            event, SpanEndSpec(output=getattr(event, "result", None))
        )

    def _on_flow_failed(self, source: Any, event: Any) -> None:
        if not self._is_user_flow(source):
            self._assembler.close_scope(event)
            return
        self._assembler.end_span(
            event,
            SpanEndSpec(error=_text(getattr(event, "error", None)) or "Flow failed"),
        )

    def shutdown(self) -> None:
        """Unregister handlers, restore CrewAI, and close unfinished spans."""
        with self._lifecycle_lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True
            if (
                self._emit_wrapper is not None
                and self._event_bus.emit is self._emit_wrapper
            ):
                self._event_bus.emit = self._emit_original
            for event_type, handler in self._handlers:
                try:
                    self._event_bus.off(event_type, handler)
                except Exception:
                    logger.debug(
                        "Failed to unregister CrewAI handler for %s",
                        event_type,
                        exc_info=True,
                    )
            self._handlers.clear()
            self._restore_token_usage_tracking()
            with self._usage_lock:
                self._usage_by_call_id.clear()
            self._assembler.shutdown()
