"""Normalize owned native LiveKit spans before downstream exporters."""

from __future__ import annotations

from livekit.agents.telemetry import trace_types
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv_ai import SpanAttributes as AI
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE

from ._audio import (
    AGENT_SESSION_SPAN,
    SPEECH_LOG_TYPE,
    TRANSCRIPTION_LOG_TYPE,
    TRUNCATED_VOICE_KEY,
    TTS_NODE_SPAN,
    USER_SPEECH_SPAN,
    USER_TURN_SPAN,
    AudioClips,
    payload,
)
from ._policy import CapturePolicy, key, private_attributes
from ._serialization import text
from ._voice import AGENT_TURN_SPAN, VOICE_METADATA_PREFIX, VoiceTurns
from ._translator import (
    build_livekit_llm_attrs,
    build_native_tool_attrs,
    is_livekit_llm_span,
    usage_attributes,
)


# Where a clip goes on the span that carries it.
_CLIP_TARGETS = {
    TTS_NODE_SPAN: (SPEECH_LOG_TYPE, AI.TRACELOOP_ENTITY_OUTPUT),
    USER_SPEECH_SPAN: (TRANSCRIPTION_LOG_TYPE, AI.TRACELOOP_ENTITY_INPUT),
}


class LiveKitSpanProcessor(SpanProcessor):
    def __init__(self, capture_content=True, capture_audio=False):
        self.policy = CapturePolicy(capture_content)
        self.active = True
        self.on_drained = None
        self.source_usage = {}
        self.voice = VoiceTurns()
        # Only with capture_audio: frame copies and clips (see ``_audio``).
        self.audio = AudioClips() if capture_audio else None
        # The runtime's own tracer, for the ``user_speech`` spans.
        self.tracer = None

    def on_start(self, span, parent_context=None):
        if self.active:
            try:
                self.voice.start(
                    span, key(span), key(trace.get_current_span(parent_context))
                )
            except Exception:  # noqa: BLE001 - turn grouping cannot replace native span creation.
                pass
            try:
                self.policy.start(span, parent_context)
            except Exception:  # noqa: BLE001 - privacy observation cannot replace native span creation.
                self.policy.open[key(span)] = [
                    span,
                    False,
                    key(trace.get_current_span(parent_context)),
                ]
            if self.audio is not None:
                try:
                    self._start_audio(span)
                except Exception:  # noqa: BLE001 - audio capture never replaces native span creation.
                    pass

    def _start_audio(self, span):
        span_key = key(span)
        if span_key is None:
            return
        if span.name == USER_TURN_SPAN:
            self.audio.user_turn_started(span_key[0])
            return
        if span.name != AGENT_TURN_SPAN or self.tracer is None:
            return
        clip, transcript, is_turn_open = self.audio.take_user_speech(span_key[0])
        if clip is None or not clip.samples or not self.policy.observe(span):
            return
        speech = self.tracer.start_span(
            USER_SPEECH_SPAN, context=trace.set_span_in_context(span)
        )
        self.audio.attach(key(speech), clip)
        if is_turn_open:
            # Ended, with the transcript, when the user_turn ends.
            self.audio.wait_for_turn(span_key[0], speech)
            return
        if transcript:
            speech.set_attribute(AI.TRACELOOP_ENTITY_OUTPUT, transcript)
        speech.end()

    def on_end(self, span):
        if key(span) not in self.policy.open:
            return
        source_usage = self.source_usage.pop(key(span), {})
        try:
            capture = self.policy.end(span)
        except Exception:  # noqa: BLE001 - callbacks cannot replace native values/errors.
            state = self.policy.open.pop(key(span), [None, False, None])
            state[0] = None
            state[1] = False
            self.policy.closed[key(span)] = state
            capture = False
        # Before ``voice.end`` moves a turn's span into the turn's trace.
        span_key = key(span)
        try:
            grouped = self.voice.end(span, span_key, dict(span.attributes or {}))
        except Exception:  # noqa: BLE001 - turn grouping observes without replacing native values.
            grouped = None
        clip = None
        if self.audio is not None:
            try:
                clip = self._end_audio(span, span_key, capture)
            except Exception:  # noqa: BLE001 - audio capture observes without replacing native values.
                clip = None
        if not self.policy.owned(span):
            if grouped is not None:
                span._attributes = grouped
            return
        try:
            original = dict(span.attributes or {})
            attrs = dict(grouped if grouped is not None else original)
            if is_livekit_llm_span(span.name, attrs):
                translated = build_livekit_llm_attrs(
                    span_name=span.name,
                    attrs=attrs,
                    events=span.events,
                    capture=capture,
                    source_usage=source_usage,
                )
                for k in list(attrs):
                    if k.startswith(("gen_ai.usage.", "llm.usage.")):
                        attrs.pop(k, None)
                attrs.update(translated)
            elif (
                span.name == "function_tool"
                and getattr(span.instrumentation_scope, "name", None)
                == "livekit-agents"
            ):
                attrs.update(build_native_tool_attrs(original, capture=capture))
                if original.get(trace_types.ATTR_FUNCTION_TOOL_IS_ERROR) is True:
                    span._status = Status(StatusCode.ERROR)
            else:
                attrs.setdefault(RESPAN_LOG_TYPE, "task")
                attrs.setdefault(
                    RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value
                )
                attrs.setdefault(AI.TRACELOOP_ENTITY_NAME, text(span.name))
                attrs.setdefault(AI.TRACELOOP_ENTITY_PATH, "")
            for k in list(attrs):
                if k.startswith(("lk.", "langfuse.")) or k in {
                    "status_code",
                    ERROR_MESSAGE,
                    "tools",
                    "tool_calls",
                    "model",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_request_tokens",
                }:
                    attrs.pop(k, None)
                elif isinstance(attrs[k], str):
                    attrs[k] = text(attrs[k])
                if attrs.get(RESPAN_LOG_TYPE) == "task" and k.startswith(
                    ("gen_ai.", "llm.")
                ):
                    attrs.pop(k, None)
                if (
                    attrs.get(RESPAN_LOG_TYPE) == "tool"
                    and k.startswith("gen_ai.tool.")
                    and k != GenAI.GEN_AI_TOOL_CALL_ID
                ):
                    attrs.pop(k, None)
            if span.status.status_code is StatusCode.ERROR:
                attrs.pop(AI.TRACELOOP_ENTITY_OUTPUT, None)
                for k in list(attrs):
                    if k.startswith(AI.LLM_COMPLETIONS + "."):
                        attrs.pop(k, None)
                span._status = Status(
                    StatusCode.ERROR, text(span.status.description) if capture else None
                )
            if not capture:
                attrs = private_attributes(attrs)
            elif clip is not None and span.name in _CLIP_TARGETS:
                # After the string normalization above, which rewrites URLs.
                encoded = payload(clip)
                if encoded is not None:
                    log_type, field = _CLIP_TARGETS[span.name]
                    attrs[RESPAN_LOG_TYPE] = log_type
                    attrs[field] = encoded
                    if clip.is_truncated:
                        attrs[VOICE_METADATA_PREFIX + TRUNCATED_VOICE_KEY] = True
            span._attributes = attrs
            span._events = ()
        except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
            fallback = private_attributes(dict(span.attributes or {}))
            for name in list(fallback):
                if name.startswith(("gen_ai.usage.", "llm.usage.")):
                    fallback.pop(name, None)
            fallback.update(usage_attributes(source_usage))
            span._attributes = fallback
            span._events = ()
            if span.status.status_code is StatusCode.ERROR:
                span._status = Status(StatusCode.ERROR)
        if self.on_drained is not None:
            self.on_drained()

    def _end_audio(self, span, span_key, capture):
        """Release the span's audio state; the clip to set on it, if any.
        ``span_key`` is the span's key as it started."""
        if span_key is None:
            return None
        if span.name == USER_TURN_SPAN:
            transcript = (span.attributes or {}).get(trace_types.ATTR_USER_TRANSCRIPT)
            if not (capture and isinstance(transcript, str)):
                transcript = None
            self.audio.user_turn_ended(span_key[0], transcript)
            waiting = self.audio.take_waiting(span_key[0])
            if waiting is not None:
                if transcript:
                    waiting.set_attribute(AI.TRACELOOP_ENTITY_OUTPUT, transcript)
                waiting.end()
        elif span.name == AGENT_SESSION_SPAN:
            waiting = self.audio.end_call(span_key[0])
            if waiting is not None:
                waiting.end()
        return self.audio.take_speech(span_key)

    def shutdown(self):
        self.policy.open.clear()
        self.policy.closed.clear()
        self.source_usage.clear()

    def force_flush(self, timeout_millis=30000):
        return True
