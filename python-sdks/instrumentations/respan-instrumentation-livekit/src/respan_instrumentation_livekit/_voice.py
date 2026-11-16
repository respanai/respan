"""A LiveKit call as a Respan session: one trace per turn, timings on its root.

LiveKit traces a whole call as one trace (``agent_session`` at the root, one
``agent_turn`` per reply under it). Respan reads a session as its traces, one
per turn, so each ``agent_turn`` subtree is moved to a trace of its own and
every span of the call carries the call's original trace id as its session id
(unless the user set one). The new trace id is derived from the call's trace
id and the turn's span id, so every span of a turn agrees on it without
waiting for the others, whatever order they end in.

The turn's timings, which LiveKit records on ``agent_turn`` (wait, transcription
delay, interruption) and on its ``llm_node`` / ``tts_node`` children (time to
first token, time to first audio), are written to the turn root's
``respan.metadata.voice.*`` in milliseconds, before the ``lk.*`` keys are dropped.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from opentelemetry.trace import SpanContext
from respan_sdk.constants.span_attributes import RESPAN_METADATA, RespanSpanAttributes

AGENT_TURN_SPAN = "agent_turn"
VOICE_METADATA_PREFIX = f"{RESPAN_METADATA}.voice."
SESSION_ID_ATTRIBUTE = RespanSpanAttributes.RESPAN_SESSION_ID.value

# LiveKit attribute (seconds) -> Respan voice metadata key (milliseconds).
# On the agent_turn span itself:
TURN_TIMINGS = {
    "lk.e2e_latency": "wait_ms",
    "lk.transcription_delay": "stt_ms",
}
# On a child of the turn; the first one seen wins (a turn's first LLM call and
# first speech are what the user waited for):
CHILD_TIMINGS = {
    "lk.response.ttft": "llm_ttft_ms",
    "lk.response.ttfb": "tts_ttfa_ms",
}
INTERRUPTED_ATTRIBUTE = "lk.interrupted"
# Not content: kept on a private span until its turn ends.
VOICE_SOURCE_ATTRIBUTES = (*TURN_TIMINGS, *CHILD_TIMINGS, INTERRUPTED_ATTRIBUTE)
_MAX_CALLS = 4096


def _seconds_to_ms(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value < 0 or value == float("inf"):
        return None
    return round(value * 1000, 1)


def turn_trace_id(call_trace_id: int, turn_span_id: int) -> int:
    digest = hashlib.sha256(
        f"{call_trace_id:032x}{turn_span_id:016x}".encode()
    ).digest()
    return int.from_bytes(digest[:16], "big") or 1


def _moved(context: SpanContext, trace_id: int) -> SpanContext:
    return SpanContext(
        trace_id=trace_id,
        span_id=context.span_id,
        is_remote=context.is_remote,
        trace_flags=context.trace_flags,
        trace_state=context.trace_state,
    )


class VoiceTurns:
    """Per-span bookkeeping from start to end; every entry is dropped when its
    span ends, and turn counters are bounded per call."""

    def __init__(self):
        self.lock = threading.Lock()
        # span key -> its turn's key ((trace id, span id) of the agent_turn)
        self.turn_of = {}
        # turn key -> {voice key: value}, filled by the turn's children
        self.timings = {}
        # call trace id -> turns started so far
        self.turn_counts = OrderedDict()

    def start(self, span, span_key, parent_key):
        if span_key is None:
            return
        with self.lock:
            if getattr(span, "name", None) == AGENT_TURN_SPAN:
                self.turn_of[span_key] = span_key
                call = span_key[0]
                index = self.turn_counts.pop(call, 0)
                self.turn_counts[call] = index + 1
                while len(self.turn_counts) > _MAX_CALLS:
                    self.turn_counts.popitem(last=False)
                self.timings[span_key] = {"turn_index": index}
            elif parent_key in self.turn_of:
                self.turn_of[span_key] = self.turn_of[parent_key]

    def end(self, span, span_key, attrs):
        """Move the span into its turn's trace, stamp the call's session id,
        and on a turn root write its timings. Returns the attributes to set."""
        if span_key is None:
            return attrs
        with self.lock:
            turn = self.turn_of.pop(span_key, None)
            timings = self.timings.get(turn) if turn is not None else None
            if timings is not None and turn != span_key:
                for lk_key, voice_key in CHILD_TIMINGS.items():
                    if voice_key not in timings:
                        reading = _seconds_to_ms(attrs.get(lk_key))
                        if reading is not None:
                            timings[voice_key] = reading
            if turn == span_key:
                timings = self.timings.pop(span_key, {})
        attrs = dict(attrs)
        call_trace_id = span_key[0]
        if not attrs.get(SESSION_ID_ATTRIBUTE):
            attrs[SESSION_ID_ATTRIBUTE] = f"{call_trace_id:032x}"
        if turn is None:
            return attrs
        new_trace_id = turn_trace_id(call_trace_id, turn[1])
        span._context = _moved(span.get_span_context(), new_trace_id)
        if turn == span_key:
            # The turn's root: a trace of its own.
            span._parent = None
            for lk_key, voice_key in TURN_TIMINGS.items():
                reading = _seconds_to_ms(attrs.get(lk_key))
                if reading is not None:
                    timings[voice_key] = reading
            if INTERRUPTED_ATTRIBUTE in attrs:
                timings["interrupted"] = bool(attrs[INTERRUPTED_ATTRIBUTE])
            for voice_key, value in timings.items():
                attrs[VOICE_METADATA_PREFIX + voice_key] = value
        elif getattr(span, "_parent", None) is not None:
            span._parent = _moved(span._parent, new_trace_id)
        return attrs
