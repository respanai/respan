"""Which native LiveKit spans a Respan trace shows, and what the kept ones say.

LiveKit traces its own plumbing: the stages of a user's turn (``user_turn``,
``user_speaking``, ``eou_wait``), wrappers around each model call
(``llm_node``, ``llm_request_run``, ``tts_stream_adapter``, ``tts_request``,
``tts_request_run``), playout (``agent_speaking``) and the session's lifecycle
(``agent_session``, ``session_start``, ``on_enter`` ...). They carry timings and
ids, no input or output. A turn's timings are read from them into the turn's
``respan.metadata.voice.*`` before export (``_voice``), so nothing a user reads
is lost by leaving them out.

Each is marked at start with ``RESPAN_INTERNAL_DROP_SPAN``, and every span that
starts under one is stamped with ``RESPAN_INTERNAL_EXPORT_PARENT``, the nearest
ancestor that is kept ("" for none). The exporter drops the marked spans and
re-parents the rest, so a user's own span under ``on_enter`` or
``on_user_turn_completed`` moves up instead of vanishing. A marked span that
ends in error is kept (its children already point past it).

The spoken text is recorded on ``tts_node``'s children (``tts_stream_adapter``
holds the whole reply, ``tts_request`` one sentence each), so it is collected
here and set as ``tts_node``'s input.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from livekit.agents.telemetry import trace_types
from respan_sdk.constants.span_attributes import (
    RESPAN_INTERNAL_DROP_SPAN,
    RESPAN_INTERNAL_EXPORT_PARENT,
)

from ._audio import TTS_NODE_SPAN

LIVEKIT_SCOPE = "livekit-agents"
TTS_STREAM_ADAPTER_SPAN = "tts_stream_adapter"
TTS_REQUEST_SPAN = "tts_request"
PLUMBING_SPANS = frozenset({
    "agent_session",
    "session_start",
    "session_close",
    "setup_toolsets",
    "start_agent_activity",
    "drain_agent_activity",
    "on_enter",
    "on_exit",
    "user_turn",
    "user_speaking",
    "eou_wait",
    "on_user_turn_completed",
    "llm_node",
    "llm_request_run",
    TTS_STREAM_ADAPTER_SPAN,
    TTS_REQUEST_SPAN,
    "tts_request_run",
    "agent_speaking",
})
_MAX_SPANS = 8192


def is_plumbing(span):
    scope = getattr(getattr(span, "instrumentation_scope", None), "name", None)
    return scope == LIVEKIT_SCOPE and span.name in PLUMBING_SPANS


class _Spoken:
    def __init__(self):
        self.full = None
        self.sentences = []


class SpanStructure:
    def __init__(self):
        self.lock = threading.Lock()
        # dropped span key -> the span id its children are exported under
        self.stand_in = OrderedDict()
        # span key -> the key of the tts_node it runs under
        self.tts_of = OrderedDict()
        # tts_node key -> what it was given to speak
        self.spoken = OrderedDict()

    @staticmethod
    def _bound(table):
        while len(table) > _MAX_SPANS:
            table.popitem(last=False)

    def start(self, span, span_key, parent_key):
        if span_key is None:
            return
        with self.lock:
            stand_in = self.stand_in.get(parent_key)
            if stand_in is not None:
                span.set_attribute(RESPAN_INTERNAL_EXPORT_PARENT, stand_in)
            if is_plumbing(span):
                span.set_attribute(RESPAN_INTERNAL_DROP_SPAN, True)
                if stand_in is None:
                    stand_in = f"{parent_key[1]:016x}" if parent_key else ""
                self.stand_in[span_key] = stand_in
                self._bound(self.stand_in)
            if span.name == TTS_NODE_SPAN:
                self.tts_of[span_key] = span_key
                self.spoken[span_key] = _Spoken()
                self._bound(self.spoken)
            elif parent_key in self.tts_of:
                self.tts_of[span_key] = self.tts_of[parent_key]
            self._bound(self.tts_of)

    def end(self, span, span_key, attrs):
        """Note the text a TTS child was given. For a ``tts_node``, return
        what it spoke (None if nothing was recorded)."""
        with self.lock:
            node = self.tts_of.pop(span_key, None)
            if node is None:
                return None
            if node == span_key:
                spoken = self.spoken.pop(node, None)
                if spoken is None:
                    return None
                return spoken.full or " ".join(spoken.sentences) or None
            text = attrs.get(trace_types.ATTR_TTS_INPUT_TEXT)
            spoken = self.spoken.get(node)
            if spoken is None or not isinstance(text, str) or not text:
                return None
            if span.name == TTS_STREAM_ADAPTER_SPAN:
                spoken.full = text
            elif span.name == TTS_REQUEST_SPAN:
                spoken.sentences.append(text)
            return None
