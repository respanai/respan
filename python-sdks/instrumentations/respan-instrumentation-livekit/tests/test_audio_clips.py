"""Opt-in per-turn audio clips (``capture_audio=True``), off by default.

The two LiveKit hook points (``_tts_inference_task``, ``RecognizeStream.push_frame``)
are replaced by stand-ins before activation, so the instrumentor's own wrappers
are what the call goes through; spans are made by the real SDK tracer in the
shape LiveKit traces a call (see ``test_voice_turns``).
"""

import asyncio
import base64
import io
import json
import logging
import wave
from types import SimpleNamespace

import pytest
from livekit.agents import stt
from livekit.agents.telemetry import trace_types
from livekit.agents.voice import generation
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_livekit import LiveKitInstrumentor
from respan_instrumentation_livekit import _audio
from respan_instrumentation_livekit import _instrumentation as lifecycle

RATE = 16000
LOG_TYPE = "respan.entity.log_type"
INPUT = "traceloop.entity.input"
OUTPUT = "traceloop.entity.output"
TRUNCATED = "respan.metadata.voice.audio_truncated"


def _frame(seconds, value=1):
    samples = int(RATE * seconds)
    return SimpleNamespace(
        data=memoryview(value.to_bytes(2, "little", signed=True) * samples),
        sample_rate=RATE,
        num_channels=1,
        samples_per_channel=samples,
    )


@pytest.fixture
def hooks(monkeypatch):
    """Stand-ins for LiveKit's hook points, installed before activation."""

    async def tts_inference(node, *args, **kwargs):
        frames = await node(*args, **kwargs)
        return [frame async for frame in frames]

    monkeypatch.setattr(generation, "_tts_inference_task", tts_inference)
    monkeypatch.setattr(stt.RecognizeStream, "push_frame", lambda stream, frame: None)


def _activate(monkeypatch, **options):
    assert lifecycle._RUNTIME is None
    provider = TracerProvider()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    owner = LiveKitInstrumentor(**options)
    owner.activate()
    return provider, exporter, owner


@pytest.fixture
def recorded(monkeypatch, hooks):
    provider, exporter, owner = _activate(monkeypatch, capture_audio=True)
    try:
        yield provider, exporter
    finally:
        owner.deactivate()
        provider.shutdown()


def _speak(tracer, seconds):
    """The agent's reply: frames out of the (wrapped) TTS node."""

    async def node(text, settings):
        async def frames():
            for _ in range(int(seconds / 0.5)):
                yield _frame(0.5)

        return frames()

    with tracer.start_as_current_span("tts_node"):
        asyncio.run(generation._tts_inference_task(node, None, None))


def _call(provider, *, user_seconds=2.0, agent_seconds=1.0):
    tracer = provider.get_tracer("livekit-agents")
    with tracer.start_as_current_span("agent_session") as session:
        # Audio before the turn opens: the pre-roll keeps its last 0.5 s.
        for _ in range(4):
            stt.RecognizeStream.push_frame(None, _frame(0.25, value=7))
        with tracer.start_as_current_span("user_turn") as user_turn:
            for _ in range(int(user_seconds / 0.5)):
                stt.RecognizeStream.push_frame(None, _frame(0.5))
            user_turn.set_attribute(trace_types.ATTR_USER_TRANSCRIPT, "move my visit")
        with tracer.start_as_current_span("agent_turn"):
            _speak(tracer, agent_seconds)
    return session.get_span_context().trace_id


def _by_name(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _wav(data_url):
    prefix = "data:audio/wav;base64,"
    assert data_url.startswith(prefix)
    with wave.open(io.BytesIO(base64.b64decode(data_url[len(prefix) :]))) as wav:
        return wav.getframerate(), wav.getnframes() / wav.getframerate()


def test_off_by_default_nothing_is_hooked_or_recorded(monkeypatch, hooks):
    stand_in = generation._tts_inference_task
    provider, exporter, owner = _activate(monkeypatch)
    try:
        assert generation._tts_inference_task is stand_in
        assert lifecycle._RUNTIME.processor.audio is None
        _call(provider)
        for span in exporter.get_finished_spans():
            assert span.name != "user_speech"
            assert span.attributes.get(LOG_TYPE) not in ("speech", "transcription")
    finally:
        owner.deactivate()
        provider.shutdown()


def test_each_turn_carries_the_users_and_the_agents_clip(recorded):
    provider, exporter = recorded
    call = _call(provider, user_seconds=2.0, agent_seconds=1.0)

    (tts,) = _by_name(exporter, "tts_node")
    assert tts.attributes[LOG_TYPE] == "speech"
    assert _wav(json.loads(tts.attributes[OUTPUT])["audio_url"]) == (RATE, 1.0)

    (turn,) = _by_name(exporter, "agent_turn")
    (speech,) = _by_name(exporter, "user_speech")
    # In the turn's own trace, under the turn, not in the call's trace.
    assert speech.context.trace_id == turn.context.trace_id != call
    assert speech.parent.span_id == turn.context.span_id
    assert speech.attributes[LOG_TYPE] == "transcription"
    assert speech.attributes[OUTPUT] == "move my visit"
    # The user's 2 s, plus 0.5 s of pre-roll from before the turn opened.
    assert _wav(json.loads(speech.attributes[INPUT])["audio_url"]) == (RATE, 2.5)
    assert TRUNCATED not in speech.attributes


def test_audio_needs_content_capture(monkeypatch, hooks):
    stand_in = generation._tts_inference_task
    provider, exporter, owner = _activate(
        monkeypatch, capture_content=False, capture_audio=True
    )
    try:
        assert generation._tts_inference_task is stand_in
        _call(provider)
        assert not _by_name(exporter, "user_speech")
        (tts,) = _by_name(exporter, "tts_node")
        assert OUTPUT not in tts.attributes
    finally:
        owner.deactivate()
        provider.shutdown()


def test_a_reply_that_starts_before_the_user_turn_ends_gets_that_turns_clip(
    recorded,
):
    # LiveKit ends ``user_turn`` after ``on_user_turn_completed``, by which
    # time the reply's ``agent_turn`` has started.
    provider, exporter = recorded
    tracer = provider.get_tracer("livekit-agents")
    with tracer.start_as_current_span("agent_session"):
        user_turn = tracer.start_span("user_turn")
        with trace.use_span(user_turn):
            stt.RecognizeStream.push_frame(None, _frame(1.0))
        user_turn.set_attribute(trace_types.ATTR_USER_TRANSCRIPT, "first")
        with tracer.start_as_current_span("agent_turn") as first_reply:
            # Still open: what the user says now is not part of this turn.
            with trace.use_span(user_turn):
                stt.RecognizeStream.push_frame(None, _frame(1.0))
            user_turn.end()
            _speak(tracer, 0.5)
        with tracer.start_as_current_span("agent_turn") as second_reply:
            _speak(tracer, 0.5)

    (speech,) = _by_name(exporter, "user_speech")
    assert speech.parent.span_id == first_reply.get_span_context().span_id
    assert speech.attributes[OUTPUT] == "first"
    assert _wav(json.loads(speech.attributes[INPUT])["audio_url"]) == (RATE, 1.0)
    assert second_reply.get_span_context().span_id != speech.parent.span_id


def test_a_clip_stops_at_the_cap_and_says_so(recorded, monkeypatch):
    monkeypatch.setattr(_audio, "MAX_CLIP_SECONDS", 2)
    provider, exporter = recorded
    _call(provider, user_seconds=1.0, agent_seconds=3.0)

    (tts,) = _by_name(exporter, "tts_node")
    assert tts.attributes[TRUNCATED] is True
    assert _wav(json.loads(tts.attributes[OUTPUT])["audio_url"]) == (RATE, 2.0)


def test_every_buffer_is_released_when_the_call_ends(recorded):
    provider, _ = recorded
    tracer = provider.get_tracer("livekit-agents")
    with tracer.start_as_current_span("agent_session"):
        with tracer.start_as_current_span("user_turn"):
            stt.RecognizeStream.push_frame(None, _frame(0.5))
        # The user talks over the reply: the turn ends, its clip goes with it.
        with tracer.start_as_current_span("agent_turn") as turn:
            _speak(tracer, 1.0)
            turn.set_attribute("lk.interrupted", True)
        with tracer.start_as_current_span("user_turn"):
            stt.RecognizeStream.push_frame(None, _frame(0.5))
    audio = lifecycle._RUNTIME.processor.audio
    assert not audio.speech
    assert not audio.calls


def test_a_missing_hook_point_turns_audio_off_and_the_agent_runs(
    monkeypatch, hooks, caplog
):
    monkeypatch.delattr(generation, "_tts_inference_task")
    with caplog.at_level(logging.DEBUG, logger=lifecycle.__name__):
        provider, exporter, owner = _activate(monkeypatch, capture_audio=True)
    try:
        assert "capture_audio is off" in caplog.text
        tracer = provider.get_tracer("livekit-agents")
        with tracer.start_as_current_span("agent_session"):
            with tracer.start_as_current_span("agent_turn"):
                pass
        assert _by_name(exporter, "agent_turn")
    finally:
        owner.deactivate()
        provider.shutdown()
