"""A LiveKit call as a Respan session: one trace per turn, timings on its root.

Spans are made by the real SDK tracer under LiveKit's instrumentation scope, in
the shape LiveKit Agents traces a call: ``agent_session`` at the root, a
``user_turn`` and one ``agent_turn`` per reply under it, ``llm_node`` and
``tts_node`` under each turn. Children end before their turn, as in LiveKit.
"""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_livekit import LiveKitInstrumentor
from respan_instrumentation_livekit import _instrumentation as lifecycle

SESSION = "respan.sessions.session_identifier"
VOICE = "respan.metadata.voice."


@pytest.fixture(params=[True, False], ids=["content", "private"])
def native(monkeypatch, request):
    assert lifecycle._RUNTIME is None
    provider = TracerProvider()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    owner = LiveKitInstrumentor(capture_content=request.param)
    owner.activate()
    try:
        yield provider, exporter
    finally:
        owner.deactivate()
        provider.shutdown()


def _turn(tracer, *, e2e, stt, ttft, ttfa, interrupted):
    with tracer.start_as_current_span("agent_turn") as turn:
        with tracer.start_as_current_span("llm_node") as node:
            node.set_attribute("lk.response.ttft", ttft)
        with tracer.start_as_current_span("llm_node") as later:
            # A follow-up LLM step after a tool call: not what the user waited on.
            later.set_attribute("lk.response.ttft", 9.0)
        with tracer.start_as_current_span("tts_node") as node:
            node.set_attribute("lk.response.ttfb", ttfa)
        turn.set_attribute("lk.e2e_latency", e2e)
        turn.set_attribute("lk.transcription_delay", stt)
        turn.set_attribute("lk.interrupted", interrupted)


def _call(provider):
    tracer = provider.get_tracer("livekit-agents")
    with tracer.start_as_current_span("agent_session") as session:
        with tracer.start_as_current_span("user_turn"):
            pass
        _turn(tracer, e2e=0.82, stt=0.21, ttft=0.39, ttfa=0.18, interrupted=False)
        _turn(tracer, e2e=2.65, stt=0.24, ttft=1.9, ttfa=0.3, interrupted=True)
    return session.get_span_context().trace_id


def _by_name(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_each_turn_is_its_own_trace_in_the_calls_session(native):
    provider, exporter = native
    call = _call(provider)

    turns = _by_name(exporter, "agent_turn")
    assert len(turns) == 2
    turn_traces = {t.context.trace_id for t in turns}
    assert len(turn_traces) == 2 and call not in turn_traces
    for turn in turns:
        assert turn.parent is None
    # Every span of a turn moved with it, still under the same parent span.
    for name in ("llm_node", "tts_node"):
        for span in _by_name(exporter, name):
            assert span.context.trace_id in turn_traces
            parent = next(
                t for t in turns if t.context.trace_id == span.context.trace_id
            )
            assert span.parent.span_id == parent.context.span_id
            assert span.parent.trace_id == parent.context.trace_id
    # The call's own spans stay in the call's trace.
    for name in ("agent_session", "user_turn"):
        (span,) = _by_name(exporter, name)
        assert span.context.trace_id == call
    # One session: the call's trace id, on every span.
    assert {s.attributes.get(SESSION) for s in exporter.get_finished_spans()} == {
        f"{call:032x}"
    }


def test_a_turn_root_carries_its_voice_timings_in_ms(native):
    provider, exporter = native
    _call(provider)

    first, second = sorted(
        _by_name(exporter, "agent_turn"),
        key=lambda s: s.attributes[VOICE + "turn_index"],
    )
    voice = {
        k[len(VOICE) :]: v for k, v in first.attributes.items() if k.startswith(VOICE)
    }
    assert voice == {
        "turn_index": 0,
        "wait_ms": 820.0,
        "stt_ms": 210.0,
        "llm_ttft_ms": 390.0,
        "tts_ttfa_ms": 180.0,
        "interrupted": False,
    }
    assert second.attributes[VOICE + "turn_index"] == 1
    assert second.attributes[VOICE + "wait_ms"] == 2650.0
    assert second.attributes[VOICE + "interrupted"] is True
    # The raw LiveKit keys are still dropped, and only turn roots carry timings.
    for span in exporter.get_finished_spans():
        assert not any(k.startswith("lk.") for k in span.attributes)
        if span.name != "agent_turn":
            assert not any(k.startswith(VOICE) for k in span.attributes)


def test_a_session_the_user_set_wins(native):
    provider, exporter = native
    tracer = provider.get_tracer("livekit-agents")
    with tracer.start_as_current_span("agent_session", attributes={SESSION: "mine"}):
        with tracer.start_as_current_span("agent_turn", attributes={SESSION: "mine"}):
            pass
    assert {s.attributes.get(SESSION) for s in exporter.get_finished_spans()} == {
        "mine"
    }


def test_bad_timings_are_left_out(native):
    provider, exporter = native
    tracer = provider.get_tracer("livekit-agents")
    with tracer.start_as_current_span("agent_session"):
        with tracer.start_as_current_span("agent_turn") as turn:
            turn.set_attribute("lk.e2e_latency", -1.0)
            turn.set_attribute("lk.transcription_delay", float("nan"))
    (turn,) = _by_name(exporter, "agent_turn")
    voice = {k[len(VOICE) :] for k in turn.attributes if k.startswith(VOICE)}
    assert voice == {"turn_index"}
