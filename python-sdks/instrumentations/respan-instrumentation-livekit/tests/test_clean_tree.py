"""What a Respan trace of a LiveKit call shows: the turn, what the user said,
the LLM call and the speech, without LiveKit's plumbing spans.

Spans are made by the real SDK tracer in the shape LiveKit traces a call, then
go through the Respan exporter's own preparation, so the test sees the tree
that is sent.
"""

import contextlib

import pytest
from livekit.agents.telemetry import trace_types
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode
from respan_instrumentation_livekit import LiveKitInstrumentor
from respan_instrumentation_livekit import _instrumentation as lifecycle
from respan_tracing.exporters import respan as exporter_module

INPUT = "traceloop.entity.input"
OUTPUT = "traceloop.entity.output"
VOICE = "respan.metadata.voice."


@pytest.fixture(params=[True, False], ids=["content", "private"])
def native(monkeypatch, request):
    assert lifecycle._RUNTIME is None
    provider = TracerProvider()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    owner = LiveKitInstrumentor(capture_content=request.param)
    owner.activate()
    try:
        yield provider, memory, request.param
    finally:
        owner.deactivate()
        provider.shutdown()


def _call(provider, *, adapter=True, failing_eou=False):
    lk = provider.get_tracer("livekit-agents")
    mine = provider.get_tracer("my-agent")
    with lk.start_as_current_span("agent_session"):
        with lk.start_as_current_span("session_start"):
            with lk.start_as_current_span("start_agent_activity"):
                with lk.start_as_current_span("on_enter"):
                    with mine.start_as_current_span("load_patient"):
                        pass
        with lk.start_as_current_span("user_turn"):
            with lk.start_as_current_span("user_speaking"):
                pass
            with lk.start_as_current_span("eou_wait") as eou:
                if failing_eou:
                    eou.set_status(Status(StatusCode.ERROR, "endpointing failed"))
        with lk.start_as_current_span("agent_turn") as turn:
            with lk.start_as_current_span("llm_node") as node:
                node.set_attribute("lk.response.ttft", 0.39)
                with lk.start_as_current_span("llm_request"):
                    with lk.start_as_current_span("llm_request_run"):
                        pass
            with lk.start_as_current_span("tts_node") as tts:
                tts.set_attribute("lk.response.ttfb", 0.18)
                with (
                    lk.start_as_current_span("tts_stream_adapter")
                    if adapter
                    else contextlib.nullcontext()
                ) as ad:
                    for sentence in ("Sure, I can help.", "Which day works?"):
                        with lk.start_as_current_span("tts_request") as req:
                            req.set_attribute(trace_types.ATTR_TTS_INPUT_TEXT, sentence)
                            with lk.start_as_current_span("tts_request_run"):
                                pass
                    if adapter:
                        ad.set_attribute(
                            trace_types.ATTR_TTS_INPUT_TEXT,
                            "Sure, I can help. Which day works?",
                        )
            with lk.start_as_current_span("agent_speaking"):
                pass
            turn.set_attribute(trace_types.ATTR_USER_INPUT, "move my cleaning")
            turn.set_attribute(
                trace_types.ATTR_RESPONSE_TEXT, "Sure, I can help. Which day works?"
            )
            turn.set_attribute("lk.e2e_latency", 0.82)
            turn.set_attribute("lk.interrupted", True)


def _sent(memory):
    """The spans as the Respan exporter sends them: name -> (span, parent id)."""
    prepared = exporter_module._prepare_spans_for_export(memory.get_finished_spans())
    sent = {}
    for span in prepared:
        body = exporter_module._span_to_otlp_json(span)
        sent.setdefault(span.name, []).append((span, body.get("parentSpanId", "")))
    return sent


def _id(span):
    return f"{span.get_span_context().span_id:016x}"


def test_the_trace_shows_the_turn_and_its_work_not_the_plumbing(native):
    provider, memory, _ = native
    _call(provider)
    sent = _sent(memory)

    assert set(sent) == {"agent_turn", "llm_request", "tts_node", "load_patient"}
    ((turn, turn_parent),) = sent["agent_turn"]
    assert turn_parent == ""
    # Each kept span hangs from the nearest kept ancestor.
    assert sent["llm_request"][0][1] == _id(turn)
    assert sent["tts_node"][0][1] == _id(turn)
    # A user's own span under a dropped callback becomes a root, not an orphan.
    assert sent["load_patient"][0][1] == ""


def test_the_turn_timings_are_the_same(native):
    provider, memory, _ = native
    _call(provider)
    ((turn, _),) = _sent(memory)["agent_turn"]
    assert turn.attributes[VOICE + "wait_ms"] == 820.0
    assert turn.attributes[VOICE + "llm_ttft_ms"] == 390.0
    assert turn.attributes[VOICE + "tts_ttfa_ms"] == 180.0
    assert turn.attributes[VOICE + "interrupted"] is True


def test_speech_input_is_the_text_spoken_and_the_turn_reads_as_the_exchange(native):
    provider, memory, is_content = native
    _call(provider)
    sent = _sent(memory)
    ((tts, _),) = sent["tts_node"]
    ((turn, _),) = sent["agent_turn"]
    if not is_content:
        assert INPUT not in tts.attributes
        assert INPUT not in turn.attributes and OUTPUT not in turn.attributes
        return
    assert tts.attributes[INPUT] == "Sure, I can help. Which day works?"
    assert turn.attributes[INPUT] == "move my cleaning"
    assert turn.attributes[OUTPUT] == "Sure, I can help. Which day works?"


def test_without_a_stream_adapter_the_sentences_are_joined(native):
    provider, memory, is_content = native
    _call(provider, adapter=False)
    ((tts, _),) = _sent(memory)["tts_node"]
    if is_content:
        assert tts.attributes[INPUT] == "Sure, I can help. Which day works?"


def test_a_plumbing_span_that_fails_is_kept(native):
    provider, memory, _ = native
    _call(provider, failing_eou=True)
    assert "eou_wait" in _sent(memory)


def test_the_legacy_style_sends_every_span(native, monkeypatch):
    monkeypatch.setenv("RESPAN_SPAN_NAME_STYLE", "legacy")
    provider, memory, _ = native
    _call(provider)
    sent = _sent(memory)
    assert {"agent_session", "user_turn", "eou_wait", "llm_node"} <= set(sent)
    for span, _parent in (s for spans in sent.values() for s in spans):
        assert exporter_module._convert_attributes(span.attributes) == [
            a
            for a in exporter_module._convert_attributes(span.attributes)
            if not a["key"].startswith("respan.internal")
        ]
