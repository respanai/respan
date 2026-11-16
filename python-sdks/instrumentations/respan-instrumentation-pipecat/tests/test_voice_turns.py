"""A Pipecat turn's voice timings on its root span (DEV-14477).

Spans are made by the real SDK tracer in the shape OpenInference traces a
Pipecat pipeline: each turn a root ``pipecat.conversation.turn`` span, its
services' ``pipecat.stt`` / ``pipecat.llm`` / ``pipecat.tts`` spans under it,
ended before the turn, with metrics as span attributes.
"""

import pytest
from openinference.semconv.trace import SpanAttributes as OI
from opentelemetry import context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import set_span_in_context
from respan_instrumentation_pipecat._translator import PipecatOpenInferenceTranslator

VOICE = "respan.metadata.voice."


@pytest.fixture
def pipeline():
    provider = TracerProvider()
    provider.add_span_processor(PipecatOpenInferenceTranslator())
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer("openinference.instrumentation.pipecat"), exporter
    provider.shutdown()


def _turn(tracer, *, number, wait, ttfb, interrupted):
    turn = tracer.start_span(
        "pipecat.conversation.turn",
        context=context.Context(),
        attributes={
            OI.OPENINFERENCE_SPAN_KIND: "CHAIN",
            "conversation.turn_number": number,
        },
    )
    under = set_span_in_context(turn)
    for service, kind in (("stt", "CHAIN"), ("llm", "LLM"), ("tts", "CHAIN")):
        for repeat in range(2):
            span = tracer.start_span(
                f"pipecat.{service}",
                context=under,
                attributes={OI.OPENINFERENCE_SPAN_KIND: kind, "service.type": service},
            )
            # A later span of the same service (a follow-up LLM call) is not
            # what the user waited on: the first one counts.
            span.set_attribute("service.ttfb_seconds", ttfb[service] + repeat * 5)
            span.end()
    turn.set_attribute("conversation.user_to_bot_latency", wait)
    turn.set_attribute("conversation.was_interrupted", interrupted)
    turn.end()


def _voice(span):
    return {
        k[len(VOICE) :]: v for k, v in span.attributes.items() if k.startswith(VOICE)
    }


def test_a_turn_root_carries_its_wait_and_each_services_first_byte(pipeline):
    tracer, exporter = pipeline
    _turn(
        tracer,
        number=1,
        wait=0.82,
        ttfb={"stt": 0.21, "llm": 0.39, "tts": 0.18},
        interrupted=False,
    )
    _turn(
        tracer,
        number=2,
        wait=2.65,
        ttfb={"stt": 0.24, "llm": 1.9, "tts": 0.3},
        interrupted=True,
    )

    first, second = (
        s
        for s in exporter.get_finished_spans()
        if s.name == "pipecat.conversation.turn"
    )
    assert _voice(first) == {
        "turn_index": 0,
        "wait_ms": 820.0,
        "stt_ms": 210.0,
        "llm_ttft_ms": 390.0,
        "tts_ttfa_ms": 180.0,
        "interrupted": False,
    }
    assert _voice(second)["turn_index"] == 1
    assert _voice(second)["llm_ttft_ms"] == 1900.0
    assert _voice(second)["interrupted"] is True
    # Only turn roots carry timings, and the raw service keys are still dropped.
    for span in exporter.get_finished_spans():
        assert "service.ttfb_seconds" not in span.attributes
        if span.name != "pipecat.conversation.turn":
            assert not _voice(span)


def test_a_turn_with_no_metrics_has_only_its_position(pipeline):
    tracer, exporter = pipeline
    turn = tracer.start_span(
        "pipecat.conversation.turn",
        context=context.Context(),
        attributes={OI.OPENINFERENCE_SPAN_KIND: "CHAIN", "conversation.turn_number": 3},
    )
    turn.set_attribute("conversation.user_to_bot_latency", -1.0)
    turn.end()
    (span,) = exporter.get_finished_spans()
    assert _voice(span) == {"turn_index": 2}
