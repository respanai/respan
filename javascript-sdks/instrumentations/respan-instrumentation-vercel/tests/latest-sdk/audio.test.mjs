import test from "node:test";
import assert from "node:assert/strict";
import { BasicTracerProvider, InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { generateSpeech, transcribe, experimental_streamTranscribe, registerTelemetry } from "ai";
import { MockSpeechModelV4, MockTranscriptionModelV4 } from "ai/test";
import { OpenTelemetry } from "@ai-sdk/otel";
import { VercelAITranslator } from "../../dist/_translator.js";

const audio = new Uint8Array([1, 2, 3, 4]);
const response = { timestamp: new Date(), modelId: "audio-fixture" };
const speech = () => new MockSpeechModelV4({ provider: "test", modelId: "speech-fixture", doGenerate: async () => ({ audio, warnings: [], response, usage: { inputTokens: 3, audioSeconds: 0.5 } }) });
const transcription = () => new MockTranscriptionModelV4({ provider: "test", modelId: "transcription-fixture", doGenerate: async () => ({ text: "fixture transcript", segments: [], warnings: [], response, usage: { inputTokens: 4, outputTokens: 2 } }) });
const stream = parts => new ReadableStream({ start(controller) { for (const part of parts) controller.enqueue(part); controller.close(); } });

async function withTelemetry(fn) {
  const exporter = new InMemorySpanExporter();
  const translator = new VercelAITranslator();
  const provider = new BasicTracerProvider({ spanProcessors: [
    { onStart: (span, ctx) => translator.onStart(span, ctx), onEnd: span => translator.onEnd(span), forceFlush: async () => {}, shutdown: async () => {} },
    new SimpleSpanProcessor(exporter),
  ] });
  const integration = new OpenTelemetry({ tracer: provider.getTracer("gen_ai") });
  registerTelemetry(integration);
  try { await fn(exporter); } finally {
    const registry = globalThis.AI_SDK_TELEMETRY_INTEGRATIONS;
    registry.splice(registry.indexOf(integration), 1);
    await provider.shutdown();
  }
}

function assertAudioSpan(span, kind) {
  assert.equal(span.attributes["respan.entity.log_type"], "task");
  assert.equal(span.attributes["respan.internal.span_name.kind"], kind);
  assert.ok(!Object.keys(span.attributes).some(key => key.startsWith("ai.") || key.startsWith("gen_ai.") || key.startsWith("llm.")));
  return span.attributes;
}

test("AI SDK 7 speech retains text, audio descriptors and provider usage", () => withTelemetry(async exporter => {
  await generateSpeech({ model: speech(), text: "Speak this fixture", outputFormat: "mp3" });
  assert.equal(exporter.getFinishedSpans().length, 1);
  const attrs = assertAudioSpan(exporter.getFinishedSpans()[0], "speech");
  assert.equal(JSON.parse(attrs["traceloop.entity.input"])[0].content, "Speak this fixture");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), { audio: { size: 4, media_type: "audio/mp3", format: "mp3" } });
  assert.deepEqual(JSON.parse(attrs["respan.metadata"]), { operation: "ai.generateSpeech", model: "speech-fixture", provider: "test", streaming: false, usage: { input_tokens: 3, audio_seconds: 0.5 } });
}));

test("AI SDK 7 transcription retains audio descriptors and final transcript", () => withTelemetry(async exporter => {
  await transcribe({ model: transcription(), audio });
  const attrs = assertAudioSpan(exporter.getFinishedSpans()[0], "transcribe");
  assert.equal(JSON.parse(attrs["traceloop.entity.input"]).audio.size, 4);
  assert.equal(JSON.parse(attrs["traceloop.entity.output"]).content, "fixture transcript");
  assert.equal(JSON.parse(attrs["respan.metadata"]).usage.output_tokens, 2);
}));

test("AI SDK 7 streaming transcription ends once with final content and usage", () => withTelemetry(async exporter => {
  const model = new MockTranscriptionModelV4({ provider: "test", modelId: "stream-transcription-fixture", doStream: async () => ({ stream: stream([
    { type: "stream-start", warnings: [] },
    { type: "transcript-delta", delta: "stream fixture" },
    { type: "finish", text: "stream fixture", segments: [], usage: { durationSeconds: 1.5 } },
  ]) }) });
  const result = experimental_streamTranscribe({ model, audio: stream([audio]), inputAudioFormat: { mediaType: "audio/pcm", sampleRate: 16000, channels: 1 } });
  for await (const _ of result.fullStream) { /* consume lifecycle */ }
  assert.equal(await result.text, "stream fixture");
  assert.equal(exporter.getFinishedSpans().length, 1);
  const attrs = assertAudioSpan(exporter.getFinishedSpans()[0], "transcribe");
  assert.equal(JSON.parse(attrs["traceloop.entity.output"]).content, "stream fixture");
  assert.equal(JSON.parse(attrs["respan.metadata"]).streaming, true);
  assert.equal(JSON.parse(attrs["respan.metadata"]).usage.duration_seconds, 1.5);
}));

for (const mode of ["sdk", "global"]) {
  test(`audio content opt-out is respected (${mode})`, () => withTelemetry(async exporter => {
    const previous = process.env.RESPAN_TRACE_CONTENT;
    try {
      if (mode === "global") process.env.RESPAN_TRACE_CONTENT = "false";
      const telemetry = mode === "sdk" ? { recordInputs: false, recordOutputs: false } : undefined;
      await generateSpeech({ model: speech(), text: "PRIVATE_SPEECH", telemetry });
      await transcribe({ model: transcription(), audio, telemetry });
    } finally {
      if (previous === undefined) delete process.env.RESPAN_TRACE_CONTENT;
      else process.env.RESPAN_TRACE_CONTENT = previous;
    }
    for (const span of exporter.getFinishedSpans()) {
      assert.equal(span.attributes["traceloop.entity.input"], undefined);
      assert.equal(span.attributes["traceloop.entity.output"], undefined);
      assert.ok(!JSON.stringify(span.attributes).includes("PRIVATE_SPEECH"));
    }
  }));
}

test("failed speech preserves the error without synthesizing audio output", () => withTelemetry(async exporter => {
  await assert.rejects(generateSpeech({ model: new MockSpeechModelV4({ doGenerate: async () => { throw new Error("speech fixture failure"); } }), text: "fail", maxRetries: 0 }), /speech fixture failure/);
  assert.equal(exporter.getFinishedSpans().length, 1);
  const span = exporter.getFinishedSpans()[0];
  assertAudioSpan(span, "speech");
  assert.equal(span.status.code, 2);
  assert.equal(span.attributes["traceloop.entity.output"], undefined);
}));
