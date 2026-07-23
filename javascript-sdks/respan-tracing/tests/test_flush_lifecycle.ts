/**
 * Regression coverage for JS tracing flush lifecycle.
 *
 * The important contract is that flush drains ended spans without shutting down
 * the SDK, so short-lived examples can omit explicit shutdown while long-lived
 * processes can continue tracing after a flush.
 */

import { strict as assert } from "node:assert";
import { SpanKind, TraceFlags } from "@opentelemetry/api";
import type { ExportResult as SpanExportResult } from "@opentelemetry/core";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";
import { RespanTelemetry, getClient } from "../src/index.js";
import { MultiProcessorManager } from "../src/processor/manager.js";

class MemoryExporter implements SpanExporter {
  public spans: ReadableSpan[] = [];
  public shutdownCount = 0;
  public forceFlushCount = 0;

  export(spans: ReadableSpan[], resultCallback: (result: SpanExportResult) => void): void {
    this.spans.push(...spans);
    resultCallback({ code: 0 });
  }

  shutdown(): Promise<void> {
    this.shutdownCount += 1;
    return Promise.resolve();
  }

  forceFlush(): Promise<void> {
    this.forceFlushCount += 1;
    return Promise.resolve();
  }
}

const makeReadableSpan = (name: string): ReadableSpan => ({
  name,
  kind: SpanKind.INTERNAL,
  spanContext: () => ({
    traceId: "1".repeat(32),
    spanId: "2".repeat(16),
    traceFlags: TraceFlags.SAMPLED,
  }),
  parentSpanId: undefined,
  startTime: [0, 0],
  endTime: [0, 1],
  duration: [0, 1],
  status: { code: 0 },
  attributes: {},
  links: [],
  events: [],
  resource: { attributes: {} } as any,
  instrumentationLibrary: { name: "respan-test" } as any,
  instrumentationScope: { name: "respan-test" } as any,
  droppedAttributesCount: 0,
  droppedEventsCount: 0,
  droppedLinksCount: 0,
}) as ReadableSpan;

const assertSpanExported = (exporter: MemoryExporter, name: string) => {
  assert.ok(
    exporter.spans.some((span) => span.name === name),
    `expected span "${name}" to be exported`,
  );
};

async function testFlushDoesNotShutdown(): Promise<void> {
  const exporter = new MemoryExporter();
  const telemetry = new RespanTelemetry({
    appName: "flush-lifecycle-test",
    apiKey: "test-key",
    exporter,
    silenceInitializationMessage: true,
    disabledInstrumentations: [
      "http",
      "openAI",
      "anthropic",
      "azureOpenAI",
      "cohere",
      "bedrock",
      "googleVertexAI",
      "googleAIPlatform",
      "pinecone",
      "together",
      "langChain",
      "llamaIndex",
      "chromaDB",
      "qdrant",
    ],
  });

  await telemetry.initialize();

  await telemetry.withTask({ name: "flush_lifecycle_first" }, async () => undefined);
  await telemetry.flush();
  assertSpanExported(exporter, "flush_lifecycle_first.task");
  assert.equal(exporter.shutdownCount, 0, "flush must not shutdown the exporter");

  await telemetry.withTask({ name: "flush_lifecycle_after_flush" }, async () => undefined);
  await getClient().flush();
  assertSpanExported(exporter, "flush_lifecycle_after_flush.task");
  assert.equal(exporter.shutdownCount, 0, "client.flush must not shutdown the exporter");

  await telemetry.shutdown();
  assert.equal(exporter.shutdownCount, 1, "shutdown should remain terminal");
}

async function testDisableBatchUsesSimpleProcessor(): Promise<void> {
  const exporter = new MemoryExporter();
  const manager = new MultiProcessorManager({ disableBatch: true });

  manager.addProcessor({
    exporter,
    name: "default",
  });

  manager.onEnd(makeReadableSpan("simple_processor_span"));
  assertSpanExported(exporter, "simple_processor_span");

  await manager.shutdown();
}

await testFlushDoesNotShutdown();
await testDisableBatchUsesSimpleProcessor();

console.log("flush lifecycle tests passed");
