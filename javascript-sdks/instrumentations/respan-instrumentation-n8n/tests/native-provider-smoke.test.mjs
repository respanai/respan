import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

import { context, trace } from "@opentelemetry/api";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { N8nInstrumentor } from "../dist/index.js";

const require = createRequire(import.meta.url);

test("real NodeSDK smoke keeps one host provider and exports translated native n8n spans", async () => {
  const instrumentor = new N8nInstrumentor();
  instrumentor.activate();

  // Require only after activation so the public module hook sees NodeSDK load.
  const { NodeSDK } = require("@opentelemetry/sdk-node");
  const exporter = new InMemorySpanExporter();
  const sdk = new NodeSDK({
    resource: resourceFromAttributes({
      "service.name": "n8n",
      "service.version": "2.37.7",
      "n8n.instance.id": "smoke-instance",
      "n8n.instance.role": "main",
    }),
    traceExporter: exporter,
  });

  sdk.start();
  const provider = trace.getTracerProvider();
  const workflowTracer = trace.getTracer("n8n-workflow");
  const agentTracer = trace.getTracer("@n8n/agents");

  const workflow = workflowTracer.startSpan("workflow.execute", {
    attributes: {
      "n8n.workflow.id": "smoke-workflow",
      "n8n.workflow.name": "n8n native provider smoke",
      "n8n.workflow.node_count": 1,
      "n8n.execution.id": "smoke-execution",
    },
  });
  const workflowContext = trace.setSpan(context.active(), workflow);
  const agent = agentTracer.startSpan(
    "smoke-agent.generate",
    {
      attributes: {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "smoke-agent",
        "gen_ai.conversation.id": "smoke-thread",
        agent_id: "smoke-agent-id",
        source: "workflow",
      },
    },
    workflowContext,
  );
  const agentContext = trace.setSpan(workflowContext, agent);
  const tool = agentTracer.startSpan(
    "execute_tool lookup",
    {
      attributes: {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "lookup",
        "gen_ai.tool.call.arguments": JSON.stringify({ id: 7 }),
        "gen_ai.tool.call.result": JSON.stringify({ found: true }),
      },
    },
    agentContext,
  );

  tool.end();
  agent.end();
  workflow.end();

  // InMemorySpanExporter.shutdown() clears its buffer. Flush and inspect before
  // shutdown so this assertion covers the real BatchSpanProcessor path.
  await sdk._tracerProvider.forceFlush();

  const spans = exporter.getFinishedSpans();
  const names = spans.map((span) => span.name).sort();
  assert.deepEqual(names, ["agent.smoke-agent", "tool.lookup", "workflow"]);

  const workflowOut = spans.find((span) => span.name === "workflow");
  const agentOut = spans.find((span) => span.name === "agent.smoke-agent");
  const toolOut = spans.find((span) => span.name === "tool.lookup");
  assert.ok(workflowOut && agentOut && toolOut);
  assert.equal(agentOut.parentSpanContext.spanId, workflowOut.spanContext().spanId);
  assert.equal(toolOut.parentSpanContext.spanId, agentOut.spanContext().spanId);
  assert.strictEqual(trace.getTracerProvider(), provider);

  await sdk.shutdown();
  instrumentor.deactivate();
});
