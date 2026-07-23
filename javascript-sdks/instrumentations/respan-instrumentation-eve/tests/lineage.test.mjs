import test from "node:test";
import assert from "node:assert/strict";

import { trace } from "@opentelemetry/api";
import { withEveLineage } from "../dist/lineage.js";

function withActiveSpan(activeSpan, run) {
  const descriptor = Object.getOwnPropertyDescriptor(trace, "getActiveSpan");
  Object.defineProperty(trace, "getActiveSpan", {
    configurable: true,
    value: () => activeSpan,
  });

  try {
    return run();
  } finally {
    if (descriptor) {
      Object.defineProperty(trace, "getActiveSpan", descriptor);
    } else {
      delete trace.getActiveSpan;
    }
  }
}

test("composes an authored step.started hook and stamps delegated lineage", () => {
  const stamped = {};
  const activeSpan = {
    isRecording: () => true,
    setAttributes(attributes) {
      Object.assign(stamped, attributes);
    },
  };
  const authoredRuntimeContext = {
    tenant: "acme",
    nested: { enabled: true },
    __respan_eve: { spoofed: true },
  };
  const input = {
    session: {
      id: "session-child",
      parent: {
        callId: "call-parent",
        rootSessionId: "session-root",
        sessionId: "session-parent",
        turn: {
          id: "turn-parent",
          sequence: 4,
        },
      },
    },
  };
  let seenInput;
  let calls = 0;
  const untouchedEvent = () => undefined;
  const definition = {
    recordInputs: false,
    events: {
      "session.started": untouchedEvent,
      "step.started"(value) {
        calls += 1;
        seenInput = value;
        return { runtimeContext: authoredRuntimeContext };
      },
    },
  };

  const wrapped = withEveLineage(definition);
  const result = withActiveSpan(activeSpan, () =>
    wrapped.events["step.started"](input),
  );

  assert.notEqual(wrapped, definition);
  assert.notEqual(wrapped.events, definition.events);
  assert.equal(wrapped.recordInputs, false);
  assert.equal(wrapped.events["session.started"], untouchedEvent);
  assert.equal(calls, 1);
  assert.equal(seenInput, input);
  assert.deepEqual(result, {
    runtimeContext: {
      tenant: "acme",
      nested: { enabled: true },
      __respan_eve: {
        lineage: {
          callId: "call-parent",
          rootSessionId: "session-root",
          sessionId: "session-parent",
          turn: {
            id: "turn-parent",
            sequence: 4,
          },
        },
      },
    },
  });
  assert.deepEqual(authoredRuntimeContext, {
    tenant: "acme",
    nested: { enabled: true },
    __respan_eve: { spoofed: true },
  });
  assert.deepEqual(stamped, {
    "ai.settings.context.__respan_eve.lineage.rootSessionId":
      "session-root",
    "ai.settings.context.__respan_eve.lineage.sessionId":
      "session-parent",
    "ai.settings.context.__respan_eve.lineage.callId": "call-parent",
    "ai.settings.context.__respan_eve.lineage.turn.id": "turn-parent",
    "ai.settings.context.__respan_eve.lineage.turn.sequence": 4,
  });
});

test("uses the current session as the root when Eve has no parent lineage", () => {
  const wrapped = withEveLineage({});
  const result = withActiveSpan(undefined, () =>
    wrapped.events["step.started"]({
      session: {
        id: "session-root",
      },
    }),
  );

  assert.deepEqual(result, {
    runtimeContext: {
      __respan_eve: {
        lineage: {
          rootSessionId: "session-root",
        },
      },
    },
  });
});

test("keeps active-turn stamping best-effort", () => {
  const wrapped = withEveLineage({});
  const result = withActiveSpan(
    {
      isRecording: () => true,
      setAttributes() {
        throw new Error("custom span rejected attributes");
      },
    },
    () =>
      wrapped.events["step.started"]({
        session: {
          id: "session-root",
        },
      }),
  );

  assert.deepEqual(result, {
    runtimeContext: {
      __respan_eve: {
        lineage: {
          rootSessionId: "session-root",
        },
      },
    },
  });
});

test("preserves Eve validation for a forced async authored hook", async () => {
  const promise = Promise.resolve({
    runtimeContext: {
      tenant: "acme",
    },
  });
  const wrapped = withEveLineage({
    events: {
      "step.started": () => promise,
    },
  });

  const result = wrapped.events["step.started"]({
    session: {
      id: "session-root",
    },
  });

  assert.equal(result, promise);
  await promise;
});
