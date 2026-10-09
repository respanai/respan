import assert from "node:assert/strict";
import test from "node:test";

import { BeeAIInstrumentor } from "../dist/index.js";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

test("BeeAIInstrumentor delegates activation with the provided BeeAI module", async () => {
  class FakeBeeAIInstrumentation {}

  const calls = [];
  const sdkModule = { BeeAgent: class BeeAgent {} };
  const delegate = {
    activate() {
      calls.push(["activate"]);
    },
    deactivate() {
      calls.push(["deactivate"]);
    },
  };

  const instrumentor = new BeeAIInstrumentor({
    sdkModule,
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory(instrumentationClass, module) {
      calls.push(["factory", instrumentationClass, module]);
      return delegate;
    },
  });

  await instrumentor.activate();
  await instrumentor.activate();
  instrumentor.deactivate();

  assert.deepEqual(calls, [
    ["factory", FakeBeeAIInstrumentation, sdkModule],
    ["activate"],
    ["deactivate"],
  ]);
});

test("BeeAIInstrumentor coalesces concurrent activation", async () => {
  class FakeBeeAIInstrumentation {}

  const delegateGate = deferred();
  const calls = [];
  let factoryCalls = 0;
  const delegate = {
    activate() {
      calls.push("activate");
    },
    deactivate() {
      calls.push("deactivate");
    },
  };
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: {},
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory() {
      factoryCalls += 1;
      return delegateGate.promise;
    },
  });

  const firstActivation = instrumentor.activate();
  const secondActivation = instrumentor.activate();
  assert.equal(factoryCalls, 1);

  delegateGate.resolve(delegate);
  await Promise.all([firstActivation, secondActivation]);
  assert.deepEqual(calls, ["activate"]);

  instrumentor.deactivate();
  assert.deepEqual(calls, ["activate", "deactivate"]);
});

test("BeeAIInstrumentor cancels a pending activation and can reactivate", async () => {
  class FakeBeeAIInstrumentation {}

  const firstDelegateGate = deferred();
  const calls = [];
  let factoryCalls = 0;
  const delegate = {
    activate() {
      calls.push("activate");
    },
    deactivate() {
      calls.push("deactivate");
    },
  };
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: {},
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory() {
      factoryCalls += 1;
      return factoryCalls === 1 ? firstDelegateGate.promise : delegate;
    },
  });

  const cancelledActivation = instrumentor.activate();
  instrumentor.deactivate();
  firstDelegateGate.resolve(delegate);
  await cancelledActivation;
  assert.deepEqual(calls, []);

  await instrumentor.activate();
  assert.equal(factoryCalls, 2);
  assert.deepEqual(calls, ["activate"]);

  instrumentor.deactivate();
  assert.deepEqual(calls, ["activate", "deactivate"]);
});

test("BeeAIInstrumentor rolls back activation errors and permits retry", async () => {
  class FakeBeeAIInstrumentation {}

  const calls = [];
  let factoryCalls = 0;
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: {},
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory() {
      factoryCalls += 1;
      const attempt = factoryCalls;
      return {
        activate() {
          calls.push(`activate-${attempt}`);
          if (attempt === 1) throw new Error("activation failed");
        },
        deactivate() {
          calls.push(`deactivate-${attempt}`);
        },
      };
    },
  });

  await assert.rejects(instrumentor.activate(), /activation failed/);
  await instrumentor.activate();
  instrumentor.deactivate();

  assert.deepEqual(calls, [
    "activate-1",
    "deactivate-1",
    "activate-2",
    "deactivate-2",
  ]);
});
