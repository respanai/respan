import { N8nInstrumentor } from "./index.js";

const PRELOAD_STATE_SYMBOL = Symbol.for("respan.instrumentation.n8n.preload.v1");
const globalState = globalThis as typeof globalThis & {
  [PRELOAD_STATE_SYMBOL]?: N8nInstrumentor;
};

if (!globalState[PRELOAD_STATE_SYMBOL]) {
  const instrumentor = new N8nInstrumentor();
  instrumentor.activate();
  globalState[PRELOAD_STATE_SYMBOL] = instrumentor;
}

export {};
