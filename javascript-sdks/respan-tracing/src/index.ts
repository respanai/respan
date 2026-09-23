export { RespanTelemetry } from "./main.js";
export * from "./decorators/index.js";
export * from "./contexts/index.js";
export * from "./types/index.js";
export * from "./utils/index.js";
export { getClient } from "./utils/client.js";
export type { RespanClient, UpdateSpanOptions } from "./utils/client.js";
export { getSpanBufferManager } from "./utils/spanBuffer.js";
export type { SpanBuffer } from "./utils/spanBuffer.js";
export type { ProcessorConfig } from "./types/clientTypes.js";
export {
  getRegisteredSpanTransformerKeys,
  registerSpanTransformer,
  RESPAN_SPAN_TRANSFORMER_REGISTRY_SYMBOL,
} from "./processor/transformers.js";
export type {
  RespanSpanTransformer,
  SpanTransformerRegistration,
} from "./processor/transformers.js";
export { transformReadableSpanBatch } from "./processor/spanName.js";
