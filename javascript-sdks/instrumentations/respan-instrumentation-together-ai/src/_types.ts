import type { RespanLogType } from "@respan/respan-sdk";

export type TogetherOperationKind =
  | "chat"
  | "text"
  | "embedding"
  | "image"
  | "rerank"
  | "speech"
  | "transcription"
  | "translation";

export interface TogetherOperationSpec {
  kind: TogetherOperationKind;
  method: "create" | "generate";
  spanName: string;
  logType: RespanLogType;
  requestType: string;
}

export interface PatchedResourceTarget {
  prototype: Record<string, any>;
  method: "create" | "generate";
  original: any;
}
