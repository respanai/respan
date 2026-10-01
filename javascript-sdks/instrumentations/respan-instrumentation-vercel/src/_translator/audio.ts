import {
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_REQUEST_STREAM,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { formatCompletionOutput, formatPromptInput } from "./messages.js";
import { setDefault, setMetadata, type SpanAttributes as Attributes } from "./shared.js";

const SPEECH_OPERATION = "ai.generateSpeech";
const TRANSCRIBE_OPERATIONS = new Set(["ai.transcribe", "ai.streamTranscribe"]);

export function enrichAudio(attrs: Attributes, operation: string | undefined): void {
  if (operation !== SPEECH_OPERATION && !TRANSCRIBE_OPERATIONS.has(operation ?? "")) return;

  const speech = operation === SPEECH_OPERATION;
  attrs[RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND] = speech ? "speech" : "transcribe";
  setMetadata(attrs, "operation", operation);
  setMetadata(attrs, "model", attrs[ATTR_GEN_AI_REQUEST_MODEL]);
  setMetadata(attrs, "provider", attrs[ATTR_GEN_AI_PROVIDER_NAME]);
  setMetadata(attrs, "streaming", attrs[ATTR_GEN_AI_REQUEST_STREAM]);

  // Audio providers report different usage units (tokens, seconds, characters).
  // Preserve their units on task spans rather than interpreting them as chat tokens.
  const usage = Object.fromEntries(Object.entries(attrs)
    .filter(([key]) => key.startsWith("gen_ai.usage."))
    .map(([key, value]) => [key.slice("gen_ai.usage.".length), value]));
  if (Object.keys(usage).length) setMetadata(attrs, "usage", usage);

  const input = speech ? formatPromptInput(attrs) : audioSummary(attrs, "request");
  const output = speech ? audioSummary(attrs, "response") : formatCompletionOutput(attrs);
  if (input) setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, input);
  if (output) setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output);
}

function audioSummary(attrs: Attributes, direction: "request" | "response"): string | undefined {
  // The native adapter exposes descriptors only; it does not emit audio bytes.
  const audio = Object.fromEntries(["size", "media_type", "format"]
    .map(key => [key, attrs[`ai.${direction}.audio.${key}`]])
    .filter(([, value]) => value !== undefined));
  return Object.keys(audio).length ? JSON.stringify({ audio }) : undefined;
}
