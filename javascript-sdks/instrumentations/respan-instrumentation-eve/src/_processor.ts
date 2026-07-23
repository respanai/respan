/**
 * Translate Eve and its nested AI SDK spans to the Respan span contract.
 *
 * Eve emits an `ai.eve.turn` parent plus AI SDK model, tool, and agent spans.
 * This processor adds Eve lifecycle semantics and applies a local copy of the
 * AI SDK attribute translation used by Respan's Vercel integration.
 *
 * Two-phase enrichment:
 * - onStart(): Sets RESPAN_LOG_TYPE so the span passes CompositeProcessor filtering
 * - onEnd():   Full attribute translation (model, messages, tokens, metadata, etc.)
 */

import type { Context } from "@opentelemetry/api";
import type { ReadableSpan, Span, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_AGENT_ID,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes as TraceloopSpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  VERCEL_PARENT_SPANS,
  VERCEL_SPAN_CONFIG,
  VERCEL_STRUCTURAL_LLM_PARENT_SPANS,
} from "./constants/index.js";
import {
  EVE_SCOPE_NAME,
  EVE_SESSION_ID,
  EVE_TURN_ID,
  EVE_TURN_SEQUENCE,
  EVE_TURN_SPAN_NAME,
} from "./constants/eve.js";
import {
  EVE_RESPAN_LINEAGE_PARENT_SESSION_ID_ATTRIBUTE,
  EVE_RESPAN_LINEAGE_PARENT_TURN_ID_ATTRIBUTE,
  EVE_RESPAN_LINEAGE_PARENT_TURN_SEQUENCE_ATTRIBUTE,
  EVE_RESPAN_LINEAGE_ROOT_SESSION_ID_ATTRIBUTE,
  EVE_RESPAN_INTERNAL_EXPORT_TRACE_ID_ATTRIBUTE,
} from "./constants/lineage.js";
import {
  formatCompletionOutput,
  formatPromptInput,
  formatToolInput,
  formatToolOutput,
  parseToolChoice,
  parseToolsValue,
} from "./_translator/messages.js";
import {
  AI_AGENT_ID,
  AI_MODEL_ID,
  AI_PREFIX,
  AI_SETTINGS_CONTEXT_PREFIX,
  AI_TELEMETRY_METADATA_PREFIX,
  AI_TELEMETRY_FUNCTION_ID,
  AI_TOOL_CALL_NAME,
  formatEmbeddingInput,
  formatEmbeddingOutput,
  instrumentationScopeName,
  isModernVercelAISpanName,
  isVercelAISpan,
  isVercelAIScope,
  setMetadata,
  resolveLogType,
  safeJsonStr,
  setDefault,
} from "./_translator/shared.js";
import {
  enrichEveAttributes,
  resolveEveAgentName,
} from "./_translator/eve.js";
import { enrichMetadata, enrichModel, enrichPerformanceMetrics, enrichSystem, enrichTokens, stripRedundantAttrs } from "./_translator/span-enrichment.js";

interface PendingDelegatedUsageLineage {
  readonly recordedAt: number;
  readonly usageKey: string;
  readonly rootSessionId: string;
  readonly parentSessionId: string;
  readonly parentTurnId?: string;
  readonly parentTurnSequence?: number;
  readonly workflowName?: string;
  readonly targetTraceId?: string;
  readonly targetParentSpanId?: string;
}

interface RetainedWorkflowName {
  readonly recordedAt: number;
  readonly workflowName: string;
}

interface SessionTraceRoot {
  readonly recordedAt: number;
  readonly sessionId: string;
  readonly turnId?: string;
  readonly turnSequence?: number;
  readonly traceId: string;
  readonly spanId: string;
  readonly workflowName?: string;
}

interface DelegatedTraceCorrelation {
  readonly recordedAt: number;
  readonly targetTraceId: string;
  readonly targetParentSpanId: string;
  readonly workflowName?: string;
}

const MAX_PENDING_DELEGATED_USAGE_LINEAGES = 128;
const PENDING_DELEGATED_USAGE_LINEAGE_TTL_MS = 5 * 60 * 1000;
const MAX_RETAINED_TRACE_CONTEXTS = 128;
const RETAINED_TRACE_CONTEXT_TTL_MS = 5 * 60 * 1000;

/**
 * SpanProcessor that translates Eve and AI SDK attributes into Respan fields.
 *
 * Phase 1 (onStart): Sets RESPAN_LOG_TYPE so CompositeProcessor lets the span through.
 * Phase 2 (onEnd):   Full attribute enrichment — model, messages, tokens, metadata,
 *                     tools, performance metrics, environment, etc.
 */
export class EveSpanProcessor implements SpanProcessor {
  private _ownerCount: number;
  private readonly _inFlightSpans = new WeakSet<object>();
  /** Open structural wrapper spans: spanId → its own parentSpanId. */
  private readonly _openStructuralSpans = new Map<string, string | undefined>();
  /** Active Eve/AI SDK span IDs mapped to their inherited workflow name. */
  private readonly _openWorkflowNames = new Map<string, string>();
  /** Workflow names retained across Eve workflow-step context resumptions. */
  private readonly _openTraceWorkflowNames =
    new Map<string, RetainedWorkflowName>();
  /** Completed Eve turns addressable by exact session + turn lineage. */
  private readonly _sessionTraceRoots = new Map<string, SessionTraceRoot>();
  /** Eve child OTel trace ID → caller trace root selected by authored lineage. */
  private readonly _delegatedTraceCorrelations =
    new Map<string, DelegatedTraceCorrelation>();
  /**
   * Eve 0.26 emits caller-side subagent usage after a workflow boundary, with
   * no active trace or session attributes. Completed delegated model roots do
   * carry the authored lineage bridge, so retain a small, short-lived queue
   * and use the exact token-usage tuple to recover that caller session. When
   * concurrent matches disagree about lineage, leave the usage span ungrouped
   * rather than guessing.
   */
  private readonly _pendingDelegatedUsageLineages: PendingDelegatedUsageLineage[] = [];

  constructor({ initiallyActive = true }: { initiallyActive?: boolean } = {}) {
    this._ownerCount = initiallyActive ? 1 : 0;
  }

  acquire(): void {
    this._ownerCount += 1;
  }

  release(): void {
    this._ownerCount = Math.max(0, this._ownerCount - 1);
  }

  onStart(span: Span, _parentContext: Context): void {
    if (this._ownerCount === 0) {
      return;
    }

    const writableSpan = span as any;
    const name: string = writableSpan.name ?? "";
    const scopeName = instrumentationScopeName(writableSpan);
    if (
      name !== EVE_TURN_SPAN_NAME &&
      scopeName !== EVE_SCOPE_NAME &&
      !name.startsWith(AI_PREFIX) &&
      !isModernVercelAISpanName(name) &&
      !isVercelAIScope(scopeName)
    ) {
      return;
    }

    this._applyPendingDelegatedUsageLineage(writableSpan, scopeName);

    this._inFlightSpans.add(span as object);

    const spanContext =
      typeof writableSpan.spanContext === "function"
        ? writableSpan.spanContext()
        : undefined;
    const spanId: string | undefined = spanContext?.spanId;
    const traceId: string | undefined = spanContext?.traceId;
    const parentSpanId: string | undefined =
      writableSpan.parentSpanId ?? writableSpan.parentSpanContext?.spanId;

    if (name === EVE_TURN_SPAN_NAME) {
      this._rememberSessionTraceRoot(
        writableSpan.attributes,
        traceId,
        spanId,
      );
    }
    this._applyDelegatedTraceCorrelation(
      writableSpan.attributes,
      name,
      traceId,
      spanId,
      (key, value) => writableSpan.setAttribute(key, value),
    );
    this._applyWorkflowName(writableSpan, spanId, traceId, parentSpanId);

    if (name === EVE_TURN_SPAN_NAME) {
      writableSpan.setAttribute(
        RespanSpanAttributes.RESPAN_LOG_TYPE,
        RespanLogType.AGENT,
      );
      // Eve creates the turn beneath a framework server span that Respan does
      // not export. Promote normal turns to exported roots so the platform can
      // select the workflow name. Delegated turns replace this sentinel with
      // their caller root during onEnd correlation.
      writableSpan.setAttribute(
        RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT,
        "",
      );
      return;
    }

    // If the parent chain starts inside an open structural wrapper, stamp the
    // export-time parent (the wrapper's own parent) so the exporter can drop
    // the wrapper per-span, immune to export-batch boundaries. "" marks a
    // wrapper that was itself a root span — the child is promoted to root.
    if (parentSpanId && this._openStructuralSpans.has(parentSpanId)) {
      let exportParent: string | undefined = parentSpanId;
      const seen = new Set<string>();
      while (
        exportParent &&
        this._openStructuralSpans.has(exportParent) &&
        !seen.has(exportParent)
      ) {
        seen.add(exportParent);
        exportParent = this._openStructuralSpans.get(exportParent);
      }
      writableSpan.setAttribute(
        RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT,
        exportParent ?? ""
      );
    }

    if (
      VERCEL_STRUCTURAL_LLM_PARENT_SPANS.has(name) ||
      isEveModelAgentWrapper(name, scopeName, writableSpan.attributes)
    ) {
      // Structural wrapper: the .doGenerate/.doStream child carries the real
      // model/input/output. Eve's gen_ai invoke_agent span is likewise an AI
      // SDK transport wrapper around the real step + model spans, not another
      // framework agent. Semantic export drops these wrappers and reparents
      // their children; legacy export still preserves the emitted tree.
      writableSpan.setAttribute(RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN, true);
      if (spanId) {
        this._openStructuralSpans.set(spanId, parentSpanId);
      }
    }

    const config = VERCEL_SPAN_CONFIG[name];
    if (config) {
      writableSpan.setAttribute(RespanSpanAttributes.RESPAN_LOG_TYPE, config.logType);
      return;
    }

    const parentLogType = VERCEL_PARENT_SPANS[name];
    if (parentLogType !== undefined) {
      writableSpan.setAttribute(RespanSpanAttributes.RESPAN_LOG_TYPE, parentLogType);
      return;
    }

    writableSpan.setAttribute(RespanSpanAttributes.RESPAN_LOG_TYPE, RespanLogType.TASK);
  }

  onEnd(span: ReadableSpan): void {
    const endedSpanId = span.spanContext?.().spanId;
    if (endedSpanId) {
      this._openStructuralSpans.delete(endedSpanId);
      this._openWorkflowNames.delete(endedSpanId);
    }

    const startedWhileActive = this._inFlightSpans.delete(span as object);
    if (this._ownerCount === 0 && !startedWhileActive) {
      return;
    }

    const attrs = (span as any).attributes as Record<string, any> | undefined;
    const isEveTurn = span.name === EVE_TURN_SPAN_NAME;
    const scopeName = instrumentationScopeName(span);
    if (
      !attrs ||
      (!isEveTurn &&
        scopeName !== EVE_SCOPE_NAME &&
        !isVercelAISpan(span))
    ) {
      return;
    }

    const endedTraceId = span.spanContext?.().traceId;
    if (isEveTurn) {
      this._rememberSessionTraceRoot(attrs, endedTraceId, endedSpanId);
    }
    this._applyDelegatedTraceCorrelation(
      attrs,
      span.name,
      endedTraceId,
      endedSpanId,
      (key, value) => {
        attrs[key] = value;
      },
    );
    this._rememberDelegatedUsageLineage(attrs, scopeName, endedTraceId);

    const name = span.name;
    const config = VERCEL_SPAN_CONFIG[name];
    const parentLogType = VERCEL_PARENT_SPANS[name];
    const logType = isEveTurn
      ? RespanLogType.AGENT
      : resolveLogType(name, attrs);

    // Embedding spans (span-contract.md): input = embedded text, output = the
    // embedding vector(s) — captured, not dropped (debuggable RAG data; size is
    // handled by storage tiering, not by deleting it here). Vercel's synthetic
    // ai.usage.tokens is intentionally NOT surfaced as a token count; it's
    // stripped. Extract up front, before any early-return or metadata move, so
    // both the parent (ai.embed) and child (ai.embed.doEmbed) spans are covered.
    if (
      logType === RespanLogType.EMBEDDING ||
      config?.logType === RespanLogType.EMBEDDING ||
      parentLogType === RespanLogType.EMBEDDING
    ) {
      const embInput = formatEmbeddingInput(attrs);
      if (embInput) setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, embInput);
      const embOutput = formatEmbeddingOutput(attrs);
      if (embOutput) setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT, embOutput);
      enrichTokens(attrs);
    }

    const entityName =
      isEveTurn
        ? resolveEveAgentName(attrs)
        : logType === RespanLogType.AGENT
        ? attrs[ATTR_GEN_AI_AGENT_NAME] ??
          attrs[ATTR_GEN_AI_AGENT_ID] ??
          attrs["ai.agent.name"] ??
          attrs[AI_AGENT_ID] ??
          attrs[AI_TELEMETRY_METADATA_PREFIX + "agent_name"] ??
          name
        : name;

    enrichEveAttributes(attrs, { name, scopeName });
    enrichMetadata(attrs);
    delete attrs[TraceloopSpanAttributes.TRACELOOP_SPAN_KIND];

    attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = logType;
    setDefault(
      attrs,
      TraceloopSpanAttributes.TRACELOOP_ENTITY_NAME,
      String(entityName),
    );
    setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_PATH, "");

    if (isEveTurn) {
      setDefault(
        attrs,
        RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND,
        "agent",
      );
      setDefault(
        attrs,
        RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL,
        String(entityName),
      );
    }

    if (config) {
      // Do NOT set traceloop.span.kind for auto-emitted AI SDK spans.
      // In the Respan composite processor `traceloop.span.kind` is reserved
      // for user-decorated spans (withWorkflow / withTask / withAgent) and
      // setting it on auto spans (a) flattens the parent/child tree and
      // (b) causes LLM detail spans (doGenerate / doStream) to be classified
      // as "task" instead of LLM in the backend. The respan.entity.log_type
      // attribute (set above) carries the correct type for ingestion.
      // Matches the patterns in respan-instrumentation-openinference (see
      // _processor.ts:500) and respan-instrumentation-openai-agents
      // (see _otel_emitter.ts:398).

      if (config.isLLM) {
        setDefault(attrs, TraceloopSpanAttributes.LLM_REQUEST_TYPE, RespanLogType.CHAT);

        enrichSystem(attrs);
        enrichModel(attrs, attrs[AI_MODEL_ID]);

        const input = formatPromptInput(attrs);
        if (input) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, input);
        }

        const output = formatCompletionOutput(attrs);
        if (output) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT, output);
        }

        enrichTokens(attrs);

        const toolsValue = parseToolsValue(attrs);
        if (toolsValue) {
          attrs[TraceloopSpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJsonStr(toolsValue);
        }

        const toolChoice = parseToolChoice(attrs);
        if (toolChoice) {
          setMetadata(attrs, "tool_choice", toolChoice);
        }

        enrichPerformanceMetrics(attrs, name);
      }

      if (config.logType === RespanLogType.EMBEDDING || logType === RespanLogType.EMBEDDING) {
        // input/output/tokens are mapped in the up-front embedding block.
        setDefault(attrs, TraceloopSpanAttributes.LLM_REQUEST_TYPE, RespanLogType.EMBEDDING);
        enrichSystem(attrs);
        enrichModel(attrs, attrs[AI_MODEL_ID]);
      }

      if (config.logType === RespanLogType.TOOL || logType === RespanLogType.TOOL) {
        setToolSpanNameHint(attrs, name);

        const toolInput = formatToolInput(attrs);
        if (toolInput) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, toolInput);
        }

        const toolOutput = formatToolOutput(attrs);
        if (toolOutput) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT, toolOutput);
        }
      }

    } else {
      if (logType === RespanLogType.TEXT) {
        enrichSystem(attrs);
        enrichModel(attrs, attrs[AI_MODEL_ID]);

        enrichTokens(attrs);

        setDefault(attrs, TraceloopSpanAttributes.LLM_REQUEST_TYPE, RespanLogType.CHAT);

        const input = formatPromptInput(attrs);
        if (input) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, input);
        }

        const output = formatCompletionOutput(attrs);
        if (output) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT, output);
        }

        const toolsValue = parseToolsValue(attrs);
        if (toolsValue) {
          attrs[TraceloopSpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJsonStr(toolsValue);
        }

        const toolChoice = parseToolChoice(attrs);
        if (toolChoice) {
          setMetadata(attrs, "tool_choice", toolChoice);
        }

        enrichPerformanceMetrics(attrs, name);
      }

      if (logType === RespanLogType.EMBEDDING) {
        // input/output/tokens are mapped in the up-front embedding block.
        setDefault(attrs, TraceloopSpanAttributes.LLM_REQUEST_TYPE, RespanLogType.EMBEDDING);
        enrichSystem(attrs);
        enrichModel(attrs, attrs[AI_MODEL_ID]);
      }

      if (logType === RespanLogType.TOOL) {
        setToolSpanNameHint(attrs, name);

        const toolInput = formatToolInput(attrs);
        if (toolInput) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, toolInput);
        }

        const toolOutput = formatToolOutput(attrs);
        if (toolOutput) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT, toolOutput);
        }
      }
    }

    stripRedundantAttrs(attrs, logType);

  }

  private _applyWorkflowName(
    span: any,
    spanId: string | undefined,
    traceId: string | undefined,
    parentSpanId: string | undefined,
  ): void {
    const attrs = span.attributes as Record<string, any> | undefined;
    if (!attrs) {
      return;
    }

    this._pruneRetainedTraceContexts();
    const retainedWorkflow =
      traceId === undefined
        ? undefined
        : this._openTraceWorkflowNames.get(traceId)?.workflowName;
    const workflowName =
      nonEmptyString(
        attrs[TraceloopSpanAttributes.TRACELOOP_WORKFLOW_NAME],
      ) ??
      nonEmptyString(attrs[AI_TELEMETRY_FUNCTION_ID]) ??
      (parentSpanId === undefined
        ? undefined
        : this._openWorkflowNames.get(parentSpanId)) ??
      retainedWorkflow;
    if (workflowName === undefined) {
      return;
    }

    span.setAttribute(
      TraceloopSpanAttributes.TRACELOOP_WORKFLOW_NAME,
      workflowName,
    );
    if (spanId !== undefined) {
      this._openWorkflowNames.set(spanId, workflowName);
    }
    if (traceId !== undefined) {
      this._openTraceWorkflowNames.delete(traceId);
      this._openTraceWorkflowNames.set(traceId, {
        recordedAt: Date.now(),
        workflowName,
      });
      trimOldest(this._openTraceWorkflowNames, MAX_RETAINED_TRACE_CONTEXTS);
    }
  }

  private _applyPendingDelegatedUsageLineage(
    span: any,
    scopeName: string | undefined,
  ): void {
    const attrs = span.attributes as Record<string, any> | undefined;
    if (
      !attrs ||
      scopeName !== EVE_SCOPE_NAME ||
      attrs[ATTR_GEN_AI_OPERATION_NAME] !==
        GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT ||
      attrs[EVE_SESSION_ID] !== undefined ||
      attrs[AI_SETTINGS_CONTEXT_PREFIX + EVE_SESSION_ID] !== undefined ||
      attrs[RespanSpanAttributes.RESPAN_SESSION_ID] !== undefined
    ) {
      return;
    }

    const usageKey = delegatedUsageKey(attrs);
    if (usageKey === undefined) {
      return;
    }

    this._prunePendingDelegatedUsageLineages();
    const matches = this._pendingDelegatedUsageLineages
      .map((lineage, index) => ({ lineage, index }))
      .filter(({ lineage }) => lineage.usageKey === usageKey);
    if (matches.length === 0) {
      return;
    }

    const lineageSignatures = new Set(
      matches.map(({ lineage }) =>
        JSON.stringify([
          lineage.rootSessionId,
          lineage.parentSessionId,
          lineage.parentTurnId,
          lineage.parentTurnSequence,
          lineage.workflowName,
          lineage.targetTraceId,
          lineage.targetParentSpanId,
        ]),
      ),
    );
    if (lineageSignatures.size !== 1) {
      return;
    }

    const match = matches[0];
    if (match === undefined) {
      return;
    }
    const [lineage] = this._pendingDelegatedUsageLineages.splice(match.index, 1);
    if (lineage === undefined) {
      return;
    }

    span.setAttribute(EVE_SESSION_ID, lineage.parentSessionId);
    span.setAttribute(
      EVE_RESPAN_LINEAGE_ROOT_SESSION_ID_ATTRIBUTE,
      lineage.rootSessionId,
    );
    if (lineage.parentTurnId !== undefined) {
      span.setAttribute(EVE_TURN_ID, lineage.parentTurnId);
    }
    if (lineage.parentTurnSequence !== undefined) {
      span.setAttribute(EVE_TURN_SEQUENCE, lineage.parentTurnSequence);
    }
    if (lineage.workflowName !== undefined) {
      span.setAttribute(
        TraceloopSpanAttributes.TRACELOOP_WORKFLOW_NAME,
        lineage.workflowName,
      );
    }
    if (
      lineage.targetTraceId !== undefined &&
      lineage.targetParentSpanId !== undefined
    ) {
      span.setAttribute(
        EVE_RESPAN_INTERNAL_EXPORT_TRACE_ID_ATTRIBUTE,
        lineage.targetTraceId,
      );
      span.setAttribute(
        RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT,
        lineage.targetParentSpanId,
      );
      // The real child Eve agent subtree carries the delegated model content
      // and tokens. This late Eve usage event duplicates that call and has no
      // content of its own, so suppress it only when exact correlation exists.
      span.setAttribute(
        RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN,
        true,
      );
    }
  }

  private _rememberDelegatedUsageLineage(
    attrs: Record<string, any>,
    scopeName: string | undefined,
    sourceTraceId: string | undefined,
  ): void {
    if (
      scopeName === EVE_SCOPE_NAME ||
      attrs[ATTR_GEN_AI_OPERATION_NAME] !==
        GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT
    ) {
      return;
    }

    const usageKey = delegatedUsageKey(attrs);
    const rootSessionId = nonEmptyString(
      attrs[EVE_RESPAN_LINEAGE_ROOT_SESSION_ID_ATTRIBUTE],
    );
    const parentSessionId = nonEmptyString(
      attrs[EVE_RESPAN_LINEAGE_PARENT_SESSION_ID_ATTRIBUTE],
    );
    if (
      usageKey === undefined ||
      rootSessionId === undefined ||
      parentSessionId === undefined
    ) {
      return;
    }

    this._prunePendingDelegatedUsageLineages();
    const correlation =
      sourceTraceId === undefined
        ? undefined
        : this._delegatedTraceCorrelations.get(sourceTraceId);
    this._pendingDelegatedUsageLineages.push({
      recordedAt: Date.now(),
      usageKey,
      rootSessionId,
      parentSessionId,
      parentTurnId: nonEmptyString(
        attrs[EVE_RESPAN_LINEAGE_PARENT_TURN_ID_ATTRIBUTE],
      ),
      parentTurnSequence: finiteNumber(
        attrs[EVE_RESPAN_LINEAGE_PARENT_TURN_SEQUENCE_ATTRIBUTE],
      ),
      workflowName: nonEmptyString(
        attrs[TraceloopSpanAttributes.TRACELOOP_WORKFLOW_NAME],
      ),
      targetTraceId: correlation?.targetTraceId,
      targetParentSpanId: correlation?.targetParentSpanId,
    });
    if (
      this._pendingDelegatedUsageLineages.length >
      MAX_PENDING_DELEGATED_USAGE_LINEAGES
    ) {
      this._pendingDelegatedUsageLineages.shift();
    }
  }

  private _prunePendingDelegatedUsageLineages(): void {
    const oldestAllowed = Date.now() - PENDING_DELEGATED_USAGE_LINEAGE_TTL_MS;
    while (
      this._pendingDelegatedUsageLineages[0]?.recordedAt !== undefined &&
      this._pendingDelegatedUsageLineages[0].recordedAt < oldestAllowed
    ) {
      this._pendingDelegatedUsageLineages.shift();
    }
  }

  private _rememberSessionTraceRoot(
    attrs: Record<string, any> | undefined,
    traceId: string | undefined,
    spanId: string | undefined,
  ): void {
    if (!attrs || traceId === undefined || spanId === undefined) {
      return;
    }

    const sessionId =
      nonEmptyString(attrs[EVE_SESSION_ID]) ??
      nonEmptyString(attrs[AI_SETTINGS_CONTEXT_PREFIX + EVE_SESSION_ID]) ??
      nonEmptyString(attrs[RespanSpanAttributes.RESPAN_SESSION_ID]);
    if (sessionId === undefined) {
      return;
    }

    const turnId =
      nonEmptyString(attrs[EVE_TURN_ID]) ??
      nonEmptyString(attrs[AI_SETTINGS_CONTEXT_PREFIX + EVE_TURN_ID]);
    const turnSequence =
      finiteNumber(attrs[EVE_TURN_SEQUENCE]) ??
      finiteNumber(attrs[AI_SETTINGS_CONTEXT_PREFIX + EVE_TURN_SEQUENCE]);
    const root: SessionTraceRoot = {
      recordedAt: Date.now(),
      sessionId,
      turnId,
      turnSequence,
      traceId,
      spanId,
      workflowName:
        nonEmptyString(
          attrs[TraceloopSpanAttributes.TRACELOOP_WORKFLOW_NAME],
        ) ?? nonEmptyString(attrs[AI_TELEMETRY_FUNCTION_ID]),
    };

    this._pruneRetainedTraceContexts();
    const key = sessionTraceRootKey(sessionId, turnId, turnSequence);
    this._sessionTraceRoots.delete(key);
    this._sessionTraceRoots.set(key, root);
    trimOldest(this._sessionTraceRoots, MAX_RETAINED_TRACE_CONTEXTS);
  }

  private _applyDelegatedTraceCorrelation(
    attrs: Record<string, any> | undefined,
    spanName: string,
    sourceTraceId: string | undefined,
    sourceSpanId: string | undefined,
    setAttribute: (key: string, value: string) => void,
  ): void {
    if (!attrs || sourceTraceId === undefined) {
      return;
    }

    this._pruneRetainedTraceContexts();
    let correlation = this._delegatedTraceCorrelations.get(sourceTraceId);
    if (correlation === undefined) {
      const parentSessionId = nonEmptyString(
        attrs[EVE_RESPAN_LINEAGE_PARENT_SESSION_ID_ATTRIBUTE],
      );
      const parentTurnId = nonEmptyString(
        attrs[EVE_RESPAN_LINEAGE_PARENT_TURN_ID_ATTRIBUTE],
      );
      const parentTurnSequence = finiteNumber(
        attrs[EVE_RESPAN_LINEAGE_PARENT_TURN_SEQUENCE_ATTRIBUTE],
      );
      if (parentSessionId === undefined) {
        return;
      }

      // Eve 0.26 includes the parent sequence on delegated runtime context but
      // omits it from the caller ai.eve.turn root. Session + turn ID is still
      // an exact turn identity, so use that form only when the fully qualified
      // key is unavailable.
      const target =
        this._sessionTraceRoots.get(
          sessionTraceRootKey(
            parentSessionId,
            parentTurnId,
            parentTurnSequence,
          ),
        ) ??
        this._sessionTraceRoots.get(
          sessionTraceRootKey(parentSessionId, parentTurnId, undefined),
        );
      if (target === undefined || target.traceId === sourceTraceId) {
        return;
      }

      correlation = {
        recordedAt: Date.now(),
        targetTraceId: target.traceId,
        targetParentSpanId: target.spanId,
        workflowName: target.workflowName,
      };
      this._delegatedTraceCorrelations.delete(sourceTraceId);
      this._delegatedTraceCorrelations.set(sourceTraceId, correlation);
      trimOldest(
        this._delegatedTraceCorrelations,
        MAX_RETAINED_TRACE_CONTEXTS,
      );
    }

    setAttribute(
      EVE_RESPAN_INTERNAL_EXPORT_TRACE_ID_ATTRIBUTE,
      correlation.targetTraceId,
    );
    if (spanName === EVE_TURN_SPAN_NAME && sourceSpanId !== undefined) {
      setAttribute(
        RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT,
        correlation.targetParentSpanId,
      );
    }
    if (
      correlation.workflowName !== undefined &&
      nonEmptyString(
        attrs[TraceloopSpanAttributes.TRACELOOP_WORKFLOW_NAME],
      ) === undefined
    ) {
      setAttribute(
        TraceloopSpanAttributes.TRACELOOP_WORKFLOW_NAME,
        correlation.workflowName,
      );
    }
  }

  private _pruneRetainedTraceContexts(): void {
    const oldestAllowed = Date.now() - RETAINED_TRACE_CONTEXT_TTL_MS;
    pruneRecordedMap(this._openTraceWorkflowNames, oldestAllowed);
    pruneRecordedMap(this._sessionTraceRoots, oldestAllowed);
    pruneRecordedMap(this._delegatedTraceCorrelations, oldestAllowed);
  }

  /** Return an export-only clone for an exactly correlated delegated trace. */
  prepareForExport(span: ReadableSpan): ReadableSpan {
    const attrs = span.attributes as Record<string, any>;
    const rawTraceId = attrs[EVE_RESPAN_INTERNAL_EXPORT_TRACE_ID_ATTRIBUTE];
    if (rawTraceId === undefined) {
      return span;
    }

    const attributes = { ...attrs };
    delete attributes[EVE_RESPAN_INTERNAL_EXPORT_TRACE_ID_ATTRIBUTE];
    const clone = Object.create(Object.getPrototypeOf(span));
    Object.assign(clone, span);
    Object.defineProperty(clone, "attributes", {
      configurable: true,
      enumerable: true,
      value: attributes,
    });

    const traceId = String(rawTraceId).trim().toLowerCase();
    if (/^[0-9a-f]{32}$/.test(traceId)) {
      const originalSpanContext = span.spanContext();
      Object.defineProperty(clone, "spanContext", {
        configurable: true,
        enumerable: true,
        value: () => ({ ...originalSpanContext, traceId }),
      });
    }
    return clone as ReadableSpan;
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  shutdown(): Promise<void> {
    this._ownerCount = 0;
    this._openStructuralSpans.clear();
    this._openWorkflowNames.clear();
    this._openTraceWorkflowNames.clear();
    this._sessionTraceRoots.clear();
    this._delegatedTraceCorrelations.clear();
    this._pendingDelegatedUsageLineages.length = 0;
    return Promise.resolve();
  }
}

function isEveModelAgentWrapper(
  name: string,
  scopeName: string | undefined,
  attrs: Record<string, any> | undefined,
): boolean {
  return (
    attrs !== undefined &&
    scopeName !== EVE_SCOPE_NAME &&
    name.startsWith("invoke_agent ") &&
    attrs[ATTR_GEN_AI_OPERATION_NAME] ===
      GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT &&
    (attrs[EVE_SESSION_ID] !== undefined ||
      attrs[AI_SETTINGS_CONTEXT_PREFIX + EVE_SESSION_ID] !== undefined)
  );
}

function sessionTraceRootKey(
  sessionId: string,
  turnId: string | undefined,
  turnSequence: number | undefined,
): string {
  return JSON.stringify([sessionId, turnId ?? null, turnSequence ?? null]);
}

function trimOldest<T>(map: Map<string, T>, maximum: number): void {
  while (map.size > maximum) {
    const oldestKey = map.keys().next().value;
    if (oldestKey === undefined) {
      return;
    }
    map.delete(oldestKey);
  }
}

function pruneRecordedMap<T extends { readonly recordedAt: number }>(
  map: Map<string, T>,
  oldestAllowed: number,
): void {
  for (const [key, value] of map) {
    if (value.recordedAt >= oldestAllowed) {
      continue;
    }
    map.delete(key);
  }
}

function delegatedUsageKey(attrs: Record<string, any>): string | undefined {
  const inputTokens = finiteNumber(attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS]);
  const outputTokens = finiteNumber(attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS]);
  if (inputTokens === undefined || outputTokens === undefined) {
    return undefined;
  }

  return JSON.stringify([
    inputTokens,
    outputTokens,
    finiteNumber(attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS]) ?? 0,
    finiteNumber(attrs[ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS]) ?? 0,
  ]);
}

function finiteNumber(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

function nonEmptyString(value: unknown): string | undefined {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  return String(value);
}

/**
 * Semantic-name hint for tool spans. The exporter derives the "tool" prefix
 * from the log type, but the detail must be the tool's own name — the entity
 * name on AI SDK tool spans is the raw span name (e.g. "ai.toolCall").
 */
function setToolSpanNameHint(attrs: Record<string, any>, spanName: string): void {
  setDefault(attrs, RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND, "tool");
  setDefault(
    attrs,
    RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL,
    attrs[AI_TOOL_CALL_NAME] ?? attrs[ATTR_GEN_AI_TOOL_NAME] ?? spanName.split(".").at(-1)
  );
}
