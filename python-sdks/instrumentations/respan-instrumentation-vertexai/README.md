# Respan Vertex AI instrumentation

Observe the released `google-cloud-aiplatform` Vertex SDK with Respan's OpenTelemetry pipeline.

```bash
pip install "respan-instrumentation-vertexai[instruments]" respan-ai
```

```python
from respan import Respan
from respan_instrumentation_vertexai import VertexAIInstrumentor

respan = Respan(instrumentations=[VertexAIInstrumentor()])
```

The adapter observes native `GenerativeModel.generate_content`, `ChatSession.send_message`, and `TextEmbeddingModel.get_embeddings`, including their async variants. Native Google clients still perform request validation, conversion, transport calls and response parsing. The optional Google SDK can be absent; activation then does nothing.

Generation spans preserve native input contents, system instructions, tool schemas, history, text and function call projections, complete native response candidates, usage and available reasoning/signature fields. Embedding spans retain every returned vector element and provider-sourced input token statistics. Streams retain every observed chunk without a chunk cap. Early close captures only consumed chunks; it does not invent output or usage for an unread stream.

The adapter returns protocol proxies for native generation streams. Stream object identity and concrete generator type change; native chunk identity, iteration, send/throw, close and async equivalents delegate to the original generator. Closing before the first read closes the original and ends telemetry. Unary response and native exception identities are preserved.

Real OpenTelemetry sampling and both general and language-model suppression run before content inspection. Set `capture_content=False`, `TRACELOOP_TRACE_CONTENT=false`, `RESPAN_TRACE_CONTENT=false`, or the canonical Respan content context flag to disable content. Native Traceloop override and observed ancestor vetoes also apply. A denial remains irreversible for that invocation and its observed ancestor chain, removes previously captured content and diagnostics, and preserves native execution. Unknown local parent spans fail closed. Credentials are redacted from builtin JSON while tool schema shapes, false, zero, empty values and known native values remain intact; opaque objects are represented structurally without calling customer formatting hooks.

Native error types and source HTTP codes are retained. Error messages and descriptions obey content privacy. No success HTTP code, model result, or token usage is inferred. Instrumentation failures are contained and native results, errors and cleanup continue. Shared activation requires matching capture/provider settings; deactivation restores only owned descriptors and processors.

## SDK lifecycle

Google's [migration notice](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk) says these generative modules are deprecated and scheduled for removal after June 24, 2026. They remain present and executable in the audited released SDK 2.3.0. This adapter covers APIs actually available in the installed release; it does not provide removed APIs or instrument `google-genai`. Use the separate Google GenAI integration for that SDK.

The declared native floor is `google-cloud-aiplatform>=1.71.0`. Required Respan SDK and AI semantic convention floors are `respan-sdk>=2.6.26` and `opentelemetry-semantic-conventions-ai>=0.5.1`, respectively. They supply canonical span attributes and the source cache token attribute.

## Examples and validation

The companion `python/tracing/vertex-ai` examples use real Google SDK clients and protobuf responses over a local gRPC server by default. They cover sync/async generation, chat, streams and early close, tool execution, native errors, full embeddings and privacy. Trace export requires `RESPAN_EXAMPLE_EXPORT=1`; a live Google call requires the independent `VERTEXAI_EXAMPLE_LIVE=1` opt-in. Controlled fixtures validate native SDK behavior without proving a deployed Google service's behavior.
