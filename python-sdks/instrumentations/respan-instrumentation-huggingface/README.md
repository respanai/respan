# Hugging Face Transformers instrumentation

Trace native `transformers.TextGenerationPipeline` calls through the upstream OpenTelemetry Transformers instrumentor. This package records complete native prompt and result structures, including batches, chat histories, multiple candidates, token-ID results, consumed lazy iterators, and generation options. It does not instrument Hugging Face Hub `InferenceClient`, training, embeddings, or arbitrary `model.generate()` calls.

Install the package and the framework needed by your model:

```bash
pip install respan-instrumentation-huggingface torch
```

Activate it with your OpenTelemetry provider:

```python
from opentelemetry.sdk.trace import TracerProvider
from respan_instrumentation_huggingface import HuggingFaceInstrumentor

provider = TracerProvider()  # Add your desired span processor/exporter.
instrumentor = HuggingFaceInstrumentor(tracer_provider=provider)
instrumentor.activate()
try:
    result = generator("Explain tracing", max_new_tokens=32, do_sample=False)
finally:
    instrumentor.deactivate()
```

`generator` is an existing native `TextGenerationPipeline`. The companion examples construct a tiny random CPU GPT-2 model and tokenizer locally, without downloading weights or calling a paid provider. The generated words demonstrate the SDK behavior; they are not model-quality examples.

The supported Transformers floor is 4.45.0, including its native `continue_final_message` chat continuation contract. Native pipeline `tools` is recorded when exposed by the installed Transformers version; it is unavailable on 4.45.0. Validation covers Transformers 5.18.0 / upstream instrumentor 0.62.4 / Torch 2.14.1 and the exact minimum Transformers 4.45.0 / upstream 0.61.0 / Torch 2.5.0. The former advertised 4.0 floor also depends on tokenizers 0.9.4, which has no wheel on the tested supported Python 3.12 runtime.

Native results and exceptions pass through unchanged. A native `PipelineIterator` is returned directly and ends its span on exhaustion, native failure, abandonment, or deactivation. The instrumentor never drains it. Native streamer callbacks also remain native; the span covers the pipeline call and its returned result rather than inventing token-level spans or usage. A caller who runs generation in another thread must propagate the OpenTelemetry context to preserve its parent.

Content capture is enabled by default. Set `capture_content=False`, a canonical `respan_enable_content_tracing=False` context or ancestor attribute, either supported legacy flag (`trace_content` / `override_enable_content_tracing`), or `RESPAN_TRACE_CONTENT` / `TRACELOOP_TRACE_CONTENT` to `false`, `0`, `no`, or `off`. Initial and later vetoes cannot be widened by another flag. Unknown local ancestry fails closed; both general and language-model suppression and native sampling are honored before extraction. A late veto removes owned bodies and diagnostic text. Credential values are redacted without dropping schema property names or false/zero/empty values.

The full native request options are capture-gated under `respan.metadata.huggingface.request`. Canonical JSON bodies have no package truncation; the native OpenTelemetry attribute limit may bound indexed convenience fields. Unknown user objects, callbacks, tensors outside native returned builtin structures, and custom serialization hooks are not executed for telemetry. Native usage counts and HTTP status are not inferred for local generation.

Identical owners share one delegate. Conflicting provider/context/capture or delegate options are rejected. Final deactivation restores only owned patches and processors, leaving foreign instrumentation intact. `exception_logger`, `use_legacy_attributes`, and upstream instrumentor keyword arguments are accepted for compatibility. Content is emitted as canonical span attributes; upstream early content log events are not emitted because they cannot be removed after a late privacy veto. The standalone `HuggingFaceSpanContractProcessor` maps existing upstream indexed attributes; full native payload and privacy observation require `HuggingFaceInstrumentor`.
