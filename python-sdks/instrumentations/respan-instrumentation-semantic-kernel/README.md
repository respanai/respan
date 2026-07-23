# respan-instrumentation-semantic-kernel

Respan instrumentation plugin for [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel).

Semantic Kernel Python already emits OpenTelemetry spans for kernel functions
and chat/text completions when its experimental diagnostics are enabled. This
package activates those diagnostics, captures Semantic Kernel prompt/completion
log records on the active completion span, and registers a Respan span processor
that normalizes the resulting spans to the Respan/OpenTelemetry contract.

## Installation

```bash
pip install respan-ai respan-instrumentation-semantic-kernel
```

## Usage

```python
from respan import Respan
from respan_instrumentation_semantic_kernel import SemanticKernelInstrumentor

respan = Respan(
    instrumentations=[SemanticKernelInstrumentor()],
)
```

Any Semantic Kernel work started after initialization is traced through the
active Respan OpenTelemetry pipeline. By default the instrumentor enables
Semantic Kernel's sensitive diagnostics so prompt and completion content can be
attached to chat/text completion spans. Pass `capture_content=False` to emit
model and token metadata without prompt/completion content.
