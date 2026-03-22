# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a monorepo containing Respan SDKs for Python and JavaScript/TypeScript. The SDKs provide OpenTelemetry-based tracing for LLM applications, sending telemetry data in OpenLLMetry format to Respan.

## Project Structure

```
respan/
├── python-sdks/           # Python packages (Poetry)
│   ├── respan-sdk/            # Core types, preprocessing, API payload helpers
│   ├── respan-tracing/        # Main tracing library with OpenTelemetry
│   ├── respan-exporter-*/     # Integration exporters (litellm, agno, haystack, openai-agents, langfuse)
│   ├── respan-instrumentation-*/  # Instrumentation plugins (google-adk)
│   ├── respan/                # Standalone package
│   └── examples/              # Usage examples by framework (google-adk/, etc.)
├── javascript-sdks/       # JavaScript/TypeScript packages (Yarn)
│   ├── respan-sdk/            # Core types and SDK (@respan/respan-sdk)
│   ├── respan-tracing/        # Main tracing library (@respan/tracing)
│   └── respan-exporter-*/     # Integration exporters (n8n, vercel, openai-agents)
└── boilerplates/          # Implementation logs and guides
```

## Build Commands

### Python Packages (Poetry)

Each Python package is managed independently with Poetry:

```bash
# Navigate to package directory first
cd python-sdks/respan-tracing

# Install dependencies
poetry install

# Install with dev dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Run a single test file
poetry run pytest tests/test_client_api.py

# Run specific test
poetry run pytest tests/test_client_api.py::test_function_name

# Build package
poetry build
```

### JavaScript/TypeScript Packages (Yarn)

```bash
# Navigate to package directory
cd javascript-sdks/respan-tracing

# Install dependencies
yarn install

# Build
yarn build

# Run examples
yarn examples:basic
yarn examples:advanced

# Test build
yarn test:build
```

## Architecture

### Tracing Architecture (Python)

- **RespanTelemetry** ([python-sdks/respan-tracing/src/respan_tracing/main.py](python-sdks/respan-tracing/src/respan_tracing/main.py)) - Main entry point, initializes OpenTelemetry tracer
- **RespanTracer** ([python-sdks/respan-tracing/src/respan_tracing/core/tracer.py](python-sdks/respan-tracing/src/respan_tracing/core/tracer.py)) - Core tracer implementation with processor management
- **RespanClient** ([python-sdks/respan-tracing/src/respan_tracing/core/client.py](python-sdks/respan-tracing/src/respan_tracing/core/client.py)) - Client for trace operations (get trace ID, update spans, etc.)
- **Decorators** ([python-sdks/respan-tracing/src/respan_tracing/decorators/](python-sdks/respan-tracing/src/respan_tracing/decorators/)) - `@workflow`, `@task`, `@agent`, `@tool` for tracing functions

### Tracing Architecture (TypeScript)

- **RespanTelemetry** ([javascript-sdks/respan-tracing/src/main.ts](javascript-sdks/respan-tracing/src/main.ts)) - Main client class with `withTask`, `withWorkflow`, `withAgent`, `withTool` wrappers
- Auto-discovery instrumentation for OpenAI, Anthropic, Azure, Cohere, Bedrock, Vertex AI, etc.
- Manual instrumentation support for Next.js/Webpack environments via `instrumentModules`

### Span Hierarchy

Traces follow OpenLLMetry conventions:
- **Workflow** - Top-level agent run/process
- **Task** - Sub-operations within a workflow (tool calls, LLM calls)
- **Agent** - Agent-specific spans
- **Tool** - Tool execution spans

### Processor Pattern

Both Python and TypeScript SDKs support multiple processors for routing spans to different destinations:

```python
# Python - add custom processor
kai.add_processor(
    exporter=RespanSpanExporter(...),
    name="production",
    filter_fn=lambda span: span.attributes.get("processor") == "prod"
)
```

```typescript
// TypeScript - add custom processor
respan.addProcessor({
    exporter: new FileExporter("./debug.json"),
    name: "debug"
});
```

## Environment Variables

- `RESPAN_API_KEY` - API key for Respan platform
- `RESPAN_BASE_URL` - API endpoint (default: `https://api.respan.ai/api`)
- `IS_RESPAN_BATCHING_ENABLED` - Enable batch processing (default: true)
- `RESPAN_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Python Version Requirements

- `respan-sdk`: Python >3.9, <4.0
- `respan-tracing`: Python >=3.11, <3.14
- Exporters vary (check individual pyproject.toml files)

## Package Dependencies

- `respan-tracing` depends on `respan-sdk`
- `respan-instrumentation-*` packages depend on `respan-tracing`
- TypeScript `@respan/tracing` depends on `@respan/respan-sdk`
- Both use OpenTelemetry SDK for tracing infrastructure

## Instrumentation Plugin Architecture

Instrumentation packages (`respan-instrumentation-*`) follow a plugin pattern that enriches third-party spans in-place as they flow through the standard OTLP pipeline.

### Pattern: How Instrumentors Work

```
Third-party framework (e.g. Google ADK) generates raw OTel spans
         ↓
Instrumentor.activate() patches OTel span processors via wrapt
         ↓
RespanSpanProcessor (patched to accept spans lacking traceloop attributes)
         ↓
BatchSpanProcessor._export() (patched to enrich spans before export)
         ↓
Enriched spans with OpenLLMetry-compatible attributes
         ↓
RespanSpanExporter → Respan backend
```

Key rules:
- **Instrumentors are stateless** — they contain no config or mutable state. Config like `environment` and `customer_identifier` belongs on `Respan()`, not on the instrumentor.
- **No custom export paths** — instrumentors enrich spans in-place through the existing pipeline, never create their own exporters.
- **Enrichment converts** third-party attributes to OpenLLMetry format (`traceloop.span.kind`, `traceloop.entity.input`, etc.) and strips redundant internal attributes.

### Example: Google ADK Instrumentation

```python
from respan_instrumentation_google_adk import GoogleAdkInstrumentor

respan = Respan(
    instrumentations=[GoogleAdkInstrumentor()],
    environment="development",
    customer_identifier="demo-user",
)
```

Package structure for `respan-instrumentation-google-adk`:
- `instrumentor.py` — Stateless entry point, calls `patch_span_processors()`
- `processor.py` — wrapt patches on `RespanSpanProcessor.on_end()`, `BatchSpanProcessor._export()`, `SimpleSpanProcessor.on_end()`
- `enrichment.py` — Two-pass span enrichment (collect cross-span info, then enrich by span type)
- `adk_detection.py` — Detects ADK spans by instrumentation scope name

## Development Rules

### Don't reinvent the wheel
Before writing any utility function, **always check** these locations first:
1. Python stdlib (e.g. `time.time_ns()`, `datetime.fromisoformat()`, `uuid.uuid4()`)
2. OpenTelemetry SDK / `opentelemetry.semconv_ai` for span attribute constants
3. `respan_sdk/utils/` for existing shared helpers
4. `respan_sdk/constants/` for existing constants

**Never create wrapper functions** for stdlib one-liners (e.g. `now_ns()` wrapping `time.time_ns()`).

### Constants: use canonical sources
- **OTEL/GenAI/Traceloop attributes**: import from `opentelemetry.semconv_ai.SpanAttributes` — never redefine as local constants
- **Respan-specific attributes** (`respan.*`): define in `respan_sdk/constants/span_attributes.py`
- **Log types**: use `respan_sdk/constants/llm_logging.py`
- **OTLP wire format keys**: use `respan_sdk/constants/otlp_constants.py`

### Shared utilities (Python `respan_sdk/utils/`)
- `data_processing/id_processing.py` — `str_to_int()`, `ensure_trace_id()`, `ensure_span_id()` (non-hex ID conversion with MD5 fallback)
- `serialization.py` — `serialize_value()` (recursive Pydantic/dict/datetime to JSON-safe types)
- `time.py` — time utilities
- `pre_processing.py` — data preprocessing
