# Respan Haystack Integration

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-exporter-haystack/)**

Respan integration for Haystack pipelines with tracing and logging support.

## Configuration

### 1. Install

```bash
pip install respan-exporter-haystack
```

### 2. Set Environment Variables

```bash
export RESPAN_API_KEY="your-respan-key"
export OPENAI_API_KEY="your-openai-key"
export HAYSTACK_CONTENT_TRACING_ENABLED="true"  # For tracing mode
```

## Quickstart

### 3. Run Script

**Gateway Mode (Auto-Logging):** Just replace `OpenAIGenerator` with `RespanGenerator`:

```python
import os
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from respan_exporter_haystack import RespanGenerator

# Create pipeline
pipeline = Pipeline()
pipeline.add_component("prompt", PromptBuilder(template="Tell me about {{topic}}."))
pipeline.add_component("llm", RespanGenerator(
    model="gpt-4o-mini",
    api_key=os.getenv("RESPAN_API_KEY")
))
pipeline.connect("prompt", "llm")

# Run
result = pipeline.run({"prompt": {"topic": "machine learning"}})
print(result["llm"]["replies"][0])
```

### 4. View Dashboard

All logs and traces appear in your Respan dashboard:

**Dashboard:** https://platform.respan.ai/logs

- **Logs view:** Individual LLM calls
- **Traces view:** Full pipeline workflows with tree visualization

## Further Reading

### Features

#### Gateway Mode
Route LLM calls through Respan gateway:
- Automatic logging (zero config)
- Model fallbacks & retries
- Load balancing
- Cost optimization
- Rate limiting & caching

#### Tracing Mode
Capture full workflow execution:
- Multi-component pipelines
- Parent-child span relationships
- Timing per component
- Input/output tracking
- RAG + Agent workflows

#### Combined Mode (Recommended)
Use both together for:
- Gateway reliability + Tracing visibility
- Production-ready monitoring

### Prompt Management

**Use platform-managed prompts** for centralized control:

```python
import os
from haystack import Pipeline
from respan_exporter_haystack import RespanGenerator

# Create prompt on platform: https://platform.respan.ai/platform/prompts
# Get your prompt_id from the platform

# Create pipeline with platform prompt (model config comes from platform)
pipeline = Pipeline()
pipeline.add_component("llm", RespanGenerator(
    prompt_id="1210b368ce2f4e5599d307bc591d9b7a",  # Your prompt ID
    api_key=os.getenv("RESPAN_API_KEY")
))

# Run with prompt variables
result = pipeline.run({
    "llm": {
        "prompt_variables": {
            "user_input": "The cat sat on the mat"
        }
    }
})

print("Response received successfully!")
print(f"Model: {result['llm']['meta'][0]['model']}")
print(f"Tokens: {result['llm']['meta'][0]['usage']['total_tokens']}")
```

**Benefits:**
- Update prompts without code changes
- Model config managed on platform (no hardcoding)
- Version control & rollback
- A/B testing
- Team collaboration

**RespanChatGenerator** supports the same pattern: pass `prompt_id` when creating the component, then use `prompt_variables` in `run()` to fill template variables. Example:

```python
from haystack.dataclasses import ChatMessage
from respan_exporter_haystack import RespanChatGenerator

generator = RespanChatGenerator(
    prompt_id="1210b368ce2f4e5599d307bc591d9b7a",
    api_key=os.getenv("RESPAN_API_KEY")
)
result = generator.run(prompt_variables={"user_input": "Hello"})
# Or with messages: generator.run(messages=[...], prompt_variables={...})
```

**See:** [`examples/prompt_example.py`](examples/prompt_example.py)

### Tracing Mode (Workflow Monitoring)

**Add `RespanConnector` to capture the entire pipeline:**

```python
import os
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from respan_exporter_haystack import RespanConnector

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

# Create pipeline with tracing
pipeline = Pipeline()
pipeline.add_component("tracer", RespanConnector("My Workflow"))
pipeline.add_component("prompt", PromptBuilder(template="Tell me about {{topic}}."))
pipeline.add_component("llm", OpenAIGenerator(model="gpt-4o-mini"))
pipeline.connect("prompt", "llm")

# Run
result = pipeline.run({"prompt": {"topic": "artificial intelligence"}})
print(result["llm"]["replies"][0])
print(f"\nTrace URL: {result['tracer']['trace_url']}")

```

**Dashboard shows:**
- Pipeline (root span)
- PromptBuilder (template processing)
- LLM (generation with tokens + cost)

**See:** [`examples/tracing_example.py`](examples/tracing_example.py)

### Combined Mode (Recommended for Production)

**Use BOTH gateway + prompt + tracing for the full stack:**

```python
import os
from haystack import Pipeline
from respan_exporter_haystack import RespanConnector, RespanGenerator

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

# Create pipeline with gateway, prompt management, and tracing
pipeline = Pipeline()
pipeline.add_component("tracer", RespanConnector("Full Stack: Gateway + Prompt + Tracing"))
pipeline.add_component("llm", RespanGenerator(
    prompt_id="1210b368ce2f4e5599d307bc591d9b7a",  # Platform-managed prompt
    api_key=os.getenv("RESPAN_API_KEY")
))

# Run with prompt variables
result = pipeline.run({
    "llm": {
        "prompt_variables": {
            "user_input": "She sells seashells by the seashore"
        }
    }
})

print("Response received successfully!")
print(f"Trace URL: {result['tracer']['trace_url']}")
```

**You get:**
1. **Gateway routing** with fallbacks, cost tracking, and reliability
2. **Platform prompts** managed centrally (no hardcoded prompts/models)
3. **Full workflow trace** with all components and timing

**See:** [`examples/combined_example.py`](examples/combined_example.py)

### What Gets Logged

#### Gateway Mode
- Model used
- Prompt & completion
- Tokens & cost
- Latency
- Request metadata

#### Tracing Mode
Each span includes:
- Component name & type
- Input data
- Output data
- Timing (latency)
- Parent-child relationships

For LLM spans, additionally:
- Model name
- Token counts
- Calculated cost (auto-computed)

### API Reference

**Important:** There is **no default model**. You must pass either `model` or `prompt_id`. Code that relied on a previous default (e.g. `gpt-3.5-turbo`) will raise `ValueError: Either 'model' or 'prompt_id' must be provided` — pass `model` explicitly or use `prompt_id` for platform-managed prompts.

#### `RespanGenerator`

Gateway component for LLM calls (text completion / non-chat).

```python
RespanGenerator(
    model: Optional[str] = None,         # Model name (e.g., "gpt-4o-mini"); required unless prompt_id is set
    api_key: Optional[str] = None,      # Respan API key (defaults to RESPAN_API_KEY env var)
    base_url: Optional[str] = None,      # API base URL (defaults to https://api.respan.ai)
    prompt_id: Optional[str] = None,     # Platform prompt ID; required unless model is set
    generation_kwargs: Optional[Dict] = None,
    timeout: float = 60.0,
)
```

**Replaces:** `OpenAIGenerator` with gateway routing.

**Note:** Either `model` or `prompt_id` is required. When using `prompt_id`, model config comes from the platform — no need to specify `model`.

**Migration:** `streaming_callback` has been removed from the constructor. Passing it will raise a `TypeError`. Remove any `streaming_callback` argument when upgrading.

**Serialization:** The API key is never written when saving pipelines (e.g. `to_dict`); it is resolved from `RESPAN_API_KEY` when the pipeline is loaded.

#### `RespanChatGenerator`

Chat-oriented gateway component with `ChatMessage` support. Same constructor contract as `RespanGenerator`: **no default model** — you must pass either `model` or `prompt_id`. Supports **prompt management** via `prompt_id` (see [Prompt Management](#prompt-management)).

```python
RespanChatGenerator(
    model: Optional[str] = None,         # Model name (e.g., "gpt-4o-mini"); required unless prompt_id is set
    api_key: Optional[str] = None,      # Respan API key (defaults to RESPAN_API_KEY env var)
    base_url: Optional[str] = None,      # API base URL (defaults to https://api.respan.ai)
    prompt_id: Optional[str] = None,     # Platform prompt ID; required unless model is set
    generation_kwargs: Optional[Dict] = None,
    timeout: float = 60.0,
)
```

**run(messages=..., generation_kwargs=..., prompt_variables=...)**  
Returns `{"replies": List[ChatMessage], "meta": List[Dict]}`.

| Parameter | Description |
|-----------|-------------|
| `messages` | List of `ChatMessage` objects. Optional if using `prompt_id` (then use `prompt_variables` instead). |
| `generation_kwargs` | Optional overrides (e.g. `temperature`, `max_tokens`). |
| `prompt_variables` | Variables for platform-managed prompt. Requires `prompt_id` set at init. Pass at run time to fill the prompt template. |

**Replaces:** Chat-style usage of `OpenAIGenerator` with gateway routing.

**Note:** Either `model` or `prompt_id` is required. If you previously relied on a default model (e.g. `gpt-3.5-turbo`), pass `model="gpt-3.5-turbo"` (or another model) or use `prompt_id` for platform-managed prompts.

**Serialization:** The API key is never written when saving pipelines; it is resolved from `RESPAN_API_KEY` when the pipeline is loaded.

#### `RespanConnector`

Tracing component for workflow monitoring.

```python
RespanConnector(
    name: str,                           # Pipeline name for dashboard
    mode: str = "tracing",                # "tracing" (default) or "gateway"
    api_key: Optional[str] = None,       # Respan API key (defaults to RESPAN_API_KEY env var)
    base_url: Optional[str] = None,      # API base URL (defaults to https://api.respan.ai)
    metadata: Optional[Dict] = None,      # Custom metadata for all spans
    max_retries: int = 3,                 # Max retries for sending traces
    base_delay: float = 1.0,              # Base delay in seconds between retries
    max_delay: float = 30.0,              # Max delay in seconds between retries
    platform_url: Optional[str] = None,  # URL for logs UI (defaults derived from base_url)
    timeout: float = 10.0,               # Request timeout in seconds
)
```

**Returns:** `{"name": str, "trace_url": str}`

**Requires:** `HAYSTACK_CONTENT_TRACING_ENABLED=true` environment variable (when `mode="tracing"`)

**Serialization:** The API key is never written when saving pipelines; it is resolved from `RESPAN_API_KEY` when the pipeline is loaded.

### Examples

Run the examples:

```bash
# Set environment variables
export RESPAN_API_KEY="your-key"
export OPENAI_API_KEY="your-openai-key"
export HAYSTACK_CONTENT_TRACING_ENABLED="true"

# Gateway mode (auto-logging)
python examples/gateway_example.py

# Tracing mode (workflow monitoring)
python examples/tracing_example.py

# Prompt management (platform prompts)
python examples/prompt_example.py

# Combined mode (gateway + prompt + tracing)
python examples/combined_example.py
```

### Breaking changes

- **`RespanChatGenerator` / `RespanGenerator`: no default model.** Either `model` or `prompt_id` must be provided. If you previously relied on a default (e.g. `gpt-3.5-turbo`), pass `model` explicitly or use `prompt_id` for platform-managed prompts.
- **`RespanGenerator`: `streaming_callback` removed.** The constructor no longer accepts `streaming_callback`. Passing it will raise a `TypeError`. Remove the argument when upgrading.

See [CHANGELOG.md](CHANGELOG.md) for full details.

### Requirements

- Python 3.10+
- `haystack-ai >= 2.24.1`
- `requests >= 2.32.5`
- `respan-sdk >= 2.3.1`

### Support

- **Documentation:** https://docs.respan.ai/
- **Dashboard:** https://platform.respan.ai/
- **Issues:** [GitHub Issues](https://github.com/Repsan/respan/issues)

### License

MIT License - see [LICENSE](LICENSE) file for details.