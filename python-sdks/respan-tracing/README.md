# Building an LLM Workflow with Respan Tracing

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-tracing/)**

This tutorial demonstrates how to build and trace complex LLM workflows using Respan Tracing. We'll create an example that generates jokes, translates them to pirate language, and simulates audience reactions - all while capturing detailed telemetry of our LLM calls.

## Prerequisites

- Python 3.7+
- OpenAI API key
- Anthropic API key
- Respan API key, you can get your API key from the [API keys page](https://platform.respan.co/platform/api/api-keys)

## Installation
```bash
pip install respan-tracing openai anthropic
```

## Initialization

### RespanTelemetry Configuration

The `RespanTelemetry` class is the main entry point for the SDK. Initialize it once at application startup:

```python
from respan_tracing import RespanTelemetry

telemetry = RespanTelemetry(
    app_name="my-app",
    api_key="respan-xxx",  # Or set RESPAN_API_KEY env var
)
```

### All Initialization Parameters

```python
RespanTelemetry(
    # Basic Configuration
    app_name: str = "respan",              # Application name for telemetry
    api_key: Optional[str] = None,             # API key (or RESPAN_API_KEY env var)
    base_url: Optional[str] = None,            # API URL (or RESPAN_BASE_URL env var)
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    
    # Performance
    is_batching_enabled: Optional[bool] = None,  # Enable background batch processing (default: True)
    
    # Instrumentation (see Instrumentation section below)
    instruments: Optional[Set[Instruments]] = None,        # Specific instruments to enable
    block_instruments: Optional[Set[Instruments]] = None,  # Instruments to disable
    
    # Advanced
    headers: Optional[Dict[str, str]] = None,              # Additional HTTP headers
    resource_attributes: Optional[Dict[str, str]] = None,  # Resource attributes
    span_postprocess_callback: Optional[Callable] = None,  # Span processing callback
    is_enabled: bool = True,                               # Enable/disable telemetry
)
```

### Parameter Reference

#### **Basic Configuration**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `app_name` | `str` | `"respan"` | Name of your application for telemetry identification |
| `api_key` | `str \| None` | `None` | Respan API key. Can also be set via `RESPAN_API_KEY` environment variable |
| `base_url` | `str \| None` | `None` | Respan API base URL. Can also be set via `RESPAN_BASE_URL` environment variable. Defaults to `https://api.respan.co/api` |

#### **Logging**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | `str` | `"INFO"` | Logging level for Respan tracing. Options: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`. Can also be set via `RESPAN_LOG_LEVEL` environment variable |

#### **Performance**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_batching_enabled` | `bool \| None` | `None` | Enable batch span processing. When `False`, uses synchronous export (no background threads). Defaults to `True`. Useful to disable for debugging or backends with custom exporters. Can also be set via `RESPAN_BATCHING_ENABLED` environment variable |

#### **Instrumentation**

See [Instrumentation section](#instrumentation) for detailed information.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruments` | `Set[Instruments] \| None` | `None` | Specific instruments to enable. If `None`, enables all available instruments. Use empty set `set()` to disable all auto-instrumentation |
| `block_instruments` | `Set[Instruments] \| None` | `None` | Instruments to explicitly disable. Use this to block specific instrumentations while enabling others |

**Examples:**

```python
# Enable only specific instruments
from respan_tracing import Instruments

telemetry = RespanTelemetry(
    instruments={Instruments.OPENAI, Instruments.ANTHROPIC}
)

# Block specific instruments
telemetry = RespanTelemetry(
    block_instruments={Instruments.REQUESTS, Instruments.URLLIB3}
)

# Disable all auto-instrumentation
telemetry = RespanTelemetry(
    instruments=set()  # Empty set = no auto-instrumentation
)
```

#### **Advanced**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `headers` | `Dict[str, str] \| None` | `None` | Additional HTTP headers to send with telemetry data |
| `resource_attributes` | `Dict[str, str] \| None` | `None` | Additional resource attributes to attach to all spans. Useful for adding environment, version, etc. |
| `span_postprocess_callback` | `Callable \| None` | `None` | Optional callback function to process spans before export. Signature: `callback(span: ReadableSpan) -> None` |
| `is_enabled` | `bool` | `True` | Enable or disable telemetry. When `False`, becomes a no-op (no spans created) |

**Note:** Threading instrumentation is ALWAYS enabled by default (even when specifying custom instruments) because it's critical for context propagation. To disable it explicitly, use `block_instruments={Instruments.THREADING}`. See [Threading Instrumentation](#important-threading-instrumentation) for details.

### Default Behavior

**A default Respan processor is automatically added when you provide an `api_key`:**

```python
from respan_tracing import RespanTelemetry

# Minimal initialization - spans automatically go to Respan!
kai = RespanTelemetry(
    app_name="my-app",
    api_key="your-key"  # ← Default processor added automatically
)

# Now all spans are automatically exported to Respan
@kai.task()
def my_task():
    pass  # This span will be exported!
```

**This ensures backward compatibility** - existing code continues to work without changes.

### Multiple Exporters (Advanced)

For advanced use cases, you can route spans to **multiple destinations** using the `add_processor()` method:

```python
from respan_tracing import RespanTelemetry

# Initialize telemetry (default processor added automatically)
kai = RespanTelemetry(app_name="my-app", api_key="your-key")

# Add additional processors for specific routing
kai.add_processor(
    exporter=FileExporter("./debug.json"),
    name="debug"  # Automatically filters for processors="debug"
)

# Use decorators to route spans
@kai.task(name="normal_task")  # Goes to default Respan processor
def normal_task():
    pass

@kai.task(name="debug_task", processors="debug")  # Only to debug processor
def debug_task():
    pass

@kai.task(name="multi_task", processors=["debug", "analytics"])  # Routes to multiple
def multi_task():
    pass
```

**Key Features:**
- ✅ **Automatic filtering**: Just provide `name` parameter - filter is auto-created!
- ✅ **Single or multiple**: `processors="debug"` or `processors=["debug", "analytics"]`
- ✅ **Custom filters**: Override with `filter_fn` for advanced logic

See [Multi-Processor Examples](#multiple-processors) for complete examples.

### Common Configuration Patterns

#### **Development (Full Visibility)**

```python
telemetry = RespanTelemetry(
    app_name="my-app-dev",
    api_key="respan-xxx",
    log_level="DEBUG",  # Verbose logging
    # All instruments enabled by default
)
```

#### **Production (Optimized)**

```python
telemetry = RespanTelemetry(
    app_name="my-app-prod",
    api_key="respan-xxx",
    log_level="WARNING",  # Less verbose
    block_instruments={
        Instruments.REQUESTS,  # Reduce noise
        Instruments.URLLIB3,
    }
)
```

#### **Backend with Custom Processor (Minimal)**

```python
from your_exporters import DirectLoggingExporter

telemetry = RespanTelemetry(
    app_name="my-backend",
    is_batching_enabled=False,  # No background threads
    instruments=set(),  # No auto-instrumentation
    block_instruments={Instruments.THREADING},  # Disable threading if single-threaded
)

# Add custom processor after initialization
telemetry.tracer.add_processor(
    exporter=DirectLoggingExporter(),
    name="logging"
)
```

#### **Testing/Disabled**

```python
telemetry = RespanTelemetry(
    app_name="my-app-test",
    is_enabled=False,  # Completely disabled (no-op)
)
```

### Environment Variables

You can configure Respan tracing using environment variables:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `RESPAN_API_KEY` | API key | None |
| `RESPAN_BASE_URL` | API base URL | `https://api.respan.co/api` |
| `RESPAN_LOG_LEVEL` | Logging level | `INFO` |
| `RESPAN_BATCHING_ENABLED` | Enable batch processing | `True` |

**Example:**

```bash
export RESPAN_API_KEY="respan-xxx"
export RESPAN_LOG_LEVEL="DEBUG"
export RESPAN_BATCHING_ENABLED="false"
```

```python
# No need to pass parameters - read from env vars
telemetry = RespanTelemetry(app_name="my-app")
```

## Tutorial

### Step 1: Initialization
```
import os
from respan_tracing.main import RespanTelemetry
from respan_tracing.decorators import workflow, task
import time

# Initialize Respan Telemetry
os.environ["RESPAN_API_KEY"] = "YOUR_RESPAN_API_KEY"
k_tl = RespanTelemetry()

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI()
```


### Step 2: First Draft - Basic Workflow
We'll start by creating a simple workflow that generates a joke, translates it to pirate speak,
and adds a signature. This demonstrates the basic usage of tasks and workflows.

- A task is a single unit of work, decorated with `@task`
- A workflow is a collection of tasks, decorated with `@workflow`
- Tasks can be used independently or as part of workflows

```python
@task(name="joke_creation")
def create_joke():
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        temperature=0.5,
        max_tokens=100,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["\n"],
        logprobs=True,
    )
    return completion.choices[0].message.content

@task(name="signature_generation")
def generate_signature(joke: str):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "add a signature to the joke:\n\n" + joke}
        ],
    )
    return completion.choices[0].message.content

@task(name="pirate_joke_translation")
def translate_joke_to_pirate(joke: str):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "translate the joke to pirate language:\n\n" + joke,
            }
        ],
    )
    return completion.choices[0].message.content

@workflow(name="pirate_joke_generator")
def joke_workflow():
    eng_joke = create_joke()
    pirate_joke = translate_joke_to_pirate(eng_joke)
    signature = generate_signature(pirate_joke)
    return pirate_joke + signature

if __name__ == "__main__":
    joke_workflow()
```

Run the workflow and see the trace in Respan `Traces` tab.

### Step 3: Adding Another Workflow
Let's add audience reactions to make our workflow more complex and demonstrate
what multiple workflow traces look like.

```python
@task(name="audience_laughs")
def audience_laughs(joke: str):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "This joke:\n\n" + joke + " is funny, say hahahahaha",
            }
        ],
        max_tokens=10,
    )
    return completion.choices[0].message.content

@task(name="audience_claps")
def audience_claps():
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Clap once"}],
        max_tokens=5,
    )
    return completion.choices[0].message.content

@task(name="audience_applaud")
def audience_applaud(joke: str):
    clap = audience_claps()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Applaud to the joke, clap clap! " + clap,
            }
        ],
        max_tokens=10,
    )
    return completion.choices[0].message.content

@workflow(name="audience_reaction")
def audience_reaction(joke: str):
    laughter = audience_laughs(joke=joke)
    applauds = audience_applaud(joke=joke)
    return laughter + applauds


@workflow(name="joke_and_audience_reaction") #<--------- Create the new workflow that combines both workflows together
def joke_and_audience_reaction():
    pirate_joke = joke_workflow()
    reactions = audience_reaction(pirate_joke)
```

Don't forget to update the entrypoint!
```python
if __name__ == "__main__":
    joke_and_audience_reaction() # <--------- Update the entrypoint here
```

Run the workflow again and see the trace in Respan `Traces` tab, notice the new span for the `audience_reaction` workflow in parallel with the `joke_workflow`. Congratulation! You have created a trace with multiple workflows.

### Step 4: Adding Vector Storage Capability
To demonstrate how to integrate with vector databases and embeddings,
we'll add a store_joke task that generates embeddings for our jokes.

```python
@task(name="store_joke")
def store_joke(joke: str):
    """Simulate storing a joke in a vector database."""
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=joke,
    )
    return embedding.data[0].embedding
```

Update create_joke to use store_joke
```python
@task(name="joke_creation")
def create_joke():
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        temperature=0.5,
        max_tokens=100,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["\n"],
        logprobs=True,
    )
    joke = completion.choices[0].message.content
    store_joke(joke)  # <--------- Add the task here
    return joke
```
Run the workflow again and see the trace in Respan `Traces` tab, notice the new span for the `store_joke` task.

Expanding the `store_joke` task, you can see the embeddings call is recognized as `openai.embeddings`.

### Step 5: Adding Arbitrary Function Calls
Demonstrate how to trace non-LLM functions by adding a logging task.

```python
@task(name="logging_joke")
def logging_joke(joke: str, reactions: str):
    """Simulates logging the process into a database."""
    print(joke + "\n\n" + reactions)
    time.sleep(1)
```

Update `joke_and_audience_reaction`
```python
@workflow(name="joke_and_audience_reaction")
def joke_and_audience_reaction():
    pirate_joke = joke_workflow()
    reactions = audience_reaction(pirate_joke)
    logging_joke(pirate_joke, reactions) # <-------- Add this workflow here
```

Run the workflow again and see the trace in Respan `Traces` tab, notice the new span for the `logging_joke` task.

This is a simple example of how to trace arbitrary functions. You can see the all the inputs and outputs of `logging_joke` task.

### Step 6: Adding Different LLM Provider (Anthropic)

Demonstrate compatibility with multiple LLM providers by adding Anthropic integration.

```python
from anthropic import Anthropic
anthropic = Anthropic()

@task(name="ask_for_comments")
def ask_for_comments(joke: str):
    completion = anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": f"What do you think about this joke: {joke}"}],
        max_tokens=100,
    )
    return completion.content[0].text

@task(name="read_joke_comments")
def read_joke_comments(comments: str):
    return f"Here is the comment from the audience: {comments}"

@workflow(name="audience_interaction")
def audience_interaction(joke: str):
    comments = ask_for_comments(joke=joke)
    read_joke_comments(comments=comments)
```

Update `joke_and_audience_reaction`
```python
@workflow(name="joke_and_audience_reaction")
def joke_and_audience_reaction():
    pirate_joke = joke_workflow()
    reactions = audience_reaction(pirate_joke)
    audience_interaction(pirate_joke) # <-------- Add this workflow here
    logging_joke(pirate_joke, reactions)
```

Running the workflow for one last time, you can see that the new `audience_interaction` can recognize the `anthropic.completion` calls.

## Instrumentation

### What is Instrumentation?

Instrumentation is the process of automatically adding telemetry (traces/spans) to library calls without modifying your code. When you enable instrumentation for a library (like OpenAI, Anthropic, LangChain), the SDK automatically captures:

- LLM requests and responses
- Model parameters (temperature, max_tokens, etc.)
- Token usage and costs
- Latency and timing
- Errors and exceptions

### Default Behavior: All Instrumentations Enabled

**By default, Respan tracing attempts to enable ALL available instrumentations.**

```python
from respan_tracing import RespanTelemetry

# This enables ALL available instrumentations (if packages are installed)
telemetry = RespanTelemetry(
    app_name="my-app",
    api_key="respan-xxx"
)
```

If a library is installed in your environment, its instrumentation will be automatically enabled. If not installed, it's silently skipped (no errors).

### Available Instrumentations

The SDK supports instrumentation for:

**AI/ML Libraries:**
- `openai` - OpenAI API
- `anthropic` - Anthropic (Claude) API
- `cohere` - Cohere API
- `mistral` - Mistral AI
- `ollama` - Ollama (local models)
- `groq` - Groq API
- `together` - Together AI
- `replicate` - Replicate
- `transformers` - Hugging Face Transformers

**Cloud AI Services:**
- `bedrock` - AWS Bedrock
- `sagemaker` - AWS SageMaker
- `vertexai` - Google Vertex AI
- `google_generativeai` - Google AI (Gemini)
- `watsonx` - IBM WatsonX
- `alephalpha` - Aleph Alpha

**Vector Databases:**
- `pinecone` - Pinecone
- `qdrant` - Qdrant
- `chroma` - Chroma
- `milvus` - Milvus
- `weaviate` - Weaviate
- `lancedb` - LanceDB
- `marqo` - Marqo

**Frameworks:**
- `langchain` - LangChain
- `llama_index` - LlamaIndex
- `haystack` - Haystack
- `crew` - CrewAI
- `mcp` - Model Context Protocol

**Infrastructure:**
- `redis` - Redis
- `requests` - HTTP requests library
- `urllib3` - urllib3 HTTP client
- `pymysql` - PyMySQL database client
- `threading` - Context propagation across threads (⚠️ **Always enabled by default**)

### Installing Instrumentation Packages

**Important:** To trace a specific library, you need to install the corresponding OpenTelemetry instrumentation package.

The SDK uses the **OpenTelemetry standard** instrumentation packages:

```bash
# Install instrumentation for the libraries you use
pip install opentelemetry-instrumentation-openai
pip install opentelemetry-instrumentation-anthropic
pip install opentelemetry-instrumentation-langchain
pip install opentelemetry-instrumentation-requests
```

**Naming convention:** `opentelemetry-instrumentation-<library-name>`

#### Example: Tracing OpenAI Calls

```bash
# 1. Install OpenAI client
pip install openai

# 2. Install OpenTelemetry instrumentation for OpenAI
pip install opentelemetry-instrumentation-openai

# 3. Install Respan tracing
pip install respan-tracing
```

```python
from respan_tracing import RespanTelemetry
from openai import OpenAI

# Initialize telemetry (OpenAI instrumentation auto-enabled)
telemetry = RespanTelemetry(
    app_name="my-app",
    api_key="respan-xxx"
)

client = OpenAI()

# This call is automatically traced!
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

The OpenAI call will automatically create a span with:
- Model name
- Prompt and completion
- Token usage
- Latency
- Cost (if available)

### Important: Threading Instrumentation

**Threading instrumentation is automatically enabled** (even when you specify custom instruments) because it's critical for context propagation across threads.

```python
# These all include threading by default:
telemetry = RespanTelemetry()  # All instruments including threading

telemetry = RespanTelemetry(
    instruments={Instruments.OPENAI}  # OpenAI + Threading (auto-added!)
)

telemetry = RespanTelemetry(
    instruments={Instruments.OPENAI, Instruments.ANTHROPIC}  # + Threading!
)
```

**To disable threading** (if you're certain your app is single-threaded):

```python
telemetry = RespanTelemetry(
    block_instruments={Instruments.THREADING}  # Explicitly disabled
)
```

### Controlling Instrumentation

#### Option 1: Disable Specific Instruments

Use `block_instruments` to disable specific instrumentations you don't want:

```python
from respan_tracing import RespanTelemetry, Instruments

telemetry = RespanTelemetry(
    app_name="my-app",
    api_key="respan-xxx",
    block_instruments={
        Instruments.REQUESTS,  # Don't trace HTTP requests
        Instruments.URLLIB3,   # Don't trace urllib3
        Instruments.REDIS,     # Don't trace Redis calls
    }
)
```

**Use case:** Reduce noise by blocking low-level HTTP instrumentations when you only care about high-level LLM calls.

#### Option 2: Enable Only Specific Instruments

Use `instruments` to enable only the instrumentations you want:

```python
from respan_tracing import RespanTelemetry, Instruments

telemetry = RespanTelemetry(
    app_name="my-app",
    api_key="respan-xxx",
    instruments={
        Instruments.OPENAI,      # Only trace OpenAI
        Instruments.ANTHROPIC,   # Only trace Anthropic
    }
)
```

**Use case:** Maximum performance and minimal noise - only instrument what you need.

#### Option 3: Disable All Instrumentation

Pass an empty set to disable all automatic instrumentation:

```python
from respan_tracing import RespanTelemetry

telemetry = RespanTelemetry(
    app_name="my-app",
    api_key="respan-xxx",
    instruments=set(),  # No auto-instrumentation
)
```

**Use case:** 
- Backend systems that only need `@workflow`/`@task` decorators
- Custom manual instrumentation only
- Minimal overhead

Even with `instruments=set()`, you can still:
- ✅ Use `@workflow` and `@task` decorators
- ✅ Create manual spans with `tracer.start_as_current_span()`
- ✅ Use `get_client()` to update spans
- ❌ Won't automatically trace OpenAI/Anthropic/etc calls

### Instrumentation Best Practices

#### 1. Install Only What You Need

```bash
# Bad: Installing all instrumentation packages (bloats dependencies)
pip install opentelemetry-instrumentation-openai \
            opentelemetry-instrumentation-anthropic \
            opentelemetry-instrumentation-langchain \
            # ... 30+ packages

# Good: Install only what you use
pip install opentelemetry-instrumentation-openai  # You use OpenAI
pip install opentelemetry-instrumentation-langchain  # You use LangChain
```

#### 2. Reduce Noise in Production

```python
# Development: Enable everything for visibility
dev_telemetry = RespanTelemetry(
    app_name="my-app-dev"
)

# Production: Block low-level instrumentations to reduce noise
prod_telemetry = RespanTelemetry(
    app_name="my-app-prod",
    block_instruments={
        Instruments.REQUESTS,
        Instruments.URLLIB3,
        Instruments.REDIS,
    }
)
```

#### 3. Backend Services: Minimal Instrumentation

```python
# Backend API: Disable auto-instrumentation, use decorators only
backend_telemetry = RespanTelemetry(
    app_name="backend-api",
    instruments=set(),  # No auto-instrumentation
    is_batching_enabled=False,  # No background threads
)

# Manual instrumentation only
@workflow(name="api_endpoint")
def api_endpoint(data):
    # Your logic here
    pass
```

### Troubleshooting Instrumentation

#### Issue: "My OpenAI calls aren't being traced"

**Solution:** Install the instrumentation package:

```bash
pip install opentelemetry-instrumentation-openai
```

Verify it's working:

```python
from respan_tracing import RespanTelemetry

telemetry = RespanTelemetry(log_level="DEBUG")
# Check logs for: "Initialized OpenAI instrumentation"
```

#### Issue: "Too many spans are being created (noise)"

**Solution:** Block unnecessary instrumentations:

```python
from respan_tracing import Instruments

telemetry = RespanTelemetry(
    block_instruments={
        Instruments.REQUESTS,  # Block HTTP client spans
        Instruments.URLLIB3,   # Block urllib3 spans
    }
)
```

#### Issue: "Performance overhead in my backend"

**Solution:** Disable auto-instrumentation entirely:

```python
telemetry = RespanTelemetry(
    instruments=set(),  # No auto-instrumentation
    is_batching_enabled=False,  # No background threads
)
```

### Summary

| Configuration | Auto-Instrumentation | Use Case |
|--------------|---------------------|----------|
| Default | All available | Development, full visibility |
| `block_instruments={...}` | All except blocked | Production, reduce noise |
| `instruments={...}` | Only specified | Targeted tracing |
| `instruments=set()` | None | Backend, minimal overhead |

**Key Points:**
- ✅ Default: All instrumentations enabled (if packages installed)
- ✅ Install: `opentelemetry-instrumentation-<library>` for each library
- ✅ Control: Use `instruments` or `block_instruments` parameters
- ✅ Performance: Disable unused instrumentations to reduce overhead
- ✅ Flexibility: Can disable all auto-instrumentation and use decorators only

## Advanced Features

### Multiple Processors (Routing Spans)

Route spans to different destinations by adding multiple processors. This is the **standard OpenTelemetry pattern** for multi-destination export.

#### Quick Example

```python
from respan_tracing import RespanTelemetry
from respan_tracing.exporters import RespanSpanExporter
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

# File exporter example
class FileExporter(SpanExporter):
    def export(self, spans):
        with open("spans.json", "a") as f:
            for span in spans:
                f.write(f"{span.name}\n")
        return SpanExportResult.SUCCESS
    def shutdown(self): pass
    def force_flush(self, timeout_millis=30000): return True

# Initialize telemetry
kai = RespanTelemetry(app_name="my-app", api_key="your-key")

# Add production processor (all spans)
kai.add_processor(
    exporter=RespanSpanExporter(
        endpoint="https://api.respan.co/api",
        api_key="prod-key"
    ),
    name="production"
)

# Add debug processor (only debug spans)
kai.add_processor(
    exporter=FileExporter(),
    name="debug"  # ← Automatically filters for processors="debug"
)

# Usage in decorators
@kai.task(processors="debug")  # → Goes to debug processor only
def debug_task():
    pass

@kai.task(processors=["production", "debug"])  # → Goes to both!
def important_task():
    pass
```

**Key Features:**
- ✅ **Automatic filtering**: Just provide `name` - filter is created automatically
- ✅ **Single or multiple**: `processors="debug"` or `processors=["debug", "analytics"]`
- ✅ **Standard OTEL**: Uses OpenTelemetry's native multi-processor support

**Use cases:**
- Send critical spans to production, debug spans to local file
- Route spans to multiple analytics systems
- Separate logging for different environments (dev/staging/prod)
- A/B testing with different monitoring backends

#### Advanced: Custom Filter Functions

For complex routing logic beyond simple name matching, provide a custom `filter_fn`.

**Important:** When both `name` and `filter_fn` are provided, **BOTH conditions must be True**:
- The span must have the processor name in its `processors` attribute (from decorator)
- The custom `filter_fn` must return `True`

This ensures decorator-based routing always works, while allowing additional filtering logic.

```python
# Example 1: Filter slow spans within "debug" processor
kai.add_processor(
    exporter=SlowDebugExporter(),
    name="debug",  # ← Must have processors="debug" in decorator
    filter_fn=lambda span: (span.end_time - span.start_time) > 1_000_000_000  # AND must be slow
)
# Only receives spans with: @task(processors="debug") AND duration > 1s

# Example 2: Filter by environment attribute
kai.add_processor(
    exporter=StagingExporter(),
    name="staging",
    filter_fn=lambda span: span.attributes.get("env") == "staging"
)
# Only receives spans with: @task(processors="staging") AND env="staging"

# Example 3: No name = only custom filter applies (all spans checked)
kai.add_processor(
    exporter=ErrorExporter(),
    filter_fn=lambda span: "error" in span.name.lower()  # No name = checks all spans
)
# Receives any span where name contains "error"
```

#### Initialization Pattern

**Initialize once at startup**, then add processors as needed:

```python
from contextvars import ContextVar
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from respan_tracing import RespanTelemetry, get_client

# Thread-safe context storage
_request_context: ContextVar[dict] = ContextVar('request_context', default={})

def set_request_context(**kwargs):
    """Set context for current request (thread-safe)."""
    _request_context.set(kwargs)

def get_request_context():
    """Get context for current request."""
    return _request_context.get()

def clear_request_context():
    """Clear context after request."""
    _request_context.set({})

class BackendExporter(SpanExporter):
    """Custom exporter that uses per-request context"""
    
    def export(self, spans):
        # Read per-request context (thread-safe)
        context = get_request_context()
        org_id = context.get('org_id')
        user_id = context.get('user_id')
        experiment_id = context.get('experiment_id')
        
        for span in spans:
            span_data = {
                'name': span.name,
                'attributes': dict(span.attributes),
                # Add context to each span
                'org_id': org_id,
                'user_id': user_id,
                'experiment_id': experiment_id,
            }
            # Send to your internal system
            self._send_to_backend(span_data)
        
        return SpanExportResult.SUCCESS
    
    def _send_to_backend(self, span_data):
        # Your backend logic (database, queue, API, etc.)
        pass
    
    def shutdown(self):
        pass
    
    def force_flush(self, timeout_millis=30000):
        return True

# At startup
exporter = BackendExporter()
telemetry = RespanTelemetry(
    app_name="backend",
    custom_exporter=exporter,
    is_batching_enabled=False,
)

# Per-request usage
def handle_api_request(org_id, user_id, experiment_id):
    # Set context at the start of request
    set_request_context(
        org_id=org_id,
        user_id=user_id,
        experiment_id=experiment_id
    )
    
    try:
        # Use get_client() for span operations
        client = get_client()
        tracer = client.get_tracer()
        
        with tracer.start_as_current_span("api_request"):
            # Your request logic
            # Exporter will read context when exporting spans
            result = process_request()
            return result
    finally:
        # Clean up context after request
        clear_request_context()
```

**Why use `contextvars`?**
- ✅ **Thread-safe**: Each request/thread has isolated context
- ✅ **Async-safe**: Works with asyncio and concurrent execution
- ✅ **Automatic propagation**: Context flows through function calls
- ✅ **Clean separation**: Exporter initialization (startup) separate from request context

See [`examples/custom_exporter_example.py`](examples/custom_exporter_example.py) and [`examples/backend_trace_collection_example.py`](examples/backend_trace_collection_example.py) for complete working examples including:
- File exporter (writing spans to JSON lines)
- Console exporter (printing spans to terminal)
- Backend exporter with per-request context
- Direct logging exporter for immediate export

### Update Span Functionality

You can dynamically update spans while they're running using the `get_client()` API. This is useful for:
- Adding Respan-specific parameters (like `customer_identifier`, `trace_group_identifier`)
- Setting custom attributes during execution
- Adding events to track progress
- Recording exceptions and errors
- Changing span names based on runtime conditions

```python
from respan_tracing import RespanTelemetry, get_client, workflow
from openai import OpenAI

telemetry = RespanTelemetry(
    app_name="my-app",
    api_key="respan-xxx"
)

client = OpenAI()

@workflow(name="data_processing")
def process_data(user_id: str, data: dict):
    # Get the client to interact with the current span
    kwai_client = get_client()
    
    # Get current trace information
    trace_id = kwai_client.get_current_trace_id()
    span_id = kwai_client.get_current_span_id()
    print(f"Processing in trace: {trace_id}")
    
    # Update span with Respan-specific parameters
    kwai_client.update_current_span(
        respan_params={
            "customer_identifier": user_id,
            "trace_group_identifier": "data-processing-pipeline",
            "metadata": {
                "data_size": len(str(data)),
                "processing_type": "batch"
            }
        }
    )
    
    # Add an event to track progress
    kwai_client.add_event("validation_started", {
        "record_count": len(data)
    })
    
    # Add custom attributes
    kwai_client.update_current_span(
        attributes={
            "custom.user_id": user_id,
            "custom.data_type": type(data).__name__
        }
    )
    
    try:
        # Call LLM
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Process: {data}"}]
        )
        
        # Update span name based on result
        kwai_client.update_current_span(
            name="data_processing.success",
            attributes={"result_length": len(response.choices[0].message.content)}
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Record exception in the span
        kwai_client.record_exception(e)
        raise

# Use the workflow
result = process_data("user-123", {"key": "value"})
```

**Available Client Methods:**
- `get_current_trace_id()` - Get the current trace ID
- `get_current_span_id()` - Get the current span ID
- `get_tracer()` - Get the OpenTelemetry tracer for manual span creation
- `update_current_span()` - Update span with params, attributes, name, or status
- `add_event()` - Add an event to the current span
- `record_exception()` - Record an exception on the current span
- `is_recording()` - Check if the current span is recording
- `flush()` - Force flush all pending spans

See [`examples/simple_span_updating_example.py`](examples/simple_span_updating_example.py) for a complete example.

### Manual Span Creation

For fine-grained control, you can manually create custom spans using the tracer directly. This is useful when:
- You need spans that don't fit the `@workflow`/`@task` pattern
- You want to instrument specific code blocks
- You're integrating with existing tracing code
- You need to create spans conditionally

```python
from respan_tracing import RespanTelemetry, get_client

telemetry = RespanTelemetry(
    app_name="my-app",
    api_key="respan-xxx"
)

# Get the tracer instance using the public API
client = get_client()
tracer = client.get_tracer()

# Create a parent span manually
with tracer.start_as_current_span("database_operation") as parent_span:
    parent_span.set_attribute("db.system", "postgresql")
    parent_span.set_attribute("db.operation", "query")
    parent_span.add_event("Connection established")
    
    # Create nested child spans
    with tracer.start_as_current_span("execute_query") as query_span:
        query_span.set_attribute("db.statement", "SELECT * FROM users")
        query_span.set_attribute("db.rows_affected", 42)
        
        # You can still use get_client() within manual spans
        client = get_client()
        client.update_current_span(
            respan_params={
                "customer_identifier": "admin-user"
            }
        )
        
        # Simulate query execution
        result = execute_database_query()
    
    with tracer.start_as_current_span("process_results") as process_span:
        process_span.set_attribute("result.count", len(result))
        processed = process_results(result)
    
    parent_span.add_event("Operation completed", {
        "total_time_ms": 150
    })
```

**Combining Manual Spans with Decorators:**

You can mix manual span creation with decorator-based spans:

```python
from respan_tracing import workflow, task, get_client

@workflow(name="hybrid_workflow")
def hybrid_workflow(data):
    client = get_client()
    tracer = client.get_tracer()
    
    # Use decorator-based task
    validated_data = validate_data(data)
    
    # Create manual span for specific instrumentation
    with tracer.start_as_current_span("custom_processing") as span:
        span.set_attribute("processing.type", "custom")
        span.set_attribute("data.size", len(validated_data))
        
        # Custom processing logic
        for item in validated_data:
            with tracer.start_as_current_span(f"process_item_{item['id']}") as item_span:
                item_span.set_attribute("item.id", item['id'])
                process_single_item(item)
    
    # Use another decorator-based task
    return finalize_results(validated_data)

@task(name="validate_data")
def validate_data(data):
    # Decorator-based span
    return [item for item in data if item.get('valid')]

@task(name="finalize_results")
def finalize_results(data):
    # Decorator-based span
    return {"processed": len(data), "data": data}
```

**Thread-Safe Context Propagation:**

When working with threads, you can manually propagate context:

```python
from opentelemetry import context as otel_context
import threading

@workflow(name="threaded_workflow")
def threaded_workflow():
    client = get_client()
    tracer = client.get_tracer()
    
    # Capture the current context
    current_context = otel_context.get_current()
    
    def worker_function():
        # Attach the captured context in the worker thread
        token = otel_context.attach(current_context)
        
        try:
            # Create spans in the worker thread
            with tracer.start_as_current_span("worker_task") as span:
                span.set_attribute("thread.name", threading.current_thread().name)
                # Your work here
                pass
        finally:
            # Detach the context
            otel_context.detach(token)
    
    # Start worker thread
    thread = threading.Thread(target=worker_function)
    thread.start()
    thread.join()
```

**Key Points:**
- Use `tracer.start_as_current_span(name)` to create custom spans
- Spans automatically nest based on context
- You can mix manual spans with `@workflow` and `@task` decorators
- Use `get_client()` within manual spans to access client API
- Threading instrumentation is enabled by default for automatic context propagation across threads

See [`examples/custom_exporter_example.py`](examples/custom_exporter_example.py) for examples of manual span creation.

### Manual Span Buffering with SpanBuffer

For advanced use cases where you need to **manually control span buffering and export timing**, use the OpenTelemetry-compliant `SpanBuffer` context manager. Unlike the SDK's automatic background batch export, `SpanBuffer` gives you complete control over when spans are created and exported.

**Key Difference:**
- **Normal SDK behavior**: Spans are automatically processed through processors in the background
- **SpanBuffer**: Spans are buffered in a local queue and processed only when you explicitly call `process_spans()`

This is particularly useful for:

- **Asynchronous span creation**: Create spans after execution completes
- **Batch ingestion**: Collect multiple spans and export them with a single API call
- **Manual export control**: Decide when and whether to export collected spans
- **Trace-level batching**: Group all spans for a trace and export together

#### Basic Usage

```python
from respan_tracing import RespanTelemetry, get_client

telemetry = RespanTelemetry(
    app_name="my-app",
    api_key="respan-xxx"
)

client = get_client()

# Collect spans for later export
collected_readable_spans = []

with client.get_span_buffer("trace-123") as buffer:
    # Create multiple spans - they go to local queue, not exported yet
    buffer.create_span("step1", {"status": "completed", "latency": 100})
    buffer.create_span("step2", {"status": "completed", "latency": 200})
    buffer.create_span("step3", {"status": "completed", "latency": 150})
    
    # Extract spans before context exits
    collected_readable_spans = buffer.get_all_spans()

# Process the collected spans through the processor pipeline
success = client.process_spans(collected_readable_spans)
```

#### Transportable Spans Pattern

The key power of `SpanBuffer` is that **spans are transportable** - you can collect them in one context and process them anywhere else in your code:

```python
def collect_workflow_spans():
    """Collect spans during workflow execution"""
    collected_spans = []
    
    with client.get_span_buffer("workflow-123") as buffer:
        buffer.create_span("data_loading", {"records": 1000})
        buffer.create_span("processing", {"duration_ms": 500})
        buffer.create_span("validation", {"errors": 0})
        
        # Extract spans before context exits
        collected_spans = buffer.get_all_spans()
    
    return collected_spans  # Spans persist as list!

def process_based_on_business_logic(spans):
    """Process spans based on business conditions"""
    if all_workflows_successful(spans):
        # Process through standard OTEL pipeline (goes to all configured processors)
        client.process_spans(spans)
    elif should_debug_failures(spans):
        # Process through pipeline (debug processor can filter)
        client.process_spans(spans)
    else:
        # Discard spans (just don't process)
        pass

# Usage: Decouple collection from processing
spans = collect_workflow_spans()  # Collect spans
# ... other business logic ...
process_based_on_business_logic(spans)  # Process when ready
```

#### Async Span Creation Pattern

Create spans after your workflow executes (useful for post-processing or logging):

```python
# Phase 1: Execute workflows (no tracing context)
results = []
for workflow in workflows:
    result = execute_workflow(workflow)  # No spans created during execution
    results.append(result)

# Phase 2: Create spans from completed results
collected_spans = []

with client.get_span_buffer("experiment-123") as buffer:
    for i, result in enumerate(results):
        buffer.create_span(
            f"workflow_{i}",
            attributes={
                "input": result["input"],
                "output": result["output"],
                "latency": result["latency"],
                "cost": result["cost"],
            }
        )
    
    # Extract spans before context exits
    collected_spans = buffer.get_all_spans()

# Phase 3: Process the collected spans through processors
client.process_spans(collected_spans)
```

#### Span Links for Pause/Resume Workflows

Use `SpanLink` when a new trace should remain separate but still point back to
the earlier causal span. This is the right pattern for pause/resume workflows.

```python
from respan_tracing import SpanLink, get_client

client = get_client()

resume_link = SpanLink(
    trace_id="0123456789abcdef0123456789abcdef",
    span_id="0123456789abcdef",
    attributes={
        "link.type": "resume",
        "link.workflow_run_id": "wr-123",
    },
)

with client.get_span_buffer("resume-trace") as buffer:
    buffer.create_span(
        "workflow_execution",
        attributes={"status": "resumed"},
        links=[resume_link],
    )

    collected_spans = buffer.get_all_spans()

client.process_spans(collected_spans)
```

#### Inspect Before Export

```python
collected_spans = []

with client.get_span_buffer("trace-123") as buffer:
    # Create spans
    buffer.create_span("task1", {"status": "completed", "score": 0.95})
    buffer.create_span("task2", {"status": "failed", "score": 0.45})
    buffer.create_span("task3", {"status": "completed", "score": 0.88})
    
    # Inspect spans before extracting
    print(f"Buffered {buffer.get_span_count()} spans")
    
    for span in buffer.get_all_spans():
        print(f"  - {span.name}: {dict(span.attributes)}")
    
    # Extract spans before context exits
    collected_spans = buffer.get_all_spans()

# Conditionally process based on inspection
if len(collected_spans) > 0:
    client.process_spans(collected_spans)
```

#### SpanBuffer API

**SpanBuffer Methods:**
- `create_span(name, attributes=None, kind=None, links=None)` - Create a span in the local queue
- `get_all_spans()` - Get list of all buffered spans for inspection
- `get_span_count()` - Get the number of buffered spans
- `clear_spans()` - Discard all buffered spans without exporting

**Client Processing Methods:**
- `client.process_spans(buffer)` - Process all spans from a SpanBuffer through the processor pipeline
- `client.process_spans(span_list)` - Process a list of ReadableSpan objects through the processor pipeline

**Context Behavior:**
- Spans created with `buffer.create_span()` go to the buffer's local queue
- **ALL spans created within the buffer context** (including via tracer or decorators) are buffered
- Spans created outside the buffer context are processed normally through processors
- **Spans are extractable as list** - spans are transportable!
- Processing happens through standard OTEL processor pipeline (filters, transformations, export)

**Key Insight: Transportable Spans**
The real power is that spans are **transportable** - collect in one place, process anywhere:

```python
# Collect spans
collected_spans = []

with client.get_span_buffer("trace-123") as buffer:
    buffer.create_span("task", {"result": "success"})
    # Extract spans before context exits
    collected_spans = buffer.get_all_spans()

# Spans persist as list - process anywhere, anytime through OTEL pipeline
if should_export():
    client.process_spans(collected_spans)  # ← Process from anywhere!
```

#### Important: Context Isolation

When you use `SpanBuffer`, **all spans created within that context** are buffered, including:
- Spans created via `buffer.create_span()`
- Spans created via `tracer.start_as_current_span()`
- Spans created by `@workflow` and `@task` decorators
- Auto-instrumented spans (OpenAI, Anthropic, etc.)

```python
client = get_client()
tracer = client.get_tracer()

# Normal span (exported immediately)
with tracer.start_as_current_span("before_buffer"):
    pass

# Buffer context - ALL spans buffered
with client.get_span_buffer("trace-123") as buffer:
    buffer.create_span("span1", {})  # Buffered
    
    with tracer.start_as_current_span("span2"):  # Also buffered!
        pass
    
    print(f"Total buffered: {buffer.get_span_count()}")  # Shows 2
    
    # Extract spans before context exits
    collected_spans = buffer.get_all_spans()

# Process through processor pipeline
client.process_spans(collected_spans)

# Normal span (exported immediately)
with tracer.start_as_current_span("after_buffer"):
    pass
```

#### Use Cases

**Backend Workflow System (Transportable Spans):**
```python
# Workflow execution system with transportable span control
def ingest_workflow_output(workflow_result, trace_id, org_id, experiment_id):
    client = get_client()
    
    # Phase 1: Collect spans (controlled timing)
    with client.get_span_buffer(trace_id) as buffer:
        # Create parent span
        buffer.create_span(
            "workflow_execution",
            attributes={
                "input": workflow_result["input"],
                "output": workflow_result["output"],
                "experiment_id": experiment_id,
                "organization_id": org_id,
            }
        )
        
        # Create child spans for each step
        for step in workflow_result["steps"]:
            buffer.create_span(
                f"step_{step['name']}",
                attributes={
                    "input": step["input"],
                    "output": step["output"],
                    "latency_ms": step["latency"],
                    "cost_usd": step["cost"],
                }
            )
    
    # Phase 2: Buffer persists - spans are now transportable!
    # Process anywhere, anytime with full control
    if should_export_to_respan(experiment_id):
        success = client.process_spans(buffer)  # Process through OTEL pipeline
    elif should_export_to_internal_db(org_id):
        spans = buffer.get_all_spans()
        success = export_to_internal_system(spans)  # Custom export
    else:
        buffer.clear_spans()  # Discard without processing
        success = True
    
    return success
```

**Experiment Logging (Conditional Export):**
```python
# Log multiple experiment runs with conditional export
experiment_results = run_experiments(configs)
collected_spans = []

with client.get_span_buffer(f"experiment-{experiment_id}") as buffer:
    for i, result in enumerate(experiment_results):
        buffer.create_span(
            f"run_{i}",
            attributes={
                "config": result["config"],
                "metrics": result["metrics"],
                "success": result["success"],
            }
        )
    
    # Extract spans before context exits
    collected_spans = buffer.get_all_spans()

# Transportable processing - decide based on experiment results
if experiment_was_successful(experiment_results):
    client.process_spans(collected_spans)  # Process successful experiments
else:
    # Just don't process (spans will be garbage collected)
    pass
```

**Benefits:**
- ✅ **Manual processing timing**: Process only when you explicitly call `client.process_spans()` (vs automatic background processing)
- ✅ **Conditional processing**: Decide whether to process based on span inspection or business logic
- ✅ **Transportable spans**: Collect spans in one place, process anywhere in your code through OTEL pipeline
- ✅ **Async span creation**: Create spans after execution completes (decouple execution from tracing)
- ✅ **OTEL-compliant**: Spans flow through standard processor pipeline (filters, transformations, export)
- ✅ **Single API call per trace**: All spans in a trace exported together (vs individual span exports)
- ✅ **Inspection capability**: Review and modify spans before sending
- ✅ **Thread-safe isolation**: Uses context variables for proper isolation

See [`examples/span_buffer_example.py`](examples/span_buffer_example.py) for complete working examples.
