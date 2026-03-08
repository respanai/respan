import copy
import importlib
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Optional, Set, Callable, List

from ..instruments import Instruments


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstrumentConfig:
    """Configuration for a single OTEL instrumentation.

    Adding a new instrumentation = one entry in INSTRUMENT_REGISTRY.
    """

    package: str
    module: str
    class_name: str
    post_init_hooks: tuple = ()


# ---------------------------------------------------------------------------
# Registry — one line per instrumentation, no boilerplate functions
# ---------------------------------------------------------------------------

INSTRUMENT_REGISTRY: dict[Instruments, InstrumentConfig] = {
    # AI/ML Libraries
    Instruments.OPENAI: InstrumentConfig(
        package="openai",
        module="opentelemetry.instrumentation.openai",
        class_name="OpenAIInstrumentor",
        post_init_hooks=("_patch_chat_prompt_capture",),
    ),
    Instruments.ANTHROPIC: InstrumentConfig(
        package="anthropic",
        module="opentelemetry.instrumentation.anthropic",
        class_name="AnthropicInstrumentor",
    ),
    Instruments.COHERE: InstrumentConfig(
        package="cohere",
        module="opentelemetry.instrumentation.cohere",
        class_name="CohereInstrumentor",
    ),
    Instruments.MISTRAL: InstrumentConfig(
        package="mistralai",
        module="opentelemetry.instrumentation.mistralai",
        class_name="MistralAiInstrumentor",
    ),
    Instruments.OLLAMA: InstrumentConfig(
        package="ollama",
        module="opentelemetry.instrumentation.ollama",
        class_name="OllamaInstrumentor",
    ),
    Instruments.GROQ: InstrumentConfig(
        package="groq",
        module="opentelemetry.instrumentation.groq",
        class_name="GroqInstrumentor",
    ),
    Instruments.TOGETHER: InstrumentConfig(
        package="together",
        module="opentelemetry.instrumentation.together",
        class_name="TogetherInstrumentor",
    ),
    Instruments.REPLICATE: InstrumentConfig(
        package="replicate",
        module="opentelemetry.instrumentation.replicate",
        class_name="ReplicateInstrumentor",
    ),
    Instruments.TRANSFORMERS: InstrumentConfig(
        package="transformers",
        module="opentelemetry.instrumentation.transformers",
        class_name="TransformersInstrumentor",
    ),

    # Cloud AI Services
    Instruments.BEDROCK: InstrumentConfig(
        package="boto3",
        module="opentelemetry.instrumentation.bedrock",
        class_name="BedrockInstrumentor",
    ),
    Instruments.SAGEMAKER: InstrumentConfig(
        package="boto3",
        module="opentelemetry.instrumentation.sagemaker",
        class_name="SageMakerInstrumentor",
    ),
    Instruments.VERTEXAI: InstrumentConfig(
        package="google.cloud.aiplatform",
        module="opentelemetry.instrumentation.vertexai",
        class_name="VertexAIInstrumentor",
    ),
    Instruments.GOOGLE_GENERATIVEAI: InstrumentConfig(
        package="google.generativeai",
        module="opentelemetry.instrumentation.google_generativeai",
        class_name="GoogleGenerativeAiInstrumentor",
    ),
    Instruments.WATSONX: InstrumentConfig(
        package="ibm_watsonx_ai",
        module="opentelemetry.instrumentation.watsonx",
        class_name="WatsonxInstrumentor",
    ),
    Instruments.ALEPHALPHA: InstrumentConfig(
        package="aleph_alpha_client",
        module="opentelemetry.instrumentation.alephalpha",
        class_name="AlephAlphaInstrumentor",
    ),

    # Vector Databases
    Instruments.PINECONE: InstrumentConfig(
        package="pinecone",
        module="opentelemetry.instrumentation.pinecone",
        class_name="PineconeInstrumentor",
    ),
    Instruments.QDRANT: InstrumentConfig(
        package="qdrant_client",
        module="opentelemetry.instrumentation.qdrant",
        class_name="QdrantInstrumentor",
    ),
    Instruments.CHROMA: InstrumentConfig(
        package="chromadb",
        module="opentelemetry.instrumentation.chromadb",
        class_name="ChromaInstrumentor",
    ),
    Instruments.MILVUS: InstrumentConfig(
        package="pymilvus",
        module="opentelemetry.instrumentation.milvus",
        class_name="MilvusInstrumentor",
    ),
    Instruments.WEAVIATE: InstrumentConfig(
        package="weaviate",
        module="opentelemetry.instrumentation.weaviate",
        class_name="WeaviateInstrumentor",
    ),
    Instruments.LANCEDB: InstrumentConfig(
        package="lancedb",
        module="opentelemetry.instrumentation.lancedb",
        class_name="LanceDBInstrumentor",
    ),
    Instruments.MARQO: InstrumentConfig(
        package="marqo",
        module="opentelemetry.instrumentation.marqo",
        class_name="MarqoInstrumentor",
    ),

    # Frameworks
    Instruments.LANGCHAIN: InstrumentConfig(
        package="langchain",
        module="opentelemetry.instrumentation.langchain",
        class_name="LangchainInstrumentor",
    ),
    Instruments.LLAMA_INDEX: InstrumentConfig(
        package="llama_index",
        module="opentelemetry.instrumentation.llama_index",
        class_name="LlamaIndexInstrumentor",
    ),
    Instruments.HAYSTACK: InstrumentConfig(
        package="haystack",
        module="opentelemetry.instrumentation.haystack",
        class_name="HaystackInstrumentor",
    ),
    Instruments.CREW: InstrumentConfig(
        package="crewai",
        module="opentelemetry.instrumentation.crewai",
        class_name="CrewAIInstrumentor",
    ),
    Instruments.MCP: InstrumentConfig(
        package="mcp",
        module="opentelemetry.instrumentation.mcp",
        class_name="MCPInstrumentor",
    ),

    # Infrastructure
    Instruments.CELERY: InstrumentConfig(
        package="celery",
        module="opentelemetry.instrumentation.celery",
        class_name="CeleryInstrumentor",
    ),
    Instruments.DJANGO: InstrumentConfig(
        package="django",
        module="opentelemetry.instrumentation.django",
        class_name="DjangoInstrumentor",
    ),
    Instruments.FASTAPI: InstrumentConfig(
        package="fastapi",
        module="opentelemetry.instrumentation.fastapi",
        class_name="FastAPIInstrumentor",
    ),
    Instruments.FLASK: InstrumentConfig(
        package="flask",
        module="opentelemetry.instrumentation.flask",
        class_name="FlaskInstrumentor",
    ),
    Instruments.SQLALCHEMY: InstrumentConfig(
        package="sqlalchemy",
        module="opentelemetry.instrumentation.sqlalchemy",
        class_name="SQLAlchemyInstrumentor",
    ),
    Instruments.PSYCOPG2: InstrumentConfig(
        package="psycopg2",
        module="opentelemetry.instrumentation.psycopg2",
        class_name="Psycopg2Instrumentor",
    ),
    Instruments.AIOHTTP_CLIENT: InstrumentConfig(
        package="aiohttp",
        module="opentelemetry.instrumentation.aiohttp_client",
        class_name="AioHttpClientInstrumentor",
    ),
    Instruments.GRPC: InstrumentConfig(
        package="grpc",
        module="opentelemetry.instrumentation.grpc",
        class_name="GrpcInstrumentorClient",
    ),
    Instruments.REDIS: InstrumentConfig(
        package="redis",
        module="opentelemetry.instrumentation.redis",
        class_name="RedisInstrumentor",
    ),
    Instruments.REQUESTS: InstrumentConfig(
        package="requests",
        module="opentelemetry.instrumentation.requests",
        class_name="RequestsInstrumentor",
    ),
    Instruments.URLLIB3: InstrumentConfig(
        package="urllib3",
        module="opentelemetry.instrumentation.urllib3",
        class_name="URLLib3Instrumentor",
    ),
    Instruments.PYMYSQL: InstrumentConfig(
        package="pymysql",
        module="opentelemetry.instrumentation.pymysql",
        class_name="PyMySQLInstrumentor",
    ),
    Instruments.THREADING: InstrumentConfig(
        package=None,  # stdlib, always available
        module="opentelemetry.instrumentation.threading",
        class_name="ThreadingInstrumentor",
    ),
}


# ---------------------------------------------------------------------------
# Post-init hooks — special-case patches that run after instrument()
# ---------------------------------------------------------------------------

_POST_INIT_HOOKS: dict[str, Callable] = {}


def _register_hook(name: str):
    """Decorator to register a post-init hook by name."""
    def decorator(fn):
        _POST_INIT_HOOKS[name] = fn
        return fn
    return decorator


@_register_hook("_patch_chat_prompt_capture")
def _patch_chat_prompt_capture():
    """
    Replace the async chat _handle_request with a sync version.

    Root cause: opentelemetry-instrumentation-openai v0.52+ has _handle_request
    as async def (for optional base64 image upload). In sync contexts, it runs
    through run_async() which either calls asyncio.run() or spawns a thread.
    Both paths can silently lose prompt attributes when:
      - _set_request_attributes (NOT @dont_throw) raises on response_format
        handling, killing the entire _handle_request before _set_prompts runs
      - asyncio.run() / thread path has environment-specific issues (Lambda, etc.)

    The embeddings wrapper is fully sync and works correctly. This patch makes
    the chat path match the embeddings path: fully synchronous with fault
    isolation between each section.

    The only async code in _set_prompts was for Config.upload_base64_image
    (rarely used). For list content (multimodal), we json.dumps as-is — the
    base64 data stays inline, which is the default behavior anyway.
    """
    try:
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from opentelemetry.instrumentation.openai.shared import (
            _set_request_attributes,
            _set_client_attributes,
            _set_functions_attributes,
            _set_span_attribute,
            set_tools_attributes,
            model_as_dict,
            propagate_trace_context,
        )
        from opentelemetry.instrumentation.openai.shared.config import Config
        from opentelemetry.instrumentation.openai.utils import (
            should_send_prompts,
            should_emit_events,
            is_openai_v1,
        )
        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAIAttributes,
        )
        from opentelemetry.semconv_ai import SpanAttributes

        def _set_prompts_sync(span, messages):
            if not span.is_recording() or messages is None:
                return

            for i, msg in enumerate(messages):
                prefix = f"{GenAIAttributes.GEN_AI_PROMPT}.{i}"
                msg = msg if isinstance(msg, dict) else model_as_dict(msg)

                _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
                if msg.get("content"):
                    content = copy.deepcopy(msg.get("content"))
                    if isinstance(content, list):
                        content = json.dumps(content)
                    _set_span_attribute(span, f"{prefix}.content", content)
                if msg.get("tool_call_id"):
                    _set_span_attribute(
                        span, f"{prefix}.tool_call_id", msg.get("tool_call_id")
                    )
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for j, tool_call in enumerate(tool_calls):
                        if is_openai_v1():
                            tool_call = model_as_dict(tool_call)
                        function = tool_call.get("function")
                        _set_span_attribute(
                            span, f"{prefix}.tool_calls.{j}.id", tool_call.get("id")
                        )
                        _set_span_attribute(
                            span, f"{prefix}.tool_calls.{j}.name", function.get("name")
                        )
                        _set_span_attribute(
                            span,
                            f"{prefix}.tool_calls.{j}.arguments",
                            function.get("arguments"),
                        )

        def _handle_request_sync(span, kwargs, instance):
            # Section 1: Request attributes (fault-isolated from prompts)
            try:
                _set_request_attributes(span, kwargs, instance)
            except Exception:
                logging.warning(
                    "respan-tracing: _set_request_attributes failed (response_format may be incompatible). "
                    "Request attributes like model/temperature may be incomplete on this span. "
                    "Error: %s",
                    traceback.format_exc(),
                )

            try:
                _set_client_attributes(span, instance)
            except Exception:
                pass

            # Section 2: Prompt/event capture
            try:
                if should_emit_events():
                    from opentelemetry.instrumentation.openai.shared.event_emitter import emit_event
                    from opentelemetry.instrumentation.openai.shared.event_models import MessageEvent
                    for message in kwargs.get("messages", []):
                        emit_event(
                            MessageEvent(
                                content=message.get("content"),
                                role=message.get("role"),
                                tool_calls=cw._parse_tool_calls(
                                    message.get("tool_calls", None)
                                ),
                            )
                        )
                else:
                    if should_send_prompts():
                        _set_prompts_sync(span, kwargs.get("messages"))
                        if kwargs.get("functions"):
                            _set_functions_attributes(span, kwargs.get("functions"))
                        elif kwargs.get("tools"):
                            set_tools_attributes(span, kwargs.get("tools"))
            except Exception:
                logging.warning(
                    "respan-tracing: chat prompt capture failed. "
                    "Input messages may not appear on the dashboard for this span. "
                    "Error: %s",
                    traceback.format_exc(),
                )

            # Section 3: Trace propagation + reasoning
            try:
                if Config.enable_trace_context_propagation:
                    propagate_trace_context(span, kwargs)
                reasoning_effort = kwargs.get("reasoning_effort")
                _set_span_attribute(
                    span,
                    SpanAttributes.LLM_REQUEST_REASONING_EFFORT,
                    reasoning_effort or (),
                )
            except Exception:
                pass

        async def _noop():
            pass

        def _patched_handle_request(span, kwargs, instance):
            _handle_request_sync(span, kwargs, instance)
            return _noop()

        cw._handle_request = _patched_handle_request
        logger.debug("respan-tracing: patched chat prompt capture to sync path")

    except Exception as e:
        logger.warning(f"respan-tracing: failed to patch chat prompt capture: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_package_installed(package_name: str) -> bool:
    """Check if a package is installed."""
    if package_name is None:
        return True  # stdlib (e.g., threading)
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def init_instrumentations(
    instruments: Optional[Set[Instruments]] = None,
    block_instruments: Optional[Set[Instruments]] = None,
) -> bool:
    """
    Initialize OpenTelemetry instrumentations for specified libraries.

    Args:
        instruments: Set of instruments to enable. If None, enables all available.
        block_instruments: Set of instruments to explicitly block.

    Returns:
        bool: True if at least one instrument was successfully initialized.

    Note:
        THREADING instrumentation is automatically enabled (unless explicitly blocked)
        because it's critical for context propagation across threads. To disable it,
        use: block_instruments={Instruments.THREADING}
    """
    block_instruments = block_instruments or set()

    # Default to all instruments if none specified
    if instruments is None:
        instruments = set(Instruments)
    else:
        # If user specified instruments, automatically include THREADING
        # unless they explicitly blocked it
        if Instruments.THREADING not in block_instruments:
            instruments = instruments | {Instruments.THREADING}

    # Remove blocked instruments
    instruments = instruments - block_instruments

    instrument_count = 0

    for instrument in instruments:
        try:
            if _init_single_instrument(instrument):
                instrument_count += 1
        except Exception as e:
            logger.warning(f"Failed to initialize {instrument.value} instrumentation: {e}")

    if instrument_count == 0:
        logger.warning("No instrumentations were successfully initialized")
        return False

    logger.info(f"Successfully initialized {instrument_count} instrumentations")
    return True


def _init_single_instrument(instrument: Instruments) -> bool:
    """Initialize a single instrument using the registry."""
    config = INSTRUMENT_REGISTRY.get(instrument)
    if config is None:
        logger.warning(f"No registry entry for instrument: {instrument}")
        return False

    if not is_package_installed(config.package):
        return False

    try:
        module = importlib.import_module(config.module)
        instrumentor_cls = getattr(module, config.class_name)
        instrumentor = instrumentor_cls()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()

        # Run post-init hooks
        for hook_name in config.post_init_hooks:
            hook = _POST_INIT_HOOKS.get(hook_name)
            if hook is not None:
                hook()
            else:
                logger.warning(f"Post-init hook '{hook_name}' not found for {instrument.value}")

        return True
    except Exception as e:
        logger.error(f"Failed to initialize {instrument.value} instrumentation: {e}")
        return False
