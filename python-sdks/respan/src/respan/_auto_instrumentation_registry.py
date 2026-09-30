"""Curated auto-instrumentation registry for direct LLM SDKs."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "respan.instrumentations"

AutoInstrumentationState = str


@dataclass(frozen=True)
class AutoInstrumentationSpec:
    """Metadata for one Respan-owned instrumentation package."""

    id: str
    category: str
    provider: str
    sdk_package: str
    instrumentation_package: str
    entry_point: str
    import_path: str
    enabled_by_default: bool = True
    priority: int = 100
    aliases: Tuple[str, ...] = ()
    conflicts_with: Tuple[str, ...] = ()
    auto_disabled_reason: Optional[str] = None
    constructor_kwargs: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class AutoInstrumentationStatus:
    """Runtime status for a registry entry."""

    id: str
    name: str
    status: AutoInstrumentationState
    provider: str
    sdk_package: str
    instrumentation_package: str
    reason: Optional[str] = None


@dataclass(frozen=True)
class AutoInstrumentationActivation:
    """Internal activation result with the public status and instrumentor."""

    status: AutoInstrumentationStatus
    instrumentor: Optional[object] = field(default=None, repr=False, compare=False)


_AGENT_FRAMEWORK_REASON = (
    "agent framework instrumentations are opt-in to avoid nested duplicate spans"
)
_FRAMEWORK_REASON = (
    "framework/wrapper instrumentations are opt-in to avoid duplicating direct SDK spans"
)
_UNBUNDLED_PROVIDER_REASON = (
    "provider instrumentation is opt-in until its adapter can be bundled without "
    "adding incompatible or heavyweight dependencies"
)
_OPENAI_COMPATIBLE_GATEWAY_REASON = (
    "provider gateway calls use OpenAI-compatible transport, so OpenAI "
    "instrumentation covers default RESPAN_API_KEY usage"
)
_PROVIDER_CREDENTIAL_REASON = (
    "native SDK instrumentation is opt-in because it requires provider credentials, "
    "cloud credentials, or a local model/runtime"
)
_OBSERVABILITY_REASON = (
    "observability bridge instrumentations are opt-in and should not run from LLM auto mode"
)
_PROTOCOL_REASON = "protocol/tooling instrumentations are opt-in"
_VECTOR_REASON = "vector database instrumentations are opt-in"


AUTO_INSTRUMENTATION_REGISTRY: Tuple[AutoInstrumentationSpec, ...] = (
    AutoInstrumentationSpec(
        id="openai",
        category="direct_llm",
        provider="OpenAI",
        sdk_package="openai",
        instrumentation_package="respan-instrumentation-openai",
        entry_point="openai",
        import_path="respan_instrumentation_openai:OpenAIInstrumentor",
        priority=10,
    ),
    AutoInstrumentationSpec(
        id="anthropic",
        category="direct_llm",
        provider="Anthropic",
        sdk_package="anthropic",
        instrumentation_package="respan-instrumentation-anthropic",
        entry_point="anthropic",
        import_path="respan_instrumentation_anthropic:AnthropicInstrumentor",
        priority=20,
    ),
    AutoInstrumentationSpec(
        id="vertexai",
        category="direct_llm",
        provider="Google Vertex AI",
        sdk_package="vertexai",
        instrumentation_package="respan-instrumentation-vertexai",
        entry_point="vertexai",
        import_path="respan_instrumentation_vertexai:VertexAIInstrumentor",
        priority=30,
    ),
    AutoInstrumentationSpec(
        id="google-genai",
        category="direct_llm",
        provider="Google Gen AI",
        sdk_package="google-genai",
        instrumentation_package="respan-instrumentation-google-genai",
        entry_point="google-genai",
        import_path="respan_instrumentation_google_genai:GoogleGenAIInstrumentor",
        priority=31,
    ),
    AutoInstrumentationSpec(
        id="aws-bedrock",
        category="direct_llm",
        provider="AWS Bedrock",
        sdk_package="boto3",
        instrumentation_package="respan-instrumentation-aws-bedrock",
        entry_point="aws-bedrock",
        import_path="respan_instrumentation_aws_bedrock:AWSBedrockInstrumentor",
        priority=40,
        aliases=("bedrock",),
    ),
    AutoInstrumentationSpec(
        id="cohere",
        category="direct_llm",
        provider="Cohere",
        sdk_package="cohere",
        instrumentation_package="respan-instrumentation-cohere",
        entry_point="cohere",
        import_path="respan_instrumentation_cohere:CohereInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_OPENAI_COMPATIBLE_GATEWAY_REASON,
        priority=50,
    ),
    AutoInstrumentationSpec(
        id="together",
        category="direct_llm",
        provider="Together AI",
        sdk_package="together",
        instrumentation_package="respan-instrumentation-together",
        entry_point="together",
        import_path="respan_instrumentation_together:TogetherInstrumentor",
        priority=60,
    ),
    AutoInstrumentationSpec(
        id="groq",
        category="direct_llm",
        provider="Groq",
        sdk_package="groq",
        instrumentation_package="respan-instrumentation-groq",
        entry_point="groq",
        import_path="respan_instrumentation_groq:GroqInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_OPENAI_COMPATIBLE_GATEWAY_REASON,
        priority=70,
    ),
    AutoInstrumentationSpec(
        id="mistralai",
        category="direct_llm",
        provider="Mistral AI",
        sdk_package="mistralai",
        instrumentation_package="respan-instrumentation-mistralai",
        entry_point="mistralai",
        import_path="respan_instrumentation_mistralai:MistralAIInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_UNBUNDLED_PROVIDER_REASON,
        priority=80,
    ),
    AutoInstrumentationSpec(
        id="ollama",
        category="direct_llm",
        provider="Ollama",
        sdk_package="ollama",
        instrumentation_package="respan-instrumentation-ollama",
        entry_point="ollama",
        import_path="respan_instrumentation_ollama:OllamaInstrumentor",
        priority=90,
    ),
    AutoInstrumentationSpec(
        id="aleph-alpha",
        category="direct_llm",
        provider="Aleph Alpha",
        sdk_package="aleph-alpha-client",
        instrumentation_package="respan-instrumentation-aleph-alpha",
        entry_point="aleph-alpha",
        import_path="respan_instrumentation_aleph_alpha:AlephAlphaInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_PROVIDER_CREDENTIAL_REASON,
        priority=100,
    ),
    AutoInstrumentationSpec(
        id="huggingface",
        category="direct_llm",
        provider="Hugging Face",
        sdk_package="transformers",
        instrumentation_package="respan-instrumentation-huggingface",
        entry_point="huggingface",
        import_path="respan_instrumentation_huggingface:HuggingFaceInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_PROVIDER_CREDENTIAL_REASON,
        priority=110,
    ),
    AutoInstrumentationSpec(
        id="litellm",
        category="llm_wrapper",
        provider="LiteLLM",
        sdk_package="litellm",
        instrumentation_package="respan-instrumentation-litellm",
        entry_point="litellm",
        import_path="respan_instrumentation_litellm:LiteLLMInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_FRAMEWORK_REASON,
        priority=120,
    ),
    AutoInstrumentationSpec(
        id="replicate",
        category="direct_llm",
        provider="Replicate",
        sdk_package="replicate",
        instrumentation_package="respan-instrumentation-replicate",
        entry_point="replicate",
        import_path="respan_instrumentation_replicate:ReplicateInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_OPENAI_COMPATIBLE_GATEWAY_REASON,
        priority=130,
    ),
    AutoInstrumentationSpec(
        id="sagemaker",
        category="direct_llm",
        provider="AWS SageMaker",
        sdk_package="boto3",
        instrumentation_package="respan-instrumentation-sagemaker",
        entry_point="sagemaker",
        import_path="respan_instrumentation_sagemaker:SageMakerInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_OPENAI_COMPATIBLE_GATEWAY_REASON,
        priority=140,
    ),
    AutoInstrumentationSpec(
        id="watsonx",
        category="direct_llm",
        provider="IBM watsonx",
        sdk_package="ibm-watsonx-ai",
        instrumentation_package="respan-instrumentation-watsonx",
        entry_point="watsonx",
        import_path="respan_instrumentation_watsonx:WatsonxInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_PROVIDER_CREDENTIAL_REASON,
        priority=150,
    ),
    AutoInstrumentationSpec(
        id="writer",
        category="direct_llm",
        provider="Writer",
        sdk_package="writerai",
        instrumentation_package="respan-instrumentation-writer",
        entry_point="writer",
        import_path="respan_instrumentation_writer:WriterInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_PROVIDER_CREDENTIAL_REASON,
        priority=160,
    ),
    AutoInstrumentationSpec(
        id="portkey",
        category="direct_llm",
        provider="Portkey",
        sdk_package="portkey-ai",
        instrumentation_package="respan-instrumentation-portkey",
        entry_point="portkey",
        import_path="respan_instrumentation_portkey:PortkeyInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_OPENAI_COMPATIBLE_GATEWAY_REASON,
        priority=170,
    ),
    AutoInstrumentationSpec(
        id="openrouter",
        category="direct_llm",
        provider="OpenRouter",
        sdk_package="openai",
        instrumentation_package="respan-instrumentation-openrouter",
        entry_point="openrouter",
        import_path="respan_instrumentation_openrouter:OpenRouterInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_OPENAI_COMPATIBLE_GATEWAY_REASON,
        priority=180,
        conflicts_with=("openai",),
        constructor_kwargs=(("normalize_all_openai_spans", False),),
    ),
    AutoInstrumentationSpec(
        id="instructor",
        category="llm_wrapper",
        provider="Instructor",
        sdk_package="instructor",
        instrumentation_package="respan-instrumentation-instructor",
        entry_point="instructor",
        import_path="respan_instrumentation_instructor:InstructorInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="langchain",
        category="framework",
        provider="LangChain",
        sdk_package="langchain",
        instrumentation_package="respan-instrumentation-langchain",
        entry_point="langchain",
        import_path="respan_instrumentation_langchain:LangChainInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="llama-index",
        category="framework",
        provider="LlamaIndex",
        sdk_package="llama-index",
        instrumentation_package="respan-instrumentation-llama-index",
        entry_point="llama-index",
        import_path="respan_instrumentation_llama_index:LlamaIndexInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="haystack",
        category="framework",
        provider="Haystack",
        sdk_package="haystack-ai",
        instrumentation_package="respan-instrumentation-haystack",
        entry_point="haystack",
        import_path="respan_instrumentation_haystack:HaystackInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="dspy",
        category="framework",
        provider="DSPy",
        sdk_package="dspy",
        instrumentation_package="respan-instrumentation-dspy",
        entry_point="dspy",
        import_path="respan_instrumentation_dspy:DSPyInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="guardrails",
        category="framework",
        provider="Guardrails",
        sdk_package="guardrails-ai",
        instrumentation_package="respan-instrumentation-guardrails",
        entry_point="guardrails",
        import_path="respan_instrumentation_guardrails:GuardrailsInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="pydantic-ai",
        category="agent_framework",
        provider="Pydantic AI",
        sdk_package="pydantic-ai",
        instrumentation_package="respan-instrumentation-pydantic-ai",
        entry_point="pydantic-ai",
        import_path="respan_instrumentation_pydantic_ai:PydanticAIInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="openai-agents",
        category="agent_framework",
        provider="OpenAI Agents",
        sdk_package="openai-agents",
        instrumentation_package="respan-instrumentation-openai-agents",
        entry_point="openai-agents",
        import_path="respan_instrumentation_openai_agents:OpenAIAgentsInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="claude-agent-sdk",
        category="agent_framework",
        provider="Claude Agent SDK",
        sdk_package="claude-agent-sdk",
        instrumentation_package="respan-instrumentation-claude-agent-sdk",
        entry_point="claude-agent-sdk",
        import_path="respan_instrumentation_claude_agent_sdk:ClaudeAgentSDKInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="google-adk",
        category="agent_framework",
        provider="Google ADK",
        sdk_package="google-adk",
        instrumentation_package="respan-instrumentation-google-adk",
        entry_point="google-adk",
        import_path="respan_instrumentation_google_adk:GoogleADKInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="strands-agents",
        category="agent_framework",
        provider="Strands Agents",
        sdk_package="strands-agents",
        instrumentation_package="respan-instrumentation-strands-agents",
        entry_point="strands-agents",
        import_path="respan_instrumentation_strands_agents:StrandsAgentsInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="beeai",
        category="agent_framework",
        provider="BeeAI",
        sdk_package="beeai-framework",
        instrumentation_package="respan-instrumentation-beeai",
        entry_point="beeai",
        import_path="respan_instrumentation_beeai:BeeAIInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="smolagents",
        category="agent_framework",
        provider="smolagents",
        sdk_package="smolagents",
        instrumentation_package="respan-instrumentation-smolagents",
        entry_point="smolagents",
        import_path="respan_instrumentation_smolagents:SmolagentsInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="crewai",
        category="agent_framework",
        provider="CrewAI",
        sdk_package="crewai",
        instrumentation_package="respan-instrumentation-crewai",
        entry_point="crewai",
        import_path="respan_instrumentation_crewai:CrewAIInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="agno",
        category="agent_framework",
        provider="Agno",
        sdk_package="agno",
        instrumentation_package="respan-instrumentation-agno",
        entry_point="agno",
        import_path="respan_instrumentation_agno:AgnoInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="autogen",
        category="agent_framework",
        provider="AutoGen",
        sdk_package="autogen-agentchat",
        instrumentation_package="respan-instrumentation-autogen",
        entry_point="autogen",
        import_path="respan_instrumentation_autogen:AutoGenInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="dify",
        category="agent_framework",
        provider="Dify",
        sdk_package="dify",
        instrumentation_package="respan-instrumentation-dify",
        entry_point="dify",
        import_path="respan_instrumentation_dify:DifyInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="exa",
        category="protocol_tooling",
        provider="Exa",
        sdk_package="exa-py",
        instrumentation_package="respan-instrumentation-exa",
        entry_point="exa",
        import_path="respan_instrumentation_exa:ExaInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_PROTOCOL_REASON,
    ),
    AutoInstrumentationSpec(
        id="agentspec",
        category="agent_framework",
        provider="AgentSpec",
        sdk_package="agentspec",
        instrumentation_package="respan-instrumentation-agentspec",
        entry_point="agentspec",
        import_path="respan_instrumentation_agentspec:AgentSpecInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="pipecat",
        category="agent_framework",
        provider="Pipecat",
        sdk_package="pipecat-ai",
        instrumentation_package="respan-instrumentation-pipecat",
        entry_point="pipecat",
        import_path="respan_instrumentation_pipecat:PipecatInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="superagent",
        category="agent_framework",
        provider="Superagent",
        sdk_package="superagent",
        instrumentation_package="respan-instrumentation-superagent",
        entry_point="superagent",
        import_path="respan_instrumentation_superagent:SuperagentInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="agentscope",
        category="agent_framework",
        provider="AgentScope",
        sdk_package="agentscope",
        instrumentation_package="respan-instrumentation-agentscope",
        entry_point="agentscope",
        import_path="respan_instrumentation_agentscope:AgentScopeInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="livekit",
        category="agent_framework",
        provider="LiveKit Agents",
        sdk_package="livekit-agents",
        instrumentation_package="respan-instrumentation-livekit",
        entry_point="livekit",
        import_path="respan_instrumentation_livekit:LiveKitInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="microsoft-agent-framework",
        category="agent_framework",
        provider="Microsoft Agent Framework",
        sdk_package="agent-framework-core",
        instrumentation_package="respan-instrumentation-microsoft-agent-framework",
        entry_point="microsoft-agent-framework",
        import_path=(
            "respan_instrumentation_microsoft_agent_framework:"
            "MicrosoftAgentFrameworkInstrumentor"
        ),
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="watson-orchestrate-adk",
        category="agent_framework",
        provider="IBM watsonx Orchestrate ADK",
        sdk_package="ibm-watsonx-orchestrate",
        instrumentation_package="respan-instrumentation-watson-orchestrate-adk",
        entry_point="watson-orchestrate-adk",
        import_path=(
            "respan_instrumentation_watson_orchestrate_adk:"
            "WatsonOrchestrateADKInstrumentor"
        ),
        enabled_by_default=False,
        auto_disabled_reason=_AGENT_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="semantic-kernel",
        category="framework",
        provider="Microsoft Semantic Kernel",
        sdk_package="semantic-kernel",
        instrumentation_package="respan-instrumentation-semantic-kernel",
        entry_point="semantic-kernel",
        import_path=(
            "respan_instrumentation_semantic_kernel:SemanticKernelInstrumentor"
        ),
        enabled_by_default=False,
        auto_disabled_reason=_FRAMEWORK_REASON,
    ),
    AutoInstrumentationSpec(
        id="cursor-sdk",
        category="protocol",
        provider="Cursor SDK",
        sdk_package="cursor",
        instrumentation_package="respan-instrumentation-cursor-sdk",
        entry_point="cursor-sdk",
        import_path="respan_instrumentation_cursor_sdk:CursorSDKInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_PROTOCOL_REASON,
    ),
    AutoInstrumentationSpec(
        id="mcp",
        category="protocol",
        provider="MCP",
        sdk_package="mcp",
        instrumentation_package="respan-instrumentation-mcp",
        entry_point="mcp",
        import_path="respan_instrumentation_mcp:MCPInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_PROTOCOL_REASON,
    ),
    AutoInstrumentationSpec(
        id="arize",
        category="observability",
        provider="Arize",
        sdk_package="arize",
        instrumentation_package="respan-instrumentation-arize",
        entry_point="arize",
        import_path="respan_instrumentation_arize:ArizeInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_OBSERVABILITY_REASON,
    ),
    AutoInstrumentationSpec(
        id="braintrust",
        category="observability",
        provider="Braintrust",
        sdk_package="braintrust",
        instrumentation_package="respan-instrumentation-braintrust",
        entry_point="braintrust",
        import_path="respan_instrumentation_braintrust:BraintrustInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_OBSERVABILITY_REASON,
    ),
    AutoInstrumentationSpec(
        id="openinference",
        category="observability",
        provider="OpenInference",
        sdk_package="openinference",
        instrumentation_package="respan-instrumentation-openinference",
        entry_point="openinference",
        import_path="respan_instrumentation_openinference:OpenInferenceInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_OBSERVABILITY_REASON,
    ),
    AutoInstrumentationSpec(
        id="langfuse",
        category="observability",
        provider="Langfuse",
        sdk_package="langfuse",
        instrumentation_package="respan-instrumentation-langfuse",
        entry_point="langfuse",
        import_path="respan_instrumentation_langfuse.instrumentor:LangfuseInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_OBSERVABILITY_REASON,
    ),
    AutoInstrumentationSpec(
        id="chroma",
        category="vector_db",
        provider="Chroma",
        sdk_package="chromadb",
        instrumentation_package="respan-instrumentation-chroma",
        entry_point="chroma",
        import_path="respan_instrumentation_chroma:ChromaInstrumentor",
        enabled_by_default=False,
        auto_disabled_reason=_VECTOR_REASON,
    ),
)


def _is_auto_enabled(spec: AutoInstrumentationSpec) -> bool:
    return spec.category == "direct_llm" and spec.enabled_by_default


def list_auto_instrumentation_specs(
    *,
    include_disabled: bool = True,
) -> Tuple[AutoInstrumentationSpec, ...]:
    """Return the curated Python auto-instrumentation registry."""

    if include_disabled:
        return AUTO_INSTRUMENTATION_REGISTRY
    return tuple(
        spec for spec in AUTO_INSTRUMENTATION_REGISTRY if _is_auto_enabled(spec)
    )


def _iter_entry_points(group: str):
    try:
        return importlib.metadata.entry_points(group=group)
    except TypeError:
        entry_points = importlib.metadata.entry_points()
        if hasattr(entry_points, "select"):
            return entry_points.select(group=group)
        return entry_points.get(group, ())


def _discover_entry_points() -> Dict[str, object]:
    try:
        return {entry_point.name: entry_point for entry_point in _iter_entry_points(ENTRY_POINT_GROUP)}
    except Exception as exc:
        logger.warning("Failed to discover Respan instrumentations: %s", exc)
        return {}


def _status(
    spec: AutoInstrumentationSpec,
    status: AutoInstrumentationState,
    *,
    name: Optional[str] = None,
    reason: Optional[str] = None,
) -> AutoInstrumentationStatus:
    return AutoInstrumentationStatus(
        id=spec.id,
        name=name or spec.id,
        status=status,
        provider=spec.provider,
        sdk_package=spec.sdk_package,
        instrumentation_package=spec.instrumentation_package,
        reason=reason,
    )


def _load_from_import_path(import_path: str):
    module_name, separator, attr_name = import_path.partition(":")
    if not separator or not module_name or not attr_name:
        raise ImportError("invalid import path: %s" % import_path)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _load_instrumentor_class(
    spec: AutoInstrumentationSpec,
    entry_points: Mapping[str, object],
):
    entry_point = entry_points.get(spec.entry_point)
    if entry_point is not None:
        return entry_point.load()
    return _load_from_import_path(spec.import_path)


def _already_activated_names(names: Iterable[str]) -> set:
    return {str(name).lower() for name in names}


def _spec_names(spec: AutoInstrumentationSpec) -> set:
    return {spec.id.lower(), spec.entry_point.lower(), *(name.lower() for name in spec.aliases)}


def _is_already_activated(
    spec: AutoInstrumentationSpec,
    activated_names: set,
) -> bool:
    return bool(_spec_names(spec) & activated_names)


def _instrumentor_is_active(instrumentor: object) -> bool:
    return getattr(instrumentor, "_is_instrumented", True) is not False


def activate_auto_instrumentations(
    *,
    already_activated: Iterable[str] = (),
    registry: Sequence[AutoInstrumentationSpec] = AUTO_INSTRUMENTATION_REGISTRY,
) -> Tuple[AutoInstrumentationActivation, ...]:
    """Activate installed default-enabled direct LLM instrumentations.

    Missing packages and opt-in-only packages are returned as status entries
    without raising. This function intentionally uses the Respan plugin entry
    point group, not ``opentelemetry_instrumentor``, so auto mode cannot turn on
    unrelated OTEL packages.
    """

    entry_points = _discover_entry_points()
    activated_names = _already_activated_names(already_activated)
    activations = []

    for spec in sorted(registry, key=lambda item: (item.priority, item.id)):
        if spec.category != "direct_llm":
            activations.append(
                AutoInstrumentationActivation(
                    status=_status(
                        spec,
                        "disabled",
                        reason=spec.auto_disabled_reason
                        or "only direct LLM SDK instrumentations are eligible for auto mode",
                    )
                )
            )
            continue

        if not spec.enabled_by_default:
            activations.append(
                AutoInstrumentationActivation(
                    status=_status(
                        spec,
                        "disabled",
                        reason=spec.auto_disabled_reason or "disabled by default",
                    )
                )
            )
            continue

        if _is_already_activated(spec, activated_names):
            activations.append(
                AutoInstrumentationActivation(
                    status=_status(
                        spec,
                        "disabled",
                        reason="already activated explicitly",
                    )
                )
            )
            continue

        try:
            instrumentor_class = _load_instrumentor_class(spec, entry_points)
        except (ImportError, AttributeError) as exc:
            activations.append(
                AutoInstrumentationActivation(
                    status=_status(
                        spec,
                        "missing",
                        reason="%s is not installed: %s"
                        % (spec.instrumentation_package, exc),
                    )
                )
            )
            continue
        except Exception as exc:
            activations.append(
                AutoInstrumentationActivation(
                    status=_status(
                        spec,
                        "failed",
                        reason="failed to load instrumentation: %s" % exc,
                    )
                )
            )
            continue

        try:
            kwargs = dict(spec.constructor_kwargs)
            instrumentor = instrumentor_class(**kwargs)
            name = str(getattr(instrumentor, "name", spec.id))
            if name.lower() in activated_names:
                activations.append(
                    AutoInstrumentationActivation(
                        status=_status(
                            spec,
                            "disabled",
                            name=name,
                            reason="already activated explicitly",
                        )
                    )
                )
                continue
            instrumentor.activate()
        except Exception as exc:
            logger.warning(
                "Failed to activate auto-instrumentation %s: %s",
                spec.id,
                exc,
            )
            activations.append(
                AutoInstrumentationActivation(
                    status=_status(
                        spec,
                        "failed",
                        reason="failed to activate instrumentation: %s" % exc,
                    )
                )
            )
            continue

        if not _instrumentor_is_active(instrumentor):
            activations.append(
                AutoInstrumentationActivation(
                    status=_status(
                        spec,
                        "missing",
                        name=name,
                        reason="target SDK is not installed or chose not to activate",
                    )
                )
            )
            continue

        activated_names.add(name.lower())
        activated_names.update(_spec_names(spec))
        activations.append(
            AutoInstrumentationActivation(
                status=_status(spec, "enabled", name=name),
                instrumentor=instrumentor,
            )
        )

    return tuple(activations)
