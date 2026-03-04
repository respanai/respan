from typing import List, Optional, Union, Dict, Literal
from datetime import datetime
from pydantic import field_validator, model_validator

from respan_sdk.respan_types.services_types.moda_types import ModaParams
from respan_sdk.respan_types.services_types.hyperspell_types import HyperspellParams

from ._internal_types import (
    RespanBaseModel,
    Message,
    Usage,
    ToolChoice,
)
from ..utils.time import parse_datetime
import json

# We need to import these directly to avoid forward reference issues
# Since log_types.py is imported by param_types.py, we need to be careful about circular imports
# But these specific types are defined early in param_types.py before RespanParams
from ..utils.mixins import PreprocessLogDataMixin
from .param_types import (
    PromptParam,
    EvaluationParams,
    CacheOptions,
    Customer,
    RespanAPIControlParams,
    LoadBalanceGroup,
    LoadBalanceModel,
    RetryParams,
    PostHogIntegration,
)
from ..constants.llm_logging import LogType
from .services_types.linkup_types import LinkupParams
from .services_types.mem0_types import Mem0Params
from .chat_completion_types import ProviderCredentialType


class RespanLogParams(PreprocessLogDataMixin, RespanBaseModel):
    """
    Public-facing logging parameters for Respan.
    These are the parameters that users can control when logging requests, used in creation method in the SDK
    """

    # region: time
    start_time: Optional[Union[str, datetime]] = None
    timestamp: Optional[Union[str, datetime]] = None
    # endregion: time

    # region: unique identifiers
    custom_identifier: Optional[Union[str, int]] = None
    # endregion: unique identifiers

    # region: log identifier/grouping
    group_identifier: Optional[Union[str, int]] = None
    evaluation_identifier: Optional[Union[str, int]] = None
    # endregion: log identifier/grouping

    # region: status
    error_message: Optional[str] = None
    warnings: Optional[str] = None
    status_code: Optional[int] = None
    # endregion: status

    # region: chat completion params
    echo: Optional[bool] = None
    frequency_penalty: Optional[float] = None
    logprobs: Optional[bool] = None
    logit_bias: Optional[Dict[str, float]] = None
    messages: List[Message] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[List[str], str]] = None
    stream: Optional[bool] = None
    stream_options: Optional[dict] = None
    temperature: Optional[float] = None
    timeout: Optional[float] = None
    tools: Optional[List[dict]] = None
    response_format: Optional[Dict] = None
    reasoning_effort: Optional[Union[str, None]] = None
    tool_choice: Optional[Union[Literal["auto", "none", "required"], ToolChoice]] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
    # endregion: chat completion params

    # region: litellm specific fields
    thinking: Optional[dict] = None  # For reasoning/thinking content
    # endregion: litellm specific fields

    # region: log input/output
    input: Optional[str] = (
        None  # There is a collision between this and the input field in BasicEmbeddingParams. Handled in the stringify_input validator
    )
    output: Optional[str] = None
    prompt_messages: Optional[List[Message]] = None
    ideal_output: Optional[str] = None
    completion_message: Optional[Message] = (
        None  # Generally for logging the response of the LLM
    )
    completion_messages: Optional[List[Message]] = (
        None  # If `n` > 1, log multiple choices of the LLM response
    )
    full_request: Optional[Union[dict, list]] = None
    full_response: Optional[Union[dict, list]] = None
    # region: special response types
    tool_calls: Optional[List[dict]] = None
    reasoning: Optional[List[dict]] = (
        None  # For tracing, logging the reasoning of the LLM
    )
    # endregion: special response types
    # endregion: log input/output

    # region: embedding
    embedding: Optional[Union[List[float], str]] = (
        None  # Either a list of floats or a base64 encoded string, convert to json string for ease of storage, handled in the stringify_embedding validator
    )
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    # endregion: embedding

    # region: cache params
    cache_enabled: Optional[bool] = None
    cache_options: Optional[CacheOptions] = None
    cache_ttl: Optional[int] = None
    # endregion: cache params

    # region: usage
    usage: Optional[Union[Usage, dict]] = (
        None  # The usage object of the LLM response, which includes the token usage details; if cannot be parsed, can be a dict
    )
    cost: Optional[float] = None
    prompt_unit_price: Optional[float] = None
    completion_unit_price: Optional[float] = None
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    prompt_cache_hit_tokens: Optional[int] = None
    prompt_cache_creation_tokens: Optional[int] = None
    # endregion: usage

    # region: user analytics
    customer_email: Optional[str] = None
    customer_name: Optional[str] = None
    customer_identifier: Optional[Union[str, int]] = None
    customer_params: Optional[Customer] = None
    # endregion: user analytics

    # region: model information
    provider: Optional[str] = None  # Top-level provider name (backward-compatible with API)
    provider_id: Optional[str] = None  # The provider id of the embedding model
    # endregion: model information

    # region: audio
    audio_input_file: Optional[str] = None  # The url of the audio input file
    audio_output_file: Optional[str] = None  # The url of the audio output file
    # endregion: audio

    # region: evaluation
    note: Optional[str] = None
    category: Optional[str] = None
    eval_params: Optional[EvaluationParams] = None
    # endregion: evaluation

    # region: custom properties
    metadata: Optional[dict] = None
    # endregion: custom properties

    # region: prompt
    prompt: Optional[Union[PromptParam, str]] = (
        None  # PromptParam when using prompt_id, str when used for logging transcription calls
    )
    variables: Optional[dict] = None
    # endregion: prompt

    # region: llm response timing metrics
    time_to_first_token: Optional[float] = None
    latency: Optional[float] = None
    # endregion: llm response timing metrics

    # region: technical integrations
    hyperspell_params: Optional[HyperspellParams] = None
    linkup_params: Optional[LinkupParams] = None
    mem0_params: Optional[Mem0Params] = None
    moda_params: Optional[ModaParams] = None
    posthog_integration: Optional[PostHogIntegration] = None
    # endregion: technical integrations

    # region: tracing
    trace_unique_id: Optional[str] = None
    trace_name: Optional[str] = None
    span_unique_id: Optional[str] = None
    span_name: Optional[str] = None
    span_parent_id: Optional[str] = None
    span_path: Optional[str] = None
    span_handoffs: Optional[List[str]] = (
        None  # OpenAI tracing, agent handoffs to other agents
    )
    span_tools: Optional[List[str]] = (
        None  # OpenAI tracing, names of the tools used in the LLM response
    )
    span_workflow_name: Optional[str] = None
    session_identifier: Optional[Union[str, int]] = None
    trace_group_identifier: Optional[Union[str, int]] = None
    # endregion: tracing

    # region: respan proxy options
    disable_fallback: Optional[bool] = False
    exclude_models: Optional[List[str]] = None
    exclude_providers: Optional[List[str]] = None
    fallback_models: Optional[List[str]] = None
    load_balance_group: Optional[LoadBalanceGroup] = None
    load_balance_models: Optional[List[LoadBalanceModel]] = None
    retry_params: Optional[RetryParams] = None
    respan_params: Optional[dict] = None
    # region: deprecated
    model_name_map: Optional[Dict[str, str]] = (
        None  #  Map an available model on Respan to a custom name at inference time
    )
    # endregion: deprecated
    # endregion: respan proxy options

    # region: respan llm response control
    field_name: Optional[str] = "data: "
    delimiter: Optional[str] = "\n\n"
    disable_log: Optional[bool] = False
    request_breakdown: Optional[bool] = False
    # endregion: respan llm response control

    @model_validator(mode="before")
    @classmethod
    def _preprocess_data_for_public(cls, data):
        """
        Preprocess data for public consumption, mapping LLM params to log params.
        Example: messages -> prompt_messages
        """
        data = super()._preprocess_data(data)
        return data

    @field_validator("timestamp")
    def validate_timestamp(cls, v):
        return parse_datetime(v)

    @field_validator("start_time")
    def validate_start_time(cls, v):
        return parse_datetime(v)

    @field_validator("input", mode="before")
    def stringify_input(cls, v):
        if isinstance(v, list) or isinstance(v, dict):
            return json.dumps(v, default=str)
        return str(v)

    @field_validator("embedding", mode="before")
    def stringify_embedding(cls, v):
        if isinstance(v, list):
            return json.dumps(v, default=str)
        return v

    @field_validator("span_name", mode="after")
    def stringify_span_name(cls, v):
        if v:
            return v[:255]  # The DB column is varchar(255)
        return v


class RespanFullLogParams(RespanLogParams):
    """
    Full logging parameters for Respan that includes all fields to be logged to the database
    NONE of these fields can be set by the user (there will be no effect if they are set)
    This is used for parsing the retrieved logs in the list/detail endpoints in the SDK
    """

    # region: authentication (missing from public)
    api_key: Optional[str] = None
    user_id: Optional[Union[int, str]] = None
    user_email: Optional[str] = None
    organization_id: Optional[Union[int, str]] = None
    organization_name: Optional[str] = None
    unique_organization_id: Optional[str] = None
    organization_key_id: Optional[str] = None
    organization_key_name: Optional[str] = None
    # endregion: authentication

    # region: environment (missing from public)
    is_test: Optional[bool] = None
    environment: Optional[str] = None
    # endregion: environment

    # region: unique identifiers (additional)
    id: Optional[Union[int, str]] = None
    unique_id: Optional[str] = None
    response_id: Optional[str] = None
    # endregion: unique identifiers

    # region: status (missing from public)
    error_bit: Optional[int] = None
    recommendations: Optional[str] = None
    recommendations_dict: Optional[dict] = None
    status: Optional[str] = None  # This is controlled by the status_code
    warnings_dict: Optional[dict] = None
    has_warnings: Optional[bool] = None
    # endregion: status

    # region: log identifier/grouping (additional)
    load_balance_group_id: Optional[str] = None
    # endregion: log identifier/grouping

    # region: log input/output (additional)
    storage_object_key: Optional[str] = None
    prompt_message_count: Optional[int] = None
    completion_message_count: Optional[int] = None
    system_text: Optional[str] = None
    prompt_text: Optional[str] = None
    completion_text: Optional[str] = None
    input_array: Optional[List[str]] = None
    is_fts_enabled: Optional[bool] = None
    full_text: Optional[str] = None
    has_tool_calls: Optional[bool] = None
    # endregion: log input/output

    # region: display
    blurred: Optional[bool] = None
    # endregion: display

    # region: cache params (additional)
    cache_hit: Optional[bool] = None
    cache_bit: Optional[int] = None
    cache_miss_bit: Optional[int] = None
    cache_key: Optional[str] = None
    redis_cache_ttl: Optional[int] = None
    cache_request_content: Optional[str] = None
    # endregion: cache params

    # region: usage (additional)
    covered_by: Optional[str] = None
    evaluation_cost: Optional[float] = None
    used_custom_credential: Optional[bool] = None
    period_start: Optional[Union[str, datetime]] = None
    period_end: Optional[Union[str, datetime]] = None
    # endregion: usage

    # region: llm proxy credentials
    credential_override: Optional[Dict[str, dict]] = None
    customer_credentials: Optional[Dict[str, ProviderCredentialType]] = None
    # endregion: llm proxy credentials

    # region: llm deployment
    models: Optional[List[str]] = None
    deployment_name: Optional[str] = None
    full_model_name: Optional[str] = None
    # endregion: llm deployment

    # region: user analytics (additional)
    customer_user_unique_id: Optional[str] = None
    # endregion: user analytics

    # region: respan logging control
    is_log_omitted: Optional[bool] = None
    respan_api_controls: Optional[RespanAPIControlParams] = None
    mock_response: Optional[str] = None
    log_method: Optional[str] = None
    log_type: Optional[LogType] = None
    # endregion: respan logging control

    # region: embedding (additional)
    base64_embedding: Optional[str] = None
    # endregion: embedding

    # region: evaluation (additional)
    for_eval: Optional[bool] = None
    positive_feedback: Optional[bool] = None
    # endregion: evaluation

    # region: request metadata
    ip_address: Optional[str] = None
    request_url_path: Optional[str] = None
    # endregion: request metadata

    # region: custom properties (additional)
    metadata_indexed_string_1: Optional[str] = None
    metadata_indexed_string_2: Optional[str] = None
    metadata_indexed_numerical_1: Optional[float] = None
    # endregion: custom properties

    # region: prompt (additional)
    prompt_id: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt_version_number: Optional[int] = None
    prompt_messages_template: Optional[List[Message]] = None
    # endregion: prompt

    # region: llm response timing metrics (additional)

    routing_time: Optional[float] = None
    tokens_per_second: Optional[float] = None
    # endregion: llm response timing metrics

    # region: usage (additional)
    total_request_tokens: Optional[int] = (
        None  # Calculated from prompt + completion tokens
    )
    # endregion: usage (additional)

    # region: tracing (additional)
    thread_identifier: Optional[Union[str, int]] = None
    thread_unique_id: Optional[str] = None
    # endregion: tracing

    # region: time (additional)
    hour_group: Optional[Union[str, datetime]] = None
    minute_group: Optional[Union[str, datetime]] = None
    # endregion: time

    # region: dataset
    dataset_id: Optional[str] = None
    ds_run_at: Optional[datetime] = None
    original_copy_storage_object_key: Optional[str] = None
    # endregion: dataset

    # Additional field validators for the new fields
    @field_validator("hour_group")
    def validate_hour_group(cls, v):
        return parse_datetime(v)

    @field_validator("minute_group")
    def validate_minute_group(cls, v):
        return parse_datetime(v)

    @field_validator("period_start")
    def validate_period_start(cls, v):
        return parse_datetime(v)

    @field_validator("period_end")
    def validate_period_end(cls, v):
        return parse_datetime(v)
