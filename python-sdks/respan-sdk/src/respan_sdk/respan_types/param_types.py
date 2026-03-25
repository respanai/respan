from typing import List, Literal, Optional, Union, Dict, Any
from typing_extensions import deprecated
from pydantic import ConfigDict, field_validator, model_validator

from respan_sdk.respan_types.services_types.moda_types import ModaParams
from respan_sdk.respan_types.services_types.hyperspell_types import HyperspellParams
from ._internal_types import (
    BasicAssistantParams,
    BasicLLMParams,
    BasicRunParams,
    BasicThreadParams,
    RespanBaseModel,
    BasicEmbeddingParams,
    LiteLLMCompletionParams,
    Message,
    Usage,
    BasicTextToSpeechParams,
)
from .eval_types import EvalInputs, EvaluatorToRun
from .chat_completion_types import ProviderCredentialType
from .services_types.linkup_types import LinkupParams
from .services_types.mem0_types import Mem0Params
from datetime import datetime
from ..utils.mixins import PreprocessLogDataMixin
from ..constants.llm_logging import (
    LogType,
)

"""
Conventions:

1. Respan as a prefix to class names
2. Params as a suffix to class names

Logging params types:
1. TEXT
2. EMBEDDING
3. AUDIO
4. GENERAL_FUNCTION
"""


class OverrideConfig(RespanBaseModel):
    messages_override_mode: Optional[Literal["override", "append"]] = "override"


class PromptParam(RespanBaseModel):
    prompt_id: Optional[str] = None
    is_custom_prompt: Optional[bool] = False
    version: Optional[Union[int, Literal["latest"]]] = None
    variables: Optional[dict] = None
    echo: Optional[bool] = True

    # v2 fields (set schema_version=2 to use v2 prompt processing)
    schema_version: Optional[int] = None
    patch: Optional[dict] = None

    # v1 fields (preserved for back-compat, ignored when schema_version=2)
    override: Optional[bool] = (
        False  # Allow prompt to override other params in the request body. (e.g. model defined in the prompt will override the model defined in the request body)
    )
    override_params: Optional[BasicLLMParams] = None
    override_config: Optional[OverrideConfig] = None

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)


class EvaluationParams(RespanBaseModel):
    evaluators: Optional[List[EvaluatorToRun]] = []
    evaluation_identifier: Union[str, int] = ""
    last_n_messages: Optional[int] = (
        1  # last n messages to consider for evaluation, 0 -> all messages
    )
    eval_inputs: Optional[EvalInputs] = (
        {}
    )  # extra params that are needed for the evaluation
    sample_percentage: Optional[float] = (
        None  # percentage of messages that trigger the evaluation, default is defined in organization settings, 0 is disabled, 100 is always.
    )

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)


class LoadBalanceModel(RespanBaseModel):
    model: str
    credentials: dict = None
    weight: int

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)

    @field_validator("weight")
    def validate_weight(cls, v):
        if v <= 0:
            raise ValueError("Weight has to be greater than 0")
        return v

    model_config = ConfigDict(protected_namespaces=())


class LoadBalanceGroup(RespanBaseModel):
    group_id: str
    models: Optional[List[LoadBalanceModel]] = None

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)


class PostHogIntegration(RespanBaseModel):
    posthog_api_key: str
    posthog_base_url: str


class Customer(RespanBaseModel):
    customer_identifier: Union[str, int, None] = None
    name: Optional[Union[str, None]] = None
    email: Optional[Union[str, None]] = None
    period_start: Optional[Union[str, datetime]] = (
        None  # ISO 8601 formatted date-string YYYY-MM-DD
    )
    period_end: Optional[Union[str, datetime]] = (
        None  # ISO 8601 formatted date-string YYYY-MM-DD
    )
    budget_duration: Optional[Literal["daily", "weekly", "monthly", "yearly"]] = None
    period_budget: Optional[float] = None
    markup_percentage: Optional[float] = None  # 20 -> original price * 1.2
    total_budget: Optional[float] = None
    metadata: Optional[dict] = None
    rate_limit: Optional[int] = None

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)

    @staticmethod
    def _validate_timestamp(v):
        if isinstance(v, str):
            from dateparser import parse

            try:
                value = datetime.fromisoformat(v)
                return value
            except Exception as e:
                try:
                    value = parse(v)
                    return value
                except Exception as e:
                    raise ValueError(
                        "timestamp has to be a valid ISO 8601 formatted date-string YYYY-MM-DD"
                    )
        return v

    @field_validator("period_start")
    def validate_period_start(cls, v):
        return cls._validate_timestamp(v)

    @field_validator("period_end")
    def validate_period_end(cls, v):
        return cls._validate_timestamp(v)


class CacheOptions(RespanBaseModel):
    cache_by_customer: Optional[bool] = None  # Create cache for each customer_user
    omit_log: Optional[bool] = None  # When cache is hit, don't log the request

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)


class RetryParams(RespanBaseModel):
    num_retries: Optional[int] = 3
    retry_after: Optional[float] = 0.2
    retry_enabled: Optional[bool] = True

    @field_validator("retry_after")
    def validate_retry_after(cls, v):
        if v <= 0:
            raise ValueError("retry_after has to be greater than 0")
        return v

    @field_validator("num_retries")
    def validate_num_retries(cls, v):
        if v <= 0:
            raise ValueError("num_retries has to be greater than 0")
        return v

    model_config = ConfigDict(extra="forbid")


class RespanAPIControlParams(RespanBaseModel):
    block: Optional[bool] = None

    def model_dump(self, *args, **kwargs):
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)


@deprecated("Use log_types.RespanLogParams instead")
class RespanParams(RespanBaseModel, PreprocessLogDataMixin):
    """
    Internal Respan parameters class that includes all fields used by the backend.
    This includes both public-facing fields and internal/backend-only fields.
    """

    # region: time
    start_time: Optional[Union[str, datetime]] = None
    timestamp: Optional[Union[str, datetime]] = (
        None  # This is the end_time in the context of being a span
    )
    hour_group: Optional[Union[str, datetime]] = None
    minute_group: Optional[Union[str, datetime]] = None
    # endregion: time

    # region: authentication
    api_key: Optional[str] = None
    user_id: Optional[Union[int, str]] = None
    user_email: Optional[str] = None  # The use email of the respan user
    organization_id: Optional[Union[int, str]] = None  # Organization ID
    organization_name: Optional[str] = None  # Organization name
    unique_organization_id: Optional[str] = (
        None  # Organization ID - the future replacement for organization_id
    )
    organization_key_id: Optional[str] = None  # Organization key ID
    organization_key_name: Optional[str] = None  # Organization key name
    # endregion: authentication

    # region: environment
    is_test: Optional[bool] = None
    environment: Optional[str] = None
    # endregion: environment

    # region: unique identifiers
    id: Optional[Union[int, str]] = None
    unique_id: Optional[str] = None
    original_copy_unique_id: Optional[str] = None
    custom_identifier: Optional[Union[str, int]] = None
    response_id: Optional[str] = None  # The id of the response from the llm provider
    # endregion: unique identifiers

    # region: status
    # region: error handling
    error_bit: Optional[int] = None
    error_message: Optional[str] = None
    recommendations: Optional[str] = None
    recommendations_dict: Optional[dict] = None
    warnings: Optional[str] = None
    warnings_dict: Optional[dict] = None
    has_warnings: Optional[bool] = (
        None  # Deprecated, covered by status's literal casings
    )
    # endregion: error handling
    status: Optional[str] = None
    status_code: Optional[int] = None
    # endregion: status

    # region: log identifier/grouping
    load_balance_group_id: Optional[str] = None
    group_identifier: Optional[Union[str, int]] = None
    evaluation_identifier: Optional[Union[str, int]] = None
    # endregion: log identifier/grouping

    # region: log input/output
    storage_object_key: Optional[str] = None
    input: Optional[Union[str, dict, list]] = None
    output: Optional[Union[str, dict, list]] = None
    prompt_messages: Optional[List[Message]] = None
    ideal_output: Optional[str] = None
    completion_message: Optional[Message] = None
    completion_messages: Optional[List[Message]] = None
    prompt_message_count: Optional[int] = None
    completion_message_count: Optional[int] = None
    completion_tokens: Optional[int] = None
    system_text: Optional[str] = None
    prompt_text: Optional[str] = None
    completion_text: Optional[str] = None
    input_array: Optional[List[str]] = None
    full_request: Optional[Union[dict, list]] = None
    full_response: Optional[Union[dict, list]] = None
    is_fts_enabled: Optional[bool] = None
    full_text: Optional[str] = (
        None  # The field that contains the full the full text representation of the request
    )
    # region: special response types
    tool_calls: Optional[List[dict]] = None
    has_tool_calls: Optional[bool] = None
    reasoning: Optional[List[dict]] = None
    # endregion: special response types
    # endregion: log input/output

    # region: display
    blurred: Optional[bool] = None
    # endregion: display

    # region: cache params
    cache_enabled: Optional[bool] = None
    cache_hit: Optional[bool] = None
    cache_bit: Optional[int] = None  # 0 or 1
    cache_miss_bit: Optional[int] = None  # 0 or 1
    cache_options: Optional[CacheOptions] = None
    cache_ttl: Optional[int] = None
    cache_key: Optional[str] = None
    redis_cache_ttl: Optional[int] = None
    cache_request_content: Optional[str] = None
    # endregion: cache params

    # region: usage
    # region: cost related
    cost: Optional[float] = None
    covered_by: Optional[str] = None
    evaluation_cost: Optional[float] = (
        None  # Deprecated, tracked by evaluation results instead
    )
    prompt_unit_price: Optional[float] = None
    completion_unit_price: Optional[float] = None
    used_custom_credential: Optional[bool] = None
    service_tier: Optional[str] = None  # Provider service tier (e.g. "priority", "default", "flex")
    # endregion: cost related

    # region: time period
    period_start: Optional[Union[str, datetime]] = None
    period_end: Optional[Union[str, datetime]] = None
    # endregion: time period

    # region: token usage
    prompt_tokens: Optional[int] = None
    prompt_cache_hit_tokens: Optional[int] = None
    prompt_cache_creation_tokens: Optional[int] = None
    usage: Optional[Union[Usage, dict]] = (
        None  # The usage object of the LLM response, which includes the token usage details; if cannot be parsed, can be a dict
    )
    # endregion: token usage
    # endregion: usage

    # region: llm proxy credentials
    credential_override: Optional[Dict[str, dict]] = None
    customer_credentials: Optional[Dict[str, ProviderCredentialType]] = None
    # endregion: llm proxy credentials

    # region: llm deployment
    models: Optional[List[str]] = None
    model_name_map: Optional[Dict[str, str]] = None
    deployment_name: Optional[str] = None
    full_model_name: Optional[str] = None
    # endregion: llm deployment

    # region: user analytics
    customer_email: Optional[str] = None
    customer_name: Optional[str] = None
    customer_identifier: Optional[Union[str, int]] = None
    customer_user_unique_id: Optional[str] = None
    customer_params: Optional[Customer] = None
    # endregion: user analytics

    # region: respan llm response control
    field_name: Optional[str] = "data: "
    delimiter: Optional[str] = "\n\n"
    disable_log: Optional[bool] = False
    request_breakdown: Optional[bool] = False
    # endregion: respan llm response control

    # region: respan logging control
    is_log_omitted: Optional[bool] = (
        None  # If true, logging will be omitted for this request
    )
    respan_api_controls: Optional[RespanAPIControlParams] = None
    mock_response: Optional[str] = None
    log_method: Optional[str] = None
    log_type: Optional[LogType] = None
    # endregion: respan logging control

    # region: respan proxy options
    disable_fallback: Optional[bool] = False
    exclude_models: Optional[List[str]] = None
    exclude_providers: Optional[List[str]] = None
    fallback_models: Optional[List[str]] = None
    load_balance_group: Optional[LoadBalanceGroup] = None
    load_balance_models: Optional[List[LoadBalanceModel]] = None
    retry_params: Optional[RetryParams] = None
    respan_params: Optional[dict] = (
        None  # Nested respan params for special cases
    )
    # endregion: respan proxy options

    # region: embedding
    embedding: Optional[Union[List[float], str]] = None
    base64_embedding: Optional[str] = None
    provider_id: Optional[str] = None
    # endregion: embedding

    # region: audio
    audio_input_file: Optional[str] = None
    audio_output_file: Optional[str] = None
    # endregion: audio

    # region: evaluation
    note: Optional[str] = None
    category: Optional[str] = None
    eval_params: Optional[EvaluationParams] = None
    for_eval: Optional[bool] = None  # Deprecated
    positive_feedback: Optional[bool] = (
        None  # Deprecated, creating positive feedback in a join table instead
    )
    # endregion: evaluation

    # region: request metadata
    ip_address: Optional[str] = None
    request_url_path: Optional[str] = None
    # endregion: request metadata

    # region: technical integrations
    hyperspell_params: Optional[HyperspellParams] = None
    linkup_params: Optional[LinkupParams] = None
    mem0_params: Optional[Mem0Params] = None
    moda_params: Optional[ModaParams] = None
    posthog_integration: Optional[PostHogIntegration] = None
    # endregion: technical integrations

    # region: custom properties
    metadata: Optional[dict] = None  # Map(String, String) — all values coerced to strings
    properties: Optional[dict] = None  # Native JSON — preserves types (numbers, booleans, nested objects)
    # region: Deprecated, clickhouse allow filters to be applied efficiently enough
    metadata_indexed_string_1: Optional[str] = None
    metadata_indexed_string_2: Optional[str] = None
    metadata_indexed_numerical_1: Optional[float] = None
    # endregion: deprecated
    # endregion: custom properties

    # region: prompt
    prompt: Optional[Union[PromptParam, str]] = (
        None  # PromptParam when using prompt_id, str when used for logging transcription calls
    )
    prompt_id: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt_version_number: Optional[int] = None
    prompt_messages_template: Optional[List[Message]] = (
        None  # This is for logging the raw messages from Prompt users
    )
    variables: Optional[dict] = (
        None  # This is for logging the variables from Prompt users
    )
    # endregion: prompt

    # region: llm response timing metrics
    latency: Optional[float] = None
    time_to_first_token: Optional[float] = None
    routing_time: Optional[float] = None
    tokens_per_second: Optional[float] = None
    # endregion: llm response timing metrics

    # region: tracing
    total_request_tokens: Optional[int] = None
    trace_unique_id: Optional[str] = None
    trace_name: Optional[str] = None
    span_unique_id: Optional[str] = None
    span_name: Optional[str] = None
    span_parent_id: Optional[str] = None
    span_path: Optional[str] = None
    span_handoffs: Optional[List[str]] = None
    span_tools: Optional[List[str]] = None
    span_workflow_name: Optional[str] = None
    session_identifier: Optional[Union[str, int]] = None
    trace_group_identifier: Optional[Union[str, int]] = (
        None  # The customizable id for grouping traces together
    )
    # region: thread, deprecated
    thread_identifier: Optional[Union[str, int]] = (
        None  # 2025-06-04: Deprecated, merged into tracing as a special case
    )
    thread_unique_id: Optional[str] = None
    # endregion: thread, deprecated

    # region: dataset
    dataset_id: Optional[str] = None
    ds_run_at: Optional[datetime] = None  # The time when the dataset run was triggered
    original_copy_storage_object_key: Optional[str] = None
    # endregion: dataset

    # endregion: tracing

    @model_validator(mode="before")
    @classmethod
    def _preprocess_data(cls, data):
        data = super()._preprocess_data(data)

        # Handle related fields
        for field_name in cls.__annotations__:
            if field_name.endswith("_id"):
                related_model_name = field_name[:-3]  # Remove '_id' from the end
                cls._assign_related_field(related_model_name, field_name, data)

        return data

    @classmethod
    def _assign_related_field(
        cls, related_model_name: str, assign_to_name: str, data: dict
    ):
        related_model_value = data.get(related_model_name)
        if not isinstance(related_model_value, (int, str)):
            return
        data[assign_to_name] = related_model_value

    def model_dump(self, *args, **kwargs):
        # Set exclude_none to True if not explicitly provided
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    @field_validator("timestamp")
    def validate_timestamp(cls, v):
        from respan_sdk.utils.time import parse_datetime

        return parse_datetime(v)

    @field_validator("start_time")
    def validate_start_time(cls, v):
        from respan_sdk.utils.time import parse_datetime

        return parse_datetime(v)

    @field_validator("hour_group")
    def validate_hour_group(cls, v):
        from respan_sdk.utils.time import parse_datetime

        return parse_datetime(v)

    @field_validator("minute_group")
    def validate_minute_group(cls, v):
        from respan_sdk.utils.time import parse_datetime

        return parse_datetime(v)

    @field_validator("customer_identifier")
    def validate_customer_identifier(cls, v):
        if v and isinstance(v, str) and len(v) > 254:
            v = v[:254]
        return v

    @field_validator("input")
    def validate_input(cls, v):
        import json
        from respan_sdk.utils.serialization import json_serial

        if v:
            if isinstance(v, dict):
                return json.dumps(
                    v, default=json_serial
                )  # Eats any unknown type as string
            elif isinstance(v, list):
                return json.dumps(v, default=json_serial)
            else:
                return v
        return v

    @field_validator("span_name", mode="after")
    def validate_span_name(cls, v):
        if v:
            return v[:255]  # The DB column is varchar(255)
        return v

    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)


@deprecated("Use log_types.RespanFullLogParams instead")
class RespanTextLogParams(
    RespanParams, LiteLLMCompletionParams, BasicEmbeddingParams
):
    """
    A type definition of the input parameters for creating a Respan RequestLog object.
    This is the INTERNAL type. Only used in respan backend
    """

    @field_validator("customer_params", mode="after")
    def validate_customer_params(cls, v: Union[Customer, None]):
        if v is None:
            return None
        if v.customer_identifier is None:
            return None
        return v

    @model_validator(mode="before")
    def _preprocess_data(cls, data):

        data = RespanParams._preprocess_data(data)
        # Special response format handling for backward compatibility
        if "response_format" in data:
            if type(data["response_format"]) == str:
                data["response_format"] = {"type": data["response_format"]}
        return data

    def serialize_for_logging(
        self, exclude_fields: List[str] = [], extra_fields: List[str] = []
    ) -> dict:
        # Define fields to include based on Django model columns
        # Using a set for O(1) lookup
        FIELDS_TO_INCLUDE = {
            "ip_address",
            "blurred",
            "custom_identifier",
            "status",
            "unique_id",
            "trace_unique_id",
            "span_unique_id",
            "trace_group_identifier",
            "session_identifier",
            "span_name",
            "span_parent_id",
            "span_path",
            "span_handoffs",
            "span_tools",
            "span_workflow_name",
            "prompt_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_creation_tokens",
            "prompt_id",
            "completion_tokens",
            "total_request_tokens",
            "cost",
            "amount_to_pay",
            "latency",
            "user_id",
            "organization_id",
            "model",
            "provider_id",
            "full_model_name",
            "start_time",
            "timestamp",
            "minute_group",
            "hour_group",
            "prompt_id",
            "prompt_name",
            "positive_feedback",  # This is a boolean, bad naming
            "error_bit",
            "time_to_first_token",
            "metadata",
            "metadata_indexed_string_1",
            "metadata_indexed_string_2",
            "metadata_indexed_numerical_1",
            "stream",
            "stream_options",
            "thread_identifier",
            "status_code",
            "cached",
            "cache_bit",
            "cache_miss_bit",
            "cache_key",
            "prompt_messages",
            "completion_message",
            "respan_params",
            "full_request",
            "full_response",
            "completion_messages",
            "system_text",
            "prompt_text",
            "completion_text",
            "error_message",
            "warnings",
            "recommendations",
            "storage_object_key",
            "tokens_per_second",
            "is_test",
            "environment",
            "temperature",
            "max_tokens",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "evaluation_cost",
            "evaluation_identifier",
            "for_eval",
            "prompt_id",
            "customer_identifier",
            "customer_email",
            "used_custom_credential",
            "covered_by",
            "log_method",
            "log_type",
            "input",
            "input_array",
            "output",
            "embedding",
            "base64_embedding",
            "tools",
            "tool_choice",
            "tool_calls",
            "has_tool_calls",
            "response_format",
            "parallel_tool_calls",
            "organization_key_id",
            "has_warnings",
            "prompt_version_number",
            "deployment_name",
            # region: dataset
            "dataset_id",
            "ds_run_at",
            "original_copy_storage_object_key",
            # endregion: dataset
        }
        FIELDS_TO_INCLUDE = (set(FIELDS_TO_INCLUDE) - set(exclude_fields)) | set(
            extra_fields
        )
        if self.disable_log:
            FIELDS_TO_INCLUDE.discard("full_request")
            FIELDS_TO_INCLUDE.discard("full_response")
            FIELDS_TO_INCLUDE.discard("tool_calls")
            FIELDS_TO_INCLUDE.discard("prompt_messages")
            FIELDS_TO_INCLUDE.discard("completion_messages")
            FIELDS_TO_INCLUDE.discard("completion_message")

        # Get all non-None values using model_dump
        data = self.model_dump(exclude_none=True)

        # Filter to only include fields that exist in Django model
        to_return = {}
        for key, value in data.items():
            if key in FIELDS_TO_INCLUDE:
                if key.endswith("_identifier"):
                    to_return[key] = str(value)[:120]
                else:
                    to_return[key] = value
        return to_return

    model_config = ConfigDict(from_attributes=True)


class EmbeddingParams(BasicEmbeddingParams, RespanParams):
    pass


class TextToSpeechParams(BasicTextToSpeechParams, RespanParams):
    pass


class AssistantParams(BasicAssistantParams, RespanParams):
    pass


class ThreadParams(BasicThreadParams, RespanParams):
    pass


class RunParams(BasicRunParams, RespanParams):
    pass


class LLMParams(BasicLLMParams, RespanParams):
    @model_validator(mode="after")
    @classmethod
    def validate_messages(cls, values):
        """
        Either prompt or messages must be provided
        Returns:
            [type]: [description]
        """
        if not values.prompt and not values.messages:
            raise ValueError("Either prompt or messages must be provided")
        return values
