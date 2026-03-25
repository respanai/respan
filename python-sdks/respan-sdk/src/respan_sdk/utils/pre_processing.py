from respan_sdk.respan_types._internal_types import LiteLLMCompletionParams
from respan_sdk.respan_types.param_types import RespanParams
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from respan_sdk.respan_types.log_types import RespanLogParams


def validate_and_separate_params(
    params: dict,
) -> tuple[LiteLLMCompletionParams, RespanParams]:
    """
    Validate and separate the params into llm_params and respan_params using Pydantic models
    Returns:
    basic_llm: LiteLLMCompletionParams
    respan: RespanParams
    """

    basic_llm = LiteLLMCompletionParams.model_validate(params)
    respan = RespanParams.model_validate(params)

    return basic_llm, respan


def validate_and_separate_log_and_llm_params(
    params: dict,
) -> tuple[LiteLLMCompletionParams, "RespanLogParams"]:
    """
    Validate and separate the params into llm_params and public respan_log_params using Pydantic models.
    This function is intended for public-facing APIs and handles mapping of common LLM params to log params.

    Returns:
    basic_llm: LiteLLMCompletionParams
    respan_log: RespanLogParams
    """
    from respan_sdk.respan_types.log_types import RespanLogParams

    basic_llm = LiteLLMCompletionParams.model_validate(params)
    respan_log = RespanLogParams.model_validate(params)

    return basic_llm, respan_log
