# Main SDK exports
from .respan_types import (
    # Public logging types - recommended for users
    RespanLogParams,
    
    # Internal types
    RespanParams,
    RespanFullLogParams,
    RespanTextLogParams,
    
    # Common parameter types
    EvaluationParams,
    RetryParams,
    Message,
    Usage,
)

from .respan_types.filter_types import (
    MetricFilterParam,
    FilterBundle,
    FilterParamDict,
    MetricFilterParamPydantic,
    FilterBundlePydantic,
    FilterParamDictPydantic,
)
from .respan_types.mixin_types.filter_mixin import MetricFilterValueType

from .utils.pre_processing import (
    validate_and_separate_params,
    validate_and_separate_log_and_llm_params,
)

__version__ = "1.0.0"

__all__ = [
    # Public types (recommended)
    "RespanLogParams",
    "RespanFullLogParams",
    "RespanTextLogParams",
    
    # Internal types (backward compatibility)
    "RespanParams",
    
    # Parameter types
    "EvaluationParams", 
    "RetryParams",
    "Message",
    "Usage",
    
    # Filter types
    "MetricFilterParam",
    "FilterBundle",
    "FilterParamDict",
    "MetricFilterValueType",
    "MetricFilterParamPydantic",
    "FilterBundlePydantic",
    "FilterParamDictPydantic",

    # Utility functions
    "validate_and_separate_params",
    "validate_and_separate_log_and_llm_params",
]
