"""
Centralized filter mixin types for Respan.
Provides base functionality for filters, conditions, and other components that need filtering capabilities.
"""

from typing import Dict, List, Union, Literal, Optional, Any
from typing_extensions import TypedDict
from pydantic import Field, ConfigDict
from respan_sdk.respan_types._internal_types import RespanBaseModel


MetricFilterValueType = Union[
    str, int, float, bool, List[str], List[int], List[float], List[bool]
]


class BaseFilterMixinTypedDict(TypedDict, total=False):
    """
    Base mixin for common filter/condition functionality (TypedDict version).
    This provides the core fields that both filters and conditions share.
    Use this version for dictionary-based type definitions.
    """

    operator: Literal[
        # region:equals
        "",
        "=",
        "==",
        "eq",
        "equals",
        # endregion:equals
        "in",  # for array values
        "not",
        "contains",
        "icontains",
        "startswith",  # casing following clickhouse
        "endswith",  # casing following clickhouse
        "gt",
        "gte",
        "lt",
        "lte",
        "isnull",
        "regex",
        "ilike",
        "trigram_word_similar",
        "full_text_search",
        "empty",
        "notEmpty",  # casing following clickhouse
        "not_empty",  # casing following django lookups
    ]
    connector: Literal["AND", "OR"]  # AND is "all" and OR is "any"
    value: Union[List[MetricFilterValueType], MetricFilterValueType]


class BaseFilterMixinPydantic(RespanBaseModel):
    """
    Base mixin for common filter/condition functionality (Pydantic version).
    This provides the core fields that both filters and conditions share.
    Use this version for Pydantic model-based type definitions.
    """

    operator: Literal[
        # region:equals
        "",
        "=",
        "==",
        "eq",
        "equals",
        # endregion:equals
        "in",  # for array values
        "not",
        "contains",
        "icontains",
        "startswith",  # casing following clickhouse
        "endswith",  # casing following clickhouse
        "gt",
        "gte",
        "lt",
        "lte",
        "isnull",
        "regex",
        "ilike",
        "trigram_word_similar",
        "full_text_search",
        "empty",
        "notEmpty",  # casing following clickhouse
        "not_empty",  # casing following django lookups
    ] = Field(..., description="The comparison operator")
    connector: Optional[Literal["AND", "OR"]] = Field(
        default="AND", description="How to connect this rule with the next one"
    )
    value: Union[List[MetricFilterValueType], MetricFilterValueType] = Field(
        ..., description="The value to compare against"
    )


# Additional Pydantic models for filter types
class MetricFilterParamPydantic(BaseFilterMixinPydantic):
    """
    Pydantic model for MetricFilterParam.
    Represents a filter parameter for a specific metric.
    """

    operator_function: Optional[Literal["mapContainsKey"]] = Field(
        None, description="The function to apply to the operator"
    )
    operator_args: Optional[List[str]] = Field(
        None, description="Arguments for the operator function"
    )

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, str_strip_whitespace=True
    )


class FilterBundlePydantic(RespanBaseModel):
    """
    Pydantic model for FilterBundle.
    Represents a bundle of filter parameters that can be applied together.
    """

    connector: Optional[Literal["AND", "OR"]] = Field(
        default="AND",
        description="How the bundle is connected to the previous conditions",
    )
    filter_params: "FilterParamDictPydantic" = Field(
        ..., description="The filter parameters in this bundle"
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FilterParamDictPydantic(RespanBaseModel):
    """
    Pydantic model for FilterParamDict.
    A dictionary that maps metric names to their filter parameters.

    Each key is a metric name (str), and each value can be:
    - A single MetricFilterParamPydantic (one condition)
    - A List[MetricFilterParamPydantic] (multiple conditions for same metric)
    - A FilterBundlePydantic (nested filter bundle with connector)

    Note: Uses extra="allow" for dynamic metric name fields.
    The __pydantic_extra__ annotation tells Pydantic what types to expect for
    extra fields, and generates typed additionalProperties in JSON Schema.
    """

    __pydantic_extra__: Dict[str, Union["MetricFilterParamPydantic", List["MetricFilterParamPydantic"], "FilterBundlePydantic"]]

    model_config = ConfigDict(
        extra="allow",  # Allow dynamic field names (metric names)
        validate_assignment=True,
    )

    def __init__(self, **data: Any):
        """
        Initialize FilterParamDict with dynamic fields.
        Each field can be a MetricFilterParam, List[MetricFilterParam], or FilterBundle.
        """
        # Validate that all values are of the correct type
        for key, value in data.items():
            if isinstance(value, dict):
                # Check if it's a FilterBundle or MetricFilterParam
                if "filter_params" in value:
                    # It's a FilterBundle
                    data[key] = FilterBundlePydantic.model_validate(value)
                else:
                    # It's a MetricFilterParam
                    data[key] = MetricFilterParamPydantic.model_validate(value)
            elif isinstance(value, list):
                # It's a List[MetricFilterParam]
                data[key] = [
                    MetricFilterParamPydantic.model_validate(item) for item in value
                ]
            else:
                # Let Pydantic handle the validation error
                pass

        super().__init__(**data)

    def __getitem__(
        self, key: str
    ) -> Union[
        MetricFilterParamPydantic, List[MetricFilterParamPydantic], FilterBundlePydantic
    ]:
        """Allow dictionary-style access"""
        return getattr(self, key)

    def __setitem__(
        self,
        key: str,
        value: Union[
            MetricFilterParamPydantic,
            List[MetricFilterParamPydantic],
            FilterBundlePydantic,
        ],
    ):
        """Allow dictionary-style assignment"""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Union[
        MetricFilterParamPydantic,
        List[MetricFilterParamPydantic],
        FilterBundlePydantic,
        Any,
    ]:
        """Dictionary-style get method"""
        return getattr(self, key, default)

    def items(self):
        """Dictionary-style items method"""
        return self.model_dump().items()

    def keys(self):
        """Dictionary-style keys method"""
        return self.model_dump().keys()

    def values(self):
        """Dictionary-style values method"""
        return self.model_dump().values()


# Update forward references
FilterBundlePydantic.model_rebuild()
FilterParamDictPydantic.model_rebuild()


# Convenient aliases for backward compatibility and easier imports
BaseFilterMixin = BaseFilterMixinTypedDict  # Default to TypedDict version
BaseFilterMixinDict = BaseFilterMixinTypedDict
BaseFilterMixinModel = BaseFilterMixinPydantic

# Export both versions for different use cases
__all__ = [
    "MetricFilterValueType",
    "BaseFilterMixinTypedDict",
    "BaseFilterMixinPydantic",
    "MetricFilterParamPydantic",
    "FilterBundlePydantic",
    "FilterParamDictPydantic",
    "BaseFilterMixin",
    "BaseFilterMixinDict",
    "BaseFilterMixinModel",
]
