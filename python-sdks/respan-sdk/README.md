## Respan SDK

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-sdk/)**

Light weight library for Respan type definitions and API payload preprocessing

Features:
- Definition of types, what data structure Respan expects to receive in API calls.
- Preprocessing, separating Respan-specific parameters from LLM-specific parameters
- Conversion, converting input types from Anthropic API format into OpenAI API format.
- **Filter types** — typed vocabulary for the Respan filter system, shared across BE and SDKs.

For **tracing**, please go to [Respan Tracing](https://github.com/respan-ai/respan-sdks/tree/main/python-sdks/respan-tracing) instead.

### Filter Types

The SDK exports the full Respan filter vocabulary so every package (backend, tracing, exporters) uses the same typed definitions instead of raw dicts.

```python
from respan_sdk import (
    # TypedDict versions (lightweight, no validation)
    FilterParamDict,      # Dict[str, MetricFilterParam | List[MetricFilterParam] | FilterBundle]
    MetricFilterParam,    # Single filter condition (operator + value)
    FilterBundle,         # Nested group with connector (AND/OR)

    # Pydantic versions (with validation)
    FilterParamDictPydantic,
    MetricFilterParamPydantic,
    FilterBundlePydantic,

    # Value type
    MetricFilterValueType,  # Union[str, int, float, bool, List[...]]
)
```

**Operators** — the full operator vocabulary available in `operator` field:

| Category | Operators |
|----------|-----------|
| Equality | `""`, `"="`, `"=="`, `"eq"`, `"equals"` |
| Negation | `"not"` |
| Numeric | `"gt"`, `"gte"`, `"lt"`, `"lte"` |
| String | `"contains"`, `"icontains"`, `"startswith"`, `"endswith"`, `"ilike"`, `"regex"` |
| Membership | `"in"` |
| Null/Empty | `"isnull"`, `"empty"`, `"notEmpty"`, `"not_empty"` |
| Search | `"trigram_word_similar"`, `"full_text_search"` |

**Example — filter spans by status:**
```python
export_filter: FilterParamDict = {
    "status_code": {"operator": "", "value": "ERROR"},
}
```

**Example — numeric range:**
```python
export_filter: FilterParamDict = {
    "latency": {"operator": "gte", "value": 1000},
}
```

**Example — nested bundle with OR:**
```python
export_filter: FilterParamDict = {
    "model": {"operator": "in", "value": ["gpt-4", "claude-3"]},
    "filter_bundle": {
        "connector": "OR",
        "filter_params": {
            "status_code": {"operator": "", "value": "ERROR"},
            "latency": {"operator": "gte", "value": 5000},
        },
    },
}
```
