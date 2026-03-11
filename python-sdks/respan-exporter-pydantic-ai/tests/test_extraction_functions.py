"""Unit tests for pure extraction functions in instrument.py.

Each test targets a single code path with fixed dict inputs — no agents,
telemetry providers, or span processors involved.
"""

import json

import pytest
from respan_exporter_pydantic_ai.constants import (
    RESPAN_RESPONSE_FORMAT_ATTR,
    RESPAN_TOOLS_ATTR,
    PYDANTIC_AI_REQUEST_PARAMETERS_ATTR,
    PYDANTIC_AI_TOOL_DEFINITIONS_ATTR,
)
from respan_exporter_pydantic_ai.instrument import (
    _extract_response_format,
    _extract_tools,
    _normalize_tool_definition,
)


# ── _normalize_tool_definition ──────────────────────────────────────────────


class TestNormalizeToolDefinition:
    def test_function_payload_dict(self):
        """Recognises a tool definition with nested {"function": {...}}."""
        tool_def = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
        result = _normalize_tool_definition(tool_def)
        assert result is not None
        dumped = result.model_dump(exclude_none=True)
        assert dumped["type"] == "function"
        assert dumped["function"]["name"] == "get_weather"
        assert dumped["function"]["description"] == "Get weather info"
        assert "properties" in dumped["function"]["parameters"]

    def test_flat_name_and_parameters(self):
        """Builds FunctionTool from a flat dict with 'name' + 'parameters'."""
        tool_def = {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
        }
        result = _normalize_tool_definition(tool_def)
        assert result is not None
        dumped = result.model_dump(exclude_none=True)
        assert dumped["type"] == "function"
        assert dumped["function"]["name"] == "add"
        assert dumped["function"]["description"] == "Add two numbers"

    def test_flat_name_with_parameters_json_schema(self):
        """Falls back to 'parameters_json_schema' when 'parameters' is absent."""
        tool_def = {
            "name": "search",
            "parameters_json_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
        result = _normalize_tool_definition(tool_def)
        assert result is not None
        dumped = result.model_dump(exclude_none=True)
        assert dumped["function"]["name"] == "search"
        assert "properties" in dumped["function"]["parameters"]

    def test_flat_name_with_strict(self):
        """Passes through the 'strict' field."""
        tool_def = {"name": "tool", "strict": True}
        result = _normalize_tool_definition(tool_def)
        dumped = result.model_dump(exclude_none=True)
        assert dumped["function"]["strict"] is True

    def test_flat_name_no_parameters(self):
        """Creates a FunctionTool even without parameters or description."""
        tool_def = {"name": "noop"}
        result = _normalize_tool_definition(tool_def)
        assert result is not None
        dumped = result.model_dump(exclude_none=True)
        assert dumped["function"]["name"] == "noop"
        assert "parameters" not in dumped["function"]
        assert "description" not in dumped["function"]

    def test_no_function_no_name_returns_none(self):
        """Returns None when neither 'function' dict nor 'name' is present."""
        assert _normalize_tool_definition({}) is None
        assert _normalize_tool_definition({"type": "function"}) is None

    def test_custom_type_preserved(self):
        """Uses the dict's 'type' value instead of defaulting to 'function'."""
        tool_def = {"name": "t", "type": "custom_tool"}
        result = _normalize_tool_definition(tool_def)
        assert result.model_dump(exclude_none=True)["type"] == "custom_tool"


# ── _extract_tools ──────────────────────────────────────────────────────────


class TestExtractTools:
    def test_tools_attr_list(self):
        """Picks up tools from a pre-parsed 'tools' list attribute."""
        attrs = {
            RESPAN_TOOLS_ATTR: [
                {"type": "function", "function": {"name": "add"}},
                {"type": "function", "function": {"name": "sub"}},
            ]
        }
        result = _extract_tools(attrs)
        assert result is not None
        names = [t["function"]["name"] for t in result]
        assert names == ["add", "sub"]

    def test_tools_attr_json_string(self):
        """Parses tools from a JSON-encoded string in 'tools' attribute."""
        tools_list = [{"type": "function", "function": {"name": "lookup"}}]
        attrs = {RESPAN_TOOLS_ATTR: json.dumps(tools_list)}
        result = _extract_tools(attrs)
        assert result is not None
        assert result[0]["function"]["name"] == "lookup"

    def test_tool_definitions_attr(self):
        """Falls back to gen_ai.tool.definitions when 'tools' is absent."""
        defs = [{"name": "fetch", "description": "Fetch data"}]
        attrs = {PYDANTIC_AI_TOOL_DEFINITIONS_ATTR: json.dumps(defs)}
        result = _extract_tools(attrs)
        assert result is not None
        assert result[0]["function"]["name"] == "fetch"

    def test_request_parameters_fallback(self):
        """Extracts from function_tools + output_tools in model_request_parameters."""
        params = {
            "function_tools": [{"name": "alpha"}],
            "output_tools": [{"name": "beta"}],
        }
        attrs = {PYDANTIC_AI_REQUEST_PARAMETERS_ATTR: json.dumps(params)}
        result = _extract_tools(attrs)
        assert result is not None
        names = [t["function"]["name"] for t in result]
        assert "alpha" in names
        assert "beta" in names

    def test_request_parameters_function_tools_only(self):
        """Works when output_tools is absent."""
        params = {"function_tools": [{"name": "only_func"}]}
        attrs = {PYDANTIC_AI_REQUEST_PARAMETERS_ATTR: json.dumps(params)}
        result = _extract_tools(attrs)
        assert result is not None
        assert len(result) == 1

    def test_empty_tools_returns_none(self):
        """Returns None when tools list is present but empty."""
        assert _extract_tools({RESPAN_TOOLS_ATTR: []}) is None

    def test_no_tools_anywhere_returns_none(self):
        """Returns None when no tool source exists."""
        assert _extract_tools({}) is None
        assert _extract_tools({"unrelated": "value"}) is None

    def test_skips_non_dict_entries(self):
        """Filters out non-dict entries from the tools list."""
        attrs = {
            RESPAN_TOOLS_ATTR: [
                {"type": "function", "function": {"name": "valid"}},
                "not-a-dict",
                42,
                None,
            ]
        }
        result = _extract_tools(attrs)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "valid"

    def test_all_unnormalisable_entries_returns_none(self):
        """Returns None when every entry fails normalisation."""
        attrs = {RESPAN_TOOLS_ATTR: [{"type": "function"}]}  # no name, no function
        assert _extract_tools(attrs) is None


# ── _extract_response_format ───────────────────────────────────────────────


class TestExtractResponseFormat:
    def test_existing_dict(self):
        """Returns response_format when it's already a dict attribute."""
        attrs = {RESPAN_RESPONSE_FORMAT_ATTR: {"type": "json_object"}}
        result = _extract_response_format(attrs)
        assert result is not None
        assert result["type"] == "json_object"

    def test_existing_json_string(self):
        """Parses response_format from a JSON-encoded string."""
        attrs = {RESPAN_RESPONSE_FORMAT_ATTR: json.dumps({"type": "text"})}
        result = _extract_response_format(attrs)
        assert result is not None
        assert result["type"] == "text"

    def test_output_mode_text(self):
        """Maps output_mode='text' from request_parameters."""
        params = {"output_mode": "text"}
        attrs = {PYDANTIC_AI_REQUEST_PARAMETERS_ATTR: json.dumps(params)}
        result = _extract_response_format(attrs)
        assert result == {"type": "text"}

    def test_output_mode_image(self):
        """Maps output_mode='image' from request_parameters."""
        params = {"output_mode": "image"}
        attrs = {PYDANTIC_AI_REQUEST_PARAMETERS_ATTR: json.dumps(params)}
        result = _extract_response_format(attrs)
        assert result == {"type": "image"}

    @pytest.mark.parametrize("mode", ["native", "prompted"])
    def test_output_mode_json_schema_with_object(self, mode):
        """Maps native/prompted to json_schema with full schema details."""
        params = {
            "output_mode": mode,
            "output_object": {
                "json_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
                "name": "MyOutput",
                "description": "structured output",
                "strict": True,
            },
        }
        attrs = {PYDANTIC_AI_REQUEST_PARAMETERS_ATTR: json.dumps(params)}
        result = _extract_response_format(attrs)
        assert result is not None
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "MyOutput"
        assert result["json_schema"]["description"] == "structured output"
        assert result["json_schema"]["strict"] is True
        assert "properties" in result["json_schema"]["schema"]

    @pytest.mark.parametrize("mode", ["native", "prompted"])
    def test_output_mode_json_schema_without_schema(self, mode):
        """Falls back to bare json_schema type when output_object has no json_schema."""
        params = {"output_mode": mode, "output_object": {}}
        attrs = {PYDANTIC_AI_REQUEST_PARAMETERS_ATTR: json.dumps(params)}
        result = _extract_response_format(attrs)
        assert result is not None
        assert result["type"] == "json_schema"
        assert "json_schema" not in result  # no schema payload

    @pytest.mark.parametrize("mode", ["native", "prompted"])
    def test_output_mode_json_schema_no_output_object(self, mode):
        """Handles missing output_object by producing bare json_schema."""
        params = {"output_mode": mode}
        attrs = {PYDANTIC_AI_REQUEST_PARAMETERS_ATTR: json.dumps(params)}
        result = _extract_response_format(attrs)
        assert result is not None
        assert result["type"] == "json_schema"

    def test_unknown_output_mode_returns_none(self):
        """Returns None for unrecognised output_mode values."""
        params = {"output_mode": "some_future_mode"}
        attrs = {PYDANTIC_AI_REQUEST_PARAMETERS_ATTR: json.dumps(params)}
        assert _extract_response_format(attrs) is None

    def test_no_output_mode_returns_none(self):
        """Returns None when request_parameters exists but has no output_mode."""
        params = {"model_name": "gpt-4o"}
        attrs = {PYDANTIC_AI_REQUEST_PARAMETERS_ATTR: json.dumps(params)}
        assert _extract_response_format(attrs) is None

    def test_no_request_parameters_returns_none(self):
        """Returns None when there is nothing to extract from."""
        assert _extract_response_format({}) is None

    def test_invalid_json_string_returns_none(self):
        """Returns None when response_format is a non-JSON string and no params."""
        attrs = {RESPAN_RESPONSE_FORMAT_ATTR: "not-valid-json{"}
        assert _extract_response_format(attrs) is None
