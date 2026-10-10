"""No customer serialization hooks and no adapter truncation."""

import json

import ollama
from respan_instrumentation_ollama._privacy import json_text, text, value
from respan_instrumentation_ollama._translator import native_value


def test_builtin_recursive_cycle_and_native_fields():
    class Unknown:
        def __str__(self):
            raise AssertionError("str")

        def model_dump(self):
            raise AssertionError("model_dump")

        def items(self):
            raise AssertionError("items")

    cycle = {}
    cycle["self"] = cycle
    assert value(cycle) == {"self": None}
    assert value(Unknown()) is None
    response = ollama.ChatResponse(
        message=ollama.Message(role="assistant", content="", thinking="long" * 5000),
        prompt_eval_count=0,
    )
    result = native_value(response)
    assert (
        result["message"]["content"] == ""
        and len(result["message"]["thinking"]) == 20000
        and result["prompt_eval_count"] == 0
    )


def test_sanitizer_schema_and_text_credentials():
    source = {
        "format": {
            "properties": {"token": {"type": "string", "examples": ["private"]}}
        },
        "api_key": "private",
        "content": 'Bearer private-token password="private" https://user:pass@example.com/?token=private',
    }
    result = json_text(source)
    assert "private" not in result
    assert json.loads(result)["format"]["properties"]["token"]["type"] == "string"
    assert (
        json.loads(result)["format"]["properties"]["token"]["examples"] == "[REDACTED]"
    )


def test_redaction_idempotent_and_sensitive_schema_example():
    for source in (
        "password=controlled",
        'api_key="controlled"',
        'Bearer "controlled"',
        "Basic controlled",
        "https://user:pass@example.com/?token=controlled",
    ):
        once = text(source)
        assert text(once) == once and "controlled" not in once
    schema = {
        "format": {
            "properties": {
                "api_key": {
                    "type": "string",
                    "example": "controlled",
                    "examples": ["controlled"],
                    "default": "controlled",
                }
            }
        }
    }
    first = json_text(schema)
    assert "controlled" not in first
    assert json_text(json.loads(first)) == first
    assert json.loads(first)["format"]["properties"]["api_key"]["type"] == "string"
