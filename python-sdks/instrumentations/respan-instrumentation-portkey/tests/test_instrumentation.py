from respan_instrumentation_portkey._serialization import json_dumps, jsonable


def test_serialization_never_calls_unknown_hooks():
    class Hostile:
        def __str__(self):
            raise AssertionError("arbitrary string hook")

        def model_dump(self):
            raise AssertionError("arbitrary dump hook")

    assert "Hostile" in json_dumps(Hostile())


def test_serialization_preserves_complete_vectors_tools_and_redacts_credentials():
    data = {
        "vectors": list(range(3072)),
        "tools": [
            {"name": str(i), "parameters": {"field" + str(j): "x" for j in range(120)}}
            for i in range(65)
        ],
        "api_key": "secret",
        "text": "Bearer abc https://user:password@example.test/a?key=secret",
    }
    result = jsonable(data, complete=True)
    assert (
        len(result["vectors"]) == 3072
        and len(result["tools"]) == 65
        and len(result["tools"][0]["parameters"]) == 120
    )
    text = json_dumps(data, complete=True)
    assert (
        "secret" not in text
        and "password@example" not in text
        and "Bearer abc" not in text
    )


def test_quoted_credentials_with_spaces_and_escapes_redact_full_values():
    from respan_instrumentation_portkey._serialization import safe_text

    for text in [
        '{"api_key":"synthetic secret with spaces"}',
        "{'password':'synthetic secret with spaces'}",
        '{"authorization":"Basic synthetic secret with spaces"}',
        '{"secret":"synthetic \\"quoted\\" secret"}',
    ]:
        redacted = safe_text(text, complete=True)
        assert (
            "synthetic" not in redacted
            and " with spaces" not in redacted
            and "quoted" not in redacted
        )
        assert "<redacted>" in redacted
