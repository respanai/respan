from __future__ import annotations

import json
from collections.abc import Iterator, Mapping

from respan_instrumentation_mirascope._serialization import json_string


def test_serialization_redacts_secrets_and_bounds_large_values() -> None:
    payload = {
        "api_key": "do-not-export",
        "nested": {"authorization": "Bearer do-not-export"},
        "prompt": "x" * 40_000,
    }

    encoded = json_string(payload)

    assert len(encoded) <= 16_000
    assert "do-not-export" not in encoded
    assert json.loads(encoded)


def test_serialization_does_not_fall_back_to_object_repr() -> None:
    class SecretRepr:
        __slots__ = ()

        def __repr__(self) -> str:
            return "token-from-repr"

    encoded = json_string({"value": SecretRepr()})

    assert "token-from-repr" not in encoded
    assert json.loads(encoded) == {"value": "<SecretRepr>"}


def test_serialization_does_not_stringify_arbitrary_mapping_keys() -> None:
    class SecretKey:
        stringify_calls = 0

        def __str__(self) -> str:
            type(self).stringify_calls += 1
            return "api_key=key-secret"

    encoded = json_string({SecretKey(): "safe"})

    assert SecretKey.stringify_calls == 0
    assert "key-secret" not in encoded
    assert json.loads(encoded) == {}


def test_serialization_redacts_secrets_embedded_in_direct_text() -> None:
    encoded = json_string(
        {
            "message": "Authorization: Bearer direct-text-secret",
            "query": "api_key=direct-query-secret",
        }
    )

    assert "direct-text-secret" not in encoded
    assert "direct-query-secret" not in encoded
    assert encoded.count("[REDACTED]") >= 2


def test_serialization_reads_only_the_bounded_mapping_prefix() -> None:
    class CountingMapping(Mapping[str, int]):
        def __init__(self) -> None:
            self.iterated = 0

        def __getitem__(self, key: str) -> int:
            return int(key)

        def __iter__(self) -> Iterator[str]:
            for index in range(10_000):
                self.iterated += 1
                yield str(index)

        def __len__(self) -> int:
            return 10_000

    value = CountingMapping()
    encoded = json_string(value)

    assert value.iterated == 50
    assert len(json.loads(encoded)) == 50


def test_escaped_quoted_auth_arguments_remain_json_and_idempotent():
    from respan_instrumentation_mirascope._serialization import safe_text

    original = json.dumps(
        {
            "token": 'Bearer fixture-token\\" quoted suffix',
            "api_key": 'secret with spaces "quoted"',
        }
    )
    redacted = safe_text(original, complete=True)
    assert json.loads(redacted)
    assert "fixture-token" not in redacted and "secret with spaces" not in redacted
    assert safe_text(redacted, complete=True) == redacted
    assert "fixture-token" in original
