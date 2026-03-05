"""Unit tests for respan_exporter_crewai.utils (no external deps)."""
from datetime import datetime, timezone
import importlib.util
import os
import sys

import pytest

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_here, "..", "src"))


@pytest.fixture(autouse=True, scope="module")
def _isolated_utils_module(request):
    """Load utils module in isolation and restore sys.modules after tests."""
    if _src not in sys.path:
        sys.path.insert(0, _src)
    original_pkg = sys.modules.get("respan_exporter_crewai")
    original_utils = sys.modules.get("respan_exporter_crewai.utils")

    _pkg = type(sys)("respan_exporter_crewai")
    _pkg.__path__ = [os.path.join(_src, "respan_exporter_crewai")]
    sys.modules["respan_exporter_crewai"] = _pkg
    _spec = importlib.util.spec_from_file_location(
        "respan_exporter_crewai.utils",
        os.path.join(_src, "respan_exporter_crewai", "utils.py"),
    )
    _utils_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_utils_mod)
    sys.modules["respan_exporter_crewai.utils"] = _utils_mod

    _names = (
        "as_dict", "build_traces_ingest_url", "clean_payload", "coerce_datetime",
        "coerce_token_count", "extract_metadata_payload", "extract_openinference_choice_texts",
        "extract_openinference_messages", "format_rfc3339", "format_span_id", "format_trace_id",
        "get_attr", "group_spans_by_trace", "infer_trace_start_time", "is_blank_value",
        "is_crewai_span", "is_hex_string", "merge_openinference_metadata", "messages_to_text",
        "normalize_respan_base_url_for_gateway", "normalize_span_id", "normalize_trace_id",
        "ns_to_datetime", "otel_span_to_dict", "parse_json_value", "pick_metadata_value",
        "serialize_value", "find_root_span", "to_completion_message", "to_prompt_messages",
    )
    for name in _names:
        setattr(request.module, name, getattr(_utils_mod, name))

    yield

    if original_pkg is None:
        sys.modules.pop("respan_exporter_crewai", None)
    else:
        sys.modules["respan_exporter_crewai"] = original_pkg
    if original_utils is None:
        sys.modules.pop("respan_exporter_crewai.utils", None)
    else:
        sys.modules["respan_exporter_crewai.utils"] = original_utils


# --- is_crewai_span ---


def test_is_crewai_span_scope_name_contains_crewai():
    """CrewAI scope name is detected."""
    scope = type("Scope", (), {"name": "openinference-instrumentation-crewai"})()
    span = type("Span", (), {"instrumentation_scope": scope, "attributes": {}})()
    assert is_crewai_span(span) is True


def test_is_crewai_span_scope_name_case_insensitive():
    """Scope name check is case-insensitive."""
    scope = type("Scope", (), {"name": "CREWAI-sdk"})()
    span = type("Span", (), {"instrumentation_scope": scope, "attributes": {}})()
    assert is_crewai_span(span) is True


def test_is_crewai_span_crewai_attribute_prefix():
    """crewai.* attribute prefix is detected."""
    span = type("Span", (), {"instrumentation_scope": None, "attributes": {"crewai.task.id": "x"}})()
    assert is_crewai_span(span) is True


def test_is_crewai_span_rejects_openinference_only():
    """Spans with only openinference.span.kind (e.g. LlamaIndex) are not CrewAI."""
    span = type("Span", (), {"instrumentation_scope": None, "attributes": {"openinference.span.kind": "AGENT"}})()
    assert is_crewai_span(span) is False


def test_is_crewai_span_rejects_graph_node_id_only():
    """Spans with only graph.node.id are not CrewAI."""
    span = type("Span", (), {"instrumentation_scope": None, "attributes": {"graph.node.id": "n1"}})()
    assert is_crewai_span(span) is False


def test_is_crewai_span_no_scope_no_attributes():
    """Span with no scope and no attributes is not CrewAI."""
    span = type("Span", (), {"instrumentation_scope": None, "attributes": {}})()
    assert is_crewai_span(span) is False


def test_is_crewai_span_uses_instrumentation_library_fallback():
    """Falls back to instrumentation_library if instrumentation_scope missing."""
    lib = type("Lib", (), {"name": "crewai-instrumentation"})()
    span = type("Span", (), {"instrumentation_scope": None, "instrumentation_library": lib, "attributes": {}})()
    assert is_crewai_span(span) is True


# --- ns_to_datetime, format_trace_id, format_span_id ---


def test_ns_to_datetime_none():
    assert ns_to_datetime(None) is None


def test_ns_to_datetime_zero():
    """Epoch zero (0 ns) converts to 1970-01-01 00:00:00 UTC."""
    dt = ns_to_datetime(0)
    assert dt is not None
    assert dt.year == 1970 and dt.month == 1 and dt.day == 1
    assert dt.hour == 0 and dt.minute == 0 and dt.second == 0


def test_ns_to_datetime_conversion():
    ts_ns = 1609459200 * 10**9  # 2021-01-01 00:00:00 UTC
    dt = ns_to_datetime(ts_ns)
    assert dt is not None
    assert dt.year == 2021 and dt.month == 1 and dt.day == 1


def test_format_trace_id():
    assert len(format_trace_id(0x1234567890ABCDEF)) == 32
    assert format_trace_id(0).startswith("0")


def test_format_span_id():
    assert len(format_span_id(0x12345678)) == 16


# --- otel_span_to_dict ---


def test_otel_span_to_dict_minimal_span():
    """Minimal span-like object with context and attributes."""
    ctx = type("Ctx", (), {"trace_id": 0xAB, "span_id": 0xCD})()
    span = type("Span", (), {
        "context": ctx,
        "attributes": {"openinference.span.kind": "AGENT"},
        "name": "test",
        "parent": None,
        "start_time": None,
        "end_time": None,
    })()
    d = otel_span_to_dict(span)
    assert d["trace_id"] == "000000000000000000000000000000ab"
    assert d["span_id"] == "00000000000000cd"
    assert d["name"] == "test"
    assert d["parent_id"] is None


def test_otel_span_to_dict_with_parent():
    """Span with parent sets parent_id."""
    ctx = type("Ctx", (), {"trace_id": 1, "span_id": 2})()
    parent = type("Parent", (), {"span_id": 1})()
    span = type("Span", (), {
        "context": ctx,
        "attributes": {},
        "name": "child",
        "parent": parent,
        "start_time": None,
        "end_time": None,
    })()
    d = otel_span_to_dict(span)
    assert d["parent_id"] == "0000000000000001"


def test_otel_span_to_dict_no_context_handled():
    """Span with no context does not crash."""
    span = type("Span", (), {"context": None, "attributes": {}, "name": "x", "parent": None})()
    d = otel_span_to_dict(span)
    assert d["trace_id"] is None
    assert d["span_id"] is None


# --- group_spans_by_trace ---


def test_group_spans_by_trace_empty():
    assert group_spans_by_trace([]) == {}


def test_group_spans_by_trace_groups_by_trace_id():
    spans = [
        {"trace_id": "aaa", "span_id": "1"},
        {"trace_id": "aaa", "span_id": "2"},
        {"trace_id": "bbb", "span_id": "1"},
    ]
    out = group_spans_by_trace(spans)
    assert set(out.keys()) == {"aaa", "bbb"}
    assert len(out["aaa"]) == 2
    assert len(out["bbb"]) == 1


def test_group_spans_by_trace_skips_invalid_trace_id():
    spans = [
        {"trace_id": "", "span_id": "1"},
        {"trace_id": None, "span_id": "2"},
        {"span_id": "3"},
    ]
    out = group_spans_by_trace(spans)
    assert out == {}


# --- get_attr ---


def test_get_attr_object():
    obj = type("O", (), {"foo": 42})()
    assert get_attr(obj, "foo") == 42
    assert get_attr(obj, "missing", default=0) == 0


def test_get_attr_dict():
    d = {"a": 1, "b": 2}
    assert get_attr(d, "a") == 1
    assert get_attr(d, "c", "a") == 1


def test_get_attr_none():
    assert get_attr(None, "x", default=3) == 3


# --- as_dict ---


def test_as_dict_none():
    assert as_dict(None) is None


def test_as_dict_dict():
    assert as_dict({"a": 1}) == {"a": 1}


def test_as_dict_model_dump():
    class M:
        def model_dump(self):
            return {"x": 1}
    assert as_dict(M()) == {"x": 1}


# --- pick_metadata_value ---


def test_pick_metadata_value():
    m = {"a": 1, "b": 2}
    assert pick_metadata_value(m, "a") == 1
    assert pick_metadata_value(m, "c", "b") == 2
    assert pick_metadata_value(m, "z") is None


def test_pick_metadata_value_none_metadata():
    assert pick_metadata_value(None, "a") is None


# --- coerce_datetime ---


def test_coerce_datetime_none():
    assert coerce_datetime(None) is None


def test_coerce_datetime_datetime():
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert coerce_datetime(dt) == dt


def test_coerce_datetime_timestamp():
    # Unix timestamp for 2024-01-01 00:00:00 UTC
    assert coerce_datetime(1704067200) is not None


# --- coerce_token_count ---


def test_coerce_token_count():
    assert coerce_token_count(100) == 100
    assert coerce_token_count("50") == 50
    assert coerce_token_count(None) is None
    assert coerce_token_count(True) is None


# --- clean_payload ---


def test_clean_payload_removes_none_and_empty():
    payload = {"a": 1, "b": None, "c": [], "d": {}, "e": 0}
    out = clean_payload(payload)
    assert out == {"a": 1, "e": 0}


# --- parse_json_value ---


def test_parse_json_value():
    assert parse_json_value('{"x":1}') == {"x": 1}
    assert parse_json_value("  [1,2]  ") == [1, 2]
    assert parse_json_value(42) == 42


# --- extract_metadata_payload, merge_openinference_metadata ---


def test_extract_metadata_payload():
    assert extract_metadata_payload(None) == {}
    assert extract_metadata_payload({}) == {}
    assert extract_metadata_payload({"metadata": {"k": "v"}}) == {"k": "v"}


def test_merge_openinference_metadata():
    m = {"a": 1, "metadata": {"b": 2}}
    out = merge_openinference_metadata(m)
    assert out.get("a") == 1
    assert out.get("b") == 2


# --- find_root_span ---


def test_find_root_span_empty():
    assert find_root_span([]) is None


def test_find_root_span_single():
    s = type("S", (), {"span_id": "1", "parent_id": None})()
    assert find_root_span([s]) is s


def test_find_root_span_parent_not_in_list():
    child = type("S", (), {"span_id": "2", "parent_id": "1"})()
    root = type("S", (), {"span_id": "1", "parent_id": None})()
    assert find_root_span([child, root]) is root


# --- to_prompt_messages, to_completion_message ---


def test_to_prompt_messages_list_of_roles():
    msgs = [{"role": "user", "content": "hi"}]
    assert to_prompt_messages(msgs) == msgs


def test_to_completion_message_dict():
    m = {"role": "assistant", "content": "hello"}
    assert to_completion_message(m) == m


# --- extract_openinference_messages ---


def test_extract_openinference_messages_empty():
    assert extract_openinference_messages(None, "llm.input_messages") is None
    assert extract_openinference_messages({}, "llm.input_messages") is None


def test_extract_openinference_messages_parses_prefix_keys():
    metadata = {
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "hello",
    }
    out = extract_openinference_messages(metadata, "llm.input_messages")
    assert out == [{"role": "user", "content": "hello"}]


# --- extract_openinference_choice_texts ---


def test_extract_openinference_choice_texts():
    metadata = {
        "llm.choices.0.completion.text": "first",
        "llm.choices.1.completion.text": "second",
    }
    out = extract_openinference_choice_texts(metadata)
    assert out == ["first", "second"]


# --- messages_to_text ---


def test_messages_to_text():
    assert messages_to_text([{"content": "a"}, {"content": "b"}]) == "a\nb"
    assert messages_to_text([]) is None


# --- is_blank_value ---


def test_is_blank_value():
    assert is_blank_value(None) is True
    assert is_blank_value("") is True
    assert is_blank_value("  ") is True
    assert is_blank_value("[]") is True
    assert is_blank_value("text") is False
    assert is_blank_value(0) is False


# --- normalize_trace_id, normalize_span_id, is_hex_string ---


def test_is_hex_string():
    assert is_hex_string("a" * 32, 32) is True
    assert is_hex_string("g", 1) is False
    assert is_hex_string("ab", 3) is False


def test_normalize_trace_id():
    assert len(normalize_trace_id("abc")) == 32
    assert normalize_trace_id("a" * 32) == "a" * 32


def test_normalize_span_id():
    assert len(normalize_span_id("x", "trace1")) == 16


# --- format_rfc3339, serialize_value ---


def test_format_rfc3339():
    dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    s = format_rfc3339(dt)
    assert "2024" in s and "12" in s


def test_serialize_value():
    assert serialize_value({"a": 1}) == '{"a": 1}' or '"a": 1' in serialize_value({"a": 1})


# --- infer_trace_start_time ---


def test_infer_trace_start_time_empty():
    assert infer_trace_start_time([]) is None


def test_infer_trace_start_time_from_span():
    # Seconds since epoch for 2024-01-01
    span = type("S", (), {"start_time": 1704067200})()
    out = infer_trace_start_time([span])
    assert out is not None
    assert out.year == 2024


# --- build_traces_ingest_url, normalize_respan_base_url_for_gateway ---


def test_build_traces_ingest_url_none_returns_default():
    default = "https://default.example/v1/traces/ingest"
    assert build_traces_ingest_url(None, default_endpoint=default) == default


def test_build_traces_ingest_url_already_ingest():
    assert build_traces_ingest_url("https://api.example/v1/traces/ingest").endswith("/v1/traces/ingest")
    assert "ingest" in build_traces_ingest_url("https://api.example/v1/traces/ingest")


def test_build_traces_ingest_url_v1_traces_adds_ingest():
    out = build_traces_ingest_url("https://api.example/v1/traces")
    assert out == "https://api.example/v1/traces/ingest"


def test_build_traces_ingest_url_api_adds_path():
    out = build_traces_ingest_url("https://api.example/api")
    assert out == "https://api.example/api/v1/traces/ingest"


def test_build_traces_ingest_url_bare_host_adds_api_and_path():
    out = build_traces_ingest_url("https://api.example")
    assert out == "https://api.example/api/v1/traces/ingest"


def test_normalize_respan_base_url_for_gateway_strips_ingest():
    out = normalize_respan_base_url_for_gateway("https://api.respan.ai/api/v1/traces/ingest")
    assert out == "https://api.respan.ai/api"


def test_normalize_respan_base_url_for_gateway_adds_api():
    out = normalize_respan_base_url_for_gateway("https://api.respan.ai")
    assert out == "https://api.respan.ai/api"
