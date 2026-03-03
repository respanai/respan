"""Tests for the export filter evaluator."""
import pytest
from respan_tracing.filters.evaluator import (
    evaluate_export_filter,
    apply_operator,
    normalize_operator,
)


# ============================================================================
# normalize_operator tests
# ============================================================================


class TestNormalizeOperator:
    def test_none_returns_equal(self):
        assert normalize_operator(None) == ""

    def test_empty_string_returns_equal(self):
        assert normalize_operator("") == ""

    def test_symbol_equal(self):
        assert normalize_operator("==") == ""
        assert normalize_operator("=") == ""

    def test_symbol_not_equal(self):
        assert normalize_operator("!=") == "not"
        assert normalize_operator("<>") == "not"

    def test_word_aliases_equal(self):
        assert normalize_operator("eq") == ""
        assert normalize_operator("equals") == ""
        assert normalize_operator("EQ") == ""

    def test_word_aliases_not_equal(self):
        assert normalize_operator("ne") == "not"
        assert normalize_operator("not_equal") == "not"
        assert normalize_operator("not_equals") == "not"

    def test_emptiness_aliases(self):
        assert normalize_operator("notEmpty") == "not_empty"
        assert normalize_operator("not_empty") == "not_empty"
        assert normalize_operator("empty") == "empty"
        assert normalize_operator("is_empty") == "empty"
        assert normalize_operator("isempty") == "empty"

    def test_alpha_lowercase(self):
        assert normalize_operator("GT") == "gt"
        assert normalize_operator("Contains") == "contains"
        assert normalize_operator("ilike") == "ilike"

    def test_non_alpha_passthrough(self):
        assert normalize_operator(">=") == ">="


# ============================================================================
# apply_operator tests
# ============================================================================


class TestApplyOperator:
    # --- Equality ---
    def test_equal_match(self):
        assert apply_operator("ERROR", "", "ERROR") is True

    def test_equal_no_match(self):
        assert apply_operator("OK", "", "ERROR") is False

    def test_not_equal(self):
        assert apply_operator("OK", "not", "ERROR") is True
        assert apply_operator("ERROR", "not", "ERROR") is False

    # --- Null handling ---
    def test_null_log_value_empty_check(self):
        assert apply_operator(None, "empty", None) is True

    def test_null_log_value_not_empty_check(self):
        assert apply_operator(None, "not_empty", None) is False

    def test_null_log_value_equal_to_none(self):
        assert apply_operator(None, "", None) is True

    def test_null_log_value_equal_to_something(self):
        assert apply_operator(None, "", "ERROR") is False

    def test_null_log_value_isnull_true(self):
        assert apply_operator(None, "isnull", True) is True

    def test_null_log_value_isnull_false(self):
        assert apply_operator(None, "isnull", False) is False

    # --- Numeric ---
    def test_gt(self):
        assert apply_operator(10, "gt", 5) is True
        assert apply_operator(5, "gt", 10) is False

    def test_gte(self):
        assert apply_operator(10, "gte", 10) is True
        assert apply_operator(9, "gte", 10) is False

    def test_lt(self):
        assert apply_operator(5, "lt", 10) is True
        assert apply_operator(10, "lt", 5) is False

    def test_lte(self):
        assert apply_operator(10, "lte", 10) is True
        assert apply_operator(11, "lte", 10) is False

    def test_numeric_with_strings(self):
        assert apply_operator("10", "gt", "5") is True

    # --- String ---
    def test_contains(self):
        assert apply_operator("hello world", "contains", "world") is True
        assert apply_operator("hello world", "contains", "xyz") is False

    def test_icontains(self):
        assert apply_operator("Hello World", "icontains", "hello") is True

    def test_ilike(self):
        assert apply_operator("Hello World", "ilike", "hello") is True

    def test_startswith(self):
        assert apply_operator("hello world", "startswith", "hello") is True
        assert apply_operator("hello world", "startswith", "world") is False

    def test_endswith(self):
        assert apply_operator("hello world", "endswith", "world") is True
        assert apply_operator("hello world", "endswith", "hello") is False

    def test_regex(self):
        assert apply_operator("error_123", "regex", r"error_\d+") is True
        assert apply_operator("warning_abc", "regex", r"error_\d+") is False

    # --- Membership ---
    def test_in_list(self):
        assert apply_operator("ERROR", "in", ["ERROR", "CRITICAL"]) is True
        assert apply_operator("OK", "in", ["ERROR", "CRITICAL"]) is False

    def test_in_scalar(self):
        assert apply_operator("ERROR", "in", "ERROR") is True

    def test_not_in_list(self):
        assert apply_operator("OK", "not_in", ["ERROR", "CRITICAL"]) is True
        assert apply_operator("ERROR", "not_in", ["ERROR", "CRITICAL"]) is False

    # --- Existence ---
    def test_not_empty(self):
        assert apply_operator("something", "not_empty", None) is True
        assert apply_operator("", "not_empty", None) is False

    def test_empty(self):
        assert apply_operator("", "empty", None) is True
        assert apply_operator(None, "empty", None) is True
        assert apply_operator("something", "empty", None) is False

    # --- Error handling ---
    def test_unknown_operator_returns_false(self):
        assert apply_operator("a", "unknown_op_xyz", "b") is False

    def test_type_error_returns_false(self):
        assert apply_operator("not_a_number", "gt", 5) is False


# ============================================================================
# evaluate_export_filter tests
# ============================================================================


class TestEvaluateExportFilter:
    def test_none_filter_exports_all(self):
        assert evaluate_export_filter({"status_code": "OK"}, None) is True

    def test_empty_filter_exports_all(self):
        assert evaluate_export_filter({"status_code": "OK"}, {}) is True

    def test_single_condition_match(self):
        span_data = {"status_code": "ERROR", "name": "my_task.task"}
        export_filter = {"status_code": {"operator": "", "value": "ERROR"}}
        assert evaluate_export_filter(span_data, export_filter) is True

    def test_single_condition_no_match(self):
        span_data = {"status_code": "OK", "name": "my_task.task"}
        export_filter = {"status_code": {"operator": "", "value": "ERROR"}}
        assert evaluate_export_filter(span_data, export_filter) is False

    def test_and_logic_all_match(self):
        span_data = {"status_code": "ERROR", "name": "critical_task.task"}
        export_filter = {
            "status_code": {"operator": "", "value": "ERROR"},
            "name": {"operator": "startswith", "value": "critical_"},
        }
        assert evaluate_export_filter(span_data, export_filter) is True

    def test_and_logic_one_fails(self):
        span_data = {"status_code": "OK", "name": "critical_task.task"}
        export_filter = {
            "status_code": {"operator": "", "value": "ERROR"},
            "name": {"operator": "startswith", "value": "critical_"},
        }
        assert evaluate_export_filter(span_data, export_filter) is False

    def test_missing_field_in_span_data(self):
        span_data = {"name": "my_task.task"}
        export_filter = {"status_code": {"operator": "", "value": "ERROR"}}
        assert evaluate_export_filter(span_data, export_filter) is False

    def test_numeric_filter(self):
        span_data = {"duration_ms": 500, "status_code": "OK"}
        export_filter = {"duration_ms": {"operator": "gt", "value": 100}}
        assert evaluate_export_filter(span_data, export_filter) is True

    def test_single_element_list_unwrap(self):
        span_data = {"status_code": "ERROR"}
        export_filter = {"status_code": {"operator": "", "value": ["ERROR"]}}
        assert evaluate_export_filter(span_data, export_filter) is True

    def test_multi_element_list_converts_to_in(self):
        span_data = {"status_code": "ERROR"}
        export_filter = {"status_code": {"operator": "", "value": ["ERROR", "CRITICAL"]}}
        assert evaluate_export_filter(span_data, export_filter) is True

    def test_multi_element_list_no_match(self):
        span_data = {"status_code": "OK"}
        export_filter = {"status_code": {"operator": "", "value": ["ERROR", "CRITICAL"]}}
        assert evaluate_export_filter(span_data, export_filter) is False

    def test_non_dict_condition_skipped(self):
        span_data = {"status_code": "OK"}
        export_filter = {"status_code": "not_a_dict"}
        assert evaluate_export_filter(span_data, export_filter) is True

    def test_span_attribute_filter(self):
        span_data = {
            "traceloop.entity.name": "critical_workflow",
            "status_code": "OK",
        }
        export_filter = {
            "traceloop.entity.name": {"operator": "startswith", "value": "critical_"},
        }
        assert evaluate_export_filter(span_data, export_filter) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
