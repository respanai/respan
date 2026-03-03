"""
Lightweight filter evaluator for span export filtering.

Ported from automation/utils/core.py — no Django dependencies.
Supports the same operator vocabulary as the Respan platform's FilterParamDict.
"""
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Canonical operator constants (mirror backend utils/filter/constants.py Filter class)
_OP_EQUAL = ""
_OP_NOT = "not"
_OP_GT = "gt"
_OP_GTE = "gte"
_OP_LT = "lt"
_OP_LTE = "lte"
_OP_CONTAINS = "contains"
_OP_ICONTAINS = "icontains"
_OP_ILIKE = "ilike"
_OP_STARTS_WITH = "startswith"
_OP_ENDS_WITH = "endswith"
_OP_IN = "in"
_OP_NOT_IN = "not_in"
_OP_IS_NULL = "isnull"
_OP_REGEX = "regex"
_OP_HAS_KEY = "has_key"
_OP_EMPTY = "empty"
_OP_NOT_EMPTY = "not_empty"


def normalize_operator(operator_str: Any) -> str:
    """Normalize operator aliases to canonical forms.

    Accepts aliases like "eq", "==", "equals", "notEmpty" and returns
    the canonical operator string used by apply_operator().
    """
    if operator_str is None:
        return _OP_EQUAL

    op = str(operator_str).strip()
    if op == "":
        return _OP_EQUAL

    # Symbol aliases
    if op in {"==", "="}:
        return _OP_EQUAL
    if op in {"!=", "<>"}:
        return _OP_NOT

    lowered = op.lower()

    # Equality aliases
    if lowered in {"eq", "equals"}:
        return _OP_EQUAL
    if lowered in {"ne", "not_equal", "not_equals"}:
        return _OP_NOT

    # Emptiness aliases
    if lowered in {"notempty", "not_empty"}:
        return _OP_NOT_EMPTY
    if lowered in {"empty", "is_empty", "isempty"}:
        return _OP_EMPTY

    # Alpha operators → lowercase canonical form
    if op.isalpha():
        return lowered

    return op


def apply_operator(log_value: Any, operator_str: str, rule_value: Any) -> bool:
    """Apply the specified operator to compare log_value with rule_value.

    Expects scalar rule_value — list unwrapping is done by the caller.
    """
    try:
        operator_str = normalize_operator(operator_str)

        # Handle null values
        if log_value is None:
            if operator_str in (_OP_HAS_KEY, _OP_NOT_EMPTY):
                return False
            if operator_str == _OP_EMPTY:
                return True
            if operator_str == _OP_EQUAL:
                return rule_value is None
            if operator_str == _OP_NOT:
                return rule_value is not None
            if operator_str == _OP_IS_NULL:
                return bool(rule_value)
            return False

        # Equality
        if operator_str == _OP_EQUAL:
            return log_value == rule_value
        if operator_str == _OP_NOT:
            return log_value != rule_value
        if operator_str == _OP_IS_NULL:
            return not bool(rule_value)

        # Numeric
        if operator_str == _OP_GT:
            return float(log_value) > float(rule_value)
        if operator_str == _OP_GTE:
            return float(log_value) >= float(rule_value)
        if operator_str == _OP_LT:
            return float(log_value) < float(rule_value)
        if operator_str == _OP_LTE:
            return float(log_value) <= float(rule_value)

        # String
        if operator_str == _OP_CONTAINS:
            return str(rule_value) in str(log_value)
        if operator_str in (_OP_ICONTAINS, _OP_ILIKE):
            return str(rule_value).lower() in str(log_value).lower()
        if operator_str == _OP_STARTS_WITH:
            return str(log_value).startswith(str(rule_value))
        if operator_str == _OP_ENDS_WITH:
            return str(log_value).endswith(str(rule_value))
        if operator_str == _OP_REGEX:
            return bool(re.search(str(rule_value), str(log_value)))

        # Membership
        if operator_str == _OP_IN:
            return log_value in rule_value if isinstance(rule_value, list) else log_value == rule_value
        if operator_str == _OP_NOT_IN:
            return log_value not in rule_value if isinstance(rule_value, list) else log_value != rule_value

        # Existence
        if operator_str in (_OP_HAS_KEY, _OP_NOT_EMPTY):
            return log_value is not None and log_value != ""
        if operator_str == _OP_EMPTY:
            return log_value is None or log_value == ""

        logger.debug(f"Unknown operator: {operator_str}")
        return False

    except Exception as e:
        logger.debug(f"Error applying operator {operator_str}: {e}")
        return False


def evaluate_export_filter(
    span_data: Dict[str, Any],
    export_filter: Optional[Dict[str, Any]],
) -> bool:
    """Evaluate an export filter dict against span data.

    Uses AND logic — all field conditions must pass for the span to be exported.

    Args:
        span_data: Flat dict of span attributes (key → value).
        export_filter: Filter dict where each key is a field name mapping to
            {"operator": str, "value": any}. None means export everything.

    Returns:
        True if the span should be exported, False if it should be dropped.
    """
    if not export_filter:
        return True

    for field_name, field_condition in export_filter.items():
        if not isinstance(field_condition, dict):
            continue

        operator = field_condition.get("operator", "")
        rule_value = field_condition.get("value")
        actual_value = span_data.get(field_name)

        # Unwrap single-element lists (frontend sends lists)
        if isinstance(rule_value, list):
            if len(rule_value) == 1:
                rule_value = rule_value[0]
            elif not operator or operator in (_OP_EQUAL, "=", "==", "eq"):
                operator = _OP_IN

        if not apply_operator(
            log_value=actual_value,
            operator_str=operator,
            rule_value=rule_value,
        ):
            return False

    return True
