import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from respan_sdk.constants.llm_logging import LOG_TYPE_GENERATION
from respan_sdk.respan_types import RespanParams
from respan_sdk.utils.export import send_payloads, validate_payload
from respan_sdk.utils.serialization import safe_json_dumps
from respan_sdk.utils.time import now_utc

from respan_exporter_dify.constants import (
    EMPTY_VALUES,
    MESSAGE_ID_PATHS,
    MESSAGE_TYPE_PATHS,
    NESTED_MESSAGE_PATHS,
    NESTED_MESSAGES_PATHS,
    RESPONSE_CONTENT_PATHS,
    SESSION_ID_PATHS,
    TOTAL_TOKENS_PATH,
    USAGE_PATHS,
)


logger = logging.getLogger(__name__)

# Max recursion depth for _to_serializable to avoid hitting Python's recursion limit
# on deeply nested Dify responses (e.g. long conversation histories).
_TO_SERIALIZABLE_MAX_DEPTH = 50


def _to_serializable(value: Any, max_depth: int = _TO_SERIALIZABLE_MAX_DEPTH) -> Any:
    """
    Convert value to JSON-serializable form (dict/list/primitives).

    Recursion is capped at max_depth to avoid stack overflow on deeply nested
    structures; beyond that, containers are stringified.
    """
    if max_depth <= 0:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        return str(value)

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _to_serializable(v, max_depth - 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item, max_depth - 1) for item in value]
    if hasattr(value, "model_dump"):
        try:
            d = value.model_dump(mode="json", exclude_none=True)
            return _to_serializable(d, max_depth - 1)
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            d = value.dict()
            return _to_serializable(d, max_depth - 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                key: _to_serializable(val, max_depth - 1)
                for key, val in vars(value).items()
                if not key.startswith("_")
            }
        except Exception:
            pass
    return value


def _class_name(value: Any) -> str:
    return getattr(getattr(value, "__class__", None), "__name__", "")


def _get_field(value: Any, field: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _get_nested(value: Any, path: Sequence[str]) -> Any:
    current = value
    for part in path:
        current = _get_field(current, part)
        if current is None:
            return None
    return current


def _first_non_empty_nested(value: Any, paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        nested_value = _get_nested(value, path)
        if nested_value not in EMPTY_VALUES:
            return nested_value
    return None


def _coerce_usage(usage_value: Any) -> Optional[Dict[str, Any]]:
    if usage_value is None:
        return None

    usage = _to_serializable(usage_value)
    if not isinstance(usage, dict):
        return None

    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = usage.get("input_tokens")

    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = usage.get("output_tokens")

    total_tokens = usage.get("total_tokens")

    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens

    return usage


def _extract_usage(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None

    for path in USAGE_PATHS:
        usage = _coerce_usage(_get_nested(value, path))
        if usage:
            return usage

    total_tokens = _get_nested(value, TOTAL_TOKENS_PATH)
    if isinstance(total_tokens, int):
        return {"total_tokens": total_tokens}

    return None


def _extract_message_id(message: Any) -> Optional[str]:
    value = _first_non_empty_nested(message, MESSAGE_ID_PATHS)
    if value is not None:
        return str(value)
    return None


def _is_assistant_message_with_usage(message: Any) -> bool:
    """True if message is an AssistantMessage instance that carries usage (one complete assistant turn)."""
    class_name = _class_name(message)
    has_usage = _get_field(message, "usage") is not None
    return class_name == "AssistantMessage" and has_usage


def _is_assistant_message(message: Any) -> bool:
    """True if the message is from the assistant (by class name, role, or type), regardless of usage."""
    class_name = _class_name(message)
    if class_name == "AssistantMessage":
        return True

    role = _get_field(message, "role")
    if isinstance(role, str) and role.lower() == "assistant":
        return True

    message_type = _get_field(message, "type")
    if hasattr(message_type, "value"):
        message_type = message_type.value
    if isinstance(message_type, str):
        normalized = message_type.lower()
        if normalized in ("assistant", "assistant_message"):
            return True

    return False


def _extract_message_usage(message: Any) -> Optional[Dict[str, Any]]:
    if _is_assistant_message_with_usage(message):
        usage_value = _to_serializable(_get_field(message, "usage"))
        if isinstance(usage_value, dict):
            return usage_value
        return None

    if _is_assistant_message(message) and _get_field(message, "usage") is not None:
        return _coerce_usage(_get_field(message, "usage"))

    return _extract_usage(message)


def _extract_session_id(
    *,
    message: Any,
    hook_input: Any,
) -> Optional[Union[str, int]]:
    message_session_id = _first_non_empty_nested(message, SESSION_ID_PATHS)
    if message_session_id is not None:
        return message_session_id
    return _first_non_empty_nested(hook_input, SESSION_ID_PATHS)


def _extract_message_type(*, method_name: str, message: Any) -> str:
    for path in MESSAGE_TYPE_PATHS:
        value = _get_nested(message, path)
        if value is None:
            continue
        if hasattr(value, "value"):
            value = value.value
        value_str = str(value).strip()
        if value_str:
            return value_str
    return method_name


def _extract_response_content(*, message: Any, result: Any) -> Any:
    for path in RESPONSE_CONTENT_PATHS:
        value = _get_nested(message, path)
        if value is not None:
            return value

    if message is not None:
        return message
    return result


def _extract_messages(result: Any, kwargs: Dict[str, Any]) -> List[Any]:
    if isinstance(result, list):
        messages = list(result)
    elif isinstance(result, tuple):
        messages = list(result)
    else:
        messages = []
        for path in NESTED_MESSAGES_PATHS:
            nested_messages = _get_nested(result, path)
            if isinstance(nested_messages, list):
                messages = nested_messages
                break
            if isinstance(nested_messages, tuple):
                messages = list(nested_messages)
                break

        if not messages:
            for path in NESTED_MESSAGE_PATHS:
                nested_message = _get_nested(result, path)
                if nested_message is not None:
                    messages = [nested_message]
                    break

        if not messages and result is not None:
            messages = [result]

    if messages:
        return messages

    request_payload = kwargs.get("req")
    if request_payload is not None:
        return [request_payload]

    return [{"event": _method_name_from_kwargs(kwargs=kwargs)}]


def _method_name_from_kwargs(*, kwargs: Dict[str, Any]) -> str:
    return str(kwargs.get("method_name") or "unknown")


def _usage_to_payload_fields(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Build payload fields from already-normalized usage (e.g. from _coerce_usage)."""
    out: Dict[str, Any] = {}
    if usage.get("prompt_tokens") is not None:
        out["prompt_tokens"] = usage["prompt_tokens"]
    if usage.get("completion_tokens") is not None:
        out["completion_tokens"] = usage["completion_tokens"]
    if usage.get("total_tokens") is not None:
        out["total_request_tokens"] = usage["total_tokens"]
    return out


def build_payload(
    *,
    method_name: str,
    start_time: datetime,
    end_time: datetime,
    status: str,
    input_value: Any,
    output_value: Any,
    error_message: Optional[str],
    export_params: Optional[RespanParams],
    span_name: Optional[str] = None,
    session_identifier: Optional[Union[str, int]] = None,
    usage: Optional[Dict[str, Any]] = None,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params = export_params or RespanParams()

    payload: Dict[str, Any] = {
        "span_workflow_name": params.span_workflow_name or "dify",
        "span_name": span_name or params.span_name or f"dify.{method_name}",
        "log_type": params.log_type or LOG_TYPE_GENERATION,
        "start_time": start_time.isoformat(),
        "timestamp": end_time.isoformat(),
        "latency": (end_time - start_time).total_seconds(),
        "status": status,
    }

    if input_value is not None:
        payload["input"] = safe_json_dumps(input_value) if not isinstance(input_value, str) else input_value
    if output_value is not None:
        payload["output"] = safe_json_dumps(output_value) if not isinstance(output_value, str) else output_value
    if error_message:
        payload["error_message"] = error_message

    if params.trace_unique_id:
        payload["trace_unique_id"] = params.trace_unique_id
        payload["trace_name"] = params.trace_name or payload["span_workflow_name"]

    if params.span_unique_id:
        payload["span_unique_id"] = params.span_unique_id
    if params.span_parent_id:
        payload["span_parent_id"] = params.span_parent_id

    chosen_session_id = (
        session_identifier
        if session_identifier not in (None, "")
        else params.session_identifier
    )
    if chosen_session_id not in (None, ""):
        payload["session_identifier"] = chosen_session_id

    if params.customer_identifier:
        payload["customer_identifier"] = params.customer_identifier

    if usage:
        normalized_usage = _coerce_usage(usage)
        if normalized_usage:
            payload["usage"] = normalized_usage
            payload.update(_usage_to_payload_fields(normalized_usage))

    metadata: Dict[str, Any] = {}
    if params.metadata:
        metadata.update(params.metadata)
    metadata["integration"] = "dify"
    metadata["method"] = method_name
    if metadata_extra:
        metadata.update(metadata_extra)

    if metadata:
        payload["metadata"] = metadata

    return payload


def _build_export_payloads(
    *,
    method_name: str,
    start_time: datetime,
    end_time: datetime,
    status: str,
    kwargs: Dict[str, Any],
    result: Any,
    error_message: Optional[str],
    params: RespanParams,
) -> List[Dict[str, Any]]:
    result_usage = _extract_usage(result)
    hook_input = kwargs.get("req") or kwargs
    messages = _extract_messages(result=result, kwargs={**kwargs, "method_name": method_name})

    payloads: List[Dict[str, Any]] = []
    seen_turn_ids: Set[str] = set()
    for message in messages:
        is_assistant_message_with_usage = _is_assistant_message_with_usage(message)
        message_id = _extract_message_id(message)
        if is_assistant_message_with_usage and message_id:
            if message_id in seen_turn_ids:
                continue
            seen_turn_ids.add(message_id)

        message_type = _extract_message_type(method_name=method_name, message=message)
        span_name = params.span_name or f"dify.{message_type}"
        message_usage = _extract_message_usage(message)
        if message_usage is None and not is_assistant_message_with_usage:
            message_usage = result_usage
        session_id = _extract_session_id(message=message, hook_input=hook_input)
        metadata_extra: Dict[str, Any] = {"message_type": message_type}
        if message_id:
            metadata_extra["message_id"] = message_id

        payload = build_payload(
            method_name=method_name,
            start_time=start_time,
            end_time=end_time,
            status=status,
            input_value=_to_serializable(message),
            output_value=_to_serializable(
                _extract_response_content(message=message, result=result)
            ),
            error_message=error_message,
            export_params=params,
            span_name=span_name,
            session_identifier=session_id,
            usage=message_usage,
            metadata_extra=metadata_extra,
        )
        payloads.append(validate_payload(payload))

    return payloads


def export_dify_call(
    *,
    api_key: Optional[str],
    endpoint: str,
    timeout: int,
    method_name: str,
    start_time: datetime,
    end_time: datetime,
    status: str,
    kwargs: Dict[str, Any],
    result: Any,
    error_message: Optional[str],
    params: RespanParams,
) -> None:
    """
    Build Respan payloads and send them to ingest (fire-and-forget).

    Called synchronously from the Dify client's _call_and_export (main thread).
    Payload building runs here on the caller's thread; only the HTTP POST is
    offloaded to a daemon thread inside send_payloads.
    """
    if not api_key:
        return

    try:
        payloads = _build_export_payloads(
            method_name=method_name,
            start_time=start_time,
            end_time=end_time,
            status=status,
            kwargs=kwargs,
            result=result,
            error_message=error_message,
            params=params,
        )
    except Exception as exc:
        logger.exception("Failed to build Respan payloads: %s", exc)
        return

    if not payloads:
        return

    send_payloads(
        api_key=api_key,
        endpoint=endpoint,
        timeout=timeout,
        payloads=payloads,
        context="respan dify ingest",
    )
