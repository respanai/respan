"""Translate known released Vertex wrappers/protobufs without customer hooks."""

from __future__ import annotations

import dataclasses
import enum
import importlib

from respan_instrumentation_vertexai._serialization import (
    json_dumps,
    safe_text,
    to_jsonable,
)

_WRAPPERS = {
    "GenerationResponse": "_raw_response",
    "Candidate": "_raw_candidate",
    "Content": "_raw_content",
    "Part": "_raw_part",
    "Tool": "_raw_tool",
    "FunctionDeclaration": "_raw_function_declaration",
    "GenerationConfig": "_raw_generation_config",
    "ToolConfig": "_gapic_tool_config",
    "SafetySetting": "_raw_safety_setting",
}


def _known_type(value, prefix):
    cls = type(value)
    module = type.__getattribute__(cls, "__module__")
    name = type.__getattribute__(cls, "__name__")
    if type(module) is not str or not module.startswith(prefix):
        return False
    try:
        if getattr(importlib.import_module(module), name, None) is cls:
            return True
        if module == "vertexai.language_models":
            return (
                getattr(
                    importlib.import_module(
                        "vertexai.language_models._language_models"
                    ),
                    name,
                    None,
                )
                is cls
            )
        return False
    except ImportError:
        return False


def native_value(value):
    """Convert only builtin containers and types owned by the installed SDK."""
    kind = type(value)
    if kind in {dict, list, tuple}:
        if kind is dict:
            return {
                k
                if type(k) is str
                else (
                    native_value(k)
                    if _known_type(k, "google.cloud.aiplatform_v")
                    and issubclass(type(k), enum.Enum)
                    else f"<{type.__getattribute__(type(k), '__name__')}>"
                ): native_value(v)
                for k, v in value.items()
            }
        return [native_value(v) for v in value]
    if value is None or kind in {str, bool, int, float, bytes, bytearray, memoryview}:
        return to_jsonable(value)
    if _known_type(value, "google.cloud.aiplatform_v"):
        if issubclass(kind, enum.Enum):
            return enum.Enum.name.__get__(value)
        from google.protobuf.json_format import MessageToDict

        raw = object.__getattribute__(value, "_pb")
        return MessageToDict(raw, preserving_proto_field_name=True)
    if _known_type(value, "vertexai.generative_models"):
        name = type.__getattribute__(kind, "__name__")
        raw = _WRAPPERS.get(name)
        if raw:
            state = object.__getattribute__(value, "__dict__")
            return native_value(state[raw]) if raw in state else {"type": name}
    if _known_type(value, "vertexai.language_models"):
        name = type.__getattribute__(kind, "__name__")
        if name in {"TextEmbedding", "TextEmbeddingInput", "TextEmbeddingStatistics"}:
            state = object.__getattribute__(value, "__dict__")
            return {
                f.name: native_value(state[f.name])
                for f in dataclasses.fields(kind)
                if not f.name.startswith("_") and f.name in state
            }
    return {"type": type.__getattribute__(kind, "__name__")}


def _schema(value):
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if key == "properties" and type(item) is dict:
                result[key] = {
                    name: _schema(definition) for name, definition in item.items()
                }
            else:
                result["type" if key == "type_" else key] = _schema(item)
        if type(result.get("type")) is str:
            result["type"] = result["type"].lower()
        return result
    if type(value) is list:
        return [_schema(item) for item in value]
    return value


def _tool_call(value):
    call = {"type": "function", "function": {"name": value.get("name", "")}}
    if "args" in value:
        call["function"]["arguments"] = json_dumps(value["args"])
    if value.get("id"):
        call["id"] = value["id"]
    return call


def _content(value, role="user"):
    if type(value) is str:
        return {"role": role, "content": value}
    if type(value) is not dict:
        return {"role": role, "content": value}
    source_role = value.get("role", role)
    parts = value.get("parts", [value])
    text = []
    other = []
    calls = []
    response_ids = []
    for part in parts:
        if type(part) is str:
            text.append(part)
        elif type(part) is dict:
            if "text" in part:
                text.append(part["text"])
            elif "function_call" in part:
                calls.append(_tool_call(part["function_call"]))
            else:
                other.append(part)
                response = part.get("function_response")
                if type(response) is dict:
                    source_role = "tool"
                    if response.get("id"):
                        response_ids.append(response["id"])
    result = {"role": "assistant" if source_role == "model" else source_role}
    if "parts" in value:
        result["parts"] = parts
    if text:
        result["content"] = "".join(text) if all(type(t) is str for t in text) else text
    if other:
        result["content"] = (
            [result["content"], *other]
            if "content" in result
            else (other[0] if len(other) == 1 else other)
        )
    if calls:
        result["tool_calls"] = calls
    if response_ids:
        result["tool_call_id"] = (
            response_ids[0] if len(response_ids) == 1 else response_ids
        )
    return result


def normalize_input_messages(contents, *, system_instruction=None):
    value = native_value(contents)
    messages = []
    if system_instruction is not None:
        system = native_value(system_instruction)
        messages.append(
            _content(
                system
                if type(system) is dict
                else {"parts": system if type(system) is list else [system]},
                "system",
            )
        )
    if type(value) is list:
        if all(type(item) is dict and "parts" in item for item in value):
            messages.extend(_content(item) for item in value)
        else:
            messages.append(_content({"parts": value}))
    elif value is not None:
        messages.append(_content(value))
    return messages


def request_payload_from_call(*, instance, args, kwargs, embedding=False):
    # Only installed SDK instance storage is eligible; arbitrary subclass hooks
    # and properties are never consulted merely for observability.
    known = _known_type(instance, "vertexai.generative_models") or _known_type(
        instance, "vertexai.language_models"
    )
    state = object.__getattribute__(instance, "__dict__") if known else {}
    nested = state.get("_model")
    if nested is not None and _known_type(nested, "vertexai.generative_models"):
        model_state = object.__getattribute__(nested, "__dict__")
    else:
        model_state = state
    contents = kwargs.get(
        "texts" if embedding else "contents",
        kwargs.get("content", args[0] if args else None),
    )
    if type(state.get("_history")) is list:
        contents = (
            [*state["_history"], contents]
            if not (
                type(contents) is list
                and all(_known_type(c, "vertexai.generative_models") for c in contents)
            )
            else [*state["_history"], *contents]
        )
        # Normalize the new chat turn as a Content, alongside complete history.
        turns = []
        for item in contents:
            converted = native_value(item)
            if type(converted) is dict and "parts" in converted:
                turns.append(converted)
            else:
                turns.append(
                    {
                        "role": "user",
                        "parts": converted if type(converted) is list else [converted],
                    }
                )
        contents = turns
    model = model_state.get("_model_name", model_state.get("_model_id"))
    model = model.split("/models/")[-1] if type(model) is str else None
    return {
        "model": model,
        "contents": contents,
        "system_instruction": model_state.get("_system_instruction"),
        "tools": kwargs.get("tools", model_state.get("_tools")),
        "generation_config": kwargs.get(
            "generation_config", model_state.get("_generation_config")
        ),
        "tool_config": kwargs.get("tool_config", model_state.get("_tool_config")),
        "safety_settings": kwargs.get(
            "safety_settings", model_state.get("_safety_settings")
        ),
        "labels": kwargs.get("labels", model_state.get("_labels")),
        "supplied_parameters": kwargs,
        "stream": kwargs.get("stream") is True,
        "embedding": embedding,
    }


def extract_tools(tools):
    values = native_value(tools)
    if type(values) is not list:
        return []
    result = []
    for value in values:
        if type(value) is not dict:
            continue
        for definition in value.get("function_declarations", []):
            function = {
                key: item
                for key, item in definition.items()
                if key in {"name", "description"}
            }
            parameters = definition.get(
                "parameters_json_schema", definition.get("parameters")
            )
            if parameters is not None:
                function["parameters"] = _schema(parameters)
            if function.get("name"):
                result.append({"type": "function", "function": function})
    return result


def response_values(response):
    values = response if type(response) is list else [response]
    return [native_value(value) for value in values]


def extract_usage(response):
    values = response_values(response)
    for value in reversed(values):
        usage = value.get("usage_metadata") if type(value) is dict else None
        if type(usage) is dict:
            return {
                key: item
                for key, item in usage.items()
                if key
                in {
                    "prompt_token_count",
                    "candidates_token_count",
                    "thoughts_token_count",
                    "total_token_count",
                    "cached_content_token_count",
                }
                and type(item) is int
            }
    return {}


def response_messages(response):
    merged = {}
    for value in response_values(response):
        if type(value) is not dict:
            continue
        for offset, candidate in enumerate(value.get("candidates", [])):
            index = candidate.get("index", offset)
            message = _content(candidate.get("content", {}), "assistant")
            target = merged.setdefault(index, {"role": message["role"]})
            if "content" in message:
                if (
                    type(message["content"]) is str
                    and type(target.get("content", "")) is str
                ):
                    target["content"] = target.get("content", "") + message["content"]
                else:
                    target["content"] = (
                        message["content"]
                        if "content" not in target
                        else [target["content"], message["content"]]
                    )
            if message.get("tool_calls"):
                target.setdefault("tool_calls", []).extend(message["tool_calls"])
    return [merged[index] for index in sorted(merged)]


def extract_tool_calls(response_or_chunks):
    return [
        call
        for message in response_messages(response_or_chunks)
        for call in message.get("tool_calls", [])
    ]


def format_input(contents, *, system_instruction=None):
    return json_dumps(
        normalize_input_messages(contents, system_instruction=system_instruction)
    )


def format_output(response_or_chunks):
    return json_dumps(response_messages(response_or_chunks))


def safe_json(value):
    return json_dumps(value)


def to_json_attr(value):
    return safe_text(value) if type(value) is str else json_dumps(value)
