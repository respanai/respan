"""Structured response schemas from Instructor's public response models."""

from collections.abc import Mapping
from types import NoneType, UnionType
from typing import (
    Any,
    Literal,
    NotRequired,
    Required,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)


def _response_model_name(model):
    return getattr(model, "__name__", None)


def _json_schema_for_annotation(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in {Required, NotRequired}:
        wrapped_annotation = args[0] if args else Any
        return _json_schema_for_annotation(wrapped_annotation)

    if origin is None:
        if annotation is str:
            return {"type": "string"}
        if annotation is int:
            return {"type": "integer"}
        if annotation is float:
            return {"type": "number"}
        if annotation is bool:
            return {"type": "boolean"}
        if annotation is None or annotation is NoneType:
            return {"type": "null"}
        if is_typeddict(annotation):
            return _typed_dict_json_schema(annotation)
        return {}

    if origin is Literal:
        values = list(args)
        schema: dict[str, Any] = {"enum": values}
        if values:
            value_type = type(values[0])
            if value_type is str:
                schema["type"] = "string"
            elif value_type is int:
                schema["type"] = "integer"
            elif value_type is float:
                schema["type"] = "number"
            elif value_type is bool:
                schema["type"] = "boolean"
        return schema

    if origin in {Union, UnionType}:
        return {"anyOf": [_json_schema_for_annotation(arg) for arg in args]}

    if origin in {list, tuple, set}:
        item_annotation = args[0] if args else Any
        return {
            "type": "array",
            "items": _json_schema_for_annotation(item_annotation),
        }

    if origin in {dict, Mapping}:
        return {"type": "object"}

    return {}


def _typed_dict_json_schema(response_model: Any) -> dict[str, Any]:
    try:
        annotations = get_type_hints(response_model, include_extras=True)
    except (NameError, TypeError):
        annotations = getattr(response_model, "__annotations__", {})
    required_keys = getattr(response_model, "__required_keys__", frozenset())
    properties = {
        field_name: _json_schema_for_annotation(field_type)
        | {
            "title": field_name.replace("_", " ").title(),
        }
        for field_name, field_type in annotations.items()
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "title": _response_model_name(response_model),
    }
    if required_keys:
        schema["required"] = sorted(required_keys)
    return schema


def _response_model_function_schema(response_model: Any) -> list[dict[str, Any]] | None:
    if response_model is None:
        return None

    model_json_schema = getattr(response_model, "model_json_schema", None)
    if callable(model_json_schema):
        schema = model_json_schema()
    elif is_typeddict(response_model):
        schema = _typed_dict_json_schema(response_model)
    else:
        return None

    model_name = _response_model_name(response_model) or "response_model"
    description = schema.get("description") if isinstance(schema, dict) else None

    return [
        {
            "type": "function",
            "function": {
                "name": model_name,
                "description": description or f"Structured response for {model_name}",
                "parameters": schema,
            },
        }
    ]
