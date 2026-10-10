"""Read exact released LanceDB/Arrow storage without customer conversion hooks."""

from __future__ import annotations

import datetime
import decimal
import inspect
import sys
import types
import uuid

import numpy as np
import pyarrow as pa

from ._serialization import json_dumps, safe_text, to_jsonable


def native_type(value):
    kind = type(value)
    fields = type.__dict__["__dict__"].__get__(kind)
    module = fields.get("__module__")
    name = type.__dict__["__name__"].__get__(kind)
    if type(module) is not str or not module.startswith("lancedb."):
        return False
    loaded = sys.modules.get(module)
    return loaded is not None and vars(loaded).get(name) is kind


def storage(value):
    return _raw_storage(value) if native_type(value) else {}


def _raw_storage(value):
    kind = type(value)
    for owner in type.__dict__["__mro__"].__get__(kind):
        descriptor = type.__dict__["__dict__"].__get__(owner).get("__dict__")
        if (
            type(descriptor) is types.GetSetDescriptorType
            or type(descriptor) is types.MemberDescriptorType
        ):
            result = descriptor.__get__(value, kind)
            return result if type(result) is dict else {}
    return {}


def _safe_axis(axis, pandas):
    if type(axis) is pandas.RangeIndex:
        return type(_raw_storage(axis).get("_range")) is range
    if type(axis) is not pandas.Index:
        return False
    data = _raw_storage(axis).get("_data")
    if type(data) is np.ndarray:
        labels = np.ndarray.tolist(data)
    else:
        from pandas.core.arrays.string_arrow import ArrowStringArray

        array = (
            _raw_storage(data).get("_pa_array")
            if type(data) is ArrowStringArray
            else None
        )
        if type(array) is not pa.ChunkedArray or not _arrow_safe(array.type):
            return False
        labels = pa.ChunkedArray.to_pylist(array)
    return all(
        any(type(label) is kind for kind in (str, bool, int, float)) for label in labels
    )


def _safe_block(block):
    from pandas.core.arrays.arrow.array import ArrowExtensionArray
    from pandas.core.arrays.string_arrow import ArrowStringArray
    from pandas.core.internals.blocks import ExtensionBlock, NumpyBlock

    if type(block) is NumpyBlock:
        return type(block.values) is np.ndarray
    if type(block) is not ExtensionBlock:
        return False
    value = block.values
    if type(value) is not ArrowStringArray and type(value) is not ArrowExtensionArray:
        return False
    array = _raw_storage(value).get("_pa_array")
    return type(array) is pa.ChunkedArray and _arrow_safe(array.type)


def _arrow_safe(dtype):
    # ExtensionScalar.as_py can be customer Python. Refuse extension storage;
    # builtin Arrow scalar conversion is implemented by the native SDK.
    if isinstance(dtype, pa.BaseExtensionType):
        return False
    if pa.types.is_struct(dtype):
        return all(_arrow_safe(field.type) for field in dtype)
    if any(
        fn(dtype)
        for fn in (
            pa.types.is_list,
            pa.types.is_large_list,
            pa.types.is_fixed_size_list,
        )
    ):
        return _arrow_safe(dtype.value_type)
    if pa.types.is_map(dtype):
        return _arrow_safe(dtype.key_type) and _arrow_safe(dtype.item_type)
    if pa.types.is_dictionary(dtype):
        return _arrow_safe(dtype.value_type)
    return True


def native_json(value, seen=None):
    kind = type(value)
    if any(kind is item for item in (dict, list, tuple)):
        active = seen if seen is not None else set()
        if id(value) in active:
            return "<cycle>"
        active.add(id(value))
        try:
            if kind is dict:
                return {safe_text(k): native_json(v, active) for k, v in value.items()}
            return [native_json(v, active) for v in value]
        finally:
            active.discard(id(value))
    if kind is np.ndarray:
        return native_json(np.ndarray.tolist(value), seen)
    pandas = sys.modules.get("pandas")
    if pandas is not None and kind is vars(pandas).get("DataFrame"):
        from pandas.core.internals.managers import BlockManager

        manager = _raw_storage(value).get("_mgr")
        if (
            type(manager) is BlockManager
            and all(_safe_block(block) for block in manager.blocks)
            and all(_safe_axis(axis, pandas) for axis in manager.axes)
        ):
            return native_json(pandas.DataFrame.to_dict(value, orient="records"), seen)
        return {"type": "DataFrame"}
    if any(kind is item for item in (pa.Table, pa.RecordBatch)):
        if all(_arrow_safe(field.type) for field in value.schema):
            rows = kind.to_pylist(value)
            return native_json(rows, seen)
        return {"type": kind.__name__, "schema": native_json(value.schema, seen)}
    if kind is pa.Schema:

        def metadata(value):
            return {
                k.decode("utf-8", errors="replace"): v.decode("utf-8", errors="replace")
                for k, v in (value or {}).items()
            }

        return {
            "fields": [
                {
                    "name": f.name,
                    "type": str(f.type)
                    if any(
                        type(f.type) is candidate
                        for candidate in vars(pa.lib).values()
                        if isinstance(candidate, type)
                    )
                    else {"type": type.__dict__["__name__"].__get__(type(f.type))},
                    "nullable": f.nullable,
                    "metadata": metadata(f.metadata),
                }
                for f in value
            ],
            "metadata": metadata(value.metadata),
        }
    if any(kind is item for item in (datetime.datetime, datetime.date, datetime.time)):
        return kind.isoformat(value)
    if kind is datetime.timedelta:
        return {
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    if kind is decimal.Decimal or kind is uuid.UUID:
        return str(value)
    # PyO3 result classes report module=builtins. Verify native export identity
    # and invoke only their actual C storage getsets (no Python properties).
    native = sys.modules.get("lancedb._lancedb")
    if native is not None and any(
        kind is candidate
        for candidate in vars(native).values()
        if isinstance(candidate, type)
    ):
        fields = vars(kind)
        result = {
            name: native_json(descriptor.__get__(value, kind), seen)
            for name, descriptor in fields.items()
            if type(descriptor) is types.GetSetDescriptorType
            and not name.startswith("_")
        }
        return result if result else {"type": type.__dict__["__name__"].__get__(kind)}
    if native_type(value):
        fields = storage(value)
        # Dataclasses and released Pydantic SDK models have safe native fields.
        cls_fields = vars(kind)
        if "__dataclass_fields__" in cls_fields or "__pydantic_fields__" in cls_fields:
            return native_json(
                {
                    k: v
                    for k, v in fields.items()
                    if type(k) is str and not k.startswith("_")
                },
                seen,
            )
    return to_jsonable(value)


def dumps(value):
    return json_dumps(native_json(value))


def schemas(value):
    if type(value) is pa.Table or type(value) is pa.RecordBatch:
        return [value.schema]
    if type(value) is list or type(value) is tuple:
        return [schema for item in value for schema in schemas(item)]
    return []


def arguments(original, instance, args, kwargs):
    try:
        bound = inspect.signature(original).bind_partial(instance, *args, **kwargs)
        return {k: v for k, v in bound.arguments.items() if k != "self"}
    except (TypeError, ValueError):
        return {"args": args, "kwargs": kwargs}


def table_name(value):
    fields = storage(value)
    for _ in range(3):
        table = fields.get("_table")
        if native_type(table):
            fields = storage(table)
        else:
            break
    inner = fields.get("_inner")
    native = sys.modules.get("lancedb._lancedb")
    cls = vars(native).get("Table") if native is not None else None
    if cls is not None and type(inner) is cls:
        name = vars(cls)["name"](inner)
        return safe_text(name) if type(name) is str else None
    return None


def query_fields(value):
    fields = storage(value)
    # Copy only SDK-owned query configuration. No reranker, table, URI, or
    # opaque Rust query getters are called solely for telemetry.
    return {
        k.lstrip("_"): v
        for k, v in fields.items()
        if k not in {"_table", "_inner", "_reranker"} and type(k) is str
    }
