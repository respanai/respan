"""Exact builtin and installed native storage without customer conversion hooks."""

from __future__ import annotations

import struct
import sys
import types
from array import array

import psycopg

try:
    from pgvector import Bit, HalfVector, SparseVector, Vector
except ImportError:
    from pgvector.utils import Bit, HalfVector, SparseVector, Vector
from psycopg import sql
from psycopg.types.json import Json, Jsonb

from ._serialization import safe_text, safe_type_name, to_jsonable

CONNECTIONS = (psycopg.Connection, psycopg.AsyncConnection)
CURSORS = (
    psycopg.Cursor,
    psycopg.ServerCursor,
    psycopg.AsyncCursor,
    psycopg.AsyncServerCursor,
)
VECTORS = (Vector, HalfVector, SparseVector, Bit)


def namespace(cls):
    return type.__dict__["__dict__"].__get__(cls)


def storage(value):
    for base in type.__dict__["__mro__"].__get__(type(value)):
        descriptor = namespace(base).get("__dict__")
        if type(descriptor) is types.GetSetDescriptorType:
            result = descriptor.__get__(value, type(value))
            return result if type(result) is dict else {}
    return {}


def member(value, name, default=None):
    state = storage(value)
    if name in state:
        return state[name]
    for base in type.__dict__["__mro__"].__get__(type(value)):
        descriptor = namespace(base).get(name)
        if any(
            type(descriptor) is cls
            for cls in (types.MemberDescriptorType, types.GetSetDescriptorType)
        ):
            return descriptor.__get__(value, type(value))
    return default


def known_get(value, name):
    for base in type.__dict__["__mro__"].__get__(type(value)):
        descriptor = namespace(base).get(name)
        if type(descriptor) is property or any(
            type(descriptor) is cls
            for cls in (types.MemberDescriptorType, types.GetSetDescriptorType)
        ):
            return descriptor.__get__(value, type(value))
    return None


def native_json(value, seen=None):
    kind = type(value)
    if any(kind is cls for cls in (dict, list, tuple)):
        active = seen if seen is not None else set()
        if id(value) in active:
            return "<cycle>"
        active.add(id(value))
        try:
            if kind is dict:
                return {safe_text(k): native_json(v, active) for k, v in value.items()}
            return [native_json(item, active) for item in value]
        finally:
            active.discard(id(value))
    if kind is array:
        return native_json(array.tolist(value), seen)
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        if kind is vars(numpy).get("ndarray"):
            if value.dtype.kind in "biuf":
                return native_json(numpy.ndarray.tolist(value), seen)
            return {"type": "ndarray"}
        names = (
            "bool_",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "longdouble",
        )
        if any(kind is vars(numpy).get(name) for name in names):
            generic = vars(numpy).get("generic")
            descriptor = namespace(generic).get("dtype")
            item = namespace(generic).get("item")
            if (
                type(descriptor) is types.GetSetDescriptorType
                and type(item) is types.MethodDescriptorType
            ):
                dtype = descriptor.__get__(value, kind)
                if dtype.kind in "biuf":
                    return native_json(item(value), seen)
    if kind is Vector:
        return native_json(member(value, "_value"), seen)
    if kind is HalfVector:
        raw = member(value, "_value")
        if type(raw) is array and raw.typecode == "H":
            return list(struct.unpack(f"{len(raw)}e", raw))
        return native_json(raw, seen)
    if kind is SparseVector:
        return native_json(
            {
                "dimensions": member(value, "_dim"),
                "indices": member(value, "_indices"),
                "values": member(value, "_values"),
            },
            seen,
        )
    if kind is Bit:
        length = member(value, "_length")
        raw = member(value, "_data")
        if type(length) is int and type(raw) is bytes and 0 <= length <= 8 * len(raw):
            return {
                "length": length,
                "bits": "".join(format(item, "08b") for item in raw)[:length],
            }
        raw = native_json(member(value, "_value"), seen)
        if type(raw) is list and all(type(item) is bool for item in raw):
            return {
                "length": len(raw),
                "bits": "".join("1" if item else "0" for item in raw),
            }
        return {"type": "Bit"}
    if kind is Json or kind is Jsonb:
        return {
            "type": safe_type_name(value),
            "value": native_json(member(value, "obj"), seen),
        }
    if any(
        kind is cls
        for cls in (sql.SQL, sql.Identifier, sql.Placeholder, sql.Literal, sql.Composed)
    ):
        return {
            "type": safe_type_name(value),
            "value": native_json(member(value, "_obj"), seen),
        }
    if any(kind is cls for cls in CONNECTIONS):
        return {"type": safe_type_name(value)}
    if any(kind is cls for cls in CURSORS):
        return {"type": safe_type_name(value), **cursor_metadata(value)}
    return to_jsonable(value)


def cursor_metadata(cursor):
    if not any(type(cursor) is cls for cls in CURSORS):
        return {}
    result = {}
    for source, name in (
        ("_rowcount", "rowcount"),
        ("_pos", "position"),
        ("_closed", "closed"),
    ):
        item = member(cursor, source)
        if any(type(item) is cls for cls in (int, bool)):
            result[name] = item
    raw = member(cursor, "pgresult")
    if type(raw) is not psycopg.pq.PGresult:
        return result
    tag = raw.command_status
    if type(tag) is bytes:
        result["statusmessage"] = safe_text(tag.decode("utf-8", errors="replace"))
    # Read the actual libpq row description, not Column()'s adapter lookups.
    result["columns"] = [
        {
            "name": safe_text(raw.fname(i).decode("utf-8", errors="replace")),
            "type_code": raw.ftype(i),
            "format": raw.fformat(i),
            "size": raw.fsize(i),
            "type_modifier": raw.fmod(i),
            "table_oid": raw.ftable(i),
            "table_column": raw.ftablecol(i),
        }
        for i in range(raw.nfields)
    ]
    return result


def connection_of(value):
    if any(type(value) is cls for cls in CONNECTIONS):
        return value
    if any(type(value) is cls for cls in CURSORS):
        return member(value, "_conn")
    return None


def database_metadata(value):
    connection = connection_of(value)
    if connection is None:
        return {}
    raw = member(connection, "pgconn")
    if type(raw) is not psycopg.pq.PGconn:
        return {}
    result = {}
    for field, name in (("db", "database"), ("host", "host"), ("port", "port")):
        item = known_get(raw, field)
        if type(item) is bytes:
            item = item.decode("utf-8", errors="replace")
        if type(item) is str:
            result[name] = safe_text(item)
    return result


def compiled_query(value):
    if not any(type(value) is cls for cls in CURSORS):
        return None
    query = member(value, "_query")
    if type(query) is psycopg._queries.PostgresQuery:
        raw = member(query, "query")
        if type(raw) is bytes:
            return raw.decode("utf-8", errors="replace")
    return None
