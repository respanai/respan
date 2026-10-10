"""Exact native Elastic storage; no customer conversion/getter hooks."""

from __future__ import annotations

import types

import elastic_transport as transport
from elastic_transport import _response
from elastic_transport._transport import TransportApiResponse
from elastic_transport.client_utils import DEFAULT

from ._serialization import to_jsonable

OMIT = object()


def namespace(cls):
    return type.__dict__["__dict__"].__get__(cls)


def storage(obj):
    for base in type.__dict__["__mro__"].__get__(type(obj)):
        descriptor = namespace(base).get("__dict__")
        if type(descriptor) is types.GetSetDescriptorType:
            return descriptor.__get__(obj, type(obj))
    return {}


def primitive(data):
    return data is None or any(
        type(data) is k
        for k in (str, int, float, bool, bytes, bytearray, dict, list, tuple)
    )


def value(data, seen=None):
    kind = type(data)
    if primitive(data):
        if type(data) is dict:
            active = set() if seen is None else seen
            if id(data) in active:
                return "<cycle>"
            active.add(id(data))
            try:
                return {
                    k: value(v, active)
                    for k, v in data.items()
                    if type(k) is str and v is not DEFAULT
                }
            finally:
                active.remove(id(data))
        if type(data) is list or type(data) is tuple:
            active = set() if seen is None else seen
            if id(data) in active:
                return "<cycle>"
            active.add(id(data))
            try:
                return [value(v, active) for v in data if v is not DEFAULT]
            finally:
                active.remove(id(data))
        return to_jsonable(data)
    if data is DEFAULT:
        return OMIT
    if kind is transport.HttpHeaders:
        return value(
            namespace(transport.HttpHeaders)["_internal"].__get__(data, kind), seen
        )
    if kind is transport.ApiResponseMeta:
        state = storage(data)
        return value({k: v for k, v in state.items() if k not in ("node",)}, seen)
    if any(
        kind is cls
        for cls in (
            _response.ApiResponse,
            _response.ObjectApiResponse,
            _response.ListApiResponse,
            _response.TextApiResponse,
            _response.BinaryApiResponse,
            _response.HeadApiResponse,
        )
    ):
        body = namespace(_response.ApiResponse)["_body"].__get__(data, kind)
        return value(body, seen)
    if kind is TransportApiResponse:
        return value(tuple.__getitem__(data, 1), seen)
    return to_jsonable(data)


def meta(data):
    if any(
        type(data) is cls
        for cls in (
            _response.ApiResponse,
            _response.ObjectApiResponse,
            _response.ListApiResponse,
            _response.TextApiResponse,
            _response.BinaryApiResponse,
            _response.HeadApiResponse,
        )
    ):
        return namespace(_response.ApiResponse)["_meta"].__get__(data, type(data))
    if type(data) is TransportApiResponse:
        return tuple.__getitem__(data, 0)
    # Native errors use builtin exception storage, never property getters.
    if any(
        base is BaseException for base in type.__dict__["__mro__"].__get__(type(data))
    ):
        return BaseException.__dict__["__dict__"].__get__(data).get("meta")
    return None


def status(data):
    m = meta(data)
    if type(m) is transport.ApiResponseMeta:
        n = storage(m).get("status")
        return n if type(n) is int else None
    return None


def body(data):
    if any(
        type(data) is cls
        for cls in (
            _response.ApiResponse,
            _response.ObjectApiResponse,
            _response.ListApiResponse,
            _response.TextApiResponse,
            _response.BinaryApiResponse,
            _response.HeadApiResponse,
        )
    ):
        return namespace(_response.ApiResponse)["_body"].__get__(data, type(data))
    if type(data) is TransportApiResponse:
        return tuple.__getitem__(data, 1)
    return data
