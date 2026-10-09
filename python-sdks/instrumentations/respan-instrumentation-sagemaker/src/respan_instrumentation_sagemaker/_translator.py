"""Native SageMaker JSON and consumed application stream data, without guesses."""

from __future__ import annotations

import base64
import json

from ._serialization import json_dumps, safe_text, to_jsonable


def decode(data):
    if type(data) is str:
        try:
            return json.loads(data)
        except ValueError:
            return data
    if type(data) in (bytes, bytearray):
        try:
            return decode(bytes(data).decode("utf-8"))
        except UnicodeDecodeError:
            return {"base64": base64.b64encode(data).decode("ascii")}
    return to_jsonable(data)


def request_body(params):
    return (
        decode(params.get("Body"))
        if type(params) is dict and type(params.get("Body")) in (str, bytes, bytearray)
        else None
    )


def first(mapping, keys):
    for name in keys:
        v = mapping.get(name)
        if type(v) is int and v >= 0:
            return v
    return None


def usage(payload):
    if type(payload) is not dict:
        return {}
    source = payload.get("usage")
    if type(source) is not dict:
        source = payload.get("details")
    if type(source) is not dict:
        return {}
    result = {}
    for name, keys in [
        (
            "input_tokens",
            ("input_tokens", "inputTokens", "prompt_tokens", "promptTokens"),
        ),
        (
            "output_tokens",
            ("output_tokens", "outputTokens", "completion_tokens", "generated_tokens"),
        ),
        ("total_tokens", ("total_tokens", "totalTokens")),
        ("cache_read", ("cache_read_input_tokens",)),
        ("cache_creation", ("cache_creation_input_tokens",)),
    ]:
        number = first(source, keys)
        if number is not None:
            result[name] = number
    details = source.get("prompt_tokens_details")
    if type(details) is dict and type(details.get("cached_tokens")) is int:
        result["cache_read"] = details["cached_tokens"]
    details = source.get("completion_tokens_details")
    if type(details) is dict and type(details.get("reasoning_tokens")) is int:
        result["reasoning"] = details["reasoning_tokens"]
    return result


def embedding(payload):
    if (
        type(payload) is dict
        and type(payload.get("data")) is list
        and payload["data"]
        and all(
            type(v) is dict
            and v.get("object") == "embedding"
            and type(v.get("embedding")) is list
            for v in payload["data"]
        )
    ):
        return [v["embedding"] for v in payload["data"]]
    return None


class StreamData:
    """Incremental UTF-8/JSON/NDJSON/SSE decoding of actual consumed PayloadPart bytes."""

    def __init__(self):
        self.data = bytearray()
        self.events = []
        self.count = 0

    def add(self, event):
        if type(event) is not dict:
            return
        part = event.get("PayloadPart")
        if type(part) is dict and type(part.get("Bytes")) is bytes:
            self.data.extend(part["Bytes"])
            self.count += 1

    def clear(self):
        self.data.clear()
        self.events.clear()

    def payload(self):
        if not self.data:
            return None
        try:
            raw = self.data.decode("utf-8")
        except UnicodeDecodeError:
            return decode(bytes(self.data))
        decoder = json.JSONDecoder()
        pending = raw.strip()
        frames = []
        while pending:
            if pending.startswith("data:"):
                line, separator, tail = pending.partition("\n")
                candidate = line[5:].strip()
                pending = tail.strip() if separator else ""
                if candidate == "[DONE]":
                    continue
                try:
                    frames.append(json.loads(candidate))
                except ValueError:
                    return {"frames": frames, "partial": safe_text(candidate)}
                continue
            try:
                frame, end = decoder.raw_decode(pending)
            except ValueError:
                return (
                    {"frames": frames, "partial": safe_text(pending)}
                    if frames
                    else safe_text(raw)
                )
            frames.append(frame)
            pending = pending[end:].strip()
        if len(frames) == 1:
            return frames[0]
        choices = {}
        texts = []
        last_usage = None
        model = None
        for frame in frames:
            if type(frame) is not dict:
                continue
            if type(frame.get("model")) is str:
                model = frame["model"]
            if type(frame.get("usage")) is dict:
                last_usage = frame["usage"]
            token = frame.get("token")
            if type(token) is dict and type(token.get("text")) is str:
                texts.append(token["text"])
            for choice in (
                frame.get("choices", []) if type(frame.get("choices")) is list else []
            ):
                if type(choice) is not dict:
                    continue
                idx = choice.get("index", 0)
                delta = choice.get("delta")
                if type(idx) is not int or type(delta) is not dict:
                    continue
                target = choices.setdefault(idx, {"content": "", "tool_calls": {}})
                if type(delta.get("role")) is str:
                    target["role"] = delta["role"]
                if type(delta.get("content")) is str:
                    target["content"] += delta["content"]
                for tool in (
                    delta.get("tool_calls", [])
                    if type(delta.get("tool_calls")) is list
                    else []
                ):
                    if type(tool) is not dict:
                        continue
                    tid = tool.get("index", 0)
                    if type(tid) is not int:
                        continue
                    item = target["tool_calls"].setdefault(
                        tid, {"function": {"arguments": ""}}
                    )
                    for k in ("id", "type"):
                        if k in tool:
                            item[k] = tool[k]
                    fn = tool.get("function")
                    if type(fn) is dict:
                        if type(fn.get("name")) is str:
                            item["function"]["name"] = fn["name"]
                        if type(fn.get("arguments")) is str:
                            item["function"]["arguments"] += fn["arguments"]
        result = {"frames": frames}
        if choices:
            result["choices"] = []
            for idx, target in sorted(choices.items()):
                target["tool_calls"] = [
                    v for _, v in sorted(target["tool_calls"].items())
                ]
                result["choices"].append({"index": idx, "message": target})
        if texts:
            result["generated_text"] = "".join(texts)
        if last_usage is not None:
            result["usage"] = last_usage
        if model is not None:
            result["model"] = model
        return result


safe_json = json_dumps
redact_text = safe_text
