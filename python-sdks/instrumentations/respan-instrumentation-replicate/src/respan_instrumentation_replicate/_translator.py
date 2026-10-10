"""Map actual native Replicate values and metrics without inferred results."""

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.semconv_ai import SpanAttributes as AI
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA

from ._serialization import REDACTED, json_dumps, native_value, safe_text


def usage(prediction):
    data = (
        object.__getattribute__(prediction, "__dict__")
        if prediction is not None
        else {}
    )
    metrics = data.get("metrics") or {}
    if type(metrics) is not dict:
        return {}
    attrs = {}
    for keys, targets in [
        (
            (
                "input_token_count",
                "input_tokens",
                "prompt_token_count",
                "prompt_tokens",
            ),
            (GenAI.GEN_AI_USAGE_INPUT_TOKENS, AI.LLM_USAGE_PROMPT_TOKENS),
        ),
        (
            (
                "output_token_count",
                "output_tokens",
                "completion_token_count",
                "completion_tokens",
            ),
            (GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, AI.LLM_USAGE_COMPLETION_TOKENS),
        ),
        (("total_token_count", "total_tokens"), (AI.LLM_USAGE_TOTAL_TOKENS,)),
        (("cache_read_input_tokens",), (AI.LLM_USAGE_CACHE_READ_INPUT_TOKENS,)),
        (("cache_creation_input_tokens",), (AI.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,)),
    ]:
        value = next((metrics[k] for k in keys if type(metrics.get(k)) is int), None)
        if value is not None:
            attrs.update({t: value for t in targets})
    return attrs


def attributes(
    name, request, result, prediction, frames, *, has_result, model, operation
):
    attrs = {RESPAN_LOG_TYPE: "task", AI.TRACELOOP_ENTITY_NAME: name}
    options = request.get("kwargs", {})
    body = options.get("input")
    if (
        body is None
        and operation in ("run", "async_run", "stream", "async_stream")
        and len(request.get("args", [])) > 1
    ):
        body = request["args"][1]
    messages = body.get("messages") if type(body) is dict else None
    prompt = body.get("prompt") if type(body) is dict else None
    file_result = type(result) is dict and "url" in result
    if type(result) is list and result:
        file_result = all(type(v) is dict and "url" in v for v in result)
    embedding = (
        type(result) is dict
        and result.get("object") == "embedding"
        and type(result.get("embedding")) is list
    )
    management = operation not in ("run", "async_run", "stream", "async_stream")
    kind = (
        "embedding"
        if embedding
        else "task"
        if management or file_result
        else "chat"
        if type(messages) is list
        else "text"
        if type(prompt) is str
        else "task"
    )
    attrs[RESPAN_LOG_TYPE] = kind
    if kind in ("chat", "text", "embedding"):
        attrs[AI.LLM_SYSTEM] = "replicate"
        attrs[AI.LLM_REQUEST_TYPE] = "embedding" if embedding else "chat"
        if model is not None:
            attrs[AI.LLM_REQUEST_MODEL] = model
        attrs.update(usage(prediction))
    if type(messages) is list:
        for i, message in enumerate(messages):
            if type(message) is dict:
                for field in ("role", "content", "tool_calls"):
                    if field in message:
                        attrs[f"{AI.LLM_PROMPTS}.{i}.{field}"] = (
                            safe_text(message[field])
                            if type(message[field]) is str
                            else json_dumps(message[field])
                        )
    elif type(prompt) is str:
        attrs[AI.LLM_PROMPTS + ".0.role"] = "user"
        attrs[AI.LLM_PROMPTS + ".0.content"] = safe_text(prompt)
    if kind in ("text", "chat"):
        text = (
            result
            if type(result) is str
            else "".join(result)
            if type(result) is list and all(type(v) is str for v in result)
            else None
        )
        if frames:
            output_frames = [
                frame["data"]
                for frame in frames
                if type(frame) is dict
                and frame.get("event") == "output"
                and type(frame.get("data")) is str
            ]
            if output_frames:
                text = "".join(output_frames)
                if safe_text(text) != text:
                    # Credentials can span native SSE frames. Keep IDs/types, clear affected output data.
                    frames = [
                        {**frame, "data": REDACTED}
                        if type(frame) is dict and frame.get("event") == "output"
                        else frame
                        for frame in frames
                    ]
        if text is not None:
            attrs[AI.LLM_COMPLETIONS + ".0.role"] = "assistant"
            attrs[AI.LLM_COMPLETIONS + ".0.content"] = safe_text(text)
    if type(body) is dict and type(body.get("tools")) is list:
        attrs[AI.LLM_REQUEST_FUNCTIONS] = json_dumps(body["tools"])
    # Write complete canonical bodies last so native OTel indexed limits cannot evict them.
    attrs[AI.TRACELOOP_ENTITY_INPUT] = json_dumps(request)
    if frames:
        attrs[AI.TRACELOOP_ENTITY_OUTPUT] = json_dumps(frames)
    elif has_result:
        attrs[AI.TRACELOOP_ENTITY_OUTPUT] = json_dumps(
            result["embedding"] if embedding else result
        )
    if embedding and has_result:
        attrs[RESPAN_METADATA + ".replicate.result"] = json_dumps(result)
    native = native_value(prediction) if prediction is not None else None
    if native is not None:
        attrs[RESPAN_METADATA + ".replicate.prediction"] = json_dumps(native)
    return attrs
