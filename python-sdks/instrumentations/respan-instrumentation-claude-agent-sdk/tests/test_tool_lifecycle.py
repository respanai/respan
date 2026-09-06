"""Offline lifecycle regressions using real SDK messages and upstream OTel hooks."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ResultError, ResultMessage, ToolResultBlock, UserMessage
from opentelemetry.instrumentation.claude_agent_sdk import _context, _hooks
from opentelemetry.instrumentation.claude_agent_sdk._constants import (
    ERROR_TYPE,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_CALL_RESULT,
)
from opentelemetry.instrumentation.claude_agent_sdk._context import InvocationContext
from opentelemetry.instrumentation.claude_agent_sdk._instrumentor import (
    ClaudeAgentSdkInstrumentor as UpstreamInstrumentor,
)
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.trace import StatusCode

from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor, _instrumentation
from respan_instrumentation_claude_agent_sdk._tool_lifecycle import (
    _capture_failure_output,
    _reconcile_sdk_message,
)


class _Recorder(SpanProcessor):
    def __init__(self):
        self.spans = []

    def on_end(self, span):
        self.spans.append(span)


@pytest.fixture
def otel():
    provider = TracerProvider(shutdown_on_exit=False)
    recorder = _Recorder()
    provider.add_span_processor(recorder)
    tracer = provider.get_tracer("offline-claude-lifecycle")
    prior = _context.get_invocation_context()
    contexts = []

    def context(*, capture_content=True):
        value = InvocationContext(tracer.start_span("invocation"), capture_content=capture_content)
        contexts.append(value)
        return value

    yield SimpleNamespace(provider=provider, recorder=recorder, tracer=tracer, context=context)
    for value in contexts:
        value.cleanup_unclosed_spans()
        if value.invocation_span.is_recording():
            value.invocation_span.end()
    _context.set_invocation_context(prior)
    provider.shutdown()


@pytest.fixture
def patched_helpers():
    owner = ClaudeAgentSDKInstrumentor()
    assert owner._patch_upstream_helpers()
    try:
        yield owner
    finally:
        owner._restore_upstream_helpers()


async def _hook(otel, ctx, event, tool_id, data):
    prior = _context.get_invocation_context()
    _context.set_invocation_context(ctx)
    try:
        hooks = _hooks.build_instrumentation_hooks(otel.tracer, capture_content=ctx.capture_content)
        return await hooks[event][0].hooks[0](data, tool_id, None)
    finally:
        _context.set_invocation_context(prior)


async def _start_tool(otel, ctx, tool_id="toolu_pending"):
    await _hook(otel, ctx, "PreToolUse", tool_id, {"tool_name": "Bash", "tool_input": {"command": "offline fixture"}})
    return ctx.active_tool_spans[tool_id]


def _result(**kwargs):
    fields = dict(subtype="success", duration_ms=1, duration_api_ms=1,
                  is_error=False, num_turns=1, session_id="offline-session")
    fields.update(kwargs)
    return ResultMessage(**fields)


def _tool_result(tool_id, *, is_error=False, content="SDK result"):
    return UserMessage(content=[ToolResultBlock(tool_use_id=tool_id, content=content, is_error=is_error)])


def _end_counter(monkeypatch, span):
    calls = []
    original = span.end

    def end(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(span, "end", end)
    return calls


def test_failure_annotation_does_not_take_hook_end_ownership(otel):
    span = otel.tracer.start_span("tool")
    _capture_failure_output(span, "tool failed", capture_content=True)
    assert json.loads(span.attributes[GEN_AI_TOOL_CALL_RESULT]) == {"error": "tool failed"}
    assert span.is_recording()
    assert span.status.status_code is StatusCode.UNSET
    assert not otel.recorder.spans
    span.end()


def test_real_failure_hook_records_error_output_and_closes_once(otel, patched_helpers, monkeypatch):
    async def scenario():
        ctx = otel.context()
        span = await _start_tool(otel, ctx)
        endings = _end_counter(monkeypatch, span)
        await _hook(otel, ctx, "PostToolUseFailure", "toolu_pending", {"error": "command rejected"})
        assert not ctx.active_tool_spans
        assert json.loads(span.attributes[GEN_AI_TOOL_CALL_RESULT]) == {"error": "command rejected"}
        assert span.status.status_code is StatusCode.ERROR
        ctx.cleanup_unclosed_spans()
        assert len(endings) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("is_error", [False, True])
def test_sdk_tool_result_closes_missing_post_hook_before_cleanup(otel, is_error):
    async def scenario():
        ctx = otel.context()
        span = await _start_tool(otel, ctx)
        content = [{"type": "text", "text": "explicit SDK outcome"}]
        _reconcile_sdk_message(ctx, _tool_result("toolu_pending", is_error=is_error, content=content))
        assert not ctx.active_tool_spans
        assert not span.is_recording()
        assert span.attributes[GEN_AI_TOOL_CALL_ID] == "toolu_pending"
        assert json.loads(span.attributes[GEN_AI_TOOL_CALL_RESULT]) == content
        assert (span.status.status_code is StatusCode.ERROR) is is_error
        if is_error:
            assert span.attributes[ERROR_TYPE] == "tool_error"
        ctx.cleanup_unclosed_spans()
        assert span.status.description != "Span not properly closed"
    asyncio.run(scenario())


def test_permission_denials_are_explicit_outcomes_for_matching_tools_only(otel):
    async def scenario():
        ctx = otel.context()
        denied = await _start_tool(otel, ctx, "toolu_denied")
        orphan = await _start_tool(otel, ctx, "toolu_orphan")
        subagent = otel.tracer.start_span("subagent")
        ctx.active_subagent_spans["toolu_subagent"] = subagent
        _reconcile_sdk_message(ctx, _result(permission_denials=[
            {"tool_use_id": "toolu_denied", "tool_name": "Bash", "tool_input": {"command": "not stderr"}},
            {"tool_use_id": "toolu_subagent"}, {"tool_use_id": "unknown"}, {"tool_use_id": 7}, "invalid",
        ]))
        assert not denied.is_recording()
        assert denied.attributes[ERROR_TYPE] == "permission_denied"
        assert denied.status.status_code is StatusCode.ERROR
        assert json.loads(denied.attributes[GEN_AI_TOOL_CALL_RESULT]) == {
            "error": "permission_denied", "source": "ResultMessage.permission_denials",
        }
        assert orphan.is_recording() and subagent.is_recording()
        assert list(ctx.active_tool_spans) == ["toolu_orphan"]
        assert list(ctx.active_subagent_spans) == ["toolu_subagent"]
        ctx.cleanup_unclosed_spans()
        for span in (orphan, subagent):
            assert span.status.status_code is StatusCode.ERROR
            assert span.status.description == "Span not properly closed"
    asyncio.run(scenario())


@pytest.mark.parametrize("first", ["hook", "result"])
@pytest.mark.parametrize("is_error", [False, True])
def test_hook_result_duplicate_and_late_hook_end_each_invocation_once(otel, patched_helpers, monkeypatch, first, is_error):
    async def scenario():
        ctx = otel.context()
        span = await _start_tool(otel, ctx)
        endings = _end_counter(monkeypatch, span)
        event = "PostToolUseFailure" if is_error else "PostToolUse"
        hook_data = {"error": "hook failure"} if is_error else {"tool_response": "hook result"}
        message = _tool_result("toolu_pending", is_error=is_error)
        if first == "hook":
            await _hook(otel, ctx, event, "toolu_pending", hook_data)
            original_attrs = dict(span.attributes)
            _reconcile_sdk_message(ctx, message)
        else:
            _reconcile_sdk_message(ctx, message)
            original_attrs = dict(span.attributes)
            await _hook(otel, ctx, event, "toolu_pending", hook_data)
        _reconcile_sdk_message(ctx, message)
        _reconcile_sdk_message(ctx, _result(permission_denials=[{"tool_use_id": "toolu_pending"}]))
        await _hook(otel, ctx, "PostToolUseFailure", "toolu_pending", {"error": "late contradictory error"})
        ctx.cleanup_unclosed_spans()
        assert len(endings) == 1
        assert dict(span.attributes) == original_attrs
        assert (span.status.status_code is StatusCode.ERROR) is is_error
    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["success", "tool_error", "permission_denied", "failure_hook"])
def test_structure_only_keeps_status_without_sensitive_result_content(otel, patched_helpers, outcome):
    async def scenario():
        ctx = otel.context(capture_content=False)
        span = await _start_tool(otel, ctx)
        sentinel = "sensitive-failure-sentinel"
        if outcome == "failure_hook":
            await _hook(otel, ctx, "PostToolUseFailure", "toolu_pending", {"error": sentinel})
        elif outcome == "permission_denied":
            _reconcile_sdk_message(ctx, _result(permission_denials=[{"tool_use_id": "toolu_pending", "tool_input": {"command": sentinel}}]))
        else:
            _reconcile_sdk_message(ctx, _tool_result("toolu_pending", is_error=outcome == "tool_error", content=sentinel))
        assert not span.is_recording()
        assert GEN_AI_TOOL_CALL_RESULT not in span.attributes
        assert sentinel not in json.dumps(dict(span.attributes))
        assert sentinel not in (span.status.description or "")
        assert (span.status.status_code is StatusCode.ERROR) is (outcome != "success")
    asyncio.run(scenario())


def test_successful_agent_result_does_not_invent_outcomes_for_orphans(otel):
    async def scenario():
        ctx = otel.context()
        span = await _start_tool(otel, ctx)
        _reconcile_sdk_message(ctx, UserMessage(content="Tool permission denied, but no explicit tool result"))
        _reconcile_sdk_message(ctx, _result())
        assert span.is_recording()
        assert GEN_AI_TOOL_CALL_RESULT not in span.attributes
        ctx.cleanup_unclosed_spans()
        ctx.cleanup_unclosed_spans()
        assert span.status.description == "Span not properly closed"
        assert len(otel.recorder.spans) == 1
    asyncio.run(scenario())


def test_identical_tool_ids_in_separate_invocations_are_isolated(otel):
    async def scenario():
        first, second = otel.context(), otel.context()
        first_span = await _start_tool(otel, first, "toolu_reused")
        second_span = await _start_tool(otel, second, "toolu_reused")
        _context.set_invocation_context(second)
        _reconcile_sdk_message(first, _tool_result("toolu_reused"))
        assert not first_span.is_recording()
        assert second_span.is_recording()
        assert _context.get_invocation_context() is second
        _reconcile_sdk_message(second, _tool_result("toolu_reused", is_error=True))
        assert first_span.status.status_code is StatusCode.UNSET
        assert second_span.status.status_code is StatusCode.ERROR
    asyncio.run(scenario())


def _configure_upstream(monkeypatch, otel):
    upstream = UpstreamInstrumentor()
    meter = NoOpMeterProvider().get_meter("offline-claude")
    for name, value in {"_tracer": otel.tracer, "_agent_name": "offline", "_capture_content": True,
                        "_token_histogram": meter.create_histogram("tokens"), "_duration_histogram": meter.create_histogram("duration")}.items():
        monkeypatch.setattr(upstream, name, value, raising=False)
    return upstream


@pytest.mark.parametrize("mode", ["standalone", "client"])
@pytest.mark.parametrize("is_error", [False, True])
def test_real_wrapper_reconciles_tool_result_before_upstream_cleanup(otel, patched_helpers, monkeypatch, mode, is_error):
    async def scenario():
        upstream = _configure_upstream(monkeypatch, otel)
        previous = otel.context()
        client = SimpleNamespace(_otel_invocation_ctx=otel.context(), _query=SimpleNamespace(_otel_invocation_ctx=None))
        _context.set_invocation_context(previous)
        seen = {}
        message = _tool_result("toolu_missing_hook", is_error=is_error)
        terminal = _result()

        async def messages(*args, **kwargs):
            ctx = _context.get_invocation_context()
            seen["ctx"] = ctx
            seen["tool"] = await _start_tool(otel, ctx, "toolu_missing_hook")
            yield message
            yield terminal

        iterator = (upstream._instrumented_query(messages, (), {"prompt": "offline", "options": ClaudeAgentOptions()})
                    if mode == "standalone" else upstream._instrumented_receive_response(messages, client, (), {}))
        assert await anext(iterator) is message
        assert not seen["tool"].is_recording()
        assert not seen["ctx"].active_tool_spans
        assert (seen["tool"].status.status_code is StatusCode.ERROR) is is_error
        assert json.loads(seen["tool"].attributes[GEN_AI_TOOL_CALL_RESULT]) == "SDK result"
        assert await anext(iterator) is terminal
        await iterator.aclose()
        assert _context.get_invocation_context() is previous
        assert seen["tool"].status.description != "Span not properly closed"
        assert seen["ctx"].invocation_span.status.status_code is not StatusCode.ERROR
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["standalone", "client"])
@pytest.mark.parametrize("terminal", [False, True])
def test_wrapper_aclose_finalizes_upstream_and_restores_prior_context(otel, patched_helpers, monkeypatch, mode, terminal):
    async def scenario():
        upstream = _configure_upstream(monkeypatch, otel)
        previous = otel.context()
        invocation = otel.context() if mode == "client" else None
        client = SimpleNamespace(_otel_invocation_ctx=invocation, _query=SimpleNamespace(_otel_invocation_ctx=None))
        _context.set_invocation_context(previous)
        seen = {}

        async def messages(*args, **kwargs):
            ctx = _context.get_invocation_context()
            seen["ctx"] = ctx
            try:
                await _start_tool(otel, ctx, "toolu_orphan")
                if terminal:
                    yield _result(permission_denials=[{"tool_use_id": "toolu_orphan"}])
                else:
                    yield UserMessage(content="stream is still active")
            finally:
                seen["source_closed"] = True

        iterator = (upstream._instrumented_query(messages, (), {"prompt": "offline", "options": ClaudeAgentOptions()})
                    if mode == "standalone" else upstream._instrumented_receive_response(messages, client, (), {}))
        await anext(iterator)
        ctx = seen["ctx"]
        assert ctx is not previous
        await iterator.aclose()
        assert _context.get_invocation_context() is previous
        assert seen["source_closed"]
        assert not ctx.invocation_span.is_recording()  # Actual upstream finally ran.
        assert not ctx.active_tool_spans
        if mode == "client":
            assert client._otel_invocation_ctx is None
            assert client._query._otel_invocation_ctx is None
        if terminal:
            assert ctx.invocation_span.status.status_code is not StatusCode.ERROR
        else:
            assert ctx.invocation_span.status.status_code is StatusCode.ERROR
            tool = next(span for span in otel.recorder.spans if span.attributes.get(GEN_AI_TOOL_CALL_ID) == "toolu_orphan")
            assert tool.status.description == "Span not properly closed"
    asyncio.run(scenario())


def test_standalone_streaming_prompt_receives_result_before_next_input(otel, patched_helpers, monkeypatch):
    async def scenario():
        upstream = _configure_upstream(monkeypatch, otel)
        previous = otel.context()
        _context.set_invocation_context(previous)
        acknowledged = asyncio.Event()
        events = []
        seen = {}
        first, second = _result(num_turns=1), _result(num_turns=2)

        async def prompt():
            yield {"type": "user", "message": {"role": "user", "content": "turn one"}}
            await acknowledged.wait()
            events.append("second_input")
            yield {"type": "user", "message": {"role": "user", "content": "turn two"}}

        async def messages(*args, **kwargs):
            seen["ctx"] = _context.get_invocation_context()
            inputs = kwargs["prompt"]
            try:
                await anext(inputs)
                events.append("first_result")
                yield first
                await anext(inputs)
                events.append("second_result")
                yield second
            finally:
                await inputs.aclose()
                events.append("source_closed")

        iterator = upstream._instrumented_query(messages, (), {"prompt": prompt(), "options": ClaudeAgentOptions()})
        # Timeout stays in this task so it does not alter the ContextVar scope.
        async with asyncio.timeout(2):
            try:
                assert await anext(iterator) is first
                assert events == ["first_result"]
                events.append("caller_acknowledged")
                acknowledged.set()
                assert await anext(iterator) is second
                with pytest.raises(StopAsyncIteration):
                    await anext(iterator)
            finally:
                await iterator.aclose()
        assert events == ["first_result", "caller_acknowledged", "second_input", "second_result", "source_closed"]
        assert _context.get_invocation_context() is previous
        assert not seen["ctx"].invocation_span.is_recording()
        assert seen["ctx"].invocation_span.status.status_code is not StatusCode.ERROR
    asyncio.run(scenario())


def test_standalone_delivers_error_result_before_sdk_result_error(otel, patched_helpers, monkeypatch):
    async def scenario():
        upstream = _configure_upstream(monkeypatch, otel)
        previous = otel.context()
        _context.set_invocation_context(previous)
        terminal = _result(subtype="error_during_execution", is_error=True)
        failure = ResultError("offline terminal failure", data={"is_error": True}, exit_code=1)
        events = []
        seen = {}

        async def messages(*args, **kwargs):
            seen["ctx"] = _context.get_invocation_context()
            try:
                yield terminal
                events.append("source_raised")
                raise failure
            finally:
                events.append("source_closed")

        iterator = upstream._instrumented_query(messages, (), {"prompt": "offline", "options": ClaudeAgentOptions()})
        async with asyncio.timeout(2):
            try:
                assert await anext(iterator) is terminal
                assert events == []
                events.append("caller_received")
                with pytest.raises(ResultError) as raised:
                    await anext(iterator)
                assert raised.value is failure
            finally:
                await iterator.aclose()
        assert events == ["caller_received", "source_raised", "source_closed"]
        assert _context.get_invocation_context() is previous
        assert not seen["ctx"].invocation_span.is_recording()
        assert seen["ctx"].invocation_span.status.status_code is StatusCode.ERROR
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["standalone", "client"])
@pytest.mark.parametrize("capture_content", [False, True])
def test_error_result_then_immediate_close_preserves_failure_without_tools(otel, patched_helpers, monkeypatch, mode, capture_content):
    async def scenario():
        upstream = _configure_upstream(monkeypatch, otel)
        monkeypatch.setattr(upstream, "_capture_content", capture_content)
        previous = otel.context()
        invocation = otel.context(capture_content=capture_content) if mode == "client" else None
        client = SimpleNamespace(_otel_invocation_ctx=invocation, _query=SimpleNamespace(_otel_invocation_ctx=None))
        _context.set_invocation_context(previous)
        sentinel = "sensitive-terminal-error-sentinel"
        terminal = _result(subtype="error_during_execution", is_error=True, errors=[sentinel], result=sentinel)
        seen = {}

        async def messages(*args, **kwargs):
            seen["ctx"] = _context.get_invocation_context()
            try:
                yield terminal
            finally:
                seen["source_closed"] = True

        iterator = (upstream._instrumented_query(messages, (), {"prompt": "offline", "options": ClaudeAgentOptions()})
                    if mode == "standalone" else upstream._instrumented_receive_response(messages, client, (), {}))
        assert await anext(iterator) is terminal
        await iterator.aclose()
        assert seen["source_closed"]
        assert _context.get_invocation_context() is previous
        span = seen["ctx"].invocation_span
        assert not span.is_recording()
        assert span.status.status_code is StatusCode.ERROR
        if not capture_content:
            assert sentinel not in json.dumps(dict(span.attributes))
            assert sentinel not in (span.status.description or "")
    asyncio.run(scenario())


def test_shared_owners_keep_compatibility_wrappers_until_last_deactivation(otel, monkeypatch):
    originals = (UpstreamInstrumentor._instrumented_query, UpstreamInstrumentor._instrumented_receive_response,
                 _hooks.set_tool_error_attributes)
    monkeypatch.setattr(_instrumentation.trace, "get_tracer_provider", lambda: otel.provider)
    first, second = ClaudeAgentSDKInstrumentor(), ClaudeAgentSDKInstrumentor()
    try:
        first.activate()
        assert first._is_instrumented
        wrapped = (UpstreamInstrumentor._instrumented_query, UpstreamInstrumentor._instrumented_receive_response,
                   _hooks.set_tool_error_attributes)
        assert all(actual is not original for actual, original in zip(wrapped, originals))
        second.activate()
        assert second._is_instrumented
        first.deactivate()
        assert (UpstreamInstrumentor._instrumented_query, UpstreamInstrumentor._instrumented_receive_response,
                _hooks.set_tool_error_attributes) == wrapped
        second.deactivate()
        assert (UpstreamInstrumentor._instrumented_query, UpstreamInstrumentor._instrumented_receive_response,
                _hooks.set_tool_error_attributes) == originals
    finally:
        second.deactivate()
        first.deactivate()
