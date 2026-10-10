"""Recording OTel and native released SDK regression contracts."""

from __future__ import annotations

import gc
import io
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from cursor_sdk import Agent, Run, SendOptions
from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_cursor_sdk import (
    CursorHookProcessor,
    CursorSDKInstrumentor,
    _native,
)
from respan_instrumentation_cursor_sdk._serialization import dumps
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from ._fixtures import Fixture, events


def event(name, **kwargs):
    return {
        "hook_event_name": name,
        "conversation_id": "conv-fixture",
        "generation_id": "gen-fixture",
        "cursor_version": "hook-v1-fixture",
        "model_id": "composer-fixture",
        **kwargs,
    }


@pytest.fixture
def recording(tmp_path, monkeypatch):
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    inst = CursorSDKInstrumentor(state_path=tmp_path / "state.json")
    inst.activate()
    yield inst, provider, memory
    inst.deactivate()
    provider.shutdown()


def test_activation_shared_provider_and_foreign_hooks(recording):
    inst, _, _ = recording
    original = _native._RUNTIME.hooks[0][2]
    wrapper = Agent.send
    peer = CursorSDKInstrumentor()
    peer.activate()
    inst.activate()
    inst.deactivate()
    assert Agent.send is wrapper
    peer.deactivate()
    assert Agent.send is original
    inst.activate()
    foreign = lambda *args, **kwargs: original(*args, **kwargs)
    Agent.send = foreign
    try:
        inst.deactivate()
        assert Agent.send is foreign
    finally:
        Agent.send = original


def test_incompatible_owner_rejected(recording):
    with pytest.raises(ValueError):
        CursorSDKInstrumentor(capture_content=False).activate()


def test_partial_activation_rolls_back(recording, monkeypatch):
    inst, _, _ = recording
    inst.deactivate()
    original = Agent.send
    install = _native.Runtime.patch
    count = 0

    def fail(self, *args):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("controlled activation failure")
        return install(self, *args)

    monkeypatch.setattr(_native.Runtime, "patch", fail)
    with pytest.raises(RuntimeError):
        inst.activate()
    assert Agent.send is original and _native._RUNTIME is None


@pytest.mark.parametrize("modern", [False, True])
def test_native_original_run_results_full_tool_values(recording, modern):
    _, _, memory = recording
    fixture = Fixture(
        rows=events(
            tools=True,
            modern=modern,
            usage={
                "inputTokens": 7,
                "outputTokens": 3,
                "totalTokens": 10,
                "cacheReadTokens": 0,
            },
        )
    )
    with fixture.client() as client:
        agent = fixture.agent(client)
        run = agent.send("Fixture prompt")
        assert type(run) is Run
        result = run.wait()
        assert result.result == "Fixture completion" and run.wait() is result
    spans = memory.get_finished_spans()
    assert len(spans) == 2
    tool = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "tool")
    root = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "agent")
    assert tool.parent.span_id == root.context.span_id
    assert tool.attributes["gen_ai.tool.call.id"] == "current-fixture-call"
    value = json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert (
        len(value["vectors"]) == 5000 and value["false"] is False and value["zero"] == 0
    )
    assert (
        json.loads(root.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["usage"][
            "cacheReadTokens"
        ]
        == 0
    )
    assert not any(
        "gen_ai.usage" in k or k in {"status_code", "tools", "tool_calls"}
        for s in spans
        for k in s.attributes
    )


@pytest.mark.asyncio
async def test_native_async_stream_callbacks_unchanged(recording):
    _, _, memory = recording
    seen = []

    async def on_delta(value):
        seen.append(value)

    options = SendOptions(on_delta=on_delta)
    fixture = Fixture()
    async with fixture.async_client() as client:
        agent = await fixture.agent(client)
        run = await agent.send("Fixture prompt", options)
        collected = [e async for e in run]
        assert collected and (await run.wait()).result == "Fixture completion"
        assert options.on_delta is on_delta and len(seen) == 1
    assert len(memory.get_finished_spans()) == 1


@pytest.mark.parametrize("status", ["error", "cancelled", "expired"])
def test_native_terminal_failure_no_invented_output_http(recording, status):
    _, _, memory = recording
    fixture = Fixture(rows=events(status=status))
    with fixture.client() as client:
        result = fixture.agent(client).send("Fixture prompt").wait()
        assert result.status == status
    span = memory.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    assert (
        "http.response.status_code" not in span.attributes
        and "status_code" not in span.attributes
    )


def test_native_actual_transport_error_identity(recording):
    _, _, memory = recording
    fixture = Fixture(error=503)
    with fixture.client() as client:
        with pytest.raises(Exception) as raised:
            fixture.agent(client).send("Fixture prompt")
        assert raised.value.status_code == 503
    span = memory.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["http.response.status_code"] == 503
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


@pytest.mark.parametrize("fault", ["serialization", "checkpoint", "flush"])
def test_native_observer_faults_preserve_results_and_close_spans(
    recording, monkeypatch, fault
):
    _, _, memory = recording

    def fail(*args, **kwargs):
        raise RuntimeError("controlled observer fault")

    target, name = (
        (_native, "dumps")
        if fault == "serialization"
        else (_native.Call, "checkpoint" if fault == "checkpoint" else "flush_tools")
    )
    monkeypatch.setattr(target, name, fail)
    with Fixture().client() as client:
        run = Fixture.agent(client).send("PRIVATE fixture")
        assert type(run) is Run
        result = run.wait()
        assert result.result == "Fixture completion" and run.wait() is result
    assert not _native._RUNTIME.pending
    assert len(memory.get_finished_spans()) == 1
    assert all(
        SpanAttributes.TRACELOOP_ENTITY_INPUT not in s.attributes
        and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in s.attributes
        for s in memory.get_finished_spans()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["checkpoint", "flush"])
async def test_async_observer_faults_preserve_result(recording, monkeypatch, fault):
    _, _, memory = recording

    def fail(*args, **kwargs):
        raise RuntimeError("controlled observer fault")

    monkeypatch.setattr(
        _native.Call, "checkpoint" if fault == "checkpoint" else "flush_tools", fail
    )
    async with Fixture().async_client() as client:
        agent = await Fixture.agent(client)
        run = await agent.send("PRIVATE fixture")
        result = await run.wait()
        assert result.result == "Fixture completion" and await run.wait() is result
    assert not _native._RUNTIME.pending
    assert len(memory.get_finished_spans()) == 1
    assert "PRIVATE" not in str(memory.get_finished_spans()[0].attributes)


def test_policy_fault_preserves_detach_and_span_lifecycle(recording, monkeypatch):
    from respan_instrumentation_cursor_sdk import _policy

    _, provider, memory = recording

    def fail(*args, **kwargs):
        raise RuntimeError("controlled policy fault")

    token = context.attach(context.set_value("fixture-token", True))
    with monkeypatch.context() as patch:
        patch.setattr(_policy, "permitted", fail)
        context.detach(token)
        with (
            provider.get_tracer("fixture").start_as_current_span("parent"),
            Fixture().client() as client,
        ):
            result = Fixture.agent(client).send("PRIVATE fixture").wait()
            assert result.result == "Fixture completion"
    assert context.get_value("fixture-token") is None
    assert not _native._RUNTIME.pending
    assert _native._RUNTIME.policy.faulted
    assert "PRIVATE" not in str([s.attributes for s in memory.get_finished_spans()])


@pytest.mark.parametrize("setting", ["start", "later", "ancestor"])
def test_native_start_end_and_finished_ancestor_privacy(
    recording, monkeypatch, setting
):
    _, provider, memory = recording
    fixture = Fixture(rows=events(tools=True))
    with fixture.client() as client:
        agent = fixture.agent(client)
        if setting == "start":
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", " false ")
        if setting == "ancestor":
            with provider.get_tracer("manual").start_as_current_span("parent"):
                run = agent.send("PRIVATE prompt")
                token = context.attach(
                    context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
                )
            context.detach(token)
        else:
            run = agent.send("PRIVATE prompt")
        if setting == "later":
            monkeypatch.setenv("RESPAN_TRACE_CONTENT", "off")
        if setting in ("start", "ancestor"):
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        run.wait()
    own = [
        s
        for s in memory.get_finished_spans()
        if s.instrumentation_scope is not None
        and s.instrumentation_scope.name == _native._SCOPE
    ]
    assert len(own) == 1
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in own[0].attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in own[0].attributes


def test_native_callback_veto_before_tool_flush(recording):
    _, _, memory = recording
    token = None

    def on_step(step):
        nonlocal token
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))

    fixture = Fixture(rows=events(tools=True, modern=True))
    try:
        with fixture.client() as client:
            fixture.agent(client).send(
                "PRIVATE prompt", SendOptions(on_step=on_step)
            ).wait()
    finally:
        if token is not None:
            context.detach(token)
    assert all(
        SpanAttributes.TRACELOOP_ENTITY_INPUT not in s.attributes
        and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in s.attributes
        for s in memory.get_finished_spans()
    )
    assert all("PRIVATE" not in str(s.attributes) for s in memory.get_finished_spans())


def test_native_terminal_status_callback_veto_clears_diagnostics(
    recording, monkeypatch
):
    _, _, memory = recording
    rows = events(status="error")
    rows[-1]["result"]["result"]["error"]["message"] = "PRIVATE controlled diagnostic"
    with Fixture(rows=rows).client() as client:
        run = Fixture.agent(client).send("PRIVATE prompt")

        def on_status(status):
            if status == "error":
                monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

        run.on_did_change_status(on_status)
        assert run.wait().status == "error"
    span = memory.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is None
    assert "PRIVATE" not in str(span.attributes)
    assert not _native._RUNTIME.pending


def test_parent_started_before_activation_cannot_enable_child_capture(
    recording, monkeypatch
):
    inst, provider, memory = recording
    inst.deactivate()
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    with provider.get_tracer("fixture").start_as_current_span("preexisting-parent"):
        inst.activate()
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        with Fixture().client() as client:
            assert (
                Fixture.agent(client).send("PRIVATE native prompt").wait().result
                == "Fixture completion"
            )
        inst.process_event(event("beforeSubmitPrompt", prompt="PRIVATE hook prompt"))
        inst.process_event(event("stop", status="completed"))
    assert "PRIVATE" not in inst._processor.state_path.read_text()
    assert all("PRIVATE" not in str(s.attributes) for s in memory.get_finished_spans())


def test_supplied_context_cannot_widen_ambient_parent_initial_bound(recording):
    inst, provider, memory = recording
    supplied = context.set_value(ENABLE_CONTENT_TRACING_KEY, True)
    private = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        with provider.get_tracer("fixture").start_as_current_span(
            "parent", context=supplied
        ):
            enabled = context.attach(
                context.set_value(ENABLE_CONTENT_TRACING_KEY, True)
            )
            try:
                with Fixture().client() as client:
                    assert (
                        Fixture.agent(client)
                        .send("PRIVATE native prompt")
                        .wait()
                        .result
                        == "Fixture completion"
                    )
                inst.process_event(event("beforeSubmitPrompt", prompt="PRIVATE hook"))
                inst.process_event(event("stop", status="completed"))
            finally:
                context.detach(enabled)
    finally:
        context.detach(private)
    assert "PRIVATE" not in inst._processor.state_path.read_text()
    assert all("PRIVATE" not in str(s.attributes) for s in memory.get_finished_spans())


def test_native_callback_error_is_same_object(recording):
    _, _, memory = recording
    error = ValueError("Controlled callback error")

    def on_delta(value):
        raise error

    fixture = Fixture()
    with fixture.client() as client:
        run = fixture.agent(client).send(
            "Fixture prompt", SendOptions(on_delta=on_delta)
        )
        with pytest.raises(ValueError) as raised:
            run.wait()
        assert raised.value is error
    assert memory.get_finished_spans()[0].status.status_code is StatusCode.ERROR


def test_native_get_usage_is_explicit_source_only(recording):
    _, _, memory = recording
    fixture = Fixture()
    with fixture.client() as client:
        agent = fixture.agent(client)
        if not hasattr(agent, "get_usage"):
            pytest.skip("Native minimum SDK lacks billed usage API")
        value = agent.get_usage()
        assert value.usage.total_tokens == 10
    span = memory.get_finished_spans()[0]
    assert span.attributes[RESPAN_LOG_TYPE] == "task"
    output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output["usage"]["usage"]["reasoningTokens"] == 2
    assert "GetUsage" in fixture.requests and "Send" not in fixture.requests


def test_sampling_and_suppression_skip_serialization(recording, monkeypatch):
    inst, _, _ = recording
    inst.deactivate()
    provider = TracerProvider(sampler=ALWAYS_OFF)
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    inst.activate()
    serialized = []

    def forbidden(value):
        serialized.append(value)
        raise AssertionError("No payload serialization allowed")

    monkeypatch.setattr(_native, "dumps", forbidden)
    fixture = Fixture()
    with fixture.client() as client:
        assert (
            fixture.agent(client).send("PRIVATE prompt").wait().result
            == "Fixture completion"
        )
    token = context.attach(context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        with fixture.client() as client:
            fixture.agent(client).send("PRIVATE prompt").wait()
    finally:
        context.detach(token)
    assert not memory.get_finished_spans()
    assert not serialized
    inst.deactivate()
    provider.shutdown()


def test_deactivate_pending_run_preserves_native_result(recording):
    inst, _, memory = recording
    fixture = Fixture()
    with fixture.client() as client:
        run = fixture.agent(client).send("PRIVATE prompt")
        inst.deactivate()
        assert run.wait().result == "Fixture completion"
    span = memory.get_finished_spans()[0]
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


def test_abandoned_native_run_discards_buffer_without_closing_user_client(recording):
    _, _, memory = recording
    fixture = Fixture()
    with fixture.client() as client:
        run = fixture.agent(client).send("PRIVATE abandoned input")
        del run
        gc.collect()
        assert len(memory.get_finished_spans()) == 1
        assert (
            SpanAttributes.TRACELOOP_ENTITY_INPUT
            not in memory.get_finished_spans()[0].attributes
        )
        assert client.ping() == ""


def test_native_observer_fault_preserves_result(recording, monkeypatch):
    _, _, memory = recording

    def broken(self, raw):
        raise RuntimeError("Controlled translator fault")

    monkeypatch.setattr(_native.Call, "observe", broken)
    fixture = Fixture()
    with fixture.client() as client:
        assert (
            fixture.agent(client).send("PRIVATE input").wait().result
            == "Fixture completion"
        )
    assert all("PRIVATE" not in str(s.attributes) for s in memory.get_finished_spans())


def test_tool_step_and_legacy_message_are_one_invocation(recording):
    _, _, memory = recording
    rows = events(tools=True)
    rows.insert(-1, {"step": {"type": "toolCall", "message": rows[-2]["sdkMessage"]}})
    with Fixture(rows=rows).client() as client:
        Fixture.agent(client).send("Fixture prompt").wait()
    assert (
        len(
            [
                s
                for s in memory.get_finished_spans()
                if s.attributes.get(RESPAN_LOG_TYPE) == "tool"
            ]
        )
        == 1
    )


def test_close_pending_hook_state_is_irreversibly_private(recording):
    inst, _, memory = recording
    path = inst._processor.state_path
    inst.process_event(event("beforeSubmitPrompt", prompt="PRIVATE abandoned prompt"))
    inst.deactivate()
    assert "PRIVATE" not in path.read_text()
    inst.activate()
    inst.process_event(event("stop", status="completed"))
    assert (
        SpanAttributes.TRACELOOP_ENTITY_INPUT
        not in memory.get_finished_spans()[-1].attributes
    )


@pytest.mark.parametrize(
    "name", ["workspaceOpen", "sessionEnd", "afterTabFileEdit", "postToolUse"]
)
def test_hook_sampling_skips_bodies(recording, monkeypatch, name):
    inst, provider, memory = recording
    provider.sampler = ALWAYS_OFF

    def forbidden(value):
        raise AssertionError("No payload serialization under drop sampling")

    from respan_instrumentation_cursor_sdk import _processor

    monkeypatch.setattr(_processor, "dumps", forbidden)
    assert not inst.process_event(
        event(name, content="PRIVATE", tool_input={"text": "PRIVATE"})
    ).emitted
    assert not memory.get_finished_spans()


def test_hook_response_is_not_terminal_stop_error(recording):
    inst, _, memory = recording
    inst.process_event(event("beforeSubmitPrompt", prompt="Fixture prompt"))
    inst.process_event(
        event("afterAgentResponse", text="Intermediate assistant message")
    )
    assert len(memory.get_finished_spans()) == 1
    result = inst.process_event(
        event("stop", status="error", error_message="Controlled failure")
    )
    assert result.emitted
    root = memory.get_finished_spans()[-1]
    assert (
        root.attributes[RESPAN_LOG_TYPE] == "agent"
        and root.status.status_code is StatusCode.ERROR
    )
    assert (
        "status_code" not in root.attributes
        and "http.response.status_code" not in root.attributes
    )
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in root.attributes
    assert not inst.process_event(event("stop", status="completed")).emitted


def test_real_command_lifecycle_persists_until_terminal_hook(
    recording, monkeypatch, capsys, tmp_path
):
    from respan_instrumentation_cursor_sdk import hook

    _, _, memory = recording
    path = tmp_path / "cli-state.json"
    monkeypatch.setenv("RESPAN_CURSOR_STATE_FILE", str(path))
    monkeypatch.setenv("TRACE_TO_RESPAN", "true")
    monkeypatch.setattr(
        hook, "RespanTelemetry", lambda **kwargs: SimpleNamespace(flush=lambda: None)
    )
    for name, fields in [
        ("beforeSubmitPrompt", {"prompt": "CLI persisted prompt"}),
        ("afterAgentResponse", {"text": "CLI result"}),
        ("stop", {"status": "completed"}),
    ]:
        monkeypatch.setattr(
            hook.sys, "stdin", io.StringIO(json.dumps(event(name, **fields)))
        )
        assert hook.main() == 0
        if name == "beforeSubmitPrompt":
            assert "CLI persisted prompt" in path.read_text()
    root = memory.get_finished_spans()[-1]
    assert (
        json.loads(root.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])["prompt"]
        == "CLI persisted prompt"
    )
    assert json.loads(root.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
        "CLI result"
    ]
    assert json.loads(capsys.readouterr().out) == {"continue": True}
    assert "CLI persisted prompt" not in path.read_text()


def test_closed_parent_veto_outlives_denial_cache(recording):
    inst, provider, memory = recording
    fixture = Fixture(rows=events(tools=True))
    with fixture.client() as client:
        agent = fixture.agent(client)
        with provider.get_tracer("manual").start_as_current_span("parent"):
            run = agent.send("PRIVATE native prompt")
            inst.process_event(
                event("beforeSubmitPrompt", prompt="PRIVATE hook prompt")
            )
            token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        context.detach(token)
        for _ in range(4200):
            with provider.get_tracer("manual").start_as_current_span("unrelated"):
                token = context.attach(
                    context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
                )
                context.detach(token)
        run.wait()
        inst.process_event(event("stop", status="completed"))
    assert all("PRIVATE" not in str(s.attributes) for s in memory.get_finished_spans())


def test_hook_new_events_current_schema_full_values(recording):
    inst, _, memory = recording
    inst.process_event(event("beforeSubmitPrompt", prompt="Fixture prompt"))
    vector = [float(i) for i in range(5000)]
    inst.process_event(
        event(
            "postToolUse",
            tool_name="lookup",
            tool_use_id="current-hook-call",
            tool_input='{"query":"fixture"}',
            tool_output=json.dumps({"vector": vector}),
        )
    )
    inst.process_event(
        event(
            "afterFileEdit",
            file_path="/synthetic/file.py",
            edits=[{"old_string": "x" * 9000, "new_string": "y" * 9000}],
        )
    )
    inst.process_event(event("preCompact", context_tokens=120000, trigger="manual"))
    inst.process_event(event("stop", status="completed"))
    spans = memory.get_finished_spans()
    assert len(spans) == 4
    tool = spans[0]
    assert tool.attributes["gen_ai.tool.call.id"] == "current-hook-call"
    assert (
        len(
            json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[
                "vector"
            ]
        )
        == 5000
    )
    assert (
        len(
            json.loads(spans[1].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[0][
                "old_string"
            ]
        )
        == 9000
    )
    assert all(s.parent.span_id == spans[-1].context.span_id for s in spans[:-1])


@pytest.mark.parametrize(
    "mode", ["environment", "context", "start_then_on", "late_veto"]
)
def test_hook_private_disk_and_spans(recording, monkeypatch, mode):
    inst, _, memory = recording
    token = None
    if mode in ("environment", "start_then_on"):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    if mode == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    inst.process_event(event("beforeSubmitPrompt", prompt="PRIVATE persisted prompt"))
    if mode == "start_then_on":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    if mode == "late_veto":
        monkeypatch.setenv("RESPAN_TRACE_CONTENT", "0")
    inst.process_event(
        event(
            "postToolUseFailure",
            tool_name="lookup",
            tool_use_id="private-call",
            tool_input={"secret": "PRIVATE"},
            error_message="PRIVATE error",
            failure_type="error",
        )
    )
    inst.process_event(event("stop", status="error"))
    if token is not None:
        context.detach(token)
    assert "PRIVATE" not in inst._processor.state_path.read_text()
    assert all(
        "PRIVATE" not in str(s.attributes)
        and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in s.attributes
        for s in memory.get_finished_spans()
    )


def test_hook_state_isolated_across_conversations(recording):
    inst, _, memory = recording
    inst.process_event(event("beforeSubmitPrompt", prompt="One"))
    inst.process_event(
        event("beforeSubmitPrompt", conversation_id="other-conversation", prompt="Two")
    )
    inst.process_event(event("stop", status="completed"))
    inst.process_event(
        event("stop", conversation_id="other-conversation", status="completed")
    )
    spans = memory.get_finished_spans()
    assert len(spans) == 2 and spans[0].context.trace_id != spans[1].context.trace_id
    assert (
        json.loads(spans[0].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])["prompt"]
        == "One"
    )
    assert (
        json.loads(spans[1].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])["prompt"]
        == "Two"
    )


def test_hook_atomic_concurrent_process_counts(recording):
    inst, _, memory = recording
    path = inst._processor.state_path
    inst.process_event(event("beforeSubmitPrompt", prompt="Fixture prompt"))

    def child(index):
        return CursorHookProcessor(state_path=path).process_event(
            event("afterAgentThought", text=f"Fixture {index}")
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(r.emitted for r in pool.map(child, range(20)))
    inst.process_event(event("stop", status="completed"))
    assert len(memory.get_finished_spans()) == 21
    assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize(
    "name",
    [
        "workspaceOpen",
        "sessionStart",
        "sessionEnd",
        "beforeTabFileRead",
        "afterTabFileEdit",
    ],
)
def test_hook_independent_lifecycle_and_tab_events(recording, name):
    inst, _, memory = recording
    assert inst.process_event(
        {
            "hook_event_name": name,
            "workspace_roots": ["/synthetic"],
            "session_id": "fixture-session",
            "edits": [{"old_string": "old", "new_string": "new"}],
        }
    ).emitted
    assert (
        len(memory.get_finished_spans()) == 1
        and memory.get_finished_spans()[0].parent is None
    )


def test_schema_sensitive_nodes_and_escaped_json_redaction():
    value = {
        "parameters": {
            "properties": {
                "api_key": {
                    "type": "string",
                    "default": "PRIVATE",
                    "examples": ["PRIVATE"],
                }
            }
        },
        "arguments": json.dumps(
            {
                "api_key": 'PRIVATE "quoted" value',
                "content": 'Bearer synthetic"quoted" value',
                "city": "Tokyo",
            }
        ),
    }
    first = dumps(value)
    assert dumps(json.loads(first)) == first
    output = json.loads(first)
    assert output["parameters"]["properties"]["api_key"]["type"] == "string"
    assert output["parameters"]["properties"]["api_key"]["default"] == "[REDACTED]"
    assert json.loads(output["arguments"])["city"] == "Tokyo"
    assert "PRIVATE" not in first and "synthetic" not in first
