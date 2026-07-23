from __future__ import annotations

import json
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_AGENT, LOG_TYPE_TASK, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)

from respan_instrumentation_cursor_sdk import CursorHookProcessor, CursorSDKInstrumentor
from respan_instrumentation_cursor_sdk import _processor


def _event(name: str, **overrides):
    payload = {
        "hook_event_name": name,
        "conversation_id": "conv_abc123",
        "generation_id": "gen_def456",
        "model": "claude-4-sonnet",
        "cursor_version": "1.0.0",
        "timestamp": "2026-06-15T08:00:00Z",
    }
    payload.update(overrides)
    return payload


def _capture_spans(monkeypatch):
    captured = []

    def fake_build_readable_span(name, **kwargs):
        span = {"name": name, **kwargs}
        captured.append(span)
        return span

    monkeypatch.setattr(_processor, "build_readable_span", fake_build_readable_span)
    monkeypatch.setattr(_processor, "inject_span", lambda span: True)
    return captured


def _assert_contract_attrs(attrs):
    banned_attrs = {
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "span_tools",
        "has_tool_calls",
        "parallel_tool_calls",
        SpanAttributes.TRACELOOP_SPAN_KIND,
    }
    assert not (set(attrs) & banned_attrs)
    assert attrs[RESPAN_LOG_METHOD] == "tracing_integration"
    assert attrs[RESPAN_THREADS_ID] == "cursor_conv_abc123"
    assert attrs[RESPAN_TRACE_GROUP_ID] == "cursor_conv_abc123"
    assert attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "cursor_conv_abc123"
    assert attrs[f"{RESPAN_METADATA}.cursor.conversation_id"] == "conv_abc123"
    assert attrs[f"{RESPAN_METADATA}.cursor.generation_id"] == "gen_def456"


def test_package_exports_instrumentor():
    assert CursorSDKInstrumentor.name == "cursor-sdk"
    instrumentor = CursorSDKInstrumentor()
    assert not instrumentor.is_instrumented
    instrumentor.activate()
    assert instrumentor.is_instrumented
    instrumentor.deactivate()
    assert not instrumentor.is_instrumented


def test_before_submit_prompt_stores_state_without_emitting(tmp_path, monkeypatch):
    captured = _capture_spans(monkeypatch)
    processor = CursorHookProcessor(state_path=tmp_path / "state.json")

    result = processor.process_event(
        _event(
            "beforeSubmitPrompt",
            prompt="Create a Fibonacci function",
            attachments=[{"path": "requirements.txt"}],
        )
    )

    assert result.emitted is False
    assert captured == []
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["gen_def456"]["prompt"] == "Create a Fibonacci function"
    assert state["gen_def456"]["attachments_count"] == 1


def test_agent_turn_replay_emits_canonical_spans(tmp_path, monkeypatch):
    captured = _capture_spans(monkeypatch)
    processor = CursorHookProcessor(state_path=tmp_path / "state.json")

    processor.process_event(_event("beforeSubmitPrompt", prompt="Build a CLI parser"))
    thought = processor.process_event(
        _event(
            "afterAgentThought",
            text="I should inspect the existing command layout.",
            duration_ms=320,
        )
    )
    shell = processor.process_event(
        _event(
            "afterShellExecution",
            command="rg -n argparse .",
            output="cli.py:12:import argparse",
            duration=140,
        )
    )
    file_edit = processor.process_event(
        _event(
            "afterFileEdit",
            file_path="/repo/cli.py",
            edits=[
                {
                    "oldText": "parser = argparse.ArgumentParser()",
                    "newText": "parser = argparse.ArgumentParser(prog='demo')",
                    "startLine": 12,
                    "endLine": 12,
                }
            ],
        )
    )
    mcp = processor.process_event(
        _event(
            "afterMCPExecution",
            tool_name="search_codebase",
            tool_input='{"query":"argparse"}',
            result_json='{"matches":1}',
            duration_ms=180,
        )
    )
    root = processor.process_event(
        _event("afterAgentResponse", text="Implemented the CLI parser.")
    )

    assert all(result.emitted for result in [thought, shell, file_edit, mcp, root])
    assert [span["parent_id"] for span in captured[:4]] == ["gen_def456-root"] * 4
    assert captured[-1]["parent_id"] is None
    assert json.loads((tmp_path / "state.json").read_text()) == {}

    thought_attrs = captured[0]["attributes"]
    assert thought_attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TASK
    assert thought_attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "thinking.1"
    assert thought_attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == (
        "I should inspect the existing command layout."
    )
    _assert_contract_attrs(thought_attrs)

    shell_attrs = captured[1]["attributes"]
    assert shell_attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert json.loads(shell_attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "command": "rg -n argparse ."
    }
    _assert_contract_attrs(shell_attrs)

    file_attrs = captured[2]["attributes"]
    assert file_attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert json.loads(file_attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[0]["new"] == (
        "parser = argparse.ArgumentParser(prog='demo')"
    )
    _assert_contract_attrs(file_attrs)

    mcp_attrs = captured[3]["attributes"]
    assert mcp_attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert json.loads(mcp_attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "query": "argparse"
    }
    _assert_contract_attrs(mcp_attrs)

    root_attrs = captured[4]["attributes"]
    assert root_attrs[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert root_attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert json.loads(root_attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "user", "content": "Build a CLI parser"}
    ]
    assert json.loads(root_attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
        {"role": "assistant", "content": "Implemented the CLI parser."}
    ]
    assert root_attrs[f"{RESPAN_METADATA}.cursor.child_count"] == 4
    _assert_contract_attrs(root_attrs)


def test_stop_cleans_generation_state(tmp_path, monkeypatch):
    _capture_spans(monkeypatch)
    processor = CursorHookProcessor(state_path=tmp_path / "state.json")

    processor.process_event(_event("beforeSubmitPrompt", prompt="Will be stopped"))
    result = processor.process_event(_event("stop", status="cancelled"))

    assert result.emitted is False
    assert json.loads((tmp_path / "state.json").read_text()) == {}


def test_unknown_event_is_ignored(tmp_path, monkeypatch):
    captured = _capture_spans(monkeypatch)
    processor = CursorHookProcessor(state_path=tmp_path / "state.json")

    result = processor.process_event(_event("afterUnknownEvent"))

    assert result.emitted is False
    assert captured == []
