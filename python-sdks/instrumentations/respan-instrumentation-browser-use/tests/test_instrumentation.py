from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager

import pytest
from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from browser_use.tokens.service import TokenCost
from browser_use.tools.service import Tools
from opentelemetry import context
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from respan_instrumentation_browser_use import BrowserUseInstrumentor
from respan_instrumentation_browser_use._constants import (
    ATTR_ACTION_NAME,
    ATTR_PAGE_TITLE,
    ATTR_PAGE_URL,
    ATTR_STEP_NUMBER,
)
from respan_instrumentation_browser_use._instrumentation import _openai_suppressor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.exporters.respan import _export_span_name
from respan_tracing.utils.preprocessing.span_processing import is_processable_span

from .conftest import ScriptedLLM, chrome_path

_PIXEL = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
_MESSAGES = [
    SystemMessage(content="You are a browser agent."),
    UserMessage(
        content=[
            ContentPartTextParam(text="What is on the page?"),
            ContentPartImageParam(
                image_url=ImageURL(url=f"data:image/png;base64,{_PIXEL}")
            ),
        ]
    ),
]


def _by_name(exporter, name):
    return [span for span in exporter.get_finished_spans() if span.name == name]


async def test_llm_call_becomes_chat_span_through_register_llm(exporter, instrumentor):
    llm = ScriptedLLM(["hello"])
    # Patching happens before TokenCost captures the bound ainvoke.
    TokenCost().register_llm(llm)

    response = await llm.ainvoke(_MESSAGES)

    assert response.completion == "hello"
    (span,) = _by_name(exporter, "browser_use.llm")
    attrs = span.attributes
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "scripted-1"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "scripted"
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 120
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 30
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 150
    assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 20
    assert SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS not in attrs
    assert attrs["gen_ai.prompt.0.role"] == "system"
    assert attrs["gen_ai.prompt.0.content"] == "You are a browser agent."
    parts = json.loads(attrs["gen_ai.prompt.1.content"])
    assert parts[0] == {"type": "text", "text": "What is on the page?"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert attrs["gen_ai.completion.0.content"] == "hello"


async def test_screenshots_off_replaces_inline_images(exporter):
    instrumentor = BrowserUseInstrumentor(capture_screenshots=False)
    instrumentor.activate()
    try:
        llm = ScriptedLLM(["ok"])
        TokenCost().register_llm(llm)
        await llm.ainvoke(_MESSAGES)
    finally:
        instrumentor.deactivate()
    (span,) = _by_name(exporter, "browser_use.llm")
    parts = json.loads(span.attributes["gen_ai.prompt.1.content"])
    assert parts[1]["type"] == "text"
    assert parts[1]["text"].startswith("[image omitted: image/png")


async def test_capture_content_off_keeps_usage_only(exporter):
    instrumentor = BrowserUseInstrumentor(capture_content=False)
    instrumentor.activate()
    try:
        llm = ScriptedLLM(["secret answer"])
        TokenCost().register_llm(llm)
        await llm.ainvoke(_MESSAGES)
    finally:
        instrumentor.deactivate()
    (span,) = _by_name(exporter, "browser_use.llm")
    assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 150
    assert not any(key.startswith("gen_ai.prompt") for key in span.attributes)
    assert not any(key.startswith("gen_ai.completion") for key in span.attributes)


async def test_inner_provider_calls_are_suppressed_and_not_nested(
    exporter, instrumentor
):
    seen = {}

    class Inner(ScriptedLLM):
        pass

    class Wrapper(ScriptedLLM):
        def __init__(self, inner):
            super().__init__([])
            self.inner = inner

        async def ainvoke(self, messages, output_format=None, **kwargs):
            seen["suppressed"] = context.get_value(
                SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
            )
            return await self.inner.ainvoke(messages, output_format, **kwargs)

    inner = Inner(["from inner"])
    outer = Wrapper(inner)
    TokenCost().register_llm(inner)
    TokenCost().register_llm(outer)

    await outer.ainvoke(_MESSAGES)

    assert seen["suppressed"] is True
    assert len(_by_name(exporter, "browser_use.llm")) == 1


async def test_respan_openai_instrumentation_is_suppressed(
    exporter, instrumentor, monkeypatch
):
    entered = []

    @contextmanager
    def suppress_openai_instrumentation():
        entered.append(True)
        yield

    fake = types.ModuleType("respan_instrumentation_openai._instrumentation")
    fake.suppress_openai_instrumentation = suppress_openai_instrumentation
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setitem(
        sys.modules,
        "respan_instrumentation_openai",
        types.ModuleType("respan_instrumentation_openai"),
    )
    _openai_suppressor.cache_clear()  # resolved once per process by design
    llm = ScriptedLLM(["ok"])
    TokenCost().register_llm(llm)
    try:
        await llm.ainvoke(_MESSAGES)
    finally:
        _openai_suppressor.cache_clear()

    assert entered == [True]


async def test_native_anthropic_usage_counts_cache_writes_as_prompt(
    exporter, instrumentor
):
    class AnthropicShaped(ScriptedLLM):
        async def ainvoke(self, messages, output_format=None, **kwargs):
            # Mirrors browser_use.llm.anthropic.chat: prompt = input + cache_read.
            return ChatInvokeCompletion(
                completion="ok",
                usage=ChatInvokeUsage(
                    prompt_tokens=100 + 20,
                    prompt_cached_tokens=20,
                    prompt_cache_creation_tokens=500,
                    prompt_image_tokens=None,
                    completion_tokens=30,
                    total_tokens=100 + 30,
                ),
            )

    AnthropicShaped.__module__ = "browser_use.llm.anthropic.chat"
    llm = AnthropicShaped([])
    TokenCost().register_llm(llm)
    await llm.ainvoke(_MESSAGES)

    (span,) = _by_name(exporter, "browser_use.llm")
    assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 620
    assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 650
    assert span.attributes[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 500


async def test_other_clients_usage_is_passed_through(exporter, instrumentor):
    # LiteLLM-style usage already includes cache writes in prompt_tokens.
    class LiteLLMShaped(ScriptedLLM):
        async def ainvoke(self, messages, output_format=None, **kwargs):
            return ChatInvokeCompletion(
                completion="ok",
                usage=ChatInvokeUsage(
                    prompt_tokens=620,
                    prompt_cached_tokens=20,
                    prompt_cache_creation_tokens=500,
                    prompt_image_tokens=None,
                    completion_tokens=30,
                    total_tokens=650,
                ),
            )

    llm = LiteLLMShaped([])
    TokenCost().register_llm(llm)
    await llm.ainvoke(_MESSAGES)

    (span,) = _by_name(exporter, "browser_use.llm")
    assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 620


async def test_content_off_records_exception_type_only(exporter):
    class Leaky(ScriptedLLM):
        async def ainvoke(self, messages, output_format=None, **kwargs):
            raise ValueError("model echoed: my-secret-prompt")

    instrumentor = BrowserUseInstrumentor(capture_content=False)
    instrumentor.activate()
    try:
        llm = Leaky([])
        TokenCost().register_llm(llm)
        with pytest.raises(ValueError):
            await llm.ainvoke(_MESSAGES)
    finally:
        instrumentor.deactivate()
    (span,) = _by_name(exporter, "browser_use.llm")
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "ValueError"
    (event,) = span.events
    assert dict(event.attributes) == {"exception.type": "ValueError"}
    assert "my-secret-prompt" not in repr(span.to_json())


async def test_llm_registered_before_activation_is_traced(exporter):
    llm = ScriptedLLM(["early", "late"])
    TokenCost().register_llm(llm)  # an Agent built before instrumentation
    instrumentor = BrowserUseInstrumentor()
    instrumentor.activate()
    try:
        TokenCost().register_llm(llm)
        await llm.ainvoke(_MESSAGES)
        instrumentor.deactivate()
        instrumentor.activate()  # reactivation must still trace the same model
        TokenCost().register_llm(llm)
        await llm.ainvoke(_MESSAGES)
    finally:
        instrumentor.deactivate()
    assert len(_by_name(exporter, "browser_use.llm")) == 2


async def test_llm_error_marks_span_and_propagates(exporter, instrumentor):
    class Broken(ScriptedLLM):
        async def ainvoke(self, messages, output_format=None, **kwargs):
            raise RuntimeError("provider down")

    llm = Broken([])
    TokenCost().register_llm(llm)
    with pytest.raises(RuntimeError, match="provider down"):
        await llm.ainvoke(_MESSAGES)
    (span,) = _by_name(exporter, "browser_use.llm")
    assert span.status.status_code is StatusCode.ERROR
    # Respan shows the status description as the span's error message.
    assert span.status.description == "RuntimeError: provider down"


async def test_failed_action_is_an_error_tool_span(exporter, instrumentor):
    tools = Tools()
    action_model = tools.registry.create_action_model()
    # An action that does not exist on the page fails inside browser-use and comes
    # back as ActionResult.error rather than an exception.
    action = action_model.model_validate({"wait": {"seconds": 0}})

    result = await tools.act(action=action, browser_session=None)

    (span,) = _by_name(exporter, "wait")
    assert span.attributes[RESPAN_LOG_TYPE] == "tool"
    assert span.attributes[ATTR_ACTION_NAME] == "wait"
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "seconds": 0
    }
    if result.error:
        assert span.status.status_code is StatusCode.ERROR
        assert span.status.description.startswith("action error: ")
    else:
        assert span.status.status_code is not StatusCode.ERROR


def test_deactivate_restores_original_methods(exporter):
    originals = (Agent.run, Agent.step, Tools.act, TokenCost.register_llm)
    instrumentor = BrowserUseInstrumentor()
    instrumentor.activate()
    assert Agent.run is not originals[0]
    second = BrowserUseInstrumentor()
    second.activate()  # shares the runtime; does not double wrap
    second.deactivate()
    assert Agent.run is not originals[0]
    instrumentor.deactivate()
    assert (Agent.run, Agent.step, Tools.act, TokenCost.register_llm) == originals


def test_conflicting_settings_are_rejected(exporter, instrumentor):
    with pytest.raises(ValueError, match="different settings"):
        BrowserUseInstrumentor(capture_content=False).activate()


@pytest.mark.integration
@pytest.mark.skipif(chrome_path() is None, reason="needs a local Chrome/Chromium")
async def test_real_agent_run_produces_full_span_tree(
    exporter, instrumentor, tmp_path, test_page
):
    page = test_page
    llm = ScriptedLLM(
        [
            {
                "evaluation_previous_goal": "Start",
                "memory": "Nothing yet",
                "next_goal": "Open the test page",
                "action": [{"navigate": {"url": page}}],
            },
            {
                "evaluation_previous_goal": "Page opened",
                "memory": "Saw the heading",
                "next_goal": "Finish",
                "action": [{"done": {"text": "Hello browser agent", "success": True}}],
            },
            # browser-use's judge grades the finished run with the same model.
            {"reasoning": "Heading read", "verdict": True, "failure_reason": ""},
        ]
    )
    session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            executable_path=chrome_path(),
            user_data_dir=str(tmp_path / "profile"),
        )
    )
    agent = Agent(task="Read the heading", llm=llm, browser_session=session)
    try:
        history = await agent.run(max_steps=4)
    finally:
        await session.kill()

    assert history.is_successful()
    spans = exporter.get_finished_spans()
    (root,) = _by_name(exporter, "browser_use.agent")
    steps = _by_name(exporter, "browser_use.step")
    llm_spans = _by_name(exporter, "browser_use.llm")
    assert len(steps) == 2
    assert len(llm_spans) == 3
    assert {s.name for s in spans} >= {"navigate", "done"}
    # respan-tracing's exporter drops spans it does not recognise; all of ours pass.
    assert all(is_processable_span(s) for s in spans)
    # ... and names them "<kind>.<detail>" in the Respan UI (task spans carry no
    # detail by exporter convention; the step number is an attribute).
    assert {_export_span_name(s) for s in spans} >= {
        "agent.browser_use",
        "task",
        "tool.navigate",
        "tool.done",
        "llm.scripted-1",
    }

    assert root.attributes[RESPAN_LOG_TYPE] == "agent"
    assert json.loads(root.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "task": agent.task
    }
    output = json.loads(root.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output["is_successful"] is True
    assert output["final_result"] == "Hello browser agent"
    assert output["steps"] == 2
    assert output["judge_verdict"] is True

    # Every span belongs to the one trace, steps under the agent, calls under steps.
    assert {s.context.trace_id for s in spans} == {root.context.trace_id}
    for step in steps:
        assert step.parent.span_id == root.context.span_id
    step_ids = {s.context.span_id for s in steps}
    for child in [*_by_name(exporter, "navigate"), *_by_name(exporter, "done")]:
        assert child.parent.span_id in step_ids
    step_llm = [s for s in llm_spans if s.parent.span_id in step_ids]
    judge_llm = [s for s in llm_spans if s.parent.span_id == root.context.span_id]
    assert (len(step_llm), len(judge_llm)) == (2, 1)

    second = max(steps, key=lambda s: s.attributes[ATTR_STEP_NUMBER])
    assert second.attributes[ATTR_PAGE_URL] == page
    # browser-use falls back to the URL when the title is not ready yet.
    assert second.attributes[ATTR_PAGE_TITLE]
    step_output = json.loads(second.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert step_output["next_goal"] == "Finish"
    assert step_output["actions"][0]["done"]["text"] == "Hello browser agent"
    assert step_output["screenshot"]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
