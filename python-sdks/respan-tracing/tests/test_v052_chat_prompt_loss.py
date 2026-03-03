"""
Reproduction tests for the chat prompt capture bug.

These tests prove the ORIGINAL (unpatched) _handle_request loses gen_ai.prompt.*
attributes under specific conditions, and that the sync patch fixes it.

Theory A: _set_request_attributes raises on response_format handling.
          It is NOT wrapped with @dont_throw, so the exception propagates to
          _handle_request's @dont_throw, which silently catches it.
          _set_prompts (and everything after _set_request_attributes) never runs.

Theory B: run_async() in a running-event-loop environment goes through
          the thread path. Test that attributes survive this path.

NOTE: The patch function is inlined here so these tests are self-contained
and don't depend on any particular respan_tracing version being installed.
"""
import asyncio
import copy
import json
import logging
import threading
import traceback
import unittest
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind

from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
from opentelemetry.instrumentation.openai.shared import (
    _set_request_attributes,
    _set_client_attributes,
    _set_functions_attributes,
    _set_span_attribute,
    set_tools_attributes,
    model_as_dict,
    propagate_trace_context,
)
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.utils import (
    run_async,
    should_send_prompts,
    should_emit_events,
    is_openai_v1,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


def _apply_sync_patch():
    """
    Inline version of _patch_chat_prompt_capture() from respan_tracing.
    Self-contained so reproduction tests don't depend on respan_tracing install.
    """
    def _set_prompts_sync(span, messages):
        if not span.is_recording() or messages is None:
            return
        for i, msg in enumerate(messages):
            prefix = f"{GenAIAttributes.GEN_AI_PROMPT}.{i}"
            msg = msg if isinstance(msg, dict) else model_as_dict(msg)
            _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
            if msg.get("content"):
                content = copy.deepcopy(msg.get("content"))
                if isinstance(content, list):
                    content = json.dumps(content)
                _set_span_attribute(span, f"{prefix}.content", content)
            if msg.get("tool_call_id"):
                _set_span_attribute(span, f"{prefix}.tool_call_id", msg.get("tool_call_id"))
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for j, tool_call in enumerate(tool_calls):
                    if is_openai_v1():
                        tool_call = model_as_dict(tool_call)
                    function = tool_call.get("function")
                    _set_span_attribute(span, f"{prefix}.tool_calls.{j}.id", tool_call.get("id"))
                    _set_span_attribute(span, f"{prefix}.tool_calls.{j}.name", function.get("name"))
                    _set_span_attribute(span, f"{prefix}.tool_calls.{j}.arguments", function.get("arguments"))

    def _handle_request_sync(span, kwargs, instance):
        try:
            _set_request_attributes(span, kwargs, instance)
        except Exception:
            logging.debug("_set_request_attributes failed: %s", traceback.format_exc())
        try:
            _set_client_attributes(span, instance)
        except Exception:
            pass
        try:
            if should_emit_events():
                pass  # Simplified — events path not tested here
            else:
                if should_send_prompts():
                    _set_prompts_sync(span, kwargs.get("messages"))
                    if kwargs.get("functions"):
                        _set_functions_attributes(span, kwargs.get("functions"))
                    elif kwargs.get("tools"):
                        set_tools_attributes(span, kwargs.get("tools"))
        except Exception:
            logging.debug("prompt capture failed: %s", traceback.format_exc())
        try:
            if Config.enable_trace_context_propagation:
                propagate_trace_context(span, kwargs)
            reasoning_effort = kwargs.get("reasoning_effort")
            _set_span_attribute(span, SpanAttributes.LLM_REQUEST_REASONING_EFFORT, reasoning_effort or ())
        except Exception:
            pass

    async def _noop():
        pass

    def _patched_handle_request(span, kwargs, instance):
        _handle_request_sync(span, kwargs, instance)
        return _noop()

    cw._handle_request = _patched_handle_request


class TestOtelOpenAIv052_ChatPromptLoss(unittest.TestCase):
    """
    Reproduce: opentelemetry-instrumentation-openai v0.52.6 chat spans lose
    gen_ai.prompt.* attributes when _set_request_attributes raises.
    Fixed in respan-tracing v2.1.3 via sync fault-isolation patch.
    """

    def setUp(self):
        self.provider = TracerProvider()
        self.tracer = self.provider.get_tracer("test")
        # Capture the original BEFORE any patching
        self._original = cw._handle_request

    def tearDown(self):
        # Always restore original after each test
        cw._handle_request = self._original

    def _make_span(self):
        return self.tracer.start_span("test.chat", kind=SpanKind.CLIENT)

    # ──────────────────────────────────────────────────────────────
    # Theory A: response_format exception kills prompts
    # ──────────────────────────────────────────────────────────────

    def test_v052_response_format_exception_kills_all_prompts(self):
        """
        BUG: _set_request_attributes (NOT @dont_throw) raises on response_format
        → propagates to _handle_request's @dont_throw → silently caught
        → _set_prompts never runs → gen_ai.prompt.* absent from span.
        """
        span = self._make_span()

        # This triggers the elif branch at line 159-164:
        #   hasattr(response_format, "model_json_schema")
        #   and callable(response_format.model_json_schema)
        # Then line 169: json.dumps(response_format.model_json_schema()) raises.
        class BrokenResponseFormat:
            def model_json_schema(self):
                raise TypeError("Cannot generate schema")

        kwargs = {
            "model": "parasail/parasail-gpt-oss-20b-fast-mem0",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Hello world"},
            ],
            "response_format": BrokenResponseFormat(),
        }
        instance = MagicMock()
        instance._client = MagicMock()

        # Call original _handle_request directly (no patch applied)
        asyncio.run(self._original(span, kwargs, instance))
        span.end()

        attrs = dict(span.attributes)

        # Request attributes set BEFORE response_format handling ARE present
        self.assertEqual(
            attrs.get("gen_ai.request.model"),
            "parasail/parasail-gpt-oss-20b-fast-mem0",
            "Request model should be set (it's before response_format in _set_request_attributes)",
        )

        # But ALL prompt attributes are MISSING — _set_prompts was never called
        self.assertNotIn(
            "gen_ai.prompt.0.role", attrs,
            "BUG REPRODUCED: prompt.0.role should be absent because _set_prompts never ran",
        )
        self.assertNotIn(
            "gen_ai.prompt.0.content", attrs,
            "BUG REPRODUCED: prompt.0.content should be absent",
        )
        self.assertNotIn(
            "gen_ai.prompt.1.role", attrs,
            "BUG REPRODUCED: prompt.1.role should be absent",
        )
        self.assertNotIn(
            "gen_ai.prompt.1.content", attrs,
            "BUG REPRODUCED: prompt.1.content should be absent",
        )

    def test_v052_response_format_exception_kills_prompts_via_run_async(self):
        """Same bug through run_async() — the actual call path in chat_wrapper."""
        span = self._make_span()

        class BrokenResponseFormat:
            def model_json_schema(self):
                raise TypeError("Cannot generate schema")

        kwargs = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "System prompt here."},
                {"role": "user", "content": "User message."},
            ],
            "response_format": BrokenResponseFormat(),
        }
        instance = MagicMock()
        instance._client = MagicMock()

        # This is exactly how chat_wrapper calls it (line 95):
        #   run_async(_handle_request(span, kwargs, instance))
        run_async(self._original(span, kwargs, instance))
        span.end()

        attrs = dict(span.attributes)
        self.assertEqual(attrs.get("gen_ai.request.model"), "test-model")
        self.assertNotIn("gen_ai.prompt.0.role", attrs, "BUG: prompts lost via run_async")

    def test_v052_response_format_exception_kills_prompts_thread_path(self):
        """Same bug with running event loop — forces run_async thread path."""
        span = self._make_span()

        class BrokenResponseFormat:
            def model_json_schema(self):
                raise TypeError("Cannot generate schema")

        kwargs = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": BrokenResponseFormat(),
        }
        instance = MagicMock()
        instance._client = MagicMock()

        # Force a running event loop → run_async takes the thread path
        async def run_with_loop():
            run_async(self._original(span, kwargs, instance))

        asyncio.run(run_with_loop())
        span.end()

        attrs = dict(span.attributes)
        self.assertNotIn("gen_ai.prompt.0.role", attrs, "BUG: prompts lost via thread path")

    # ──────────────────────────────────────────────────────────────
    # Theory A verification: confirm the chain of causation
    # ──────────────────────────────────────────────────────────────

    def test_v052_response_format_exception_also_kills_client_attrs(self):
        """api_base also lost — matches mem0 trace where gen_ai.openai.api_base absent."""
        span = self._make_span()

        class BrokenResponseFormat:
            def model_json_schema(self):
                raise TypeError("boom")

        instance = MagicMock()
        instance._client = MagicMock()
        # Set a base_url so _set_client_attributes has something to set
        instance._client.base_url = "https://api.parasail.io/v1"

        kwargs = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": BrokenResponseFormat(),
        }

        asyncio.run(self._original(span, kwargs, instance))
        span.end()

        attrs = dict(span.attributes)
        # _set_client_attributes sets gen_ai.openai.api_base — should be absent
        self.assertNotIn(
            "gen_ai.openai.api_base", attrs,
            "Client attributes should also be lost (called after _set_request_attributes)",
        )

    def test_v052_attrs_before_response_format_survive_after_attrs_lost(self):
        """model/temperature survive (set before raise), prompts lost (set after)."""
        span = self._make_span()

        class BrokenResponseFormat:
            def model_json_schema(self):
                raise TypeError("boom")

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "response_format": BrokenResponseFormat(),
        }
        instance = MagicMock()
        instance._client = MagicMock()

        asyncio.run(self._original(span, kwargs, instance))
        span.end()

        attrs = dict(span.attributes)
        # These are set BEFORE response_format handling → should survive
        self.assertEqual(attrs.get("gen_ai.request.model"), "gpt-4")
        self.assertAlmostEqual(attrs.get("gen_ai.request.temperature"), 0.7)
        self.assertEqual(attrs.get("gen_ai.request.max_tokens"), 100)
        # But prompts are gone
        self.assertNotIn("gen_ai.prompt.0.role", attrs)

    # ──────────────────────────────────────────────────────────────
    # Patch fixes the bug
    # ──────────────────────────────────────────────────────────────

    def test_v214_sync_patch_captures_prompts_despite_response_format_exception(self):
        """respan-tracing v2.1.3+ sync patch: prompts survive response_format explosion."""
        _apply_sync_patch()

        span = self._make_span()

        class BrokenResponseFormat:
            def model_json_schema(self):
                raise TypeError("Cannot generate schema")

        kwargs = {
            "model": "parasail/parasail-gpt-oss-20b-fast-mem0",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Hello world"},
            ],
            "response_format": BrokenResponseFormat(),
        }
        instance = MagicMock()
        instance._client = MagicMock()

        result = cw._handle_request(span, kwargs, instance)
        asyncio.run(result)
        span.end()

        attrs = dict(span.attributes)
        # With patch, prompts ARE captured (fault-isolated from request attrs)
        self.assertEqual(attrs.get("gen_ai.prompt.0.role"), "system")
        self.assertEqual(attrs.get("gen_ai.prompt.0.content"), "You are a helpful AI assistant.")
        self.assertEqual(attrs.get("gen_ai.prompt.1.role"), "user")
        self.assertEqual(attrs.get("gen_ai.prompt.1.content"), "Hello world")

    def test_v214_sync_patch_captures_prompts_via_run_async_despite_exception(self):
        """Same fix verified through run_async call path."""
        _apply_sync_patch()

        span = self._make_span()

        class BrokenResponseFormat:
            def model_json_schema(self):
                raise TypeError("Cannot generate schema")

        kwargs = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "User msg."},
            ],
            "response_format": BrokenResponseFormat(),
        }
        instance = MagicMock()
        instance._client = MagicMock()

        run_async(cw._handle_request(span, kwargs, instance))
        span.end()

        attrs = dict(span.attributes)
        self.assertEqual(attrs.get("gen_ai.prompt.0.role"), "system")
        self.assertEqual(attrs.get("gen_ai.prompt.0.content"), "System prompt.")
        self.assertEqual(attrs.get("gen_ai.prompt.1.role"), "user")
        self.assertEqual(attrs.get("gen_ai.prompt.1.content"), "User msg.")

    # ──────────────────────────────────────────────────────────────
    # Theory B: run_async thread path (no response_format issue)
    # Proves the thread path works when no exception is raised.
    # ──────────────────────────────────────────────────────────────

    def test_v052_no_response_format_prompts_captured_correctly(self):
        """Control: without response_format, v0.52 async path works fine."""
        span = self._make_span()

        kwargs = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }
        instance = MagicMock()
        instance._client = MagicMock()

        asyncio.run(self._original(span, kwargs, instance))
        span.end()

        attrs = dict(span.attributes)
        self.assertEqual(attrs.get("gen_ai.prompt.0.role"), "user")
        self.assertEqual(attrs.get("gen_ai.prompt.0.content"), "Hello")

    def test_v052_run_async_thread_path_works_without_exception(self):
        """Control: run_async thread path is not root cause — works when no exception."""
        span = self._make_span()

        kwargs = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        instance = MagicMock()
        instance._client = MagicMock()

        # Force running event loop → thread path
        async def run_with_loop():
            run_async(self._original(span, kwargs, instance))

        asyncio.run(run_with_loop())
        span.end()

        attrs = dict(span.attributes)
        self.assertEqual(attrs.get("gen_ai.prompt.0.role"), "system")
        self.assertEqual(attrs.get("gen_ai.prompt.0.content"), "You are helpful.")
        self.assertEqual(attrs.get("gen_ai.prompt.1.role"), "user")
        self.assertEqual(attrs.get("gen_ai.prompt.1.content"), "Hello")


if __name__ == "__main__":
    unittest.main()
