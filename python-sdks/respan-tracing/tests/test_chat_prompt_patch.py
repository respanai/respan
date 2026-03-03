"""
Unit tests for the chat prompt capture sync patch.

Tests verify that _patch_chat_prompt_capture() correctly replaces the async
_handle_request with a sync version that:
1. Sets prompt attributes (gen_ai.prompt.N.role/content) on the span
2. Isolates _set_request_attributes failures from prompt capture
3. Returns a coroutine for compatibility with both run_async() and await
4. Handles multimodal list content (json.dumps)
5. Handles tool_calls in messages
"""
import asyncio
import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.trace import SpanKind


class TestChatPromptPatch(unittest.TestCase):
    def setUp(self):
        self.provider = TracerProvider()
        self.tracer = self.provider.get_tracer("test")

    def _make_span(self):
        return self.tracer.start_span("test.chat", kind=SpanKind.CLIENT)

    def test_patch_applies_successfully(self):
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw

        original = cw._handle_request
        from respan_tracing.utils.instrumentation import _patch_chat_prompt_capture
        _patch_chat_prompt_capture()
        patched = cw._handle_request

        self.assertIsNot(patched, original)
        # Restore
        cw._handle_request = original

    def test_patched_returns_coroutine(self):
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from respan_tracing.utils.instrumentation import _patch_chat_prompt_capture

        original = cw._handle_request
        _patch_chat_prompt_capture()

        try:
            span = self._make_span()
            kwargs = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
            result = cw._handle_request(span, kwargs, MagicMock())

            self.assertTrue(asyncio.iscoroutine(result))
            # Clean up the coroutine
            asyncio.run(result)
            span.end()
        finally:
            cw._handle_request = original

    def test_prompts_set_on_span(self):
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from respan_tracing.utils.instrumentation import _patch_chat_prompt_capture

        original = cw._handle_request
        _patch_chat_prompt_capture()

        try:
            span = self._make_span()
            kwargs = {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello world"},
                ],
            }
            instance = MagicMock()
            instance._client = MagicMock()

            result = cw._handle_request(span, kwargs, instance)
            asyncio.run(result)
            span.end()

            attrs = dict(span.attributes)
            self.assertEqual(attrs.get("gen_ai.prompt.0.role"), "system")
            self.assertEqual(attrs.get("gen_ai.prompt.0.content"), "You are helpful.")
            self.assertEqual(attrs.get("gen_ai.prompt.1.role"), "user")
            self.assertEqual(attrs.get("gen_ai.prompt.1.content"), "Hello world")
        finally:
            cw._handle_request = original

    def test_long_system_prompt_captured(self):
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from respan_tracing.utils.instrumentation import _patch_chat_prompt_capture

        original = cw._handle_request
        _patch_chat_prompt_capture()

        try:
            long_prompt = "A" * 10000
            span = self._make_span()
            kwargs = {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": long_prompt},
                    {"role": "user", "content": "short"},
                ],
            }
            instance = MagicMock()
            instance._client = MagicMock()

            result = cw._handle_request(span, kwargs, instance)
            asyncio.run(result)
            span.end()

            attrs = dict(span.attributes)
            self.assertEqual(attrs.get("gen_ai.prompt.0.content"), long_prompt)
            self.assertEqual(attrs.get("gen_ai.prompt.1.content"), "short")
        finally:
            cw._handle_request = original

    def test_multimodal_list_content_json_dumped(self):
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from respan_tracing.utils.instrumentation import _patch_chat_prompt_capture
        import json

        original = cw._handle_request
        _patch_chat_prompt_capture()

        try:
            content = [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]
            span = self._make_span()
            kwargs = {
                "model": "test-model",
                "messages": [{"role": "user", "content": content}],
            }
            instance = MagicMock()
            instance._client = MagicMock()

            result = cw._handle_request(span, kwargs, instance)
            asyncio.run(result)
            span.end()

            attrs = dict(span.attributes)
            self.assertEqual(attrs.get("gen_ai.prompt.0.role"), "user")
            parsed = json.loads(attrs.get("gen_ai.prompt.0.content"))
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0]["type"], "text")
        finally:
            cw._handle_request = original

    def test_tool_calls_in_messages(self):
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from respan_tracing.utils.instrumentation import _patch_chat_prompt_capture

        original = cw._handle_request
        _patch_chat_prompt_capture()

        try:
            span = self._make_span()
            kwargs = {
                "model": "test-model",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": '{"temp": 72}',
                        "tool_call_id": "call_123",
                    },
                ],
            }
            instance = MagicMock()
            instance._client = MagicMock()

            result = cw._handle_request(span, kwargs, instance)
            asyncio.run(result)
            span.end()

            attrs = dict(span.attributes)
            self.assertEqual(attrs.get("gen_ai.prompt.0.role"), "assistant")
            self.assertEqual(attrs.get("gen_ai.prompt.0.tool_calls.0.id"), "call_123")
            self.assertEqual(attrs.get("gen_ai.prompt.0.tool_calls.0.name"), "get_weather")
            self.assertEqual(attrs.get("gen_ai.prompt.1.role"), "tool")
            self.assertEqual(attrs.get("gen_ai.prompt.1.tool_call_id"), "call_123")
            self.assertEqual(attrs.get("gen_ai.prompt.1.content"), '{"temp": 72}')
        finally:
            cw._handle_request = original

    def test_request_attrs_failure_does_not_kill_prompts(self):
        """Key test: if _set_request_attributes raises, prompts must still be set."""
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from respan_tracing.utils.instrumentation import _patch_chat_prompt_capture

        original = cw._handle_request
        _patch_chat_prompt_capture()

        try:
            span = self._make_span()
            kwargs = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test prompt"}],
            }
            instance = MagicMock()
            instance._client = MagicMock()

            # Patch _set_request_attributes to raise
            with patch(
                "opentelemetry.instrumentation.openai.shared._set_request_attributes",
                side_effect=RuntimeError("response_format explosion"),
            ):
                result = cw._handle_request(span, kwargs, instance)
                asyncio.run(result)
            span.end()

            attrs = dict(span.attributes)
            # Prompts must still be captured despite request attrs failure
            self.assertEqual(attrs.get("gen_ai.prompt.0.role"), "user")
            self.assertEqual(attrs.get("gen_ai.prompt.0.content"), "test prompt")
        finally:
            cw._handle_request = original

    def test_run_async_compatibility(self):
        """Verify the patched function works with run_async() from the sync chat_wrapper."""
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from opentelemetry.instrumentation.openai.utils import run_async
        from respan_tracing.utils.instrumentation import _patch_chat_prompt_capture

        original = cw._handle_request
        _patch_chat_prompt_capture()

        try:
            span = self._make_span()
            kwargs = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "via run_async"}],
            }
            instance = MagicMock()
            instance._client = MagicMock()

            # This is exactly how chat_wrapper calls it
            run_async(cw._handle_request(span, kwargs, instance))
            span.end()

            attrs = dict(span.attributes)
            self.assertEqual(attrs.get("gen_ai.prompt.0.content"), "via run_async")
        finally:
            cw._handle_request = original

    def test_await_compatibility(self):
        """Verify the patched function works with await from the async achat_wrapper."""
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from respan_tracing.utils.instrumentation import _patch_chat_prompt_capture

        original = cw._handle_request
        _patch_chat_prompt_capture()

        try:
            span = self._make_span()
            kwargs = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "via await"}],
            }
            instance = MagicMock()
            instance._client = MagicMock()

            # This is how achat_wrapper calls it
            async def test_async():
                await cw._handle_request(span, kwargs, instance)

            asyncio.run(test_async())
            span.end()

            attrs = dict(span.attributes)
            self.assertEqual(attrs.get("gen_ai.prompt.0.content"), "via await")
        finally:
            cw._handle_request = original

    def test_none_messages_handled(self):
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from respan_tracing.utils.instrumentation import _patch_chat_prompt_capture

        original = cw._handle_request
        _patch_chat_prompt_capture()

        try:
            span = self._make_span()
            kwargs = {"model": "test-model"}  # No messages key
            instance = MagicMock()
            instance._client = MagicMock()

            result = cw._handle_request(span, kwargs, instance)
            asyncio.run(result)
            span.end()

            attrs = dict(span.attributes)
            self.assertNotIn("gen_ai.prompt.0.role", attrs)
        finally:
            cw._handle_request = original

    def test_empty_content_not_set(self):
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from respan_tracing.utils.instrumentation import _patch_chat_prompt_capture

        original = cw._handle_request
        _patch_chat_prompt_capture()

        try:
            span = self._make_span()
            kwargs = {
                "model": "test-model",
                "messages": [{"role": "assistant", "content": None}],
            }
            instance = MagicMock()
            instance._client = MagicMock()

            result = cw._handle_request(span, kwargs, instance)
            asyncio.run(result)
            span.end()

            attrs = dict(span.attributes)
            self.assertEqual(attrs.get("gen_ai.prompt.0.role"), "assistant")
            self.assertNotIn("gen_ai.prompt.0.content", attrs)
        finally:
            cw._handle_request = original


if __name__ == "__main__":
    unittest.main()
