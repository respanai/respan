"""Unit tests for chat message conversion utils."""

import pytest
from haystack.dataclasses import ChatMessage

from respan_exporter_haystack.utils.chat_utils import (
    extract_response_text,
    to_chat_message,
    to_request_message,
)


class TestExtractResponseText:
    """Unit tests for extract_response_text."""

    def test_none_returns_empty_string(self):
        assert extract_response_text(None) == ""

    def test_string_passthrough(self):
        assert extract_response_text("hello") == "hello"

    def test_list_of_dicts_with_text(self):
        content = [{"text": "a"}, {"text": "b"}]
        assert extract_response_text(content) == "ab"

    def test_list_of_dicts_with_content_key(self):
        content = [{"content": "part1"}, {"content": "part2"}]
        assert extract_response_text(content) == "part1part2"

    def test_dict_with_text_key(self):
        assert extract_response_text({"text": "hello"}) == "hello"

    def test_dict_with_nested_text_value(self):
        assert extract_response_text({"text": {"value": "nested"}}) == "nested"

    def test_malformed_missing_keys_returns_empty_or_str(self):
        assert extract_response_text({}) == "{}"
        assert extract_response_text([]) == ""

    def test_non_string_content_fallback_to_str(self):
        assert extract_response_text(123) == "123"

    def test_list_with_mixed_items(self):
        content = [{"text": "x"}, None, {"content": "y"}, "z"]
        # None items are skipped unless we have a dict; str "z" is not a dict so goes to str(item)
        result = extract_response_text(content)
        assert "x" in result and "y" in result and "z" in result


class TestToRequestMessage:
    """Unit tests for to_request_message."""

    def test_user_message(self):
        msg = ChatMessage.from_user(text="Hi")
        out = to_request_message(msg)
        assert out["role"] == "user"
        assert out["content"] == "Hi"

    def test_system_message(self):
        msg = ChatMessage.from_system(text="You are helpful.")
        out = to_request_message(msg)
        assert out["role"] == "system"
        assert out["content"] == "You are helpful."

    def test_assistant_message(self):
        msg = ChatMessage.from_assistant(text="Hello!")
        out = to_request_message(msg)
        assert out["role"] == "assistant"
        assert out["content"] == "Hello!"


class TestToChatMessage:
    """Unit tests for to_chat_message."""

    def test_user_payload(self):
        payload = {"role": "user", "content": "hello"}
        msg = to_chat_message(payload)
        assert msg.text == "hello"
        assert "user" in str(msg.role).lower()

    def test_system_payload(self):
        payload = {"role": "system", "content": "You are a bot."}
        msg = to_chat_message(payload)
        assert msg.text == "You are a bot."
        assert "system" in str(msg.role).lower()

    def test_assistant_payload_default_role(self):
        payload = {"content": "Hi there"}  # no role
        msg = to_chat_message(payload)
        assert msg.text == "Hi there"
        assert "assistant" in str(msg.role).lower()

    def test_malformed_content_safe(self):
        payload = {"role": "assistant"}  # no content key
        msg = to_chat_message(payload)
        assert msg is not None
        assert msg.text == ""  # extract_response_text(None) -> ""

    def test_content_from_extract_response_text(self):
        payload = {"role": "assistant", "content": [{"text": "part1"}, {"text": "part2"}]}
        msg = to_chat_message(payload)
        assert msg.text == "part1part2"
