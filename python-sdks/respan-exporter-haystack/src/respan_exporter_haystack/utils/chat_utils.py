"""Chat payload conversion helper utilities."""

import logging
from typing import Any, Dict, List

from haystack.dataclasses import ChatMessage

logger = logging.getLogger(__name__)


def extract_response_text(content: Any) -> str:
    """Extract normalized text from response content payload.

    Safely handles None and malformed payloads (e.g. missing keys); callers should use
    .get() when passing choice/message data to avoid KeyError on bad API responses.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
                    continue
                if isinstance(text_value, dict):
                    nested_value = text_value.get("value")
                    if isinstance(nested_value, str):
                        text_parts.append(nested_value)
                        continue
                item_content = item.get("content")
                if isinstance(item_content, str):
                    text_parts.append(item_content)
                    continue
            if item is not None:
                text_parts.append(str(item))
        return "".join(text_parts)

    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
        if isinstance(text_value, dict):
            nested_value = text_value.get("value")
            if isinstance(nested_value, str):
                return nested_value
        item_content = content.get("content")
        if isinstance(item_content, str):
            return item_content

    return str(content)


def chat_message_text(message: ChatMessage) -> str:
    """Extract plain text from Haystack ChatMessage."""
    try:
        if getattr(message, "text", None) is not None:
            return str(message.text)
    except (AttributeError, ValueError) as e:
        logger.debug("Could not read message.text, falling back to to_dict: %s", e)

    message_dict = message.to_dict()
    content_parts = message_dict.get("content", [])
    text_parts: List[str] = []
    if isinstance(content_parts, list):
        for part in content_parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
    if text_parts:
        return "\n".join(text_parts)
    return str(content_parts)


def to_request_message(message: ChatMessage) -> Dict[str, str]:
    """Convert Haystack ChatMessage to gateway payload format."""
    role_value = getattr(getattr(message, "role", None), "value", None)
    role = role_value or str(getattr(message, "role", "user")).lower()
    if role.startswith("chatrole."):
        role = role.split(".", maxsplit=1)[1]
    return {
        "role": role,
        "content": chat_message_text(message=message),
    }


def to_chat_message(message_payload: Dict[str, Any]) -> ChatMessage:
    """Convert gateway message payload back to Haystack ChatMessage."""
    role = str(message_payload.get("role", "assistant")).lower()
    content = extract_response_text(content=message_payload.get("content"))
    if role == "user":
        return ChatMessage.from_user(text=content)
    if role == "system":
        return ChatMessage.from_system(text=content or "")
    return ChatMessage.from_assistant(text=content)
