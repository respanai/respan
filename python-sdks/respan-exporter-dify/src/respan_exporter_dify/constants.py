"""
Dify-specific constants for response extraction and payload building.

Path tuples are tried in order when extracting fields from Dify API responses.
"""

EMPTY_VALUES = (None, "")

USAGE_PATHS = (
    ("usage",),
    ("metadata", "usage"),
    ("data", "usage"),
    ("data", "execution_metadata"),
    ("execution_metadata",),
)
TOTAL_TOKENS_PATH = ("data", "total_tokens")

MESSAGE_ID_PATHS = (
    ("id",),
    ("message_id",),
    ("messageId",),
    ("uuid",),
)
SESSION_ID_PATHS = (
    ("conversation_id",),  # Dify API documented field (request and response)
)
MESSAGE_TYPE_PATHS = (
    ("event",),
    ("type",),
    ("mode",),
    ("message_type",),
    ("kind",),
)
RESPONSE_CONTENT_PATHS = (
    ("response",),
    ("result",),
    ("answer",),
    ("output",),
    ("outputs",),
    ("content",),
    ("data",),
)
NESTED_MESSAGES_PATHS = (
    ("messages",),
    ("all_messages",),
    ("assistant_messages",),
    ("items",),
    ("results",),
)
NESTED_MESSAGE_PATHS = (
    ("message",),
    ("assistant_message",),
    ("last_message",),
)
