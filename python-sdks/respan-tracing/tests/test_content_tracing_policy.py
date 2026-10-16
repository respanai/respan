import pytest
from opentelemetry import context as context_api
from respan_tracing import is_content_tracing_enabled
from respan_tracing.constants.context_constants import (
    ENABLE_CONTENT_TRACING_KEY,
    OVERRIDE_ENABLE_CONTENT_TRACING_KEY,
    TRACE_CONTENT_KEY,
)
from respan_tracing.decorators.base import _should_send_prompts


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    monkeypatch.delenv("RESPAN_TRACE_CONTENT", raising=False)


def _with(key, value):
    return context_api.set_value(key, value)


def test_on_by_default():
    assert is_content_tracing_enabled() is True


@pytest.mark.parametrize("env", ["TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT"])
@pytest.mark.parametrize("value", ["0", "false", "FALSE", " no ", "off"])
def test_either_env_var_turns_it_off(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    assert is_content_tracing_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", ""])
def test_other_env_values_leave_it_on(monkeypatch, value):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", value)
    assert is_content_tracing_enabled() is True


@pytest.mark.parametrize(
    "key", [ENABLE_CONTENT_TRACING_KEY, TRACE_CONTENT_KEY, OVERRIDE_ENABLE_CONTENT_TRACING_KEY]
)
def test_any_context_key_false_turns_it_off(key):
    assert is_content_tracing_enabled(_with(key, False)) is False


def test_override_true_wins_over_env(monkeypatch):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    assert is_content_tracing_enabled(_with(OVERRIDE_ENABLE_CONTENT_TRACING_KEY, True)) is True


def test_reads_the_active_context():
    token = context_api.attach(_with(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        assert is_content_tracing_enabled() is False
    finally:
        context_api.detach(token)


def test_private_alias_kept_for_published_instrumentations():
    assert _should_send_prompts is is_content_tracing_enabled
