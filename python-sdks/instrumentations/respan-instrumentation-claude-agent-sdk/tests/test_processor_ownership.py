"""Lifecycle regressions for independent Respan instrumentor instances."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry import trace

from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor
from test_instrumentation import (
    _install_fake_claude_agent_sdk_modules,
    _make_fake_tracer_provider,
)


@pytest.fixture
def lifecycle(monkeypatch):
    provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    fake = _install_fake_claude_agent_sdk_modules(monkeypatch)
    owners = []

    def make_owner(**kwargs):
        owner = ClaudeAgentSDKInstrumentor(**kwargs)
        owners.append(owner)
        return owner

    yield provider, fake, make_owner

    for owner in reversed(owners):
        owner.deactivate()


@pytest.mark.parametrize("first_to_deactivate", [0, 1])
def test_owners_share_processor_and_keep_helpers_until_last_deactivation(
    lifecycle, first_to_deactivate
):
    provider, fake, make_owner = lifecycle
    exporter = object()
    provider._active_span_processor._span_processors = (exporter,)
    original_seam = fake.internal_client.process_query
    owners = [make_owner(agent_name="first"), make_owner(agent_name="second")]
    owners[0].activate()
    upstream = owners[0]._otel_instrumentor
    processor = owners[0]._processor
    patched_response = fake.spans_module.set_response_content
    patched_result = fake.spans_module.set_result_attributes
    patched_seam = vars(fake.internal_client)["process_query"]
    owners[1].activate()

    assert owners[1]._processor is processor
    assert owners[1]._otel_instrumentor is upstream
    assert upstream.instrument_kwargs["agent_name"] == "first"
    assert provider._active_span_processor._span_processors == (processor, exporter)
    assert fake.spans_module.set_response_content is patched_response
    assert fake.spans_module.set_result_attributes is patched_result
    assert vars(fake.internal_client)["process_query"] is patched_seam

    owners[first_to_deactivate].deactivate()
    owners[first_to_deactivate].deactivate()

    assert upstream.uninstrument_calls == 0
    assert provider._active_span_processor._span_processors == (processor, exporter)
    assert fake.spans_module.set_response_content is patched_response
    assert vars(fake.internal_client)["process_query"] is patched_seam

    owners[1 - first_to_deactivate].deactivate()

    assert upstream.uninstrument_calls == 1
    assert provider._active_span_processor._span_processors == (exporter,)
    assert fake.spans_module.set_response_content is fake.original_set_response_content
    assert fake.spans_module.set_result_attributes is fake.original_set_result_attributes
    assert fake.internal_client.process_query is original_seam
    assert fake.claude_sdk_module.query is fake.standalone_query


def test_owner_can_reactivate_while_peer_is_active(lifecycle):
    provider, fake, make_owner = lifecycle
    first, second = make_owner(), make_owner()
    first.activate()
    second.activate()
    upstream = first._otel_instrumentor
    processor = first._processor
    first.deactivate()
    first.activate()
    first.activate()
    second.deactivate()

    assert first._otel_instrumentor is upstream
    assert first._processor is processor
    assert provider._active_span_processor._span_processors == (processor,)
    assert upstream.uninstrument_calls == 0

    first.deactivate()
    assert upstream.uninstrument_calls == 1
    assert provider._active_span_processor._span_processors == ()
    assert fake.spans_module.set_response_content is fake.original_set_response_content


def test_deactivation_uses_captured_provider(lifecycle, monkeypatch):
    provider, _, make_owner = lifecycle
    owner = make_owner()
    owner.activate()
    other_provider = _make_fake_tracer_provider()
    unrelated_processor = object()
    other_provider._active_span_processor._span_processors = (unrelated_processor,)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: other_provider)

    owner.deactivate()

    assert provider._active_span_processor._span_processors == ()
    assert other_provider._active_span_processor._span_processors == (unrelated_processor,)


def test_global_upstream_retains_first_provider_until_last_owner(lifecycle, monkeypatch):
    first_provider, _, make_owner = lifecycle
    first, second = make_owner(), make_owner()
    first.activate()
    first_processor = first._processor
    upstream = first._otel_instrumentor
    second_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: second_provider)
    second.activate()
    second_processor = second._processor

    assert second_processor is not first_processor
    assert second._otel_instrumentor is upstream
    assert upstream.instrument_kwargs["tracer_provider"] is first_provider
    first.deactivate()
    assert first_provider._active_span_processor._span_processors == (first_processor,)
    assert second_provider._active_span_processor._span_processors == (second_processor,)
    assert upstream.uninstrument_calls == 0

    second.deactivate()
    assert first_provider._active_span_processor._span_processors == ()
    assert second_provider._active_span_processor._span_processors == ()
    assert upstream.uninstrument_calls == 1


def test_simultaneous_owners_register_once(lifecycle):
    provider, _, make_owner = lifecycle
    owners = [make_owner() for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda owner: owner.activate(), owners))

    processor = owners[0]._processor
    upstream = owners[0]._otel_instrumentor
    assert all(owner._processor is processor for owner in owners)
    assert all(owner._otel_instrumentor is upstream for owner in owners)
    assert provider._active_span_processor._span_processors == (processor,)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda owner: owner.deactivate(), owners))
    assert provider._active_span_processor._span_processors == ()
    assert upstream.uninstrument_calls == 1


def test_partial_activation_rolls_back_upstream_and_can_retry(lifecycle, monkeypatch):
    provider, fake, make_owner = lifecycle
    instrument = fake.instrumentor_class.instrument
    attempted = []

    def fail_after_wrapping(instance, **kwargs):
        attempted.append(instance)
        instrument(instance, **kwargs)
        raise RuntimeError("failed after wrapping")

    monkeypatch.setattr(fake.instrumentor_class, "instrument", fail_after_wrapping)
    owner = make_owner()
    owner.activate()

    assert owner._is_instrumented is False
    assert owner._otel_instrumentor is None
    assert provider._active_span_processor._span_processors == ()
    assert attempted[0].uninstrument_calls == 1
    assert fake.spans_module.set_response_content is fake.original_set_response_content
    assert fake.claude_sdk_module.query is fake.standalone_query

    monkeypatch.setattr(fake.instrumentor_class, "instrument", instrument)
    owner.activate()
    assert owner._is_instrumented is True
    assert len(provider._active_span_processor._span_processors) == 1


def test_failed_peer_activation_preserves_existing_owner(lifecycle, monkeypatch):
    provider, fake, make_owner = lifecycle
    first, second = make_owner(), make_owner()
    first.activate()
    upstream = first._otel_instrumentor
    processor = first._processor
    patched_response = fake.spans_module.set_response_content
    other_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: other_provider)

    def fail_registration(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(second, "_register_processor", fail_registration)
    second.activate()

    assert second._is_instrumented is False
    assert first._is_instrumented is True
    assert provider._active_span_processor._span_processors == (processor,)
    assert other_provider._active_span_processor._span_processors == ()
    assert fake.spans_module.set_response_content is patched_response
    assert upstream.uninstrument_calls == 0


def test_uninstrument_failure_still_releases_ownership(lifecycle, monkeypatch):
    provider, fake, make_owner = lifecycle
    owner = make_owner()
    owner.activate()
    uninstrument = fake.instrumentor_class.uninstrument

    def fail_after_uninstrument(instance):
        uninstrument(instance)
        raise RuntimeError("uninstrument failed")

    monkeypatch.setattr(fake.instrumentor_class, "uninstrument", fail_after_uninstrument)
    with pytest.raises(RuntimeError, match="uninstrument failed"):
        owner.deactivate()

    assert owner._is_instrumented is False
    assert provider._active_span_processor._span_processors == ()
    assert fake.spans_module.set_response_content is fake.original_set_response_content
    monkeypatch.setattr(fake.instrumentor_class, "uninstrument", uninstrument)
    owner.activate()
    assert owner._is_instrumented is True


def test_add_only_provider_reuses_processor_after_reactivation(lifecycle, monkeypatch):
    _, _, make_owner = lifecycle
    provider = _make_fake_tracer_provider(composite=False)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    owner = make_owner()
    owner.activate()
    processor = owner._processor
    owner.deactivate()
    owner.activate()

    assert owner._processor is processor
    assert provider.added_processors == [processor]


def test_external_upstream_instrumentation_is_not_uninstrumented(lifecycle, monkeypatch):
    provider, fake, make_owner = lifecycle
    external = fake.instrumentor_class()
    external.instrument(tracer_provider=provider)
    external_query = fake.claude_sdk_module.query
    monkeypatch.setattr(
        fake.instrumentor_class, "is_instrumented_by_opentelemetry", True, raising=False
    )
    owner = make_owner()
    owner.activate()
    upstream = owner._otel_instrumentor
    owner.deactivate()

    assert upstream.instrument_kwargs is None
    assert upstream.uninstrument_calls == 0
    assert fake.claude_sdk_module.query is external_query
    assert fake.spans_module.set_response_content is fake.original_set_response_content
    assert provider._active_span_processor._span_processors == ()
    external.uninstrument()
