"""Real pytest collection/report protocol with recording released OTel spans."""

import json
from collections.abc import Mapping

import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from respan_instrumentation_pytest._runtime import PytestRuntimePlugin
from respan_instrumentation_pytest._serialization import json_dumps, safe_text
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

INPUT = SpanAttributes.TRACELOOP_ENTITY_INPUT
OUTPUT = SpanAttributes.TRACELOOP_ENTITY_OUTPUT


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("RESPAN_PYTEST_ENABLED", raising=False)
    monkeypatch.delenv("RESPAN_PYTEST_CAPTURE_CONTENT", raising=False)
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    monkeypatch.delenv("RESPAN_TRACE_CONTENT", raising=False)
    owned = []

    def run(
        source="def test_ok(): pass\n",
        *,
        capture=True,
        provider=None,
        prepare=None,
        plugins=(),
        args=(),
        expected=0,
    ):
        path = tmp_path / ("test_native_" + tmp_path.name.replace("-", "_") + ".py")
        path.write_text(source)
        provider = provider or TracerProvider()
        memory = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(memory))
        runtime = PytestRuntimePlugin(
            tracer=provider.get_tracer("native.pytest"), capture_content=capture
        )
        runtime.activate()
        owned.append((runtime, provider))
        if prepare:
            prepare(runtime)
        before = context.get_current()
        code = pytest.main(
            [
                "-q",
                "--import-mode=importlib",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                str(path),
                *args,
            ],
            plugins=[runtime, *plugins],
        )
        assert int(code) == expected
        assert context.get_current() == before
        assert runtime._session_state is None and runtime._test_states == {}
        spans = memory.get_finished_spans()
        runtime.deactivate()
        return spans

    yield run
    for runtime, provider in owned:
        runtime.deactivate()
        provider.shutdown()


def tasks(spans):
    return [s for s in spans if s.name == "pytest.test"]


def output(span):
    return json.loads(span.attributes[OUTPUT])


def test_native_outcomes_and_complete_parameters(native):
    spans = native("""import pytest
@pytest.mark.parametrize('value', [{'vector':list(range(5000)), 'zero':0, 'false':False, 'text':'x'*6000, 'api_key':'controlled credential'}], ids=['native-param'])
def test_pass(value): assert value['vector'][-1] == 4999
@pytest.mark.skip(reason='native')
def test_skip(): pass
@pytest.mark.xfail(reason='native')
def test_xfail(): assert False
@pytest.mark.xfail(reason='native')
def test_xpass(): pass
""")
    assert [output(s)["outcome"] for s in tasks(spans)] == [
        "passed",
        "skipped",
        "xfailed",
        "xpassed",
    ]
    first = tasks(spans)[0]
    param = json.loads(first.attributes[INPUT])["parameters"]["value"]
    assert len(param["vector"]) == 5000 and param["vector"][-1] == 4999
    assert param["zero"] == 0 and param["false"] is False and len(param["text"]) == 6000
    assert param["api_key"] == "[REDACTED]"
    session = spans[-1]
    assert all(s.parent.span_id == session.context.span_id for s in tasks(spans))
    assert all("status_code" not in s.attributes for s in spans)
    assert output(session)["outcomes"] == {
        "passed": 1,
        "skipped": 1,
        "xfailed": 1,
        "xpassed": 1,
    }


@pytest.mark.parametrize(
    "source,phase",
    [
        ('def test_fail(): raise ValueError("controlled native failure")\n', "call"),
        (
            'import pytest\n@pytest.fixture\ndef broken(): raise RuntimeError("controlled setup failure")\ndef test_fail(broken): pass\n',
            "setup",
        ),
        (
            'import pytest\n@pytest.fixture\ndef broken():\n yield\n raise RuntimeError("controlled teardown failure")\ndef test_fail(broken): pass\n',
            "teardown",
        ),
        (
            "import pytest\n@pytest.mark.xfail(strict=True)\ndef test_fail(): pass\n",
            "call",
        ),
    ],
)
def test_native_failure_phases(native, source, phase):
    spans = native(source, expected=1)
    span = tasks(spans)[0]
    assert output(span)["outcome"] == "failed"
    assert output(span)["phases"][phase]["outcome"] == "failed"
    assert span.status.status_code is StatusCode.ERROR
    assert "error" not in output(span)
    assert spans[-1].status.status_code is StatusCode.ERROR


@pytest.mark.parametrize(
    "source,expected",
    [
        ('raise RuntimeError("controlled collection")\n', 2),
        ("", 5),
        ("def test_stop(): raise KeyboardInterrupt()\n", 2),
    ],
)
def test_native_collection_empty_interruption(native, source, expected):
    spans = native(source, expected=expected)
    assert output(spans[-1])["exit_status"] == expected
    if tasks(spans):
        assert output(tasks(spans)[0])["outcome"] == "interrupted"


@pytest.mark.parametrize(
    "setting",
    [
        "constructor",
        "TRACELOOP_TRACE_CONTENT",
        "RESPAN_TRACE_CONTENT",
        "RESPAN_PYTEST_CAPTURE_CONTENT",
        "context",
    ],
)
def test_private_initial_bound(native, monkeypatch, setting):
    if setting not in ("constructor", "context"):
        monkeypatch.setenv(setting, "false")
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if setting == "context"
        else None
    )
    try:
        spans = native(
            'import pytest\n@pytest.mark.parametrize("value",["PRIVATE-PARAM"],ids=["PRIVATE-ID"])\ndef test_fail(value): raise ValueError("PRIVATE-ERROR")\n',
            capture=setting != "constructor",
            expected=1,
        )
    finally:
        if token:
            context.detach(token)
    assert spans and all(
        INPUT not in s.attributes and OUTPUT not in s.attributes for s in spans
    )
    assert "PRIVATE-" not in str(
        [(dict(s.attributes), s.status.description, s.events) for s in spans]
    )


@pytest.mark.parametrize("kind", ["env", "context"])
def test_late_private_context_exit_scrubs_already_captured_parameters(native, kind):
    code = (
        'import os\nos.environ["TRACELOOP_TRACE_CONTENT"]="false"\nos.environ["TRACELOOP_TRACE_CONTENT"]="true"'
        if kind == "env"
        else "from opentelemetry import context\nfrom respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY\nt=context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY,False))\ncontext.detach(t)"
    )
    # Environment changes have no observable callback until a span/hook boundary.
    if kind == "env":
        code = 'import os\nos.environ["TRACELOOP_TRACE_CONTENT"]="false"'
    source = (
        'import pytest\n@pytest.mark.parametrize("value",["PRIVATE-PARAM"],ids=["PRIVATE-ID"])\ndef test_private(value):\n '
        + code.replace("\n", "\n ")
        + "\n"
    )
    spans = native(source)
    assert "PRIVATE-" not in str(
        [(dict(s.attributes), s.status.description, s.events) for s in spans]
    )
    assert all(INPUT not in s.attributes and OUTPUT not in s.attributes for s in spans)


@pytest.mark.parametrize(
    "suppression",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_suppression_never_extracts(native, monkeypatch, suppression):
    from respan_instrumentation_pytest import _instrumentation

    monkeypatch.setattr(
        _instrumentation,
        "_json_dumps",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("serialization forbidden")
        ),
    )
    token = context.attach(context.set_value(suppression, True))
    try:
        assert native() == ()
    finally:
        context.detach(token)


def test_unsampled_never_extracts(native, monkeypatch):
    def prepare(runtime):
        runtime._test_input = lambda item: (_ for _ in ()).throw(
            AssertionError("sampling gate failed")
        )

    assert native(provider=TracerProvider(sampler=ALWAYS_OFF), prepare=prepare) == ()


@pytest.mark.parametrize("fault", ["input", "metadata", "finish", "end"])
def test_observer_faults_preserve_native_failure_cleanup(native, fault):
    def prepare(runtime):
        def fail(*a, **k):
            raise RuntimeError("controlled observer fault")

        if fault == "input":
            runtime._test_input = fail
        elif fault == "metadata":
            runtime._set_metadata = fail
        elif fault == "finish":
            runtime._finish_test_span = fail
        else:
            original = runtime._end

            def end(state, **kw):
                state.span.end = fail
                return original(state, **kw)

            runtime._end = end

    native(
        'def test_fail(): raise ValueError("native failure")\n',
        prepare=prepare,
        expected=1,
    )


def test_native_async_report(native):
    spans = native(
        "import pytest\n@pytest.mark.asyncio\nasync def test_async():\n import asyncio\n await asyncio.sleep(0)\n",
        plugins=["pytest_asyncio.plugin"],
    )
    assert output(tasks(spans)[0])["outcome"] == "passed"


def test_unknown_conversion_hooks_not_called_and_schema_preserved():
    class Hostile(Mapping):
        def __iter__(self):
            raise AssertionError("iterated")

        def __getitem__(self, key):
            raise AssertionError("read")

        def __len__(self):
            raise AssertionError("len")

        def model_dump(self):
            raise AssertionError("dumped")

    result = json.loads(
        json_dumps(
            {
                "unknown": Hostile(),
                "schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "default": "controlled secret"}
                    },
                    "required": ["api_key"],
                },
            }
        )
    )
    assert result["unknown"] == {"type": "Hostile"}
    assert result["schema"]["properties"]["api_key"] == {
        "type": "string",
        "default": "[REDACTED]",
    }
    assert result["schema"]["required"] == ["api_key"]


@pytest.mark.parametrize(
    "text",
    [
        '{"api_key":"controlled multi word secret","ok":false}',
        "authorization=Basic Y29udHJvbGxlZA==",
        "Bearer controlled-token",
        "https://user:password@fixture.invalid/path?token=secret#secret",
    ],
)
def test_redaction_is_idempotent_and_quoted_json_valid(text):
    result = safe_text(text)
    assert safe_text(result) == result
    assert (
        "multi word secret" not in result
        and "controlled-token" not in result
        and "Y29udHJvbGxlZA" not in result
        and "password@" not in result
    )
    if text.startswith("{"):
        assert json.loads(result)["api_key"] == "[REDACTED]"


def test_lifecycle_removes_only_owned_observer(native):
    provider = TracerProvider()
    foreign = SimpleSpanProcessor(InMemorySpanExporter())
    provider.add_span_processor(foreign)
    native(provider=provider)
    assert provider._active_span_processor._span_processors[0] is foreign
    assert len(provider._active_span_processor._span_processors) == 2


def test_unknown_recording_parent_does_not_widen(native):
    provider = TracerProvider()
    with provider.get_tracer("foreign").start_as_current_span("unobserved"):
        spans = native(provider=provider)
    assert all(INPUT not in s.attributes and OUTPUT not in s.attributes for s in spans)


def test_observed_finished_private_parent_stays_private(native):
    provider = TracerProvider()
    runtime = PytestRuntimePlugin(tracer=provider.get_tracer("observe"))
    runtime.activate()
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    parent = provider.get_tracer("native").start_span("private.parent")
    parent.end()
    context.detach(token)
    carrier = context.attach(trace.set_span_in_context(parent))
    # Keep the original observer; the newly enrolled unknown finished local parent
    # is also conservative in a second runtime.
    try:
        spans = native(provider=provider)
    finally:
        context.detach(carrier)
        runtime.deactivate()
    assert all(INPUT not in s.attributes and OUTPUT not in s.attributes for s in spans)


@pytest.mark.parametrize(
    "suppression",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_supplied_suppressed_parent_never_widens(native, suppression):
    provider = TracerProvider()
    runtime = PytestRuntimePlugin(tracer=provider.get_tracer("observe"))
    runtime.activate()
    parent = provider.get_tracer("native").start_span(
        "suppressed.parent", context=context.set_value(suppression, True)
    )
    carrier = context.attach(trace.set_span_in_context(parent))
    try:
        spans = native(provider=provider)
    finally:
        context.detach(carrier)
        parent.end()
        runtime.deactivate()
    assert all(INPUT not in s.attributes and OUTPUT not in s.attributes for s in spans)


def test_json_url_delimiters_and_token_values_survive_redaction():
    value = '{"url":"https://user:password@fixture.invalid/path?secret=value#secret","ok":false}'
    redacted = json.loads(safe_text(value))
    assert redacted == {"url": "https://fixture.invalid/path", "ok": False}
    assert json.loads(json_dumps({"token": "controlled secret", "zero": 0})) == {
        "token": "[REDACTED]",
        "zero": 0,
    }
