# respan-instrumentation-pytest

An installable Pytest plugin that emits one canonical Respan workflow span for each test session and one task span for every test protocol. The test span remains current across setup, synchronous or asynchronous execution, and teardown, so application spans created by a test are nested beneath the correct test.

The plugin records pass, skip, xfail, setup failure, assertion failure, teardown failure, phase duration, parametrization, markers, and fixture names. Failures include OpenTelemetry error status plus backend-visible `status_code` and `error.message` fields.

## Install and enable

```bash
pip install respan-ai respan-instrumentation-pytest
pytest --respan-tracing
```

Pytest discovers the package through its `pytest11` entry point. It remains opt-in after installation so a globally installed plugin does not unexpectedly export every test run. Enable it in CI with:

```bash
RESPAN_PYTEST_ENABLED=true pytest
```

The usual `RESPAN_API_KEY` and optional `RESPAN_BASE_URL` variables configure export. Configuration can also live in `pytest.ini`:

```ini
[pytest]
respan_tracing = true
respan_capture_content = true
respan_workflow_name = checkout_integration_tests
```

Equivalent environment variables are `RESPAN_PYTEST_ENABLED`, `RESPAN_PYTEST_CAPTURE_CONTENT`, and `RESPAN_PYTEST_WORKFLOW_NAME`.

## Content capture

By default, test parameters, fixture names, markers, and failure text are captured. Disable sensitive values with:

```bash
pytest --respan-tracing --no-respan-capture-content
```

With capture disabled, node IDs, outcomes, phase durations, and exception types remain visible while parameter values and failure messages are omitted. Fixture return values are never recorded. Each xdist worker creates its own session span with worker metadata. The lifecycle is idempotent.
