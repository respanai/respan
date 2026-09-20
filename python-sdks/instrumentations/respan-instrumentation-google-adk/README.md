# respan-instrumentation-google-adk

Respan instrumentation plugin for [Google Agent Development Kit](https://adk.dev/).

This package wraps the upstream OpenInference Google ADK instrumentor and
registers a Google-ADK-specific span processor. The processor composes Respan's
generic OpenInference translation and applies ADK-only normalization in this
package, so ADK runner, agent, LLM, and tool spans are exported through the same
Respan OTEL pipeline as the rest of the Python SDK.

## Installation

```bash
pip install respan-ai respan-instrumentation-google-adk
```

Install ADK's LiteLLM extension if you want to route models through an
OpenAI-compatible gateway:

```bash
pip install "google-adk[extensions]"
```

## Usage

```python
from respan import Respan
from respan_instrumentation_google_adk import GoogleADKInstrumentor

respan = Respan(instrumentations=[GoogleADKInstrumentor()])
```

Any Google ADK runs started after initialization are traced and exported to
Respan.

## Compatibility

Requires Python 3.11–3.13, Google ADK 1.5.0 or newer, and OpenInference Google
ADK instrumentation 0.1.12 or newer. Existing projects can keep their ADK pin:

```bash
pip install respan-instrumentation-google-adk "google-adk==1.5.0"
```

ADK versions below 1.17 use an iterator bridge around the upstream runner and
agent hooks. This enters OpenInference's async iterator when legacy parallel
agents advance it directly. Each iterator runs in its own task, advancing only
when the consumer requests an event and closing in its original context. It also
works with custom agents that imported ADK's legacy `_merge_agent_run` helper
before activation. Deactivation restores the upstream methods.

Activate one Respan Google ADK adapter per process. If another adapter or an
independently activated OpenInference Google ADK instrumentor already owns the
hooks, activation logs a warning and leaves that owner in control.

The processor normalizes actual response usage into both modern and legacy
token fields. This corrects ADK 1.5's native output-token field, which contains
the total token count. Tool execution stays tool content, including results
carried into the next model request.

The offline runtime tests exercise installed ADK and OpenInference packages:
`Runner.run`/`run_async`, session state and model callbacks, sequential and
parallel agents, custom `BaseAgent` subclasses, sync/async tools, SSE response
events, errors, suppression, deactivation, and legacy generator cleanup. Model
responses are deterministic fixtures; no provider credentials are needed.

| Google ADK | OpenInference Google ADK | Validation |
| --- | --- | --- |
| 1.5.0 | 0.1.12 and 0.1.25 | Legacy and common runtime tests |
| 1.17.0 | 0.1.25 | Common runtime tests; legacy-only tests skipped |
| 2.8.0 | 0.1.25 | Common runtime tests; legacy-only tests skipped |

The legacy fixtures follow the APIs used by
[MultiAgentPPT](https://github.com/johnson7788/MultiAgentPPT/tree/ce8185cee83092290bdb913a528c6e3a72ee879e)
and its `google-adk==1.5.0` pin. They cover the ADK execution path; they do not
validate that application's external services or presentation rendering.

Run the package tests from this directory in an environment with the desired
ADK version:

```bash
pip install -e . pytest "google-adk==1.5.0"
python -m pytest tests -q
```
