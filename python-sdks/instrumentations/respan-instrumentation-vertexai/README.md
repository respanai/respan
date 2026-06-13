# respan-instrumentation-vertexai

Respan instrumentation plugin for the Google Vertex AI Python SDK.

The package patches `vertexai.generative_models.GenerativeModel` and
`ChatSession` generation methods and emits canonical Respan chat spans through
the active OTEL pipeline. It captures sync calls, async calls, streamed
responses, prompt and completion content, token usage, tool declarations, and
model function calls.

## Installation

```bash
pip install respan-ai respan-instrumentation-vertexai google-cloud-aiplatform
```

## Usage

```python
import vertexai
from respan import Respan
from respan_instrumentation_vertexai import VertexAIInstrumentor
from vertexai.generative_models import GenerativeModel

respan = Respan(instrumentations=[VertexAIInstrumentor()])

vertexai.init(project="your-gcp-project", location="us-central1")

model = GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Say hello in three languages.")
print(response.text)
respan.flush()
```
