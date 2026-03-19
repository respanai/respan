# respan-instrumentation-openai-agents

Respan instrumentation plugin for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

## Installation

```bash
pip install respan-instrumentation-openai-agents
```

## Usage

When installed alongside `respan-ai`, the plugin is auto-discovered:

```python
from respan import Respan

respan = Respan(api_key="your-key")
# OpenAI Agents SDK spans are now automatically captured
```
