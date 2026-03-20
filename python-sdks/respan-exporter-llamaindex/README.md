# Respan Exporter for LlamaIndex

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-exporter-llamaindex/)**

Respan exporter for LlamaIndex traces. Captures spans via LlamaIndex's instrumentation system and sends them to Respan.

## Installation

```bash
pip install respan-exporter-llamaindex
```

## Usage

```python
import llama_index.core.instrumentation as instrument
from respan_exporter_llamaindex import RespanSpanHandler

handler = RespanSpanHandler(api_key="your-respan-api-key")
dispatcher = instrument.get_dispatcher()
dispatcher.add_span_handler(handler)

# Now run your LlamaIndex queries as usual - spans are exported automatically.
```

## Configuration

| Parameter              | Environment Variable          | Default                              |
|------------------------|-------------------------------|--------------------------------------|
| `api_key`              | `RESPAN_API_KEY`              | -                                    |
| `base_url`             | `RESPAN_BASE_URL`             | `https://api.respan.ai/api`         |
| `environment`          | `RESPAN_ENVIRONMENT`          | `production`                         |
| `customer_identifier`  | `RESPAN_CUSTOMER_IDENTIFIER`  | -                                    |
| `session_identifier`   | -                             | -                                    |
| `trace_name`           | -                             | Inferred from root span              |
| `timeout`              | -                             | `10` (seconds)                       |
