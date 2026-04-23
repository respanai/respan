# Monitors & Automation Reference

Set up monitoring alerts, online evaluations, and automated workflows.

Docs: `https://respan.ai/docs/documentation/features/monitoring/monitors.md`

---

## Monitors

A monitor watches a metric over time and sends a notification when a threshold is crossed.

### Examples

- Error rate > 5% over 10 minutes
- Total cost > $1.00 over 5 minutes
- Average latency > 3 seconds over 15 minutes
- P95 latency > 5 seconds over 1 hour

### Setup

Monitors are configured in the platform UI: **Monitoring -> Monitors -> Create**.

A monitor has two parts:

#### 1. Trigger

Template: `When [metric] of [source] is [threshold] over [time window]`

- **Metric:** cost, latency, error rate, token usage, request count, etc.
- **Source:** all requests, specific model, specific customer, etc.
- **Threshold:** condition (greater than, less than, equals)
- **Time window:** evaluation period (1 min, 5 min, 10 min, 1 hour, etc.)

Add **Where** conditions to narrow scope:
- Filter by model, project, environment, customer, or custom metadata

#### 2. Notifications

Destinations:

| Destination | Details |
|-------------|---------|
| **Email** | Customizable subject and body |
| **Slack** | Customizable message |
| **Webhook** | Raw URL — receives JSON payload |

A single monitor can send to multiple destinations.

### Manage

- **Deploy** to make a monitor live
- **Pause/Resume** deployed monitors
- **Send test alert** to validate before deploying
- **View versions** and restore earlier versions as drafts
- **Create from chart** — select a metric on the dashboard, click to create a monitor with metric/source/interval prefilled

---

## Online Evaluations (Automation)

Automatically score production traffic in real time using deployed evaluators.

### Setup

Configure in the platform UI: **Evaluators -> Automations -> Create**.

1. **Create a condition** — filter rules for which logs to evaluate:
   - **Single log** — evaluate individual spans matching criteria
   - **Aggregation** — evaluate aggregated metrics over a window

2. **Create an automation** — connect:
   - Condition (which logs)
   - Evaluator (which scoring workflow)
   - Sampling rate (0.0-1.0, fraction of matching logs to evaluate)

3. **Check results** — view eval scores in the automation table

### Use case

> "Score 10% of production chat responses for hallucination using my LLM evaluator"

```
Condition: log_type = "chat" AND model = "gpt-4o"
Evaluator: hallucination-check
Sampling: 0.1
```

---

## Webhooks

Receive event notifications via HTTP webhooks.

### Setup

1. Go to platform: **Settings -> Webhooks -> Create**
2. Enter your endpoint URL
3. Select event types to subscribe to
4. Optionally set a webhook secret for HMAC-SHA256 verification

### Verification

Respan signs webhook payloads with HMAC-SHA256. Verify the signature:

```python
import hashlib
import hmac

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

The signature is sent in the request headers.

---

## Metrics Dashboard

Track key indicators in the platform: **Dashboard -> Metrics**.

### Available metrics

- **Requests:** total count, requests per model/customer/API key
- **Tokens:** prompt tokens, completion tokens, total tokens
- **Cost:** per-request, per-model, per-customer
- **Latency:** average, P50, P95, P99
- **Errors:** error count, error rate
- **Time to first token (TTFT):** streaming latency

### Breakdowns

Slice metrics by:
- Model
- Customer (`customer_identifier`)
- API key
- Prompt template
- Environment
- Custom metadata keys

### Per-user analytics

Use `customer_identifier` in spans to enable:
- Per-user cost tracking and budgets
- Per-user rate limiting
- User-level quality metrics

---

## CLI Commands

```bash
respan traces list --limit 10          # List recent traces
respan traces get <trace-id>           # Get trace details
respan traces summary                  # Trace summary stats
respan logs list --limit 10            # List recent logs
respan logs get <log-id>               # Get log details
respan logs summary                    # Log summary stats
```
