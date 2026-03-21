"""Async Batch — Submit a batch, come back later, log results into the original trace.

Simulates the real-world pattern where batch submission and result retrieval
happen at different times (or in different processes).

    RESPAN_API_KEY=your-respan-key
    OPENAI_API_KEY=sk-proj-...
"""

import os
import json
import time

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from respan import Respan, workflow, task, get_client
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOPICS = [
    "quantum computing",
    "blockchain",
    "edge computing",
]


# ── Phase 1: Submit ──────────────────────────────────────────────────
# In production this would be an API endpoint, a CLI command, etc.

@task(name="create_and_submit")
def create_and_submit() -> dict:
    """Build requests, upload, submit batch, return handles for later."""
    requests = []
    for i, topic in enumerate(TOPICS):
        requests.append({
            "custom_id": f"topic-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-nano",
                "messages": [
                    {"role": "system", "content": "Explain in one sentence."},
                    {"role": "user", "content": f"What is {topic}?"},
                ],
            },
        })

    # Write + upload + submit
    file_path = "/tmp/respan_batch_async.jsonl"
    with open(file_path, "w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")

    batch_file = client.files.create(file=open(file_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch submitted: {batch.id} (status: {batch.status})")

    # Capture trace context — in production, save this to DB
    rc = get_client()
    trace_id = rc.get_current_trace_id()
    print(f"Trace ID saved: {trace_id}")

    return {
        "batch_id": batch.id,
        "trace_id": trace_id,
        "input_file": file_path,
    }


@workflow(name="batch_submit")
def submit():
    return create_and_submit()


# ── Phase 2: Retrieve (simulates a separate job) ────────────────────
# In production this would be a cron job, webhook handler, queue worker, etc.

def retrieve_and_log(saved: dict):
    """Check batch status, download results, log into original trace."""
    batch_id = saved["batch_id"]
    trace_id = saved["trace_id"]
    input_file = saved["input_file"]

    # Poll until done
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Status: {batch.status} ({batch.request_counts.completed}/{batch.request_counts.total} done)")
        if batch.status == "completed":
            break
        elif batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch.status}: {batch.errors}")
        time.sleep(5)

    # Download results
    content = client.files.content(batch.output_file_id).content
    results = [json.loads(line) for line in content.decode().strip().split("\n")]

    # Load original requests
    with open(input_file) as f:
        requests = [json.loads(line) for line in f]

    # Log completions back into the ORIGINAL trace
    respan.log_batch_results(requests, results, trace_id=trace_id)

    for r in results:
        idx = int(r["custom_id"].split("-")[1])
        message = r["response"]["body"]["choices"][0]["message"]["content"]
        print(f"{TOPICS[idx]}: {message}")


# ── Run both phases ──────────────────────────────────────────────────

# Phase 1: submit
saved = submit()
respan.flush()
print(f"\n--- Batch submitted. Waiting for results... ---\n")

# Phase 2: come back later and retrieve
# (In production, this would be a completely separate process/job)
retrieve_and_log(saved)
respan.flush()
print("\nDone — check the trace in Respan!")
