"""Batch API — Submit multiple requests for async processing at 50% cost.

The Batch API uses OpenAI's file and batch endpoints directly.
Respan decorators trace the batch workflow for observability.

Set OPENAI_API_KEY (direct OpenAI key) and RESPAN_API_KEY in .env:

    RESPAN_API_KEY=your-respan-key
    OPENAI_API_KEY=sk-proj-...
"""

import os
import json
import time

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from respan import Respan, workflow, task
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])

# Batch API requires direct OpenAI access (not gateway).
# The .env has a direct OPENAI_API_KEY (not the Respan gateway key).
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

TOPICS = [
    "quantum computing",
    "blockchain",
    "edge computing",
    "reinforcement learning",
    "zero-knowledge proofs",
]


@task(name="create_batch_file")
def create_batch_file() -> str:
    """Create a JSONL file with batch requests."""
    tasks = []
    for i, topic in enumerate(TOPICS):
        tasks.append({
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

    file_path = "/tmp/respan_batch_input.jsonl"
    with open(file_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Created {len(tasks)} tasks in {file_path}")
    return file_path


@task(name="upload_and_submit")
def upload_and_submit(file_path: str) -> str:
    """Upload the JSONL file and create a batch job."""
    batch_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch",
    )
    print(f"Uploaded file: {batch_file.id}")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch created: {batch.id} (status: {batch.status})")
    return batch.id


@task(name="poll_batch")
def poll_batch(batch_id: str) -> str:
    """Poll until the batch completes."""
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Status: {batch.status} ({batch.request_counts.completed}/{batch.request_counts.total} done)")

        if batch.status == "completed":
            return batch.output_file_id
        elif batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch.status}: {batch.errors}")

        time.sleep(5)


@task(name="download_results")
def download_results(output_file_id: str):
    """Download batch results and log each completion as a traced span."""
    content = client.files.content(output_file_id).content
    results = [json.loads(line) for line in content.decode().strip().split("\n")]

    with open("/tmp/respan_batch_input.jsonl") as f:
        requests = [json.loads(line) for line in f]

    # Log each batch result as an individual chat completion span
    respan.log_batch_results(requests, results)

    for r in results:
        idx = int(r["custom_id"].split("-")[1])
        message = r["response"]["body"]["choices"][0]["message"]["content"]
        print(f"{TOPICS[idx]}: {message}")


@workflow(name="batch_pipeline")
def run():
    file_path = create_batch_file()
    batch_id = upload_and_submit(file_path)
    output_file_id = poll_batch(batch_id)
    download_results(output_file_id)


run()
respan.flush()
