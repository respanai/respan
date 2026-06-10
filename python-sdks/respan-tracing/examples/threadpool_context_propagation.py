"""Trace parallel work launched through ThreadPoolExecutor.

Plain ThreadPoolExecutor workers start with a fresh thread-local context.
Use ContextPropagatingThreadPoolExecutor when parallel agent steps should stay
attached to the parent Respan workflow and inherit processor routing.
"""

from respan_tracing import (
    ContextPropagatingThreadPoolExecutor,
    RespanTelemetry,
    task,
    workflow,
)

telemetry = RespanTelemetry(app_name="threadpool-agent")


@task(name="retrieve_context")
def retrieve_context(query: str) -> str:
    return f"context for {query}"


@task(name="score_context")
def score_context(context: str) -> int:
    return len(context)


@workflow(name="parallel_retrieval_agent", processors="debug")
def parallel_retrieval_agent(queries: list[str]) -> list[int]:
    with ContextPropagatingThreadPoolExecutor(max_workers=4) as executor:
        retrievals = [executor.submit(retrieve_context, query) for query in queries]
        contexts = [future.result() for future in retrievals]

        scoring = [executor.submit(score_context, context) for context in contexts]
        return [future.result() for future in scoring]


if __name__ == "__main__":
    print(parallel_retrieval_agent(["pricing", "latency", "tool failures"]))
    telemetry.flush()
