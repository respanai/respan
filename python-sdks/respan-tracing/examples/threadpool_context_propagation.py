"""Trace parallel work launched through ThreadPoolExecutor and asyncio.

Plain ThreadPoolExecutor workers start with a fresh thread-local context.
Use ContextPropagatingThreadPoolExecutor when parallel agent steps should stay
attached to the parent Respan workflow and inherit processor routing.
"""

import asyncio
from respan_tracing import (
    ContextPropagatingThread,
    ContextPropagatingThreadPoolExecutor,
    RespanTelemetry,
    add_done_callback_with_current_context,
    get_client,
    run_in_executor_with_current_context,
    task,
    to_thread_with_current_context,
    workflow,
)

telemetry = RespanTelemetry(app_name="threadpool-agent")


@task(name="retrieve_context")
def retrieve_context(query: str) -> str:
    return f"context for {query}"


@task(name="score_context")
def score_context(context: str) -> int:
    return len(context)


def record_final_count(_future) -> None:
    with get_client().start_span("record_final_count", kind="task"):
        pass


@workflow(name="parallel_retrieval_agent", processors="debug")
def parallel_retrieval_agent(queries: list[str]) -> list[int]:
    with ContextPropagatingThreadPoolExecutor(max_workers=4) as executor:
        retrievals = [executor.submit(retrieve_context, query) for query in queries]
        contexts = [future.result() for future in retrievals]

        scoring = [executor.submit(score_context, context) for context in contexts]
        add_done_callback_with_current_context(scoring[-1], record_final_count)
        return [future.result() for future in scoring]


@workflow(name="async_parallel_retrieval_agent", processors="debug")
async def async_parallel_retrieval_agent(queries: list[str]) -> list[int]:
    with ContextPropagatingThreadPoolExecutor(max_workers=4) as executor:
        contexts = await asyncio.gather(
            *[
                run_in_executor_with_current_context(executor, retrieve_context, query)
                for query in queries
            ]
        )
        return await asyncio.gather(
            *[
                to_thread_with_current_context(score_context, context)
                for context in contexts
            ]
        )


if __name__ == "__main__":
    warmup = ContextPropagatingThread(
        target=lambda: parallel_retrieval_agent(["warmup"])
    )
    warmup.start()
    warmup.join()

    print(parallel_retrieval_agent(["pricing", "latency", "tool failures"]))
    print(asyncio.run(async_parallel_retrieval_agent(["pricing", "latency"])))
    telemetry.flush()
