"""Deterministic fixtures for the released Temporal workflow sandbox."""

from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError


@activity.defn
async def echo(value: dict) -> dict:
    return value


@activity.defn
async def fail(reason: str) -> None:
    raise ApplicationError(reason, non_retryable=True)


@workflow.defn
class Echo:
    @workflow.run
    async def run(self, value: dict) -> dict:
        return await workflow.execute_activity(
            echo, value, start_to_close_timeout=timedelta(seconds=10)
        )


@workflow.defn
class Failure:
    @workflow.run
    async def run(self, reason: str) -> None:
        await workflow.execute_activity(
            fail,
            reason,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )


@workflow.defn
class Approval:
    def __init__(self):
        self.approved = False

    @workflow.run
    async def run(self, topic: str) -> str:
        await workflow.wait_condition(lambda: self.approved)
        return f"approved:{topic}"

    @workflow.signal
    async def approve(self) -> None:
        self.approved = True

    @workflow.query
    def status(self) -> str:
        return "approved" if self.approved else "pending"
