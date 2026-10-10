"""Released Pipecat pipeline/observer fixtures; no SDK methods are replaced."""

import asyncio

from openai import _exceptions

httpx = getattr(_exceptions, "httpx", None) or _exceptions.httpx2
from openai import AuthenticationError
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    FunctionCallFromLLM,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import LLMService, LLMSettings

try:
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.workers.runner import WorkerRunner

    CURRENT = True
except ImportError:
    from pipecat.pipeline.runner import PipelineRunner as WorkerRunner
    from pipecat.pipeline.task import PipelineParams
    from pipecat.pipeline.task import PipelineTask as PipelineWorker

    CURRENT = False


class Service(LLMService):
    def __init__(
        self,
        *,
        fail=False,
        tool=False,
        veto=None,
        tool_args=None,
        cancel=False,
        partial=False,
    ):
        super().__init__(
            name="FixtureLLM",
            settings=LLMSettings(
                **{
                    f.name: (
                        "fixture-pipecat"
                        if f.name == "model"
                        else {}
                        if f.name == "extra"
                        else None
                    )
                    for f in __import__("dataclasses").fields(LLMSettings)
                }
            ),
        )
        self.tool_args = tool_args
        self.partial = partial
        self.cancel = cancel
        self.fail = fail
        self.tool = tool
        self.veto = veto
        self._tool_done = asyncio.Event()
        if tool:
            self.register_function("vector_tool", self._vector_handler)

    async def _vector_handler(self, params):
        await params.result_callback(
            {
                "dense": [i / 5000 for i in range(5000)],
                "sparse": {i: float(i) for i in range(256)},
            }
        )
        self._tool_done.set()

    def can_generate_metrics(self):
        return True

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            return await self.push_frame(frame, direction)
        await self.push_frame(LLMFullResponseStartFrame())
        if self.veto:
            self.veto()
        if self.fail:
            if self.partial:
                await self.push_frame(LLMTextFrame("Actual partial text."))
            request = httpx.Request("POST", "https://provider.test/chat")
            response = httpx.Response(
                401,
                request=request,
                json={"error": {"message": "Controlled authorization failure."}},
            )
            exception = AuthenticationError(
                "Controlled authorization failure.",
                response=response,
                body={"message": "Controlled authorization failure."},
            )
            return await self.push_frame(
                ErrorFrame(
                    error="Controlled authorization failure.",
                    exception=exception,
                    processor=self,
                )
            )
        await self.push_frame(LLMTextFrame("Actual native text."))
        if self.cancel:
            return
        if self.tool:
            await self.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name="vector_tool",
                        tool_call_id="current-call",
                        arguments=self.tool_args or {"values": list(range(120))},
                        context=frame.context,
                    )
                ]
            )
            await asyncio.wait_for(self._tool_done.wait(), 5)
        await self.start_llm_usage_metrics(
            LLMTokenUsage(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=23,
                cache_read_input_tokens=4,
                cache_creation_input_tokens=5,
                reasoning_tokens=2,
            )
        )
        await self.push_frame(LLMFullResponseEndFrame())


class Collector(FrameProcessor):
    def __init__(self, *, cancel_mode=False):
        super().__init__(name="collector", enable_direct_mode=True)
        self.cancel_mode = cancel_mode
        self.done = asyncio.Event()
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        if self.cancel_mode and isinstance(frame, LLMTextFrame):
            self.done.set()
        if isinstance(
            frame, (ErrorFrame, LLMFullResponseEndFrame, CancelFrame)
        ) or type(frame).__name__ in {"TranscriptionFrame", "TTSTextFrame"}:
            self.done.set()
        await self.push_frame(frame, direction)


async def run(
    *,
    fail=False,
    tool=False,
    veto=None,
    messages=None,
    tools=None,
    setup=None,
    service=None,
):
    collector = Collector(cancel_mode=getattr(service, "cancel", False))
    worker = PipelineWorker(
        Pipeline([service or Service(fail=fail, tool=tool, veto=veto), collector]),
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
        conversation_id="fixture-conversation",
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    if setup:
        setup(worker)
    runner = WorkerRunner(handle_sigint=False)
    if CURRENT:
        await runner.add_workers(worker)

    async def drive():
        await asyncio.sleep(0.02)
        await worker.queue_frame(
            LLMContextFrame(
                LLMContext(
                    messages=messages
                    or [{"role": "user", "content": "Actual native prompt."}],
                    **({"tools": tools} if tools is not None else {}),
                )
            )
        )
        await asyncio.wait_for(collector.done.wait(), 10)
        if collector.cancel_mode:
            await worker.cancel(reason="controlled cancellation")
        else:
            await worker.queue_frame(EndFrame())

    await asyncio.wait_for(
        asyncio.gather(runner.run() if CURRENT else runner.run(worker), drive()), 15
    )
    return collector, worker
