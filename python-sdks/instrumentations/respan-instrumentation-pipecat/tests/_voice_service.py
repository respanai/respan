"""Native voice service frame protocol fixtures; no audio model or transport claims."""

import dataclasses

from pipecat.frames.frames import (
    LLMContextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService, STTSettings
from pipecat.services.tts_service import TTSService, TTSSettings


def settings(cls, model):
    return cls(
        **{
            f.name: model if f.name == "model" else {} if f.name == "extra" else None
            for f in dataclasses.fields(cls)
        }
    )


class STT(STTService):
    def __init__(self):
        super().__init__(
            name="FixtureSTTService", settings=settings(STTSettings, "fixture-stt")
        )

    async def run_stt(self, audio):
        if False:
            yield None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self.push_frame(VADUserStartedSpeakingFrame())
            await self.push_frame(
                TranscriptionFrame(
                    "Actual native transcription.",
                    "fixture-user",
                    "2026-10-05T00:00:00Z",
                )
            )
            await self.push_frame(VADUserStoppedSpeakingFrame())


class TTS(TTSService):
    def __init__(self):
        super().__init__(
            name="FixtureTTSService",
            sample_rate=16000,
            settings=settings(TTSSettings, "fixture-tts"),
        )

    async def run_tts(self, text, context_id=None):
        if False:
            yield None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(
                TTSTextFrame("Actual native speech text.", aggregated_by="sentence")
            )
