"""Opt-in per-turn audio clips (``capture_audio=True``), off by default.

Two clips per turn, each a 16-bit PCM WAV data URL that Respan stores as log
media and the log panel plays:

- the agent's speech, the frames ``tts_node`` produced, set on that span
  (log type ``speech``, output ``{"audio_url": ...}``);
- the user's speech, the frames pushed to speech-to-text while the call's
  ``user_turn`` was open (plus a short pre-roll, so the first word is kept).
  LiveKit records ``user_turn`` in the call's trace, so the clip goes on a
  ``user_speech`` span emitted under the ``agent_turn`` that answers it (log
  type ``transcription``, input ``{"audio_url": ...}``, the transcript as
  output), in that turn's own trace. LiveKit can start that reply before it
  ends ``user_turn`` (it ends it after ``on_user_turn_completed``, and a
  preemptive reply starts before the transcript is final), so the reply takes
  the recording as it starts, and its ``user_speech`` stays open until the
  ``user_turn`` ends with the transcript.

Frames are copied as they pass and never held up; the WAV is encoded once,
when the span ends. A clip stops at ``MAX_CLIP_SECONDS`` and its span is marked
``respan.metadata.voice.audio_truncated``. Every buffer is released when its
span or turn ends, and the per-call and per-span tables are bounded.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import wave
from collections import OrderedDict, deque

from ._policy import key

TTS_NODE_SPAN = "tts_node"
USER_TURN_SPAN = "user_turn"
USER_SPEECH_SPAN = "user_speech"
AGENT_SESSION_SPAN = "agent_session"
AUDIO_URL_KEY = "audio_url"
SPEECH_LOG_TYPE = "speech"
TRANSCRIPTION_LOG_TYPE = "transcription"
TRUNCATED_VOICE_KEY = "audio_truncated"
# 60 s of 16 kHz mono 16-bit PCM is ~1.9 MB, far under the backend's 20 MB
# per-log cap; a longer turn is cut there and marked.
MAX_CLIP_SECONDS = 60
PRE_ROLL_SECONDS = 0.5
_SAMPLE_WIDTH_BYTES = 2
_MAX_OPEN_CLIPS = 64
_MAX_CALLS = 256


class Clip:
    def __init__(self):
        self.chunks = []
        self.sample_rate = None
        self.channels = None
        self.samples = 0
        self.is_truncated = False

    def add(self, pcm, *, sample_rate, channels, samples):
        if self.sample_rate is None:
            self.sample_rate, self.channels = sample_rate, channels
        elif (sample_rate, channels) != (self.sample_rate, self.channels):
            # One WAV has one format: keep what came first, mark the rest.
            self.is_truncated = True
            return
        if self.samples + samples > MAX_CLIP_SECONDS * sample_rate:
            self.is_truncated = True
            return
        self.chunks.append(pcm)
        self.samples += samples

    def data_url(self):
        if not self.samples:
            return None
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(_SAMPLE_WIDTH_BYTES)
            wav.setframerate(self.sample_rate)
            wav.writeframes(b"".join(self.chunks))
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{encoded}"


def payload(clip):
    """``{"audio_url": <data URL>}`` as the JSON a span input/output carries,
    or None for an empty clip."""
    url = clip.data_url() if clip is not None else None
    return json.dumps({AUDIO_URL_KEY: url}) if url else None


def _frame_fields(frame):
    """A copy of the frame's PCM and its format. The copy is the only work
    done on the audio path."""
    return (
        bytes(frame.data),
        {
            "sample_rate": frame.sample_rate,
            "channels": frame.num_channels,
            "samples": frame.samples_per_channel,
        },
    )


class _CallAudio:
    def __init__(self):
        self.pre_roll = deque()
        self.pre_roll_samples = 0
        self.recording = None
        self.pending = None
        self.transcript = None
        # A ``user_speech`` span waiting for its ``user_turn`` to end.
        self.waiting = None


class AudioClips:
    def __init__(self):
        self.lock = threading.Lock()
        # tts_node span key -> its Clip
        self.speech = OrderedDict()
        # call trace id -> _CallAudio
        self.calls = OrderedDict()

    def add_tts_frame(self, span, frame):
        span_key = key(span)
        if span_key is None:
            return
        pcm, fields = _frame_fields(frame)
        with self.lock:
            clip = self.speech.get(span_key)
            if clip is None:
                clip = Clip()
                self.speech[span_key] = clip
                while len(self.speech) > _MAX_OPEN_CLIPS:
                    self.speech.popitem(last=False)
            clip.add(pcm, **fields)

    def attach(self, span_key, clip):
        """Hold ``clip`` for the span ``span_key`` until it ends."""
        with self.lock:
            self.speech[span_key] = clip
            while len(self.speech) > _MAX_OPEN_CLIPS:
                self.speech.popitem(last=False)

    def take_speech(self, span_key):
        """The clip held for a span, released as the span ends."""
        with self.lock:
            return self.speech.pop(span_key, None)

    def _call(self, call_id):
        call = self.calls.get(call_id)
        if call is None:
            call = self.calls[call_id] = _CallAudio()
            while len(self.calls) > _MAX_CALLS:
                self.calls.popitem(last=False)
        return call

    def add_user_frame(self, span, frame):
        span_key = key(span)
        if span_key is None:
            return
        pcm, fields = _frame_fields(frame)
        with self.lock:
            call = self._call(span_key[0])
            if call.recording is not None:
                call.recording.add(pcm, **fields)
                return
            call.pre_roll.append((pcm, fields))
            call.pre_roll_samples += fields["samples"]
            while (
                call.pre_roll
                and call.pre_roll_samples - call.pre_roll[0][1]["samples"]
                >= PRE_ROLL_SECONDS * fields["sample_rate"]
            ):
                call.pre_roll_samples -= call.pre_roll.popleft()[1]["samples"]

    def user_turn_started(self, call_id):
        with self.lock:
            call = self._call(call_id)
            call.recording = Clip()
            for pcm, fields in call.pre_roll:
                call.recording.add(pcm, **fields)
            call.pre_roll.clear()
            call.pre_roll_samples = 0

    def user_turn_ended(self, call_id, transcript):
        with self.lock:
            call = self.calls.get(call_id)
            if call is None or call.recording is None:
                return
            call.pending, call.recording = call.recording, None
            call.transcript = transcript

    def take_user_speech(self, call_id):
        """For a reply starting in this call: the user's turn it answers, as
        ``(clip, transcript, is_turn_open)``, handed out once. A turn still
        recording is cut here (the reply has started; what follows is not
        part of it); its transcript comes when the turn ends."""
        with self.lock:
            call = self.calls.get(call_id)
            if call is None:
                return None, None, False
            if call.recording is not None:
                clip, call.recording = call.recording, None
                return clip, None, True
            clip, transcript = call.pending, call.transcript
            call.pending = call.transcript = None
            return clip, transcript, False

    def wait_for_turn(self, call_id, speech_span):
        """Hold ``speech_span`` open until this call's ``user_turn`` ends."""
        with self.lock:
            self._call(call_id).waiting = speech_span

    def take_waiting(self, call_id):
        with self.lock:
            call = self.calls.get(call_id)
            if call is None:
                return None
            span, call.waiting = call.waiting, None
            return span

    def end_call(self, call_id):
        """Release the call; a ``user_speech`` still waiting is returned to be
        ended."""
        with self.lock:
            call = self.calls.pop(call_id, None)
            return call.waiting if call is not None else None
