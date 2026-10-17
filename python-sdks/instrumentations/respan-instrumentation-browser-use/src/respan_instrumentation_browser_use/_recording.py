"""browser-use glue for respan_tracing's rrweb session recorder.

browser-use 0.13 drives Chrome through one root ``cdp-use`` client
(``BrowserSession._cdp_client_root``) with flattened sessions per target, and
``SessionManager._handle_target_attached`` is the single place every new tab or
out-of-process frame enters its pool. The recorder sends raw CDP on that root
client, attaches to the targets that exist when the run's first step/action
starts, and to every target attached afterwards (new tabs, popups, OOPIFs).
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Optional

from respan_tracing.utils.agent_instrumentation import secret_replacements
from respan_tracing.utils.session_recording import (
    SessionRecorder,
    set_replay_metadata,
    upload_recording,
)

from respan_instrumentation_browser_use._constants import INSTRUMENTATION_NAME

# The run's recording: Agent.step / Tools.act / Agent.close run inside Agent.run.
CURRENT: ContextVar[Optional[RunRecording]] = ContextVar(
    "respan_browser_use_recording", default=None
)
# Target-attach handlers run in tasks spawned by the CDP reader, outside the
# run's context, so they find the recording by browser session (pydantic models
# are unhashable, hence id()); entries live from the first step to Agent.close.
_BY_BROWSER: dict[int, RunRecording] = {}


class RunRecording:
    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.browser = getattr(agent, "browser_session", None)
        secrets = [secret for secret, _ in secret_replacements(getattr(agent, "sensitive_data", None))]
        self.recorder = SessionRecorder(self._send, secrets=secrets)
        self.started = False

    async def _send(self, method: str, params: Optional[dict], session_id: Optional[str]) -> dict:
        client = getattr(self.browser, "_cdp_client_root", None)
        if client is None:
            raise RuntimeError("browser-use CDP client is not connected")
        return await client.send_raw(method, params, session_id)

    async def ensure_started(self) -> None:
        """Attach to the open targets once the browser is up (first step/action)."""
        if self.started or self.recorder.is_stopped:
            return
        manager = getattr(self.browser, "session_manager", None)
        if manager is None or getattr(self.browser, "_cdp_client_root", None) is None:
            return  # not started yet; the next step/action retries
        self.started = True
        _BY_BROWSER[id(self.browser)] = self
        for target_id, target in list(manager.get_all_targets().items()):
            await self.attach(manager, target_id, target.target_type)
        await self.follow_focus()
        self.recorder.start()

    async def attach(self, manager: Any, target_id: str, target_type: str) -> None:
        sessions = manager.get_all_sessions_for_target(target_id)
        if sessions:
            await self.recorder.attach(
                target_id=target_id, session_id=sessions[0].session_id, target_type=target_type
            )

    async def follow_focus(self) -> None:
        await self.recorder.set_focus(getattr(self.browser, "agent_focus_target_id", None))

    async def drain(self) -> None:
        if self.started:
            await self.follow_focus()
            await self.recorder.drain()

    async def stop(self) -> None:
        """Final drain while the browser is still alive (Agent.close kills it)."""
        if self.started:
            await self.drain()
            await self.recorder.stop()
        if _BY_BROWSER.get(id(self.browser)) is self:
            del _BY_BROWSER[id(self.browser)]

    async def finish(self, span: Any) -> None:
        await self.stop()
        if not self.started:
            return
        context = span.get_span_context()
        upload = await upload_recording(
            self.recorder.recording(),
            trace_id=format(context.trace_id, "032x"),
            span_id=format(context.span_id, "016x"),
            session_id=str(getattr(self.agent, "session_id", "") or ""),
            framework=INSTRUMENTATION_NAME,
        )
        set_replay_metadata(span, upload)


async def on_target_attached(manager: Any, event: Any) -> None:
    """After browser-use pooled a new target, inject the recorder into it."""
    recording = _BY_BROWSER.get(id(getattr(manager, "browser_session", None)))
    if recording is None or recording.recorder.is_stopped:
        return
    info = event.get("targetInfo") or {}
    await recording.recorder.attach(
        target_id=info.get("targetId"),
        session_id=event.get("sessionId"),
        target_type=info.get("type", ""),
    )
