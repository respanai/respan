"""Session replay for browser agents: record the page with rrweb over CDP.

Framework-neutral. An instrumentation hands ``SessionRecorder`` a raw CDP
``send(method, params, session_id)`` coroutine and tells it which targets
exist and which one the agent is looking at; the recorder does the rest:

- injects rrweb's recorder into every page / frame target
  (``Page.addScriptToEvaluateOnNewDocument`` for future documents plus one
  ``Runtime.evaluate`` for the document already loaded), with every input
  masked and every declared secret masked out of page text;
- drains the page-global buffer with ``Runtime.evaluate`` (periodically, and
  whenever the caller asks: before an action that may navigate, at step end,
  at run end), keeping events in memory with the target they came from;
- stops collecting above a size cap and marks the recording ``truncated``.

``upload_recording`` then sends it to Respan once (gzipped JSON, see
``RECORDINGS_PATH``) and never raises: on any failure it logs a warning and
writes a local ``.json.gz`` plus a standalone ``.html`` player instead.
``set_replay_metadata`` puts the resulting link on the agent's root span.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, Optional

import requests
from respan_sdk.constants.span_attributes import RESPAN_METADATA

from respan_tracing.constants.generic_constants import TRACE_CONTENT_OFF_VALUES
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

RECORD_SESSION_ENV = "RESPAN_RECORD_SESSION"
RECORDINGS_PATH = "/browser-sessions/recordings/"
SESSION_REPLAY_URL_METADATA_KEY = "session_replay_url"
SESSION_REPLAY_ID_METADATA_KEY = "session_replay_id"

DRAIN_INTERVAL_SECONDS = 1.5
CDP_CALL_TIMEOUT_SECONDS = 3.0
UPLOAD_TIMEOUT_SECONDS = 15.0
MAX_RAW_BYTES = 20 * 1024 * 1024
FOCUS_SWITCH_SLACK_MS = 20
LOCAL_REPLAY_DIR = "respan-session-replays"

# rrweb eventWithTime.type values the recorder relies on.
RRWEB_FULL_SNAPSHOT = 2
RRWEB_META = 4

# Recorder and player share rrweb's major version (the backend's player pins the same).
RRWEB_PLAYER_VERSION = "1.0.0-alpha.4"
RRWEB_RECORD_ASSET = "rrweb-record.min.js"

# CDP target types that host a document rrweb can record.
PAGE_TARGET_TYPES = frozenset({"page", "tab"})
FRAME_TARGET_TYPES = frozenset({"iframe"})

_RECORDER_GLOBAL = "__respanRecorder"
CdpSend = Callable[[str, Optional[dict], Optional[str]], Awaitable[dict]]


def is_session_recording_requested(explicit: Optional[bool] = None) -> bool:
    """An explicit argument wins; otherwise ``RESPAN_RECORD_SESSION`` (off by default)."""
    if explicit is not None:
        return explicit
    value = os.getenv(RECORD_SESSION_ENV, "").strip().lower()
    return bool(value) and value not in TRACE_CONTENT_OFF_VALUES


@cache
def _rrweb_record_source() -> str:
    asset = resources.files("respan_tracing.utils").joinpath("assets", RRWEB_RECORD_ASSET)
    return asset.read_text(encoding="utf-8")


def recorder_script(secrets: Sequence[str] = ()) -> str:
    """The script injected into each document: rrweb + a drainable buffer.

    Same-origin child frames are skipped (the parent's recorder serializes
    them); cross-origin frames post their events to the parent, which is how
    rrweb merges out-of-process iframes into one stream.
    """
    return (
        "(function(){if(window.%(g)s&&!window.%(g)s.stopped)return;"
        "try{if(window.parent!==window){window.parent.document;return;}}catch(e){}"
        "%(rrweb)s\n"
        "var secrets=%(secrets)s;"
        "function maskText(t){for(var i=0;i<secrets.length;i++){var s=secrets[i];"
        "if(s&&t.indexOf(s)!==-1){t=t.split(s).join(new Array(s.length+1).join('*'));}}return t;}"
        "var r={buf:[],stopped:false,stop:null};"
        "var opts={emit:function(e){if(!r.stopped)r.buf.push(e);},maskAllInputs:true,"
        "recordCrossOriginIframes:true,sampling:{mousemove:50,scroll:150,input:'last'}};"
        "if(secrets.length){opts.maskTextSelector='*';opts.maskTextFn=maskText;}"
        "r.drain=function(){var b=r.buf;r.buf=[];return JSON.stringify(b);};"
        "r.snapshot=function(){try{rrwebRecord.takeFullSnapshot(true);}catch(e){}};"
        "r.halt=function(){r.stopped=true;r.buf=[];try{r.stop&&r.stop();}catch(e){}};"
        "window.%(g)s=r;"
        "try{r.stop=rrwebRecord(opts);}catch(e){}"
        "})();"
    ) % {
        "g": _RECORDER_GLOBAL,
        "rrweb": _rrweb_record_source(),
        "secrets": json.dumps([s for s in secrets if s]),
    }


DRAIN_EXPRESSION = f"window.{_RECORDER_GLOBAL}?window.{_RECORDER_GLOBAL}.drain():null"
SNAPSHOT_EXPRESSION = f"window.{_RECORDER_GLOBAL}&&window.{_RECORDER_GLOBAL}.snapshot()"
HALT_EXPRESSION = f"window.{_RECORDER_GLOBAL}&&window.{_RECORDER_GLOBAL}.halt()"


@dataclass
class Recording:
    events: list[dict[str, Any]]
    started_at: datetime
    ended_at: datetime
    raw_bytes: int
    truncated: bool
    secrets: tuple[str, ...] = ()

    @property
    def has_full_snapshot(self) -> bool:
        return any(e.get("type") == RRWEB_FULL_SNAPSHOT for e in self.events)


@dataclass
class _Target:
    target_id: str
    session_id: str
    is_page: bool
    script_id: Optional[str] = None


@dataclass
class _Tagged:
    target_id: str
    event: dict[str, Any]


class SessionRecorder:
    """Collect rrweb events from every page of one browser for one agent run."""

    def __init__(
        self,
        send: CdpSend,
        *,
        secrets: Sequence[str] = (),
        max_raw_bytes: int = MAX_RAW_BYTES,
        drain_interval: float = DRAIN_INTERVAL_SECONDS,
    ) -> None:
        self.send = send
        self.secrets = tuple(s for s in secrets if s)
        self.max_raw_bytes = max_raw_bytes
        self.drain_interval = drain_interval
        self._targets: dict[str, _Target] = {}
        self._events: list[_Tagged] = []
        # (epoch ms, target id) whenever the agent's focused tab changes.
        self._focus_changes: list[tuple[float, str]] = []
        self._focus: Optional[str] = None
        self._raw_bytes = 0
        self._truncated = False
        self._stopped = False
        self._started_at = datetime.now(timezone.utc)
        self._ended_at: Optional[datetime] = None
        self._loop_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._script: Optional[str] = None

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    async def _call(self, method: str, params: dict, session_id: str) -> Optional[dict]:
        try:
            return await asyncio.wait_for(
                self.send(method, params, session_id), CDP_CALL_TIMEOUT_SECONDS
            )
        except Exception:
            # Targets close, navigate and crash under us; a lost call is a gap in
            # the replay, never an error in the agent.
            logger.debug("session recording CDP call %s failed", method, exc_info=True)
            return None

    async def attach(self, *, target_id: str, session_id: str, target_type: str) -> None:
        """Inject the recorder into a target (idempotent per session)."""
        is_page = target_type in PAGE_TARGET_TYPES
        if self._stopped or self._truncated or not (is_page or target_type in FRAME_TARGET_TYPES):
            return
        known = self._targets.get(target_id)
        if known is not None and known.session_id == session_id:
            return
        target = self._targets[target_id] = _Target(target_id, session_id, is_page)
        if self._script is None:
            self._script = recorder_script(self.secrets)
        added = await self._call(
            "Page.addScriptToEvaluateOnNewDocument", {"source": self._script}, session_id
        )
        target.script_id = (added or {}).get("identifier")
        await self._call("Runtime.evaluate", {"expression": self._script}, session_id)

    def detach(self, target_id: str) -> None:
        self._targets.pop(target_id, None)

    async def set_focus(self, target_id: Optional[str]) -> None:
        """Follow the agent's tab: the replay switches to it at its next snapshot."""
        if not target_id or target_id == self._focus or self._stopped:
            return
        previous, self._focus = self._focus, target_id
        if previous is None:
            return
        # Page clocks report whole milliseconds; leave a little slack so the
        # snapshot requested just below is not ordered before the switch.
        self._focus_changes.append((time.time() * 1000 - FOCUS_SWITCH_SLACK_MS, target_id))
        target = self._targets.get(target_id)
        if target is not None and target.is_page:
            # A fresh Meta + FullSnapshot so the player can render the new tab.
            await self._call("Runtime.evaluate", {"expression": SNAPSHOT_EXPRESSION}, target.session_id)

    async def drain(self) -> None:
        """Pull buffered events from every page; returns quickly when nothing is there."""
        if self._stopped:
            return
        async with self._lock:
            pages = [t for t in self._targets.values() if t.is_page]
            results = await asyncio.gather(
                *(
                    self._call(
                        "Runtime.evaluate",
                        {"expression": DRAIN_EXPRESSION, "returnByValue": True},
                        t.session_id,
                    )
                    for t in pages
                )
            )
            for target, result in zip(pages, results, strict=True):
                self._collect(target.target_id, result)

    def _collect(self, target_id: str, result: Optional[dict]) -> None:
        payload = ((result or {}).get("result") or {}).get("value")
        if not isinstance(payload, str) or payload == "[]" or self._truncated:
            return
        if self._raw_bytes + len(payload) > self.max_raw_bytes:
            self._truncated = True
            logger.warning(
                "Respan session recording stopped at %d bytes (cap %d); the replay is truncated",
                self._raw_bytes,
                self.max_raw_bytes,
            )
            return
        try:
            events = json.loads(payload)
        except ValueError:
            return
        self._raw_bytes += len(payload)
        self._events.extend(_Tagged(target_id, e) for e in events if isinstance(e, dict))

    def start(self) -> None:
        """Drain every ``drain_interval`` seconds until ``stop``."""
        if self._loop_task is None and not self._stopped:
            self._loop_task = asyncio.ensure_future(self._drain_forever())

    async def _drain_forever(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self.drain_interval)
            try:
                await self.drain()
            except Exception:
                logger.debug("session recording drain failed", exc_info=True)
            if self._truncated:
                await self._halt_pages()
                return

    async def _halt_pages(self) -> None:
        """Stop the in-page recorders and unregister the script, so a browser that
        outlives the run (``keep_alive``) does not keep buffering events nobody reads."""
        calls = []
        for target in self._targets.values():
            calls.append(
                self._call("Runtime.evaluate", {"expression": HALT_EXPRESSION}, target.session_id)
            )
            if target.script_id:
                calls.append(
                    self._call(
                        "Page.removeScriptToEvaluateOnNewDocument",
                        {"identifier": target.script_id},
                        target.session_id,
                    )
                )
        await asyncio.gather(*calls)

    async def stop(self) -> None:
        """Final drain, then stop the loop and the in-page recorders. Idempotent."""
        if self._stopped:
            return
        task, self._loop_task = self._loop_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await self.drain()
        self._stopped = True
        self._ended_at = datetime.now(timezone.utc)
        await self._halt_pages()

    def recording(self) -> Recording:
        return Recording(
            events=linearize(self._events, self._focus_changes),
            started_at=self._started_at,
            ended_at=self._ended_at or datetime.now(timezone.utc),
            raw_bytes=self._raw_bytes,
            truncated=self._truncated,
            secrets=self.secrets,
        )


def linearize(events: Iterable[_Tagged], focus_changes: Sequence[tuple[float, str]]) -> list[dict]:
    """One playable stream across tabs, following the agent's focus.

    rrweb replays a single document, and incremental events refer to node ids of
    the snapshot they follow, so tabs cannot be interleaved. The stream stays on
    one tab and switches to a newly focused tab at that tab's next Meta event
    (``set_focus`` asks it for a fresh snapshot).
    """
    ordered = sorted(events, key=lambda tagged: tagged.event.get("timestamp") or 0)
    switches = sorted(focus_changes)
    current: Optional[str] = None
    pending: Optional[str] = None
    index = 0
    stream: list[dict] = []
    for tagged in ordered:
        timestamp = tagged.event.get("timestamp") or 0
        while index < len(switches) and switches[index][0] <= timestamp:
            pending = switches[index][1]
            index += 1
        if tagged.event.get("type") == RRWEB_META and (
            current is None or tagged.target_id == pending
        ):
            current, pending = tagged.target_id, None
        if tagged.target_id == current:
            stream.append(tagged.event)
    return stream


# -- upload ------------------------------------------------------------------


@dataclass
class RecordingUpload:
    recording_id: str
    view_url: str
    events_url: Optional[str] = None
    expires_at: Optional[str] = None


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _mask_json(text: str, secrets: Sequence[str]) -> str:
    """Backstop for what rrweb's masking cannot see (attributes, URLs, CSS)."""
    for secret in sorted(set(secrets), key=len, reverse=True):
        for variant in {secret, json.dumps(secret)[1:-1]}:
            text = text.replace(variant, "*" * len(secret))
    return text


def recording_body(
    recording: Recording,
    *,
    trace_id: str,
    span_id: str,
    session_id: Optional[str],
    framework: str,
) -> bytes:
    """The gzipped JSON body ``RECORDINGS_PATH`` accepts."""
    body = {
        "trace_id": trace_id,
        "span_id": span_id,
        "session_id": session_id or "",
        "started_at": _iso(recording.started_at),
        "ended_at": _iso(recording.ended_at),
        "framework": framework,
        "events": recording.events,
    }
    text = _mask_json(json.dumps(body, separators=(",", ":")), recording.secrets)
    return gzip.compress(text.encode("utf-8"))


def _resolve_endpoint(base_url: Optional[str], api_key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    # The running Respan client is the source of truth; env is the fallback.
    tracer = getattr(RespanTracer, "_instance", None)
    base_url = base_url or getattr(tracer, "api_endpoint", None) or os.getenv("RESPAN_BASE_URL")
    api_key = api_key or getattr(tracer, "api_key", None) or os.getenv("RESPAN_API_KEY")
    return base_url, api_key


def _post(url: str, api_key: str, body: bytes, timeout: float) -> requests.Response:
    return requests.post(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
        },
        timeout=timeout,
    )


async def upload_recording(
    recording: Recording,
    *,
    trace_id: str,
    span_id: str,
    session_id: Optional[str],
    framework: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = UPLOAD_TIMEOUT_SECONDS,
) -> Optional[RecordingUpload]:
    """POST the recording once. Never raises; on failure keeps a local copy."""
    if not recording.events:
        return None
    try:
        body = recording_body(
            recording,
            trace_id=trace_id,
            span_id=span_id,
            session_id=session_id,
            framework=framework,
        )
    except Exception:
        logger.warning("Respan session recording could not be serialized", exc_info=True)
        return None
    base_url, api_key = _resolve_endpoint(base_url, api_key)
    reason = None
    if not base_url or not api_key:
        reason = "no Respan base URL / API key"
    else:
        url = base_url.rstrip("/") + RECORDINGS_PATH
        try:
            response = await asyncio.to_thread(_post, url, api_key, body, timeout)
            if response.status_code == 201:
                data = response.json()
                return RecordingUpload(
                    recording_id=str(data["recording_id"]),
                    view_url=str(data["view_url"]),
                    events_url=data.get("events_url"),
                    expires_at=data.get("expires_at"),
                )
            reason = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
    paths = write_local_replay(body, trace_id=trace_id)
    logger.warning(
        "Respan session replay upload failed (%s); saved locally: %s",
        reason,
        ", ".join(str(p) for p in paths) or "(could not write)",
    )
    return None


# -- local fallback ------------------------------------------------------------

_PLAYER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Respan session replay {trace_id}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/rrweb-player@{version}/dist/style.css">
<style>body{{margin:0;font:14px system-ui,sans-serif;background:#f5f5f5}}header{{padding:10px 16px}}#player{{padding:0 16px}}</style>
</head><body>
<header>trace <code>{trace_id}</code> &middot; session <code>{session_id}</code> &middot; {duration}</header>
<div id="player"></div>
<script id="events" type="application/json">{events}</script>
<script src="https://cdn.jsdelivr.net/npm/rrweb-player@{version}/dist/index.js"></script>
<script>
var events = JSON.parse(document.getElementById("events").textContent);
window.respanReplay = new rrwebPlayer({{target: document.getElementById("player"),
  props: {{events: events, autoPlay: true, width: Math.min(window.innerWidth - 32, 1280)}}}});
</script>
</body></html>
"""


def player_html(body: dict[str, Any]) -> str:
    """A self-contained page that plays ``body['events']`` with rrweb-player."""
    events = json.dumps(body.get("events") or [], separators=(",", ":"))
    try:
        started = datetime.fromisoformat(body["started_at"].replace("Z", "+00:00"))
        ended = datetime.fromisoformat(body["ended_at"].replace("Z", "+00:00"))
        duration = f"{(ended - started).total_seconds():.1f}s"
    except Exception:
        duration = ""
    return _PLAYER_HTML.format(
        version=RRWEB_PLAYER_VERSION,
        trace_id=_html_text(body.get("trace_id")),
        session_id=_html_text(body.get("session_id")),
        duration=duration,
        # "</" would end the script element early.
        events=events.replace("</", "<\\/"),
    )


def _html_text(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z_.:-]", "", str(value or ""))


def write_local_replay(body: bytes, *, trace_id: str, directory: Optional[Path] = None) -> list[Path]:
    """Write ``<trace>.json.gz`` + ``<trace>.html``; returns the paths written."""
    try:
        folder = directory or Path(tempfile.gettempdir()) / LOCAL_REPLAY_DIR
        folder.mkdir(parents=True, exist_ok=True)
        name = _html_text(trace_id) or "recording"
        gz_path = folder / f"{name}.json.gz"
        gz_path.write_bytes(body)
        html_path = folder / f"{name}.html"
        html_path.write_text(player_html(json.loads(gzip.decompress(body))), encoding="utf-8")
        return [gz_path, html_path]
    except Exception:
        logger.debug("could not write the local session replay", exc_info=True)
        return []


def set_replay_metadata(span: Any, upload: Optional[RecordingUpload]) -> None:
    """Flat ``respan.metadata.session_replay_url`` / ``_id`` on the agent's root span."""
    if upload is None:
        return
    span.set_attribute(f"{RESPAN_METADATA}.{SESSION_REPLAY_URL_METADATA_KEY}", upload.view_url)
    span.set_attribute(f"{RESPAN_METADATA}.{SESSION_REPLAY_ID_METADATA_KEY}", upload.recording_id)
