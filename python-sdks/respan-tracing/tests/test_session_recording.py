"""SessionRecorder drain / upload / fallback logic, against a fake CDP and a stub server."""

from __future__ import annotations

import asyncio
import gzip
import json
from datetime import datetime, timezone
from typing import Optional

import pytest
from opentelemetry.sdk.trace import TracerProvider

from respan_tracing.testing.recordings import recording_stub_server
from respan_tracing.utils import session_recording
from respan_tracing.utils.session_recording import (
    RRWEB_FULL_SNAPSHOT,
    RRWEB_META,
    Recording,
    ReplayLinks,
    SessionRecorder,
    is_session_recording_requested,
    recorder_script,
    reserve_recording,
    set_replay_metadata,
    upload_recording,
)

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
SPAN_ID = "b7ad6b7169203331"
SECRET = "hunter2-secret"


class FakeBrowser:
    """Answers the CDP calls SessionRecorder makes, per session id."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, Optional[str]]] = []
        self.buffers: dict[str, list[dict]] = {}
        self.fail_sessions: set[str] = set()

    def push(self, session_id: str, *events: dict) -> None:
        self.buffers.setdefault(session_id, []).extend(events)

    async def send(self, method: str, params: Optional[dict], session_id: Optional[str]) -> dict:
        params = params or {}
        self.calls.append((method, params, session_id))
        if session_id in self.fail_sessions:
            raise RuntimeError("target closed")
        expression = params.get("expression", "")
        if method == "Page.addScriptToEvaluateOnNewDocument":
            return {"identifier": f"script-{session_id}"}
        if method == "Runtime.evaluate" and expression == session_recording.DRAIN_EXPRESSION:
            events, self.buffers[session_id] = self.buffers.get(session_id, []), []
            return {"result": {"type": "string", "value": json.dumps(events)}}
        return {"result": {}}


def meta(ts: int, href: str = "http://x/") -> dict:
    return {"type": RRWEB_META, "timestamp": ts, "data": {"href": href, "width": 1, "height": 1}}


def full(ts: int) -> dict:
    return {"type": RRWEB_FULL_SNAPSHOT, "timestamp": ts, "data": {"node": {}, "initialOffset": {}}}


def incremental(ts: int, text: str = "") -> dict:
    return {"type": 3, "timestamp": ts, "data": {"source": 0, "text": text}}


@pytest.mark.asyncio
async def test_attach_injects_the_recorder_for_new_and_current_documents():
    browser = FakeBrowser()
    recorder = SessionRecorder(browser.send, secrets=[SECRET])
    await recorder.attach(target_id="t1", session_id="s1", target_type="page")
    await recorder.attach(target_id="t1", session_id="s1", target_type="page")  # idempotent
    await recorder.attach(target_id="w1", session_id="s2", target_type="service_worker")

    methods = [(m, s) for m, _, s in browser.calls]
    assert methods == [
        ("Page.addScriptToEvaluateOnNewDocument", "s1"),
        ("Runtime.evaluate", "s1"),
    ]
    script = browser.calls[0][1]["source"]
    assert "maskAllInputs:true" in script
    assert json.dumps([SECRET]) in script  # page text equal to a secret is masked in-page
    assert "rrwebRecord" in script


@pytest.mark.asyncio
async def test_stop_halts_the_page_recorders_and_unregisters_the_script():
    browser = FakeBrowser()
    recorder = SessionRecorder(browser.send)
    await recorder.attach(target_id="t1", session_id="s1", target_type="page")
    await recorder.stop()
    tail = [(m, p.get("expression") or p.get("identifier")) for m, p, _ in browser.calls[-2:]]
    assert ("Runtime.evaluate", session_recording.HALT_EXPRESSION) in tail
    # A keep_alive browser outlives the run: new documents must not keep recording.
    assert ("Page.removeScriptToEvaluateOnNewDocument", "script-s1") in tail
    calls = len(browser.calls)
    await recorder.attach(target_id="t2", session_id="s2", target_type="page")
    await recorder.drain()
    assert len(browser.calls) == calls


@pytest.mark.asyncio
async def test_drain_collects_events_in_timestamp_order_across_pages():
    browser = FakeBrowser()
    recorder = SessionRecorder(browser.send)
    await recorder.attach(target_id="t1", session_id="s1", target_type="page")
    browser.push("s1", meta(10), full(11))
    await recorder.drain()
    browser.push("s1", incremental(30), incremental(20))
    await recorder.drain()
    await recorder.stop()

    recording = recorder.recording()
    assert [e["timestamp"] for e in recording.events] == [10, 11, 20, 30]
    assert recording.has_full_snapshot
    assert recording.raw_bytes > 0
    assert not recording.truncated


@pytest.mark.asyncio
async def test_a_dead_target_does_not_stop_the_others():
    browser = FakeBrowser()
    recorder = SessionRecorder(browser.send)
    await recorder.attach(target_id="t1", session_id="s1", target_type="page")
    await recorder.attach(target_id="t2", session_id="s2", target_type="page")
    browser.push("s1", meta(1), full(2))
    browser.fail_sessions.add("s2")
    await recorder.drain()
    assert len(recorder.recording().events) == 2


@pytest.mark.asyncio
async def test_periodic_drain_runs_until_stop():
    browser = FakeBrowser()
    recorder = SessionRecorder(browser.send, drain_interval=0.01)
    await recorder.attach(target_id="t1", session_id="s1", target_type="page")
    recorder.start()
    browser.push("s1", meta(1), full(2))
    await asyncio.sleep(0.1)
    assert len(recorder._events) == 2  # drained without an explicit call
    await recorder.stop()
    drains = sum(1 for _, p, _ in browser.calls if p.get("expression") == session_recording.DRAIN_EXPRESSION)
    await asyncio.sleep(0.05)
    assert drains == sum(
        1 for _, p, _ in browser.calls if p.get("expression") == session_recording.DRAIN_EXPRESSION
    )


@pytest.mark.asyncio
async def test_size_cap_stops_collecting_and_marks_truncated():
    browser = FakeBrowser()
    recorder = SessionRecorder(browser.send, max_raw_bytes=400)
    await recorder.attach(target_id="t1", session_id="s1", target_type="page")
    browser.push("s1", meta(1), full(2))
    await recorder.drain()
    browser.push("s1", *(incremental(10 + i, "x" * 50) for i in range(10)))
    await recorder.drain()
    recording = recorder.recording()
    assert recording.truncated
    assert len(recording.events) == 2
    assert recording.raw_bytes <= 400


@pytest.mark.asyncio
async def test_focus_switch_follows_the_new_tab_from_its_snapshot():
    browser = FakeBrowser()
    recorder = SessionRecorder(browser.send)
    await recorder.attach(target_id="a", session_id="sa", target_type="page")
    await recorder.set_focus("a")
    browser.push("sa", meta(1), full(2), incremental(3))
    await recorder.drain()
    await recorder.attach(target_id="b", session_id="sb", target_type="page")
    await recorder.set_focus("b")
    switched = recorder._focus_changes[-1][0]
    assert any(p.get("expression") == session_recording.SNAPSHOT_EXPRESSION for _, p, s in browser.calls if s == "sb")
    browser.push("sb", meta(switched + 1), full(switched + 2), incremental(switched + 3))
    browser.push("sa", incremental(switched + 4))  # background tab: not playable in this stream
    await recorder.drain()

    events = recorder.recording().events
    assert [e["timestamp"] for e in events] == [1, 2, 3, switched + 1, switched + 2, switched + 3]


def recording(events: Optional[list[dict]] = None, secrets: tuple[str, ...] = ()) -> Recording:
    now = datetime.now(timezone.utc)
    return Recording(
        events=events if events is not None else [meta(1), full(2), incremental(3, SECRET)],
        started_at=now,
        ended_at=now,
        raw_bytes=10,
        truncated=False,
        secrets=secrets,
    )


@pytest.mark.asyncio
async def test_upload_posts_the_contract_body_gzipped():
    with recording_stub_server() as stub:
        upload = await upload_recording(
            recording(secrets=(SECRET,)),
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            session_id="sess-1",
            framework="browser-use",
            base_url=stub.base_url,
            api_key="key-1",
        )
    assert upload is not None
    assert upload.view_url == stub.view_url(upload.recording_id)
    [sent] = stub.uploads
    assert sent.path == "/api/browser-sessions/recordings/"
    assert sent.headers["Authorization"] == "Bearer key-1"
    assert sent.headers["Content-Encoding"] == "gzip"
    assert sent.headers["Content-Type"] == "application/json"
    body = sent.body
    assert (body["trace_id"], body["span_id"], body["session_id"], body["framework"]) == (
        TRACE_ID,
        SPAN_ID,
        "sess-1",
        "browser-use",
    )
    assert body["started_at"].endswith("Z") and body["ended_at"].endswith("Z")
    assert [e["type"] for e in body["events"]] == [RRWEB_META, RRWEB_FULL_SNAPSHOT, 3]
    # The backstop scrub: a secret rrweb could not mask never leaves the process.
    assert SECRET not in json.dumps(body)


@pytest.mark.asyncio
async def test_upload_uses_the_running_respan_client_endpoint(monkeypatch):
    with recording_stub_server() as stub:
        tracer = type("Tracer", (), {"api_endpoint": stub.base_url, "api_key": "from-client"})()
        monkeypatch.setattr(session_recording.RespanTracer, "_instance", tracer)
        upload = await upload_recording(
            recording(), trace_id=TRACE_ID, span_id=SPAN_ID, session_id=None, framework="x"
        )
    assert upload is not None
    assert stub.uploads[0].headers["Authorization"] == "Bearer from-client"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 413, 500])
async def test_upload_failure_never_raises_and_keeps_a_local_player(tmp_path, monkeypatch, status, caplog):
    monkeypatch.setattr(session_recording.tempfile, "gettempdir", lambda: str(tmp_path))
    with recording_stub_server(status=status) as stub:
        upload = await upload_recording(
            recording(),
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            session_id="s",
            framework="x",
            base_url=stub.base_url,
            api_key="k",
        )
    assert upload is None
    folder = tmp_path / session_recording.LOCAL_REPLAY_DIR
    saved = json.loads(gzip.decompress((folder / f"{TRACE_ID}.json.gz").read_bytes()))
    assert saved["trace_id"] == TRACE_ID and len(saved["events"]) == 3
    html = (folder / f"{TRACE_ID}.html").read_text()
    assert f"rrweb-player@{session_recording.RRWEB_PLAYER_VERSION}" in html
    assert '"type":2' in html
    assert str(folder / f"{TRACE_ID}.html") in caplog.text


@pytest.mark.asyncio
async def test_unreachable_endpoint_and_missing_key_never_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(session_recording.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(session_recording.RespanTracer, "_instance", None)
    monkeypatch.delenv("RESPAN_API_KEY", raising=False)
    for kwargs in (
        {"base_url": "http://127.0.0.1:9/api", "api_key": "k", "timeout": 2},
        {"base_url": "http://127.0.0.1:9/api"},
    ):
        assert (
            await upload_recording(
                recording(), trace_id=TRACE_ID, span_id=SPAN_ID, session_id="s", framework="x", **kwargs
            )
            is None
        )
    assert (tmp_path / session_recording.LOCAL_REPLAY_DIR / f"{TRACE_ID}.html").exists()


@pytest.mark.asyncio
async def test_nothing_recorded_uploads_nothing():
    with recording_stub_server() as stub:
        assert (
            await upload_recording(
                recording(events=[]),
                trace_id=TRACE_ID,
                span_id=SPAN_ID,
                session_id="s",
                framework="x",
                base_url=stub.base_url,
                api_key="k",
            )
            is None
        )
    assert stub.uploads == []


def test_replay_link_is_flat_respan_metadata():
    span = TracerProvider().get_tracer("t").start_span("agent")
    set_replay_metadata(span, session_recording.RecordingUpload("rid", "https://view/rid"))
    assert span.attributes["respan.metadata.session_replay_url"] == "https://view/rid"
    assert span.attributes["respan.metadata.session_replay_id"] == "rid"
    span.end()


@pytest.mark.asyncio
async def test_reserve_then_upload_under_the_reserved_id_with_markers():
    with recording_stub_server() as stub:
        reserved = await reserve_recording(
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            session_id="s",
            framework="x",
            base_url=stub.base_url,
            api_key="k",
        )
        assert reserved is not None
        markers = [{"t_ms": 0, "label": "agent", "span_id": SPAN_ID, "kind": "agent"}]
        upload = await upload_recording(
            recording(),
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            session_id="s",
            framework="x",
            recording_id=reserved.recording_id,
            markers=markers,
            base_url=stub.base_url,
            api_key="k",
        )
    [sent] = stub.reservations
    assert sent.path == "/api/browser-sessions/recordings/reserve/"
    assert sent.headers["Authorization"] == "Bearer k"
    assert (sent.body["trace_id"], sent.body["span_id"]) == (TRACE_ID, SPAN_ID)
    assert upload is not None and upload.recording_id == reserved.recording_id
    body = stub.uploads[0].body
    assert body["recording_id"] == reserved.recording_id
    assert body["markers"] == markers


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 500])
async def test_reserve_failure_returns_none(status):
    with recording_stub_server(reserve_status=status) as stub:
        reserved = await reserve_recording(
            trace_id=TRACE_ID, span_id=SPAN_ID, session_id="s", framework="x",
            base_url=stub.base_url, api_key="k",
        )
    assert reserved is None
    assert await reserve_recording(
        trace_id=TRACE_ID, span_id=SPAN_ID, session_id="s", framework="x",
        base_url="http://127.0.0.1:9/api", api_key="k", timeout=2,
    ) is None


def test_replay_links_stamp_every_span_with_its_offset():
    tracer = TracerProvider().get_tracer("t")
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    links = ReplayLinks(started)
    span = tracer.start_span("step")
    assert links.stamp(span, kind="step", label="step 1") is None  # nothing reserved
    assert "respan.metadata.session_replay_url" not in span.attributes
    links.reservation = session_recording.RecordingUpload("rid", "https://view/rid.html?sig=1")
    marker = links.stamp(span, kind="step", label="step 1", t_ms=1250)
    assert span.attributes["respan.metadata.session_replay_url"] == "https://view/rid.html?sig=1#t=1250"
    assert span.attributes["respan.metadata.session_replay_id"] == "rid"
    assert marker == {
        "t_ms": 1250, "label": "step 1", "span_id": format(span.get_span_context().span_id, "016x"),
        "kind": "step",
    }
    assert links.offset_ms(datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)) == 2000
    links.stamp(span, kind="tool", label="x" * 500, t_ms=1)
    assert len(links.markers[-1]["label"]) == session_recording.MAX_MARKER_LABEL_CHARS
    span.end()


def test_player_html_seeks_to_the_hash_and_lists_markers():
    html = session_recording.player_html(
        {
            "trace_id": "t",
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:00:10Z",
            "events": [meta(1767225600500), full(1767225600501)],
            "markers": [{"t_ms": 2000, "label": "</script>step 1", "span_id": "a", "kind": "step"}],
        }
    )
    assert '"started_at_ms":1767225600000' in html
    assert "step 1" in html and "</script>step 1" not in html
    assert 'location.hash' in html and "player.goto(" in html


def test_player_html_cannot_be_broken_out_of_by_page_content():
    html = session_recording.player_html(
        {"trace_id": "t", "events": [incremental(1, "</script><script>alert(1)</script>")]}
    )
    assert "</script><script>alert(1)" not in html


@pytest.mark.parametrize(
    ("explicit", "env", "expected"),
    [(None, None, False), (None, "1", True), (None, "0", False), (False, "1", False), (True, None, True)],
)
def test_record_session_opt_in(monkeypatch, explicit, env, expected):
    if env is None:
        monkeypatch.delenv(session_recording.RECORD_SESSION_ENV, raising=False)
    else:
        monkeypatch.setenv(session_recording.RECORD_SESSION_ENV, env)
    assert is_session_recording_requested(explicit) is expected


def test_recorder_script_embeds_the_vendored_rrweb_with_its_license():
    assert "MIT License" in recorder_script()
    assert "rrweb record 2.0.0-alpha.4" in recorder_script()
