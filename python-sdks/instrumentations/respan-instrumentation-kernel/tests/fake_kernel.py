"""A local stand-in for the Kernel API, backed by a real headless Chrome.

Kernel browsers are cloud VMs, so nothing here can run against Kernel itself
without an account. This serves the same HTTP routes the ``kernel`` client calls
(point it here with ``Kernel(base_url=...)`` or ``KERNEL_BASE_URL``) and does
the work on a local Chrome:

* ``browsers.create`` returns that Chrome's CDP URL (with a ``?jwt=`` token, as
  Kernel's does), so Playwright / browser-use connect to a real browser, and
  opens ``start_url``;
* ``computer.*`` drives the focused tab over CDP (real screenshots, real
  clicks and typing);
* ``process.exec`` / ``spawn`` run the command locally;
* ``playwright.execute`` runs the code in the page with a small ``page`` shim
  (``title``, ``url``, ``evaluate``, ``content``);
* ``replays.*`` return realistic, canned replay records.

Requests are recorded on ``FakeKernel.requests`` for assertions.
"""

from __future__ import annotations

import base64
import itertools
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

JWT = "eyJhbGciOiJIUzI1NiJ9.fake-browser-session.c2lnbmF0dXJl"
MISSING_SESSION_MESSAGE = "browser session {} not found"

_PAGE_SHIM = """
const page = {
  title: async () => document.title,
  url: () => location.href,
  content: async () => document.documentElement.outerHTML,
  evaluate: async (fn, arg) => (typeof fn === 'function' ? fn(arg) : eval(fn)),
};
"""


def chrome_path() -> Optional[str]:
    if env := os.environ.get("CHROME_PATH"):
        return env
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac.exists():
        return str(mac)
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        if found := shutil.which(name):
            return found
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - local Chrome
        return json.loads(response.read())


class _CDP:
    """A minimal blocking CDP client over one WebSocket (stdlib only)."""

    def __init__(self, ws_url: str) -> None:
        parsed = urlparse(ws_url)
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            (
                f"GET {parsed.path} HTTP/1.1\r\nHost: {parsed.hostname}:{parsed.port}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        header = b""
        while b"\r\n\r\n" not in header:
            header += self.sock.recv(1)
        if b" 101 " not in header.split(b"\r\n", 1)[0]:
            raise ConnectionError(header.decode(errors="replace"))
        self.ids = itertools.count(1)

    def _recv_exact(self, size: int) -> bytes:
        data = b""
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("CDP socket closed")
            data += chunk
        return data

    def _recv_message(self) -> str:
        payload = b""
        while True:
            first, second = self._recv_exact(2)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._recv_exact(8))[0]
            payload += self._recv_exact(length)
            if first & 0x80:
                return payload.decode()

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        message_id = next(self.ids)
        data = json.dumps({"id": message_id, "method": method, "params": params}).encode()
        mask = os.urandom(4)
        header = bytes([0x81])
        if len(data) < 126:
            header += bytes([0x80 | len(data)])
        elif len(data) < 65536:
            header += bytes([0x80 | 126]) + struct.pack(">H", len(data))
        else:
            header += bytes([0x80 | 127]) + struct.pack(">Q", len(data))
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(header + mask + masked)
        while True:
            message = json.loads(self._recv_message())
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(message["error"].get("message"))
                return message.get("result", {})

    def close(self) -> None:
        self.sock.close()


class LocalChrome:
    """A headless Chrome with a DevTools port and a throwaway profile."""

    def __init__(self) -> None:
        executable = chrome_path()
        if executable is None:
            raise RuntimeError("needs a local Chrome/Chromium (or CHROME_PATH)")
        self.port = _free_port()
        self.profile = tempfile.mkdtemp(prefix="respan-kernel-chrome-")
        self.process = subprocess.Popen(  # noqa: S603 - test browser
            [
                executable,
                "--headless=new",
                f"--remote-debugging-port={self.port}",
                f"--user-data-dir={self.profile}",
                "--no-first-run",
                "--no-default-browser-check",
                "--window-size=1024,768",
                # CI runners (Ubuntu 24.04) block the namespaces Chrome's sandbox needs.
                *(
                    ["--no-sandbox", "--disable-dev-shm-usage"]
                    if sys.platform.startswith("linux")
                    else []
                ),
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 30
        while True:
            try:
                self.ws_url = _get_json(f"http://127.0.0.1:{self.port}/json/version")[
                    "webSocketDebuggerUrl"
                ]
                break
            except OSError:
                if time.monotonic() > deadline:
                    self.close()
                    raise
                time.sleep(0.2)

    def page(self) -> _CDP:
        """The focused tab: the most recently active page target."""
        targets = _get_json(f"http://127.0.0.1:{self.port}/json/list")
        pages = [t for t in targets if t.get("type") == "page"]
        if not pages:
            raise RuntimeError("no page in the browser")
        return _CDP(pages[0]["webSocketDebuggerUrl"])

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        shutil.rmtree(self.profile, ignore_errors=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_KEYS = {
    "Return": ("Enter", "Enter", 13, "\r"),
    "Enter": ("Enter", "Enter", 13, "\r"),
    "Tab": ("Tab", "Tab", 9, None),
}


class FakeKernel:
    """The fake Kernel API. Use as a context manager; ``base_url`` goes to ``Kernel(...)``."""

    def __init__(self, chrome: Optional[LocalChrome] = None) -> None:
        self.chrome = chrome or LocalChrome()
        self._owns_chrome = chrome is None
        self.requests: list[tuple[str, str, Any]] = []
        self.api_keys: set[str] = set()
        self.browsers: dict[str, dict[str, Any]] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self.replays: dict[str, list[dict[str, Any]]] = {}
        # Make playwright.execute answer 500 (a Kernel outage, for fail-closed tests).
        self.execute_fails = False
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _serve(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else {}
                auth = self.headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    fake.api_keys.add(auth[len("Bearer ") :])
                path = urlparse(self.path).path
                with fake._lock:
                    fake.requests.append((method, path, body))
                try:
                    status, payload, content_type = fake.route(method, path, body)
                except Exception as error:  # noqa: BLE001 - surface as a 500
                    status, payload, content_type = (
                        500,
                        {"code": "internal", "message": str(error)},
                        None,
                    )
                if content_type == "text/event-stream":
                    data = "".join(f"data: {json.dumps(event)}\n\n" for event in payload).encode()
                elif isinstance(payload, bytes):
                    data = payload
                elif payload is None:
                    data = b""
                else:
                    data = json.dumps(payload).encode()
                    content_type = content_type or "application/json"
                self.send_response(status)
                if content_type:
                    self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._serve("GET")

            def do_POST(self) -> None:
                self._serve("POST")

            def do_PATCH(self) -> None:
                self._serve("PATCH")

            def do_DELETE(self) -> None:
                self._serve("DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def __enter__(self) -> FakeKernel:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        for process in self.processes.values():
            if process.poll() is None:
                process.kill()
        if self._owns_chrome:
            self.chrome.close()

    def paths(self) -> list[str]:
        return [path for _, path, _ in self.requests]

    # -- routing -------------------------------------------------------------

    def route(self, method: str, path: str, body: Any) -> tuple[int, Any, Optional[str]]:
        parts = [p for p in path.split("/") if p]
        if parts[:1] != ["browsers"]:
            return 404, {"code": "not_found", "message": f"no route {path}"}, None
        if len(parts) == 1:
            return self._browsers(method, body)
        session_id, rest = parts[1], parts[2:]
        if session_id not in self.browsers:
            return (
                404,
                {"code": "not_found", "message": MISSING_SESSION_MESSAGE.format(session_id)},
                None,
            )
        if not rest:
            if method == "DELETE":
                del self.browsers[session_id]
                return 200, None, None
            if method == "PATCH":
                self.browsers[session_id].update({k: v for k, v in body.items() if k == "name"})
            return 200, self.browsers[session_id], None
        group = rest[0]
        if group == "computer":
            return self._computer("/".join(rest[1:]), body)
        if group == "process":
            return self._process(rest[1:], body)
        if group == "playwright" and rest[1:] == ["execute"]:
            if self.execute_fails:
                return 500, {"code": "internal", "message": "executor unavailable"}, None
            return 200, self._execute(body.get("code", "")), None
        if group == "replays":
            return self._replays(session_id, method, rest[1:])
        return 404, {"code": "not_found", "message": f"no route {path}"}, None

    def _browsers(self, method: str, body: Any) -> tuple[int, Any, Optional[str]]:
        if method == "GET":
            return 200, list(self.browsers.values()), None
        session_id = uuid.uuid4().hex[:24]
        browser = {
            "session_id": session_id,
            "cdp_ws_url": f"{self.chrome.ws_url}?jwt={JWT}",
            "webdriver_ws_url": f"ws://127.0.0.1:{self.chrome.port}/webdriver?jwt={JWT}",
            "browser_live_view_url": f"https://live.example.test/browser/{session_id}?jwt={JWT}",
            "created_at": _now(),
            "headless": bool(body.get("headless", False)),
            "stealth": bool(body.get("stealth", False)),
            "region": body.get("region", "us-east"),
            "timeout_seconds": body.get("timeout_seconds", 60),
            "memory": {"total_mb": 4096},
            "viewport": {"width": 1024, "height": 768},
        }
        if invocation_id := body.get("invocation_id"):
            browser["invocation_id"] = invocation_id
        if start_url := body.get("start_url"):
            page = self.chrome.page()
            try:
                page.call("Page.navigate", url=start_url)
                time.sleep(0.5)  # let the document load before the first screenshot
            finally:
                page.close()
        self.browsers[session_id] = browser
        return 200, browser, None

    # -- computer ------------------------------------------------------------

    def _computer(self, action: str, body: Any) -> tuple[int, Any, Optional[str]]:
        if action == "screenshot":
            page = self.chrome.page()
            try:
                data = page.call("Page.captureScreenshot", format="png")["data"]
            finally:
                page.close()
            return 200, base64.b64decode(data), "image/png"
        if action == "get_mouse_position":
            return 200, {"x": 0, "y": 0}, None
        if action == "clipboard/read":
            return 200, {"text": ""}, None
        if action == "cursor":
            return 200, {"ok": True}, None
        if action == "batch":
            actions = body.get("actions") or []
        else:
            kind = "type_text" if action == "type" else action
            actions = [{"type": kind, kind: body}]
        page = self.chrome.page()
        try:
            for item in actions:
                kind = item.get("type")
                self._input(page, kind, item.get(kind) or {})
        finally:
            page.close()
        return 200, None, None

    def _input(self, page: _CDP, kind: str, params: dict[str, Any]) -> None:
        if kind == "click_mouse":
            x, y = params.get("x", 0), params.get("y", 0)
            for event in ("mousePressed", "mouseReleased"):
                page.call(
                    "Input.dispatchMouseEvent",
                    type=event,
                    x=x,
                    y=y,
                    button=params.get("button", "left"),
                    clickCount=params.get("num_clicks", 1),
                )
        elif kind == "move_mouse":
            page.call(
                "Input.dispatchMouseEvent",
                type="mouseMoved",
                x=params.get("x", 0),
                y=params.get("y", 0),
            )
        elif kind == "type_text":
            page.call("Input.insertText", text=params.get("text", ""))
        elif kind == "press_key":
            for key in params.get("keys", []):
                name, code, vk, text = _KEYS.get(key, (key, key, 0, None))
                down = {"type": "keyDown", "key": name, "code": code, "windowsVirtualKeyCode": vk}
                if text:
                    down["text"] = text
                page.call("Input.dispatchKeyEvent", **down)
                page.call(
                    "Input.dispatchKeyEvent",
                    type="keyUp",
                    key=name,
                    code=code,
                    windowsVirtualKeyCode=vk,
                )
        elif kind == "scroll":
            page.call(
                "Input.dispatchMouseEvent",
                type="mouseWheel",
                x=params.get("x", 0),
                y=params.get("y", 0),
                deltaX=params.get("delta_x", 0),
                deltaY=params.get("delta_y", 0),
            )
        # drag_mouse, set_cursor, sleep: accepted, no visible effect needed here.

    # -- process -------------------------------------------------------------

    def _process(self, rest: list[str], body: Any) -> tuple[int, Any, Optional[str]]:
        if rest == ["exec"]:
            started = time.monotonic()
            done = subprocess.run(  # noqa: S603 - commands come from the tests/examples
                [body["command"], *(body.get("args") or [])],
                capture_output=True,
                env={**os.environ, **(body.get("env") or {})},
                cwd=body.get("cwd"),
                timeout=body.get("timeout_sec") or 30,
                check=False,
            )
            return (
                200,
                {
                    "exit_code": done.returncode,
                    "stdout_b64": base64.b64encode(done.stdout).decode(),
                    "stderr_b64": base64.b64encode(done.stderr).decode(),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
                None,
            )
        if rest == ["spawn"]:
            process = subprocess.Popen(  # noqa: S603 - commands come from the tests/examples
                [body["command"], *(body.get("args") or [])],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, **(body.get("env") or {})},
            )
            process_id = uuid.uuid4().hex[:12]
            self.processes[process_id] = process
            return 200, {"process_id": process_id, "pid": process.pid, "started_at": _now()}, None
        process_id, action = rest[0], "/".join(rest[1:])
        process = self.processes.get(process_id)
        if process is None:
            return 404, {"code": "not_found", "message": f"process {process_id} not found"}, None
        if action == "status":
            code = process.poll()
            return 200, {"state": "running" if code is None else "exited", "exit_code": code}, None
        if action == "stdin":
            data = base64.b64decode(body.get("data_b64", ""))
            process.stdin.write(data)
            process.stdin.flush()
            return 200, {"written_bytes": len(data)}, None
        if action == "kill":
            process.kill()
            return 200, {"ok": True}, None
        if action == "stdout/stream":
            out, err = process.communicate(timeout=30)
            events = [
                {"stream": "stdout", "data_b64": base64.b64encode(out).decode()},
                {"stream": "stderr", "data_b64": base64.b64encode(err).decode()},
                {"event": "exit", "exit_code": process.returncode},
            ]
            return 200, events, "text/event-stream"
        return 404, {"code": "not_found", "message": f"no process route {action}"}, None

    # -- playwright.execute --------------------------------------------------

    def _execute(self, code: str) -> dict[str, Any]:
        page = self.chrome.page()
        try:
            expression = f"(async () => {{ {_PAGE_SHIM} {code} }})()"
            result = page.call(
                "Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True
            )
            if details := result.get("exceptionDetails"):
                description = (details.get("exception") or {}).get("description") or details.get(
                    "text"
                )
                return {"success": False, "error": description}
            return {"success": True, "result": result.get("result", {}).get("value")}
        finally:
            page.close()

    # -- replays -------------------------------------------------------------

    def _replays(
        self, session_id: str, method: str, rest: list[str]
    ) -> tuple[int, Any, Optional[str]]:
        replays = self.replays.setdefault(session_id, [])
        if not rest:
            if method == "GET":
                return 200, replays, None
            replay_id = uuid.uuid4().hex[:16]
            replay = {
                "replay_id": replay_id,
                "started_at": _now(),
                "replay_view_url": f"https://replays.example.test/{session_id}/{replay_id}?jwt={JWT}",
            }
            replays.append(replay)
            return 200, replay, None
        replay = next((r for r in replays if r["replay_id"] == rest[0]), None)
        if replay is None:
            return 404, {"code": "not_found", "message": f"replay {rest[0]} not found"}, None
        if rest[1:] == ["stop"]:
            replay["finished_at"] = _now()
            return 200, None, None
        return 200, b"\x00\x00\x00\x18ftypmp42fake-replay-video", "video/mp4"


def jwt_of(url: str) -> Optional[str]:
    values = parse_qs(urlparse(url).query).get("jwt")
    return values[0] if values else None
