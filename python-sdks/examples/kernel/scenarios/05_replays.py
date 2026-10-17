"""Kernel's Replays API: record a browser session as a video. The replay id and
view link land on the replay span and on the action's root span (Respan's interim
for session replay).

    python scenarios/05_replays.py           # KERNEL_FAKE=1 for the local fake Kernel
"""

import kernel
from _harness import TARGET_URL, Expect, Scenario, main
from kernel import Kernel

client = Kernel()

app = kernel.App("python-replays")


@app.action("record")
async def record(ctx: kernel.KernelContext, payload: dict) -> dict:
    kernel_browser = client.browsers.create(
        invocation_id=ctx.invocation_id, start_url=payload["url"]
    )
    session = kernel_browser.session_id
    try:
        replay = client.browsers.replays.start(session)
        client.browsers.computer.scroll(session, x=400, y=300, delta_y=3)
        client.browsers.replays.stop(replay.replay_id, id_or_name=session)
        replays = client.browsers.replays.list(session)
        return {
            "replay_id": replay.replay_id,
            "replays": len(replays),
            "view": replay.replay_view_url,
        }
    finally:
        client.browsers.delete_by_id(session)


SCENARIO = Scenario(
    name="replays",
    title="Replays: start, stop, list",
    source="kernel.sh/docs/browsers/replays",
    look_for="replays.start span and the action root carry kernel.replay.id and kernel.replay.view_url",
    app=app,
    action="record",
    payload={"url": TARGET_URL},
    expect=Expect(calls_include=("replays.start", "replays.stop", "replays.list")),
)

if __name__ == "__main__":
    main(SCENARIO)
