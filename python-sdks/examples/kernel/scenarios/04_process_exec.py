"""Kernel's Process API: run commands in the browser's VM. Output is decoded
from base64; environment values are masked.

    python scenarios/04_process_exec.py      # KERNEL_FAKE=1 for the local fake Kernel
"""

import base64

import kernel
from _harness import Expect, Scenario, main
from kernel import Kernel

client = Kernel()

app = kernel.App("python-process")

DEPLOY_TOKEN = "tok_live_51f0c9e2a7"


@app.action("inspect-vm")
async def inspect_vm(ctx: kernel.KernelContext) -> dict:
    kernel_browser = client.browsers.create(invocation_id=ctx.invocation_id)
    session = kernel_browser.session_id
    process = client.browsers.process
    try:
        uname = process.exec(session, command="uname", args=["-s"])
        process.exec(
            session,
            command="sh",
            args=["-c", 'echo "token is $DEPLOY_TOKEN"; echo "warning" >&2'],
            env={"DEPLOY_TOKEN": DEPLOY_TOKEN},
        )
        spawned = process.spawn(session, command="cat")
        process.stdin(
            spawned.process_id, id_or_name=session, data_b64=base64.b64encode(b"hello\n").decode()
        )
        for event in process.stdout_stream(spawned.process_id, id_or_name=session):
            if event.event == "exit":
                break
        status = process.status(spawned.process_id, id_or_name=session)
        return {
            "os": base64.b64decode(uname.stdout_b64 or "").decode().strip(),
            "cat": status.state,
        }
    finally:
        client.browsers.delete_by_id(session)


SCENARIO = Scenario(
    name="process_exec",
    title="Process API: exec, env, spawn + stdin + stdout",
    source="kernel.sh/docs/browsers/process",
    look_for="process.exec spans with decoded stdout/stderr and exit code; env shown as <masked>; the token as <secret>DEPLOY_TOKEN</secret>",
    app=app,
    action="inspect-vm",
    expect=Expect(
        calls_include=(
            "process.exec",
            "process.spawn",
            "process.stdin",
            "process.stdout_stream",
            "process.status",
        ),
        forbidden=(DEPLOY_TOKEN,),
    ),
)

if __name__ == "__main__":
    main(SCENARIO)
