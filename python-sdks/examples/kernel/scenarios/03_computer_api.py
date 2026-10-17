"""Kernel's Computer Controls API: screenshots, mouse and keyboard at the OS level,
the primitives every computer-use template (Anthropic / OpenAI / Gemini) drives.
The password is declared with `secrets(...)`, so it is exported as
<secret>password</secret>.

    python scenarios/03_computer_api.py      # KERNEL_FAKE=1 for the local fake Kernel
"""

from urllib.parse import quote

import kernel
from _harness import Expect, Scenario, main
from kernel import Kernel
from respan_instrumentation_kernel import secrets

client = Kernel()

app = kernel.App("python-computer")

PASSWORD = "correct-horse-battery-7"
LOGIN_FORM = (
    "<title>Sign in</title><style>input{position:absolute;left:40px;width:320px;height:32px}"
    "</style><input id=user style='top:40px'><input id=password type=password style='top:100px'>"
)


@app.action("sign-in")
async def sign_in(ctx: kernel.KernelContext) -> dict:
    kernel_browser = client.browsers.create(
        invocation_id=ctx.invocation_id,
        start_url="data:text/html," + quote(LOGIN_FORM),
        viewport={"width": 1024, "height": 768},
    )
    session = kernel_browser.session_id
    computer = client.browsers.computer
    try:
        computer.capture_screenshot(session)  # what a computer-use model sees first
        computer.click_mouse(session, x=120, y=56)
        computer.type_text(session, text="ada.lovelace")
        computer.click_mouse(session, x=120, y=116)
        with secrets({"password": PASSWORD}):
            computer.type_text(session, text=PASSWORD)
        computer.batch(
            session,
            actions=[
                {"type": "press_key", "press_key": {"keys": ["Tab"]}},
                {"type": "move_mouse", "move_mouse": {"x": 500, "y": 400}},
            ],
        )
        position = computer.get_mouse_position(session)
        computer.capture_screenshot(session)
        return {"mouse": position.model_dump()}
    finally:
        client.browsers.delete_by_id(session)


SCENARIO = Scenario(
    name="computer_api",
    title="Computer Controls: screenshot, click, type, batch",
    source="kernel.sh/docs/browsers/computer-controls",
    look_for="computer.capture_screenshot spans with a JPEG; click/type spans; the password typed as <secret>password</secret>",
    app=app,
    action="sign-in",
    expect=Expect(
        calls_include=("computer.click_mouse", "computer.type_text", "computer.batch"),
        screenshots=("computer.capture_screenshot",),
        forbidden=(PASSWORD,),
    ),
)

if __name__ == "__main__":
    main(SCENARIO)
