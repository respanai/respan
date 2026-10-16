"""An LLM drives Playwright through tool calls: the "agent over Playwright"
pattern. The model is called with OpenAI-compatible chat through the Respan
gateway (one key for the model and the traces).

    python scenarios/14_llm_agent.py
"""

import json

from openai import OpenAI
from playwright.sync_api import sync_playwright
from respan import agent

from _harness import API_KEY, BASE_URL, MODEL, SITE, Expect, Scenario, launch_options, main

TASK = (
    f"Open {SITE}, go to the sign-up form, sign up with email llm@example.com, "
    "the Pro plan and the terms accepted (leave the password empty), "
    "then tell me the heading of the page you land on."
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in params},
                "required": list(params),
            },
        },
    }
    for name, description, params in [
        ("goto", "Open a URL", ["url"]),
        ("read_page", "Visible text and form fields of the page", []),
        ("click", "Click the element with this visible text", ["text"]),
        ("fill", "Fill the input with this label", ["label", "value"]),
        ("select", "Select an option of the dropdown with this label", ["label", "value"]),
        ("check", "Tick the checkbox with this label", ["label"]),
    ]
]


def act(page, name: str, args: dict) -> str:
    if name == "goto":
        page.goto(args["url"])
    elif name == "click":
        page.get_by_text(args["text"], exact=False).first.click()
    elif name == "fill":
        page.get_by_label(args["label"]).fill(args["value"])
    elif name == "select":
        page.get_by_label(args["label"]).select_option(args["value"])
    elif name == "check":
        page.get_by_label(args["label"]).check()
    fields = page.eval_on_selector_all(
        "input, select, button", "els => els.map(e => e.name || e.textContent).filter(Boolean)"
    )
    return json.dumps({"url": page.url, "text": page.inner_text("body")[:1500], "fields": fields})


@agent(name="playwright_agent")
def run_agent(page, llm: OpenAI) -> str:
    messages = [
        {"role": "system", "content": "You control a browser with the given tools. Be brief."},
        {"role": "user", "content": TASK},
    ]
    for _ in range(12):
        reply = llm.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        message = reply.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content or ""
        for call in message.tool_calls:
            try:
                result = act(page, call.function.name, json.loads(call.function.arguments or "{}"))
            except Exception as error:  # the model sees the failure and retries
                result = f"{type(error).__name__}: {error}"[:500]
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "gave up"


def run():
    llm = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_options())
        page = browser.new_page()
        print(run_agent(page, llm))
        browser.close()


SCENARIO = Scenario(
    name="llm_agent",
    title="LLM tool-calling agent over Playwright",
    source="LLM-in-the-loop (OpenAI tool calls -> Playwright actions)",
    look_for="agent span: LLM calls (tokens, cost) interleaved with the Playwright calls each tool call made",
    run=run,
    expect=Expect(calls_include=("Page.goto", "Locator.fill", "Locator.click"), has_llm=True),
)

if __name__ == "__main__":
    main(SCENARIO)
