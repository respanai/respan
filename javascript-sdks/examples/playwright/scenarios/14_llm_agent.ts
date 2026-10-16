/**
 * An LLM drives Playwright through tool calls: the "agent over Playwright"
 * pattern. The model is called with OpenAI-compatible chat through the Respan
 * gateway (one key for the model and the traces).
 */
import OpenAI from "openai";
import { chromium, type Page } from "playwright";
import { withAgent } from "@respan/tracing";
import { API_KEY, BASE_URL, MODEL, launchOptions, type Scenario } from "../harness.js";

const TOOLS: Array<[string, string, string[]]> = [
  ["goto", "Open a URL", ["url"]],
  ["read_page", "Visible text and form fields of the page", []],
  ["click", "Click the element with this visible text", ["text"]],
  ["fill", "Fill the input with this label", ["label", "value"]],
  ["select", "Select an option of the dropdown with this label", ["label", "value"]],
  ["check", "Tick the checkbox with this label", ["label"]],
];

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = TOOLS.map(([name, description, params]) => ({
  type: "function",
  function: {
    name,
    description,
    parameters: {
      type: "object",
      properties: Object.fromEntries(params.map((key) => [key, { type: "string" }])),
      required: params,
    },
  },
}));

async function act(page: Page, name: string, args: Record<string, string>): Promise<string> {
  if (name === "goto") await page.goto(args.url);
  else if (name === "click") await page.getByText(args.text, { exact: false }).first().click();
  else if (name === "fill") await page.getByLabel(args.label).fill(args.value);
  else if (name === "select") await page.getByLabel(args.label).selectOption(args.value);
  else if (name === "check") await page.getByLabel(args.label).check();
  const fields = await page.$$eval("input, select, button", (els) =>
    els.map((e) => (e as HTMLInputElement).name || e.textContent).filter(Boolean),
  );
  return JSON.stringify({ url: page.url(), text: (await page.innerText("body")).slice(0, 1500), fields });
}

async function runAgent(page: Page, llm: OpenAI, task: string): Promise<string> {
  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: "system", content: "You control a browser with the given tools. Be brief." },
    { role: "user", content: task },
  ];
  for (let turn = 0; turn < 12; turn++) {
    const reply = await llm.chat.completions.create({ model: MODEL, messages, tools });
    const message = reply.choices[0].message;
    messages.push(message);
    if (!message.tool_calls?.length) return message.content ?? "";
    for (const call of message.tool_calls) {
      if (call.type !== "function") continue;
      let result: string;
      try {
        result = await act(page, call.function.name, JSON.parse(call.function.arguments || "{}"));
      } catch (error) {
        result = `${(error as Error).name}: ${(error as Error).message}`.slice(0, 500); // the model sees it and retries
      }
      messages.push({ role: "tool", tool_call_id: call.id, content: result });
    }
  }
  return "gave up";
}

export const scenario: Scenario = {
  name: "llm_agent",
  title: "LLM tool-calling agent over Playwright",
  source: "LLM-in-the-loop (OpenAI tool calls -> Playwright actions)",
  lookFor: "agent span: LLM calls (tokens, cost) interleaved with the Playwright calls each tool call made",
  async run(site) {
    const task =
      `Open ${site}, go to the sign-up form, sign up with email llm@example.com, ` +
      "the Pro plan and the terms accepted (leave the password empty), " +
      "then tell me the heading of the page you land on.";
    const llm = new OpenAI({ apiKey: API_KEY, baseURL: BASE_URL });
    const browser = await chromium.launch(launchOptions());
    try {
      const page = await browser.newPage();
      console.log(await withAgent({ name: "playwright_agent" }, () => runAgent(page, llm, task)));
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["page.goto", "locator.fill", "locator.click"], hasLlm: true },
};
