/**
 * Run the Playwright scenarios, then write runs/<run id>.json for the live check.
 *
 *   npx tsx run_all.ts                          # every scenario
 *   npx tsx run_all.ts forms network_route      # some of them (by name)
 *   npx tsx run_all.ts --list
 *   npx tsx run_all.ts --headed --screenshot-on action --no-content
 *
 * Then: python verify_live.py runs/<run id>.json  (reads every trace back from Respan)
 */
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith("--")));
const screenshotOnIndex = args.indexOf("--screenshot-on");
const screenshotOn = (screenshotOnIndex >= 0 ? args[screenshotOnIndex + 1] : "never") as "never" | "navigation" | "action";
const names = args.filter((a, i) => !a.startsWith("--") && !(screenshotOnIndex >= 0 && i === screenshotOnIndex + 1));
if (flags.has("--headed")) process.env.HEADED = "1";

const harness = await import("./harness.js");
const dir = join(harness.HERE, "scenarios");
const scenarios: import("./harness.js").Scenario[] = [];
for (const file of readdirSync(dir).filter((f) => /^\d\d_.*\.ts$/.test(f)).sort()) {
  scenarios.push((await import(pathToFileURL(join(dir, file)).href)).scenario);
}

if (flags.has("--list")) {
  for (const s of scenarios) console.log(`${s.name.padEnd(22)} ${s.title.padEnd(44)} ${s.source}`);
  process.exit(0);
}
const unknown = names.filter((n) => !scenarios.some((s) => s.name === n));
if (unknown.length) throw new Error(`unknown scenarios: ${unknown.join(", ")}`);
const selected = names.length ? scenarios.filter((s) => names.includes(s.name)) : scenarios;

const settings = {
  captureContent: !flags.has("--no-content"),
  captureScreenshots: !flags.has("--no-screenshots"),
  screenshotOn,
};
await harness.init(settings);
const site = await harness.serveSite();
const outcomes: import("./harness.js").Outcome[] = [];
try {
  for (const scenario of selected) {
    console.log(`\n== ${scenario.name}: ${scenario.title}`);
    const outcome = await harness.execute(scenario, site.url);
    console.log(outcome.error ? `   run raised ${outcome.error}` : "   ok");
    outcomes.push(outcome);
  }
} finally {
  await site.close();
}
const path = await harness.writeRun(outcomes, settings);
await harness.shutdown();
console.log(`\nRun record: ${path}\nCheck it in Respan:  python verify_live.py ${path}`);
