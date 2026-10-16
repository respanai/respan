/**
 * Run the Stagehand scenarios, then write runs/<run id>.json for the live check.
 *
 *   npx tsx run_all.ts                       # every scenario
 *   npx tsx run_all.ts hello act_variables   # some of them (by name)
 *   npx tsx run_all.ts --list
 *   npx tsx run_all.ts --headless --no-screenshots
 *
 * Then: python verify_live.py runs/<run id>.json  (reads every trace back from Respan)
 */
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith("--")));
const names = args.filter((a) => !a.startsWith("--"));
if (flags.has("--headless")) process.env.HEADLESS = "1";

const harness = await import("./harness.js");
const dir = join(harness.HERE, "scenarios");
const scenarios: import("./harness.js").Scenario[] = [];
for (const file of readdirSync(dir).filter((f) => /^\d\d_.*\.ts$/.test(f)).sort()) {
  scenarios.push((await import(pathToFileURL(join(dir, file)).href)).scenario);
}

if (flags.has("--list")) {
  for (const s of scenarios) console.log(`${s.name.padEnd(22)} ${s.title.padEnd(48)} ${s.source}`);
  process.exit(0);
}
const unknown = names.filter((n) => !scenarios.some((s) => s.name === n));
if (unknown.length) throw new Error(`unknown scenarios: ${unknown.join(", ")}`);
const selected = names.length ? scenarios.filter((s) => names.includes(s.name)) : scenarios;

await harness.init({ captureContent: !flags.has("--no-content"), captureScreenshots: !flags.has("--no-screenshots") });
const site = await harness.serveFixtures();
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
const path = await harness.writeRun(outcomes);
await harness.shutdown();
console.log(`\nRun record: ${path}\nCheck it in Respan:  python verify_live.py ${path}`);
