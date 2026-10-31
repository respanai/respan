#!/usr/bin/env node
// Checks chatgpt-plugin/ against OpenAI's plugin submission rules, then zips it
// for upload at https://platform.openai.com/plugins.
//
//   node scripts/build-chatgpt-plugin.mjs   # check + write chatgpt-plugin/dist/respan-chatgpt-plugin.zip
//   node scripts/build-chatgpt-plugin.mjs --check   # check only
//
// The limits mirror https://developers.openai.com/plugins/deploy/submission.
// Every failed rule is printed, and nothing is zipped until all pass.
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const pluginDir = join(here, '..', 'chatgpt-plugin');
const zipPath = join(pluginDir, 'dist', 'respan-chatgpt-plugin.zip');
// Only these go into the ZIP; README.md and dist/ stay out. The plugin is
// MCP-only for now: skills can be added in a later version, but an MCP server
// cannot be added to a plugin first published without one.
const PACKAGE_ENTRIES = ['plugin.json', 'mcp.json', 'assets'];

const errors = [];
const check = (ok, message) => {
  if (!ok) errors.push(message);
};

const manifest = JSON.parse(readFileSync(join(pluginDir, 'plugin.json'), 'utf8'));
const mcp = JSON.parse(readFileSync(join(pluginDir, 'mcp.json'), 'utf8'));
const openai = manifest.extensions?.['com.openai'] ?? {};
const ui = openai.interface ?? {};

// --- Agent Plugins manifest ---
check(
  manifest.$schema === 'https://agent-plugins.org/schemas/1.0.0/plugin.schema.json',
  'plugin.json: $schema must be the Agent Plugins 1.0.0 plugin schema',
);
check(
  /^(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$/.test(manifest.name ?? '') && manifest.name.length <= 64,
  'plugin.json: name must be lowercase kebab-case, at most 64 characters',
);
check(!('apps' in openai), 'plugin.json: extensions.com.openai.apps cannot be submitted; declare servers in mcp.json');
check(!existsSync(join(pluginDir, '.app.json')), '.app.json cannot be submitted');

// --- Listing ---
const text = (field, max) => {
  const value = ui[field];
  check(typeof value === 'string' && value.trim().length > 0, `interface.${field} is required`);
  check(typeof value !== 'string' || value.length <= max, `interface.${field} must be at most ${max} characters (is ${value?.length})`);
};
text('displayName', 30);
text('shortDescription', 30);
text('longDescription', 4000);
text('developerName', 80);
text('category', 200);

const capabilities = ui.capabilities ?? [];
check(capabilities.length <= 20, 'interface.capabilities: at most 20 labels');
for (const label of capabilities) {
  check(label.length <= 120, `interface.capabilities: "${label}" is over 120 characters`);
}

const prompts = ui.defaultPrompt ?? [];
check(prompts.length <= 3, 'interface.defaultPrompt: at most 3 prompts');
check(new Set(prompts).size === prompts.length, 'interface.defaultPrompt: prompts must be unique');
for (const prompt of prompts) {
  check(prompt.length <= 128, `interface.defaultPrompt: "${prompt}" is over 128 characters`);
  check(!/@\w/.test(prompt), `interface.defaultPrompt: "${prompt}" must not @mention an app`);
}

const isHttpsUrl = (value) => {
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && !url.username && !url.password && value.length <= 1024;
  } catch {
    return false;
  }
};
for (const field of ['websiteURL', 'supportURL', 'privacyPolicyURL', 'termsOfServiceURL']) {
  check(isHttpsUrl(ui[field]), `interface.${field} must be an https URL without credentials, at most 1024 characters`);
}

// WCAG contrast ratio between two #RRGGBB colors.
const luminance = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const contrast = (a, b) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};
for (const [field, background] of [['brandColor', '#FFFFFF'], ['brandColorDark', '#212121']]) {
  const color = ui[field];
  if (color === undefined) continue;
  check(/^#[0-9A-Fa-f]{6}$/.test(color), `interface.${field} must be #RRGGBB`);
  if (/^#[0-9A-Fa-f]{6}$/.test(color)) {
    check(contrast(color, background) >= 2, `interface.${field} needs at least 2:1 contrast against ${background}`);
  }
}

// Square images of at least 48x48. Only SVG viewBoxes are measured here.
const IMAGE_FIELDS = ['logo', 'logoDark', 'composerIcon', 'composerIconDark'];
for (const field of IMAGE_FIELDS) {
  const path = ui[field];
  if (path === undefined) continue;
  check(path.startsWith('./') && !path.includes('..'), `interface.${field} must be a ./ path inside the plugin`);
  const file = join(pluginDir, path);
  check(existsSync(file), `interface.${field}: ${path} does not exist`);
  if (existsSync(file) && path.endsWith('.svg')) {
    const viewBox = /viewBox="\s*[\d.-]+\s+[\d.-]+\s+([\d.]+)\s+([\d.]+)\s*"/.exec(readFileSync(file, 'utf8'));
    check(viewBox && viewBox[1] === viewBox[2] && Number(viewBox[1]) >= 48, `interface.${field}: ${path} must be square and at least 48x48`);
  }
}
check(ui.logo !== undefined, 'interface.logo is required');
check(ui.composerIcon !== undefined, 'interface.composerIcon is required');

// --- MCP server ---
const servers = Object.entries(mcp.mcpServers ?? {});
check(mcp.$schema === 'https://agent-plugins.org/schemas/1.0.0/mcp.schema.json', 'mcp.json: $schema must be the Agent Plugins 1.0.0 MCP schema');
check(servers.length === 1, 'mcp.json: declare exactly one server (only one can be connected)');
for (const [name, server] of servers) {
  check(server.type === 'streamable-http', `mcp.json: ${name} must be type streamable-http`);
  check(isHttpsUrl(server.url), `mcp.json: ${name} needs an https url`);
}

// --- Review materials ---
const review = openai.review ?? {};
const positive = review.test_cases?.positive ?? [];
const negative = review.test_cases?.negative ?? [];
check(positive.length >= 5, `review: needs 5 positive test cases (has ${positive.length})`);
check(negative.length >= 3, `review: needs 3 negative test cases (has ${negative.length})`);
positive.forEach((testCase, i) => {
  for (const field of ['description', 'prompt', 'tools_triggered', 'expected_behavior']) {
    check(typeof testCase[field] === 'string' && testCase[field].trim(), `review: positive case ${i + 1} needs ${field}`);
  }
});
negative.forEach((testCase, i) => {
  for (const field of ['description', 'prompt']) {
    check(typeof testCase[field] === 'string' && testCase[field].trim(), `review: negative case ${i + 1} needs ${field}`);
  }
});
check(isHttpsUrl(review.demo_recording_url ?? ''), 'review.demo_recording_url must be the https URL of the demo video');
check(!('test_credentials' in review) && !('reviewer_instructions' in review), 'review: credentials go in the dashboard, not the ZIP');
check(typeof openai.publication?.release_notes === 'string', 'publication.release_notes is required');

if (errors.length > 0) {
  console.error(`chatgpt-plugin is not ready to submit (${errors.length} problem${errors.length === 1 ? '' : 's'}):`);
  for (const error of errors) console.error(`  - ${error}`);
  process.exit(1);
}
console.log('chatgpt-plugin passes the submission checks.');

if (!process.argv.includes('--check')) {
  mkdirSync(dirname(zipPath), { recursive: true });
  rmSync(zipPath, { force: true });
  execFileSync('zip', ['-r', '-X', '-q', zipPath, ...PACKAGE_ENTRIES], { cwd: pluginDir, stdio: 'inherit' });
  console.log(`Wrote ${zipPath}`);
}
