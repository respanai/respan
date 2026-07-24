import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  formatRuntimeEnvironmentAsShell,
  resolveGatewayBaseUrl,
  resolveRuntimeEnvironment,
  runtimeEnvironmentJson,
} from '../dist/lib/runtime-env.js';
import { extractEnvVar } from '../dist/lib/integrate.js';
import { resolveAuth } from '../dist/lib/auth.js';

const auth = {
  apiKey: "secret'key",
  baseUrl: 'https://api.respan.ai/',
  source: 'project_env',
};

test('runtime environment separates API and gateway URLs', () => {
  const runtime = resolveRuntimeEnvironment(auth, {
    RESPAN_GATEWAY_BASE_URL: 'https://gateway.example/api/chat/completions',
    RESPAN_MODEL: 'test-model',
  });

  assert.equal(runtime.apiBaseUrl, 'https://api.respan.ai');
  assert.equal(runtime.gatewayBaseUrl, 'https://gateway.example/api');
  assert.equal(runtime.model, 'test-model');
  assert.equal(runtime.authSource, 'project_env');
  assert.equal(runtime.gatewayReady, true);
});

test('runtime output hides the API key unless explicitly requested', () => {
  const runtime = resolveRuntimeEnvironment(auth, {});
  const safeShell = formatRuntimeEnvironmentAsShell(runtime);
  const safeJson = runtimeEnvironmentJson(runtime);

  assert.doesNotMatch(safeShell, /secret/);
  assert.equal('apiKey' in safeJson, false);

  const secretShell = formatRuntimeEnvironmentAsShell(runtime, true);
  const secretJson = runtimeEnvironmentJson(runtime, true);
  assert.match(secretShell, /RESPAN_API_KEY='secret'"'"'key'/);
  assert.equal(secretJson.apiKey, "secret'key");
});

test('gateway URL normalization appends the API path once', () => {
  assert.equal(
    resolveGatewayBaseUrl({ RESPAN_BASE_URL: 'https://enterprise.example/' }),
    'https://enterprise.example/api',
  );
  assert.equal(
    resolveGatewayBaseUrl({ RESPAN_GATEWAY_BASE_URL: 'https://enterprise.example/api/' }),
    'https://enterprise.example/api',
  );
});

test('dotenv parsing accepts whitespace, export, and matching quotes', () => {
  assert.equal(extractEnvVar('RESPAN_API_KEY = secret\n', 'RESPAN_API_KEY'), 'secret');
  assert.equal(extractEnvVar('export RESPAN_API_KEY = "secret"\n', 'RESPAN_API_KEY'), 'secret');
  assert.equal(extractEnvVar("RESPAN_API_KEY='secret value'\n", 'RESPAN_API_KEY'), 'secret value');
});

test('explicit env-file auth does not inherit the active profile base URL', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'respan-env-file-'));
  const envFile = path.join(root, '.env');
  fs.writeFileSync(envFile, 'RESPAN_API_KEY = file-secret\n');
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));

  const previousKey = process.env.RESPAN_API_KEY;
  const previousBaseUrl = process.env.RESPAN_API_BASE_URL;
  delete process.env.RESPAN_API_KEY;
  delete process.env.RESPAN_API_BASE_URL;
  t.after(() => {
    if (previousKey === undefined) delete process.env.RESPAN_API_KEY;
    else process.env.RESPAN_API_KEY = previousKey;
    if (previousBaseUrl === undefined) delete process.env.RESPAN_API_BASE_URL;
    else process.env.RESPAN_API_BASE_URL = previousBaseUrl;
  });

  const resolved = resolveAuth({}, { envFile });
  assert.equal(resolved.source, 'env_file');
  assert.equal(resolved.baseUrl, 'https://api.respan.ai');
});
