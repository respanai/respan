import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import { syncBuiltinESMExports } from 'node:module';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { pathToFileURL } from 'node:url';

const posixOnly = { skip: process.platform === 'win32' };
const credential = (apiKey) => ({ type: 'api_key', apiKey, baseUrl: 'https://api.respan.ai/api' });

const modeOf = (target) => fs.statSync(target).mode & 0o777;
const pathsIn = (home) => ({
  dir: join(home, '.respan'),
  credentials: join(home, '.respan', 'credentials.json'),
  config: join(home, '.respan', 'config.json'),
});

// config.ts resolves its paths from HOME at load, so each test imports a fresh copy.
async function withHome(run) {
  const home = fs.mkdtempSync(join(tmpdir(), 'respan-credentials-'));
  const previousHome = process.env.HOME;
  process.env.HOME = home;
  try {
    const config = await import(`../dist/lib/config.js?home=${encodeURIComponent(home)}`);
    await run(config, pathsIn(home), home);
  } finally {
    process.env.HOME = previousHome;
    fs.rmSync(home, { recursive: true, force: true });
  }
}

// What an older CLI left behind. chmod is explicit so the test ignores the umask.
function seedExisting(paths, { fileMode = 0o644, dirMode = 0o755 } = {}) {
  fs.mkdirSync(paths.dir, { recursive: true });
  fs.writeFileSync(paths.credentials, JSON.stringify({ default: credential('old-key') }));
  fs.chmodSync(paths.credentials, fileMode);
  fs.chmodSync(paths.dir, dirMode);
}

function withPatchedFs(name, replacement, run) {
  const original = fs[name];
  fs[name] = replacement;
  syncBuiltinESMExports();
  try {
    return run();
  } finally {
    fs[name] = original;
    syncBuiltinESMExports();
  }
}

test('a fresh write creates a 0700 directory and a 0600 credentials file', posixOnly, async () => {
  await withHome(async (config, paths, home) => {
    config.setCredential('default', credential('new-key'));

    assert.equal(modeOf(paths.dir), 0o700);
    assert.equal(modeOf(paths.credentials), 0o600);
    assert.equal(config.getCredential('default').apiKey, 'new-key');

    // config.json holds no secrets, so it keeps the default file mode.
    const control = join(home, 'control');
    fs.writeFileSync(control, '');
    assert.equal(modeOf(paths.config), modeOf(control));
  });
});

test('writing to an existing 644 file tightens it and keeps other profiles', posixOnly, async () => {
  await withHome(async (config, paths) => {
    seedExisting(paths);

    config.setCredential('staging', credential('new-key'));

    assert.equal(modeOf(paths.dir), 0o700);
    assert.equal(modeOf(paths.credentials), 0o600);
    const stored = JSON.parse(fs.readFileSync(paths.credentials, 'utf-8'));
    assert.equal(stored.default.apiKey, 'old-key');
    assert.equal(stored.staging.apiKey, 'new-key');
  });
});

test('reading an existing 644 file tightens it and returns the credential', posixOnly, async () => {
  await withHome(async (config, paths) => {
    seedExisting(paths);

    assert.equal(config.getCredential('default').apiKey, 'old-key');

    assert.equal(modeOf(paths.dir), 0o700);
    assert.equal(modeOf(paths.credentials), 0o600);
  });
});

test('deleting a credential leaves an existing 644 file at 0600', posixOnly, async () => {
  await withHome(async (config, paths) => {
    seedExisting(paths);

    config.deleteCredential('default');

    assert.equal(modeOf(paths.credentials), 0o600);
    assert.deepEqual(JSON.parse(fs.readFileSync(paths.credentials, 'utf-8')), {});
  });
});

test('config writes leave an existing config.json mode alone', posixOnly, async () => {
  await withHome(async (config, paths) => {
    seedExisting(paths);
    fs.writeFileSync(paths.config, '{}');
    fs.chmodSync(paths.config, 0o644);

    config.setConfigValue('activeProfile', 'staging');

    assert.equal(modeOf(paths.config), 0o644);
    assert.equal(config.getActiveProfile(), 'staging');
  });
});

test('tightening never adds access the owner removed', posixOnly, async () => {
  await withHome(async (config, paths) => {
    seedExisting(paths, { fileMode: 0o400, dirMode: 0o700 });

    assert.equal(config.getCredential('default').apiKey, 'old-key');

    assert.equal(modeOf(paths.credentials), 0o400);
  });
});

test('a chmod failure does not block reading or writing', posixOnly, async () => {
  await withHome(async (config, paths) => {
    seedExisting(paths);
    const denied = () => {
      throw Object.assign(new Error('operation not permitted'), { code: 'EPERM' });
    };

    withPatchedFs('chmodSync', denied, () => {
      assert.equal(config.getCredential('default').apiKey, 'old-key');
      config.setCredential('default', credential('new-key'));
      assert.equal(config.getCredential('default').apiKey, 'new-key');
    });
  });
});

test('Windows skips chmod and still reads and writes credentials', async () => {
  await withHome(async (config) => {
    const realPlatform = Object.getOwnPropertyDescriptor(process, 'platform');
    Object.defineProperty(process, 'platform', { ...realPlatform, value: 'win32' });
    let chmodCalls = 0;
    try {
      withPatchedFs('chmodSync', () => { chmodCalls += 1; }, () => {
        config.setCredential('default', credential('new-key'));
        assert.equal(config.getCredential('default').apiKey, 'new-key');
      });
    } finally {
      Object.defineProperty(process, 'platform', realPlatform);
    }
    assert.equal(chmodCalls, 0);
  });
});

// shared.ts imports config.js without a cache-busting query, so this needs its own process.
function resolveHookCredentials(home) {
  const shared = pathToFileURL(resolve('dist/hooks/shared.js')).href;
  const result = spawnSync(
    process.execPath,
    ['--input-type=module', '-e', `
      import { resolveCredentials } from ${JSON.stringify(shared)};
      console.log(JSON.stringify(resolveCredentials()));
    `],
    { env: { PATH: process.env.PATH, HOME: home }, encoding: 'utf8', timeout: 10000 },
  );
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

test('the hook reader tightens an existing 644 file and reads it at 0600', posixOnly, () => {
  const home = fs.mkdtempSync(join(tmpdir(), 'respan-credentials-hook-'));
  try {
    const paths = pathsIn(home);
    seedExisting(paths);

    assert.equal(resolveHookCredentials(home).apiKey, 'old-key');
    assert.equal(modeOf(paths.dir), 0o700);
    assert.equal(modeOf(paths.credentials), 0o600);

    assert.equal(resolveHookCredentials(home).apiKey, 'old-key');
  } finally {
    fs.rmSync(home, { recursive: true, force: true });
  }
});
