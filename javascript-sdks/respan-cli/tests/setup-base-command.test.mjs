import assert from 'node:assert/strict';
import test from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const cli = resolve('bin/run.js');
for (const [args, modes] of [
  [['setup'], ['tracing', 'gateway']],
  [['setup', 'tracing'], ['tracing']],
  [['setup', 'gateway'], ['gateway']],
  [['setup', 'gateway', '--agent', 'codex-cli', '--no-instrument'], ['gateway']],
]) {
  test(`${args.join(' ')} hands off to docs without credentials or side effects`, () => {
    const cwd = mkdtempSync(join(tmpdir(), 'respan-setup-handoff-'));
    try {
      const result = spawnSync(process.execPath, [cli, ...args], {
        cwd,
        env: { PATH: process.env.PATH, HOME: cwd, NO_COLOR: '1' },
        encoding: 'utf8',
        input: '',
        timeout: 10000,
      });
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stderr);
      assert.match(result.stdout, /deprecated/);
      for (const mode of ['tracing', 'gateway']) {
        assert.equal(result.stdout.includes(`Respan ${mode}: fetch and follow`), modes.includes(mode));
      }
      assert.match(result.stdout, /https:\/\/www\.respan\.ai\/docs\/documentation\/agent-setup\.md/);
      assert.deepEqual(readdirSync(cwd), [], 'must not write credentials, skills or agent config');
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });
}
