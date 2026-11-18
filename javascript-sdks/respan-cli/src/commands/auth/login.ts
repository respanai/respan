import { Flags } from '@oclif/core';
import { select, input } from '@inquirer/prompts';
import * as http from 'node:http';
import * as open from 'node:child_process';
import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import { BaseCommand } from '../../lib/base-command.js';
import { setCredential, setActiveProfile } from '../../lib/config.js';
import { printBanner, printLoginSuccess } from '../../lib/banner.js';
import { DEFAULT_BASE_URL, ENTERPRISE_BASE_URL } from '../../lib/auth.js';

const CALLBACK_PORT = 18392;
const CALLBACK_PATH = '/callback';
const LOGIN_TIMEOUT_MS = 120_000;
const LOGIN_URL_BASE = 'https://platform.respan.ai/login';
const ENTERPRISE_LOGIN_URL_BASE = 'https://enterprise.respan.ai/login';
// Overrides the web login page, e.g. a staging frontend. Pair it with
// --base-url for that deployment's API.
const LOGIN_URL_ENV = 'RESPAN_LOGIN_URL';
const CODE_EXCHANGE_PATH = '/auth/cli/token/';

// PKCE (RFC 7636): the browser sends this listener only a one-time code, and
// the code is worthless without the verifier, which never leaves this process.
// `state` ties the callback to this login attempt.
interface LoginAttempt {
  codeVerifier: string;
  codeChallenge: string;
  state: string;
}

function base64Url(buffer: Buffer): string {
  return buffer.toString('base64url');
}

function newLoginAttempt(): LoginAttempt {
  const codeVerifier = base64Url(randomBytes(32));
  return {
    codeVerifier,
    codeChallenge: base64Url(createHash('sha256').update(codeVerifier).digest()),
    state: base64Url(randomBytes(16)),
  };
}

function isSameState(expected: string, received: string | null): boolean {
  if (!received) return false;
  const expectedBytes = Buffer.from(expected);
  const receivedBytes = Buffer.from(received);
  return expectedBytes.length === receivedBytes.length && timingSafeEqual(expectedBytes, receivedBytes);
}

async function exchangeCode(baseUrl: string, code: string, attempt: LoginAttempt, redirectUri: string): Promise<BrowserLoginResult> {
  const origin = baseUrl.replace(/\/+$/, '').replace(/\/api$/, '');
  const response = await fetch(`${origin}${CODE_EXCHANGE_PATH}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, code_verifier: attempt.codeVerifier, redirect_uri: redirectUri }),
  });
  if (!response.ok) {
    throw new Error('The sign-in code was rejected or expired. Please run `respan auth login` again.');
  }
  const body = (await response.json()) as { access?: string; refresh?: string; email?: string };
  if (!body.access || !body.refresh) {
    throw new Error('The sign-in response was missing a token. Please try again.');
  }
  return { token: body.access, refreshToken: body.refresh, email: body.email };
}

interface BrowserLoginResult {
  token: string;
  refreshToken: string;
  email?: string;
}

function openBrowser(url: string): void {
  const cmd = process.platform === 'darwin' ? 'open' : process.platform === 'win32' ? 'start' : 'xdg-open';
  open.exec(`${cmd} "${url}"`);
}

function successHtml(): string {
  return `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Respan CLI</title>
<style>body{font-family:-apple-system,system-ui,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#0a0a0f;color:#fff}
.card{text-align:center;padding:2rem}.check{font-size:3rem;margin-bottom:1rem;color:#6483F0}p{color:#aaa;margin-top:0.5rem}</style></head>
<body><div class="card"><div class="check">&#10003;</div><h2>Login successful!</h2><p>You can close this window and return to the terminal.</p></div></body></html>`;
}

function errorHtml(msg: string): string {
  return `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Respan CLI</title>
<style>body{font-family:-apple-system,system-ui,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#0a0a0f;color:#fff}
.card{text-align:center;padding:2rem}.icon{font-size:3rem;margin-bottom:1rem;color:#f06464}p{color:#aaa;margin-top:0.5rem}</style></head>
<body><div class="card"><div class="icon">&#10007;</div><h2>Login failed</h2><p>${msg}</p></div></body></html>`;
}

function waitForBrowserLogin(enterprise: boolean, baseUrl: string): Promise<BrowserLoginResult> {
  const attempt = newLoginAttempt();
  const redirectUri = `http://localhost:${CALLBACK_PORT}${CALLBACK_PATH}`;
  return new Promise((resolve, reject) => {
    let isSettled = false;
    const fail = (res: http.ServerResponse, message: string) => {
      res.writeHead(400, { 'Content-Type': 'text/html' });
      res.end(errorHtml(message));
      cleanup();
      reject(new Error(message));
    };
    const succeed = (res: http.ServerResponse, result: BrowserLoginResult) => {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(successHtml());
      cleanup();
      resolve(result);
    };

    const server = http.createServer((req, res) => {
      const url = new URL(req.url || '/', `http://localhost:${CALLBACK_PORT}`);
      if (url.pathname !== CALLBACK_PATH || isSettled) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }

      const code = url.searchParams.get('code');
      if (code) {
        // A callback for another login attempt (or a crafted link) is ignored
        // without spending this attempt.
        if (!isSameState(attempt.state, url.searchParams.get('state'))) {
          res.writeHead(400, { 'Content-Type': 'text/html' });
          res.end(errorHtml('This sign-in link is not from your current login. Please try again.'));
          return;
        }
        isSettled = true;
        exchangeCode(baseUrl, code, attempt, redirectUri).then(
          (result) => succeed(res, result),
          (error: Error) => fail(res, error.message),
        );
        return;
      }

      // A web app that predates the code flow sends the tokens themselves.
      const token = url.searchParams.get('token');
      const refreshToken = url.searchParams.get('refresh_token');
      const email = url.searchParams.get('email') || undefined;

      isSettled = true;
      if (!token || !refreshToken) {
        fail(res, 'Missing sign-in code. Please try again.');
        return;
      }
      succeed(res, { token, refreshToken, email });
    });

    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error('Login timed out after 120 seconds. Please try again.'));
    }, LOGIN_TIMEOUT_MS);

    function cleanup() {
      clearTimeout(timeout);
      server.close();
    }

    server.on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EADDRINUSE') {
        reject(new Error(`Port ${CALLBACK_PORT} is in use. Close the other process and try again.`));
      } else {
        reject(err);
      }
    });

    server.listen(CALLBACK_PORT, '127.0.0.1', () => {
      const loginUrlBase = process.env[LOGIN_URL_ENV] || (enterprise ? ENTERPRISE_LOGIN_URL_BASE : LOGIN_URL_BASE);
      const loginUrl = new URL(loginUrlBase);
      loginUrl.searchParams.set('mode', 'cli');
      loginUrl.searchParams.set('redirect_uri', redirectUri);
      loginUrl.searchParams.set('code_challenge', attempt.codeChallenge);
      loginUrl.searchParams.set('code_challenge_method', 'S256');
      loginUrl.searchParams.set('state', attempt.state);

      console.log('');
      console.log('  Opening browser to log in...');
      console.log(`  If the browser doesn't open, visit:`);
      console.log(`  ${loginUrl.toString()}`);
      console.log('');
      console.log('  Waiting for login (timeout: 120s)...');
      openBrowser(loginUrl.toString());
    });
  });
}

export default class AuthLogin extends BaseCommand {
  static description = 'Log in to Respan';
  static flags = {
    ...BaseCommand.baseFlags,
    'api-key': Flags.string({ description: 'API key to store (skips interactive prompt)' }),
    'profile-name': Flags.string({ description: 'Profile name', default: 'default' }),
    'base-url': Flags.string({ description: 'API base URL (auto-detected from login method if not set)' }),
    enterprise: Flags.boolean({ description: 'Use enterprise SSO login', default: false }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(AuthLogin);
    this.globalFlags = flags;
    const profile = flags['profile-name'] || 'default';
    setActiveProfile(profile);

    await printBanner();

    // Step 1: Determine environment (skip if --enterprise flag or --api-key with flag)
    const enterprise = flags.enterprise || (!flags['api-key'] && await select({
      message: 'Select your environment:',
      choices: [
        { name: 'Respan Platform', value: false },
        { name: 'Enterprise', value: true },
      ],
    }));
    const baseUrl = flags['base-url'] || (enterprise ? ENTERPRISE_BASE_URL : DEFAULT_BASE_URL);

    // If --api-key passed directly, skip auth method prompt
    if (flags['api-key']) {
      setCredential(profile, { type: 'api_key', apiKey: flags['api-key'], baseUrl });
      await printLoginSuccess(undefined, profile);
      return;
    }

    // Step 2: Choose auth method
    const method = await select({
      message: 'How would you like to authenticate?',
      choices: [
        { name: 'API key (recommended)', value: 'api_key' },
        { name: 'Browser login', value: 'browser' },
      ],
    });

    if (method === 'api_key') {
      const apiKey = await input({ message: 'Enter your Respan API key:' });
      setCredential(profile, { type: 'api_key', apiKey, baseUrl });
      await printLoginSuccess(undefined, profile);
      return;
    }

    const result = await waitForBrowserLogin(enterprise, baseUrl);

    setCredential(profile, {
      type: 'jwt',
      accessToken: result.token,
      refreshToken: result.refreshToken,
      email: result.email || '',
      baseUrl,
    });
    await printLoginSuccess(result.email, profile);
  }
}
