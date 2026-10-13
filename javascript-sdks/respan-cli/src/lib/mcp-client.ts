import { readFileSync } from 'node:fs';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import type { AuthConfig } from './auth.js';

export function resolveMcpUrl(baseUrl: string, override?: string): URL {
  if (override) {
    const url = new URL(override);
    if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname))) {
      throw new Error('MCP URL must use HTTPS (HTTP is allowed for localhost).');
    }
    if (url.username || url.password) throw new Error('MCP URL must not contain credentials.');
    return url;
  }
  const origin = new URL(baseUrl).origin;
  if (origin === 'https://api.respan.ai') return new URL('https://mcp.respan.ai/api/mcp');
  if (origin === 'https://endpoint.respan.ai') return new URL('https://mcp.respan.ai/api/mcp/enterprise');
  throw new Error('For a custom API endpoint, specify its MCP server with --mcp-url.');
}

export async function withMcpClient<T>(auth: AuthConfig, override: string | undefined, run: (client: Client) => Promise<T>): Promise<T> {
  const token = auth.apiKey || auth.accessToken;
  if (!token) throw new Error('No API key or access token available.');
  const { version } = JSON.parse(readFileSync(new URL('../../package.json', import.meta.url), 'utf8'));
  const client = new Client({ name: 'respan-cli', version });
  const transport = new StreamableHTTPClientTransport(resolveMcpUrl(auth.baseUrl, override), {
    requestInit: { headers: { Authorization: `Bearer ${token}` } },
  });
  try {
    await client.connect(transport);
    return await run(client);
  } finally {
    await client.close();
  }
}

export async function listMcpTools(client: Client) {
  const tools: Awaited<ReturnType<Client['listTools']>>['tools'] = [];
  const seen = new Set<string>();
  let cursor: string | undefined;
  do {
    const result = await client.listTools(cursor ? { cursor } : undefined);
    tools.push(...result.tools);
    cursor = result.nextCursor;
    if (cursor && seen.has(cursor)) throw new Error('MCP server returned a repeated pagination cursor.');
    if (cursor) seen.add(cursor);
  } while (cursor);
  return tools;
}
