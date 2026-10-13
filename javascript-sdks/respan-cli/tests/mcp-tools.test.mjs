import assert from 'node:assert/strict';
import test from 'node:test';
import { createServer } from 'node:http';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { resolve } from 'node:path';
import { resolveMcpUrl } from '../dist/lib/mcp-client.js';

const exec = promisify(execFile);
const cli = resolve('bin/run.js');
const tool = (name) => ({ name, description: name, inputSchema: { type: 'object', properties: {} } });

test('MCP commands use live schemas, follow pagination, and forward exact arguments', async () => {
  const requests = [];
  const server = createServer(async (req, res) => {
    if (req.method !== 'POST') { res.writeHead(405).end(); return; }
    let body = '';
    for await (const chunk of req) body += chunk;
    const request = JSON.parse(body);
    requests.push(request);
    assert.equal(req.headers.authorization, 'Bearer test-token');
    if (request.id === undefined) { res.writeHead(202).end(); return; }
    let result;
    if (request.method === 'initialize') {
      result = { protocolVersion: request.params.protocolVersion, capabilities: { tools: {} }, serverInfo: { name: 'test', version: '1.0.0' } };
    } else if (request.method === 'tools/list') {
      result = request.params?.cursor ? { tools: [tool('trace_get')] } : { tools: [tool('log_list')], nextCursor: 'next' };
    } else if (request.method === 'tools/call') {
      result = { content: [{ type: 'text', text: JSON.stringify(request.params.arguments) }], isError: request.params.name === 'failure' };
    } else { res.writeHead(400).end(); return; }
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jsonrpc: '2.0', id: request.id, result }));
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}/mcp`;
  const options = { env: { ...process.env, RESPAN_API_KEY: 'test-token', NO_COLOR: '1' }, timeout: 10000 };
  try {
    const list = await exec(process.execPath, [cli, 'tools', 'list', '--mcp-url', url, '--json'], options);
    assert.deepEqual(JSON.parse(list.stdout).map((t) => t.name), ['log_list', 'trace_get']);
    const input = { page_size: 7, environment: 'test', filters: [{ field: 'latency', operator: 'gt', value: [5] }] };
    const call = await exec(process.execPath, [cli, 'tools', 'call', 'log_list', '--mcp-url', url, '--input', JSON.stringify(input)], options);
    assert.deepEqual(JSON.parse(JSON.parse(call.stdout).content[0].text), input);
    const sent = requests.find((r) => r.method === 'tools/call');
    assert.deepEqual(sent.params, { name: 'log_list', arguments: input });
    await assert.rejects(exec(process.execPath, [cli, 'tools', 'call', 'failure', '--mcp-url', url], options), (err) => err.code === 1 && JSON.parse(err.stdout).isError === true);
    const count = requests.length;
    await assert.rejects(exec(process.execPath, [cli, 'tools', 'call', 'log_list', '--mcp-url', url, '--input', '[]'], options), /Tool input must be a JSON object/);
    assert.equal(requests.length, count, 'invalid input must fail before connecting');
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('MCP endpoint selection respects Platform, Enterprise and explicit custom servers', () => {
  assert.equal(resolveMcpUrl('https://api.respan.ai/api').href, 'https://mcp.respan.ai/api/mcp');
  assert.equal(resolveMcpUrl('https://endpoint.respan.ai').href, 'https://mcp.respan.ai/api/mcp/enterprise');
  assert.throws(() => resolveMcpUrl('https://private.example.com'), /--mcp-url/);
  assert.throws(() => resolveMcpUrl('https://api.respan.ai', 'http://example.com/mcp'), /HTTPS/);
});

test('removed integrate commands are unavailable', async () => {
  await assert.rejects(exec(process.execPath, [cli, 'integrate', 'pi'], { env: { ...process.env, NO_COLOR: '1' }, timeout: 10000 }), (err) => err.code !== 0 && /is not a respan command/.test(err.stderr));
});
