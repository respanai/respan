#!/usr/bin/env node
/**
 * Debug test to verify traces are actually being exported to the backend
 */

import { RespanTelemetry } from '../src/main.js';
import Anthropic from '@anthropic-ai/sdk';

console.log('\n🔍 Export Debug Test\n');

const respan = new RespanTelemetry({
    apiKey: process.env.RESPAN_API_KEY || "test-key",
    baseURL: process.env.RESPAN_BASE_URL || "http://127.0.0.1:8000",
    appName: 'export-debug-test',
    disableBatch: true, // Should send immediately
    logLevel: 'debug',
    traceContent: true,
    instrumentModules: {
        anthropic: Anthropic,
    }
});

await respan.initialize();
console.log('✓ Initialized\n');

const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
});

console.log('📤 Making Anthropic API call...\n');

try {
    await respan.withWorkflow(
        { name: 'export_debug_workflow', version: 1 },
        async () => {
            const message = await anthropic.messages.create({
                model: 'claude-haiku-4-5-20251001',
                max_tokens: 50,
                messages: [{ role: 'user', content: 'Say test' }]
            });
            
            console.log(`✓ Got response: "${message.content[0].type === 'text' ? message.content[0].text : 'N/A'}"`);
            console.log(`✓ Tokens: ${message.usage.input_tokens} + ${message.usage.output_tokens} = ${message.usage.input_tokens + message.usage.output_tokens}\n`);
            
            return message;
        }
    );
    
    console.log('✓ Workflow completed\n');
    console.log('⏳ Manually flushing traces...\n');
    
    // Try to manually flush/shutdown to force export
    await respan.shutdown();
    
    console.log('✓ Shutdown complete - traces should have been exported\n');
    console.log('📊 Check your backend logs for:');
    console.log('   - POST /api/v1/traces');
    console.log('   - Content-Type: application/x-protobuf');
    console.log(`   - Authorization: Bearer ${process.env.RESPAN_API_KEY?.substring(0, 15)}...`);
    console.log('   - Workflow name: export_debug_workflow\n');
    
    process.exit(0);
    
} catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
}

