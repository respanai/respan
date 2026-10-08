import { RespanTelemetry } from '../src/main';
import Anthropic from '@anthropic-ai/sdk';
import dotenv from 'dotenv';

dotenv.config({ override: true });

/**
 * Test to verify Anthropic instrumentation properly records metrics (tokens, cost, etc.)
 * after adding enrichTokens: true option
 */

const respan = new RespanTelemetry({
    apiKey: process.env.RESPAN_API_KEY || "test-key",
    baseURL: process.env.RESPAN_BASE_URL || "https://api.respan.ai",
    appName: 'anthropic-metrics-test',
    disableBatch: true,
    logLevel: 'debug',
    traceContent: true,
    instrumentModules: {
        anthropic: Anthropic,
    }
});

// Wait for initialization
await respan.initialize();

console.log('[Test] Respan initialized with Anthropic instrumentation');

// Create Anthropic client
const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
});

console.log('[Test] Anthropic client created');

// Test basic message creation
const testAnthropicMetrics = async () => {
    console.log('[Test] Starting Anthropic metrics test...');
    
    try {
        const result = await respan.withWorkflow(
            { name: 'anthropic_metrics_test', version: 1 },
            async () => {
                console.log('[Test] Inside workflow, creating message...');
                
                const message = await anthropic.messages.create({
                    model: 'claude-haiku-4-5-20251001',
                    max_tokens: 100,
                    messages: [
                        { role: 'user', content: 'Say hello in 10 words or less.' }
                    ]
                });
                
                console.log('[Test] Message created successfully');
                console.log('[Test] Response:', message.content[0].type === 'text' ? message.content[0].text : 'N/A');
                console.log('[Test] Usage:', message.usage);
                
                // The instrumentation should automatically capture:
                // - prompt_tokens
                // - completion_tokens
                // - total_tokens
                // - estimated cost
                
                return message.content[0].type === 'text' ? message.content[0].text : '';
            }
        );
        
        console.log('[Test] Workflow completed successfully');
        console.log('[Test] Result:', result);
        console.log('\n[Test] ✓ Anthropic metrics test passed!');
        console.log('[Test] Check your Respan dashboard to verify that:');
        console.log('[Test]   - Token counts are recorded for the span');
        console.log('[Test]   - Cost metrics are calculated and displayed');
        console.log('[Test]   - All span attributes include usage information');
        
    } catch (error) {
        console.error('[Test] ✗ Anthropic metrics test failed:', error);
        throw error;
    }
};

// Run the test
testAnthropicMetrics()
    .then(async () => {
        console.log('\n[Test] All tests completed successfully!');
        console.log('[Test] Shutting down and flushing traces...\n');
        
        // Properly shutdown to flush all traces and show detailed metrics
        await respan.shutdown();
        console.log('[Test] ✓ Traces flushed to backend\n');
        
        process.exit(0);
    })
    .catch(async (error) => {
        console.error('\n[Test] Test suite failed:', error);
        await respan.shutdown();
        process.exit(1);
    });

export { respan, testAnthropicMetrics };

