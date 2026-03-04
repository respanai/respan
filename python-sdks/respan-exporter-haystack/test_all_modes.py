"""
Comprehensive Test Suite - Respan Haystack Integration

Tests all 5 integration modes:
1. Gateway only (no tracing, no prompt)
2. Gateway with prompt (from platform)
3. Tracing only (OpenAI direct)
4. Tracing + Gateway
5. Tracing + Gateway + Prompt (full stack!)
"""

import os
import time
import traceback
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from respan_exporter_haystack.connector import RespanConnector
from respan_exporter_haystack.gateway import RespanGenerator


def check_env():
    """Check required environment variables."""
    if not os.getenv(key="RESPAN_API_KEY"):
        print("ERROR: RESPAN_API_KEY not set")
        return False
    if not os.getenv(key="OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return False
    return True


def test_1_gateway_only():
    """Test 1: Gateway only - basic auto-logging."""
    print("\n" + "="*80)
    print("TEST 1: GATEWAY ONLY (No tracing, no prompt)")
    print("="*80)
    print("Expected: Single LLM log in dashboard (not a trace)")
    print("-"*80)
    
    pipeline = Pipeline()
    pipeline.add_component(name="prompt", instance=PromptBuilder(template="Tell me a joke about {{topic}}."))
    pipeline.add_component(name="llm", instance=RespanGenerator(model="gpt-4o-mini"))
    pipeline.connect(sender="prompt", receiver="llm")
    
    result = pipeline.run({"prompt": {"topic": "Python"}})
    
    if "llm" in result and "replies" in result["llm"]:
        response = result['llm']['replies'][0][:100].encode('ascii', 'replace').decode('ascii')
        print(f"\n[RESPONSE] {response}...")
        meta = result["llm"]["meta"][0]
        print(f"\n[META] Model: {meta['model']}, Tokens: {meta['total_tokens']}")
    
    print("\n[CHECK DASHBOARD]")
    print("  URL: https://platform.respan.ai/logs")
    print("  What to see: Single log entry (NOT in traces view)")
    print("  Log type: Chat completion")
    return True


def test_2_gateway_with_prompt():
    """Test 2: Gateway with platform prompt."""
    print("\n" + "="*80)
    print("TEST 2: GATEWAY WITH PROMPT")
    print("="*80)
    print("Expected: Single LLM log using platform prompt")
    print("-"*80)
    
    PROMPT_ID = "1210b368ce2f4e5599d307bc591d9b7a"
    
    pipeline = Pipeline()
    pipeline.add_component(name="llm", instance=RespanGenerator(
        model="gpt-4o-mini",
        prompt_id=PROMPT_ID
    ))
    
    result = pipeline.run({
        "llm": {
            "prompt_variables": {
                "user_input": "The cat sat on the mat"
            }
        }
    })
    
    if "llm" in result and "replies" in result["llm"]:
        response = result['llm']['replies'][0][:150].encode('ascii', 'replace').decode('ascii')
        print(f"\n[RESPONSE] {response}...")
        meta = result["llm"]["meta"][0]
        print(f"\n[META] Model: {meta['model']}, Tokens: {meta['total_tokens']}")
    
    print("\n[CHECK DASHBOARD]")
    print("  URL: https://platform.respan.ai/logs")
    print("  What to see: Log shows prompt_id in metadata")
    print("  Filter by: Prompt name (if set on platform)")
    return True


def test_3_trace_only():
    """Test 3: Tracing only with direct OpenAI."""
    print("\n" + "="*80)
    print("TEST 3: TRACING ONLY (Direct OpenAI, no gateway)")
    print("="*80)
    print("Expected: Full trace with all components")
    print("-"*80)
    
    os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"
    
    pipeline = Pipeline()
    pipeline.add_component(name="tracer", instance=RespanConnector(name="Test 3: Trace Only"))
    pipeline.add_component(name="prompt", instance=PromptBuilder(template="Tell me about {{topic}}."))
    pipeline.add_component(name="llm", instance=OpenAIGenerator(model="gpt-4o-mini"))
    pipeline.connect(sender="prompt", receiver="llm")
    
    result = pipeline.run({"prompt": {"topic": "machine learning"}})
    
    if "llm" in result and "replies" in result["llm"]:
        response = result['llm']['replies'][0][:100].encode('ascii', 'replace').decode('ascii')
        print(f"\n[RESPONSE] {response}...")
    
    if "tracer" in result:
        print(f"\n[TRACE URL] {result['tracer']['trace_url']}")
    
    print("\n[CHECK DASHBOARD]")
    print("  URL: https://platform.respan.ai/logs")
    print("  View: Switch to 'Traces' tab")
    print("  What to see:")
    print("    - Test 3: Trace Only (root)")
    print("    - prompt (PromptBuilder span)")
    print("    - llm (OpenAI span with tokens/cost)")
    return True


def test_4_trace_with_gateway():
    """Test 4: Tracing + Gateway combined."""
    print("\n" + "="*80)
    print("TEST 4: TRACING + GATEWAY")
    print("="*80)
    print("Expected: Both individual log AND full trace")
    print("-"*80)
    
    os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"
    
    pipeline = Pipeline()
    pipeline.add_component(name="tracer", instance=RespanConnector(name="Test 4: Gateway + Trace"))
    pipeline.add_component(name="prompt", instance=PromptBuilder(template="Explain {{topic}} in one sentence."))
    pipeline.add_component(name="llm", instance=RespanGenerator(model="gpt-4o-mini"))
    pipeline.connect(sender="prompt", receiver="llm")
    
    result = pipeline.run({"prompt": {"topic": "neural networks"}})
    
    if "llm" in result and "replies" in result["llm"]:
        response = result['llm']['replies'][0][:100].encode('ascii', 'replace').decode('ascii')
        print(f"\n[RESPONSE] {response}...")
    
    if "tracer" in result:
        print(f"\n[TRACE URL] {result['tracer']['trace_url']}")
    
    print("\n[CHECK DASHBOARD]")
    print("  URL: https://platform.respan.ai/logs")
    print("  What to see:")
    print("    1. LOGS TAB: Individual LLM call from gateway")
    print("    2. TRACES TAB: Full workflow trace with:")
    print("       - Test 4: Gateway + Trace (root)")
    print("       - prompt (PromptBuilder)")
    print("       - llm (Respan gateway)")
    return True


def test_5_full_stack():
    """Test 5: Tracing + Gateway + Prompt (everything!)."""
    print("\n" + "="*80)
    print("TEST 5: FULL STACK (Tracing + Gateway + Prompt)")
    print("="*80)
    print("Expected: Log + Trace using platform prompt")
    print("-"*80)
    
    PROMPT_ID = "1210b368ce2f4e5599d307bc591d9b7a"
    os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"
    
    pipeline = Pipeline()
    pipeline.add_component(name="tracer", instance=RespanConnector(name="Test 5: Full Stack"))
    pipeline.add_component(name="llm", instance=RespanGenerator(
        model="gpt-4o-mini",
        prompt_id=PROMPT_ID
    ))
    
    result = pipeline.run({
        "llm": {
            "prompt_variables": {
                "user_input": "She sells seashells by the seashore"
            }
        }
    })
    
    if "llm" in result and "replies" in result["llm"]:
        response = result['llm']['replies'][0][:150].encode('ascii', 'replace').decode('ascii')
        print(f"\n[RESPONSE] {response}...")
    
    if "tracer" in result:
        print(f"\n[TRACE URL] {result['tracer']['trace_url']}")
    
    print("\n[CHECK DASHBOARD]")
    print("  URL: https://platform.respan.ai/logs")
    print("  What to see:")
    print("    1. LOGS TAB: LLM call with prompt_id metadata")
    print("    2. TRACES TAB: Trace with:")
    print("       - Test 5: Full Stack (root)")
    print("       - llm (gateway call using platform prompt)")
    print("  This is the ULTIMATE setup!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RESPAN HAYSTACK INTEGRATION - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    if not check_env():
        print("\nPlease set required environment variables:")
        print("  export RESPAN_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='your-openai-key'")
        return
    
    print("\nRunning 5 test scenarios...")
    print("After each test, check the Respan dashboard to verify results.")
    
    tests = [
        ("Test 1: Gateway Only", test_1_gateway_only),
        ("Test 2: Gateway + Prompt", test_2_gateway_with_prompt),
        ("Test 3: Trace Only", test_3_trace_only),
        ("Test 4: Gateway + Trace", test_4_trace_with_gateway),
        ("Test 5: Full Stack", test_5_full_stack),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\n\n>>> Running {name}...")
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
            print(f"\n>>> {name} completed. Check dashboard before next test.\n")
            print("Waiting 3 seconds...")
            time.sleep(3)
        except Exception as e:
            print(f"\n[ERROR] {name} failed: {e}")
            results.append((name, "FAIL"))
            traceback.print_exc()
    
    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, status in results:
        print(f"  {status:6} | {name}")
    
    print("\n" + "="*80)
    print("DASHBOARD CHECK")
    print("="*80)
    print("  Logs: https://platform.respan.ai/logs")
    print("  Traces: https://platform.respan.ai/logs (switch to Traces tab)")
    print("\nExpected results:")
    print("  - 5 LLM logs (Tests 1, 2, 4, 5 + Test 3 direct)")
    print("  - 3 Traces (Tests 3, 4, 5)")
    print("  - 2 with prompt_id metadata (Tests 2, 5)")


if __name__ == "__main__":
    main()
