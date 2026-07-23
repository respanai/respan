"""
Example demonstrating the SpanCollector feature for batch span collection and export.

This example shows:
1. Basic usage of SpanCollector
2. Batch collection of multiple spans
3. Manual export control
4. Async span creation (creating spans after execution completes)
5. Isolation from normal spans
"""

import time
from respan_tracing import RespanTelemetry, get_client

# Initialize Respan telemetry
telemetry = RespanTelemetry(
    app_name="span-collector-example",
    api_key="your-api-key",  # Set your API key
    log_level="DEBUG",  # Enable debug logging to see what's happening
)


def simulate_workflow(workflow_id: str, duration: float = 0.1):
    """Simulate a workflow execution"""
    print(f"  Executing workflow {workflow_id}...")
    time.sleep(duration)
    return {
        "workflow_id": workflow_id,
        "input": f"input_{workflow_id}",
        "output": f"output_{workflow_id}",
        "duration": duration,
        "status": "completed"
    }


def example_1_basic_usage():
    """Example 1: Basic SpanCollector usage"""
    print("\n=== Example 1: Basic SpanCollector Usage ===\n")
    
    client = get_client()
    
    # Create spans in a collector context
    with client.get_span_collector("trace-basic-example") as collector:
        print("Creating spans...")
        
        # Create multiple spans - they go to local queue, not exported yet
        collector.create_span(
            "step1_data_loading",
            attributes={
                "status": "completed",
                "latency_ms": 150,
                "records_loaded": 1000
            }
        )
        
        collector.create_span(
            "step2_processing",
            attributes={
                "status": "completed",
                "latency_ms": 450,
                "records_processed": 1000
            }
        )
        
        collector.create_span(
            "step3_saving",
            attributes={
                "status": "completed",
                "latency_ms": 200,
                "records_saved": 1000
            }
        )
        
        # Check how many spans we collected
        print(f"Collected {collector.get_span_count()} spans")
        
        # Export all spans as a single batch
        print("Exporting all spans as a batch...")
        success = collector.export_spans()
        print(f"Export {'succeeded' if success else 'failed'}")
    
    print("✅ Example 1 complete\n")


def example_2_async_span_creation():
    """Example 2: Async span creation (create spans after execution)"""
    print("\n=== Example 2: Async Span Creation ===\n")
    
    client = get_client()
    
    # Phase 1: Execute workflows (no tracing context)
    print("Phase 1: Executing workflows without creating spans...")
    results = []
    for i in range(3):
        result = simulate_workflow(f"workflow_{i}", duration=0.05)
        results.append(result)
    
    print(f"Executed {len(results)} workflows")
    
    # Phase 2: Create spans from results and export as batch
    print("\nPhase 2: Creating spans from results...")
    with client.get_span_collector("trace-async-example") as collector:
        for result in results:
            collector.create_span(
                f"workflow_{result['workflow_id']}",
                attributes={
                    "input": result["input"],
                    "output": result["output"],
                    "duration": result["duration"],
                    "status": result["status"],
                }
            )
        
        print(f"Created {collector.get_span_count()} spans from {len(results)} results")
        
        # Single batch export
        print("Exporting all spans as a batch...")
        collector.export_spans()
    
    print("✅ Example 2 complete\n")


def example_3_span_inspection():
    """Example 3: Inspecting spans before export"""
    print("\n=== Example 3: Span Inspection ===\n")
    
    client = get_client()
    
    with client.get_span_collector("trace-inspection-example") as collector:
        # Create some spans
        collector.create_span("task1", {"status": "completed", "score": 0.95})
        collector.create_span("task2", {"status": "failed", "score": 0.45})
        collector.create_span("task3", {"status": "completed", "score": 0.88})
        
        # Inspect spans before export
        all_spans = collector.get_all_spans()
        print(f"Collected {len(all_spans)} spans:")
        for span in all_spans:
            print(f"  - {span.name}")
        
        # Decide whether to export based on inspection
        print("\nExporting spans...")
        collector.export_spans()
    
    print("✅ Example 3 complete\n")


def example_4_isolation():
    """Example 4: SpanCollector isolation from normal spans"""
    print("\n=== Example 4: Isolation from Normal Spans ===\n")
    
    client = get_client()
    tracer = client.get_tracer()
    
    # Create a normal span (will be exported immediately)
    print("Creating normal span (exported immediately)...")
    with tracer.start_as_current_span("normal_span") as span:
        span.set_attribute("type", "normal")
        span.set_attribute("exported", "immediately")
        time.sleep(0.01)
    
    print("Normal span exported ✓")
    
    # Create spans in collector (batched, not exported yet)
    print("\nCreating spans in collector (batched)...")
    with client.get_span_collector("trace-isolation-example") as collector:
        collector.create_span("collected_span_1", {"type": "collected"})
        collector.create_span("collected_span_2", {"type": "collected"})
        
        # Create another normal span while collector is active
        print("Creating another normal span (still exported immediately)...")
        with tracer.start_as_current_span("another_normal_span") as span:
            span.set_attribute("type", "normal")
            span.set_attribute("exported", "immediately")
            time.sleep(0.01)
        
        print(f"\nCollector has {collector.get_span_count()} spans")
        print("Exporting collector spans as batch...")
        collector.export_spans()
    
    print("✅ Example 4 complete - shows that normal spans are unaffected\n")


def example_5_multiple_collectors():
    """Example 5: Multiple collectors can coexist"""
    print("\n=== Example 5: Multiple Collectors ===\n")
    
    client = get_client()
    
    # Collector 1
    print("Creating collector 1...")
    with client.get_span_collector("trace-collector-1") as collector1:
        collector1.create_span("collector1_span1", {"source": "collector1"})
        collector1.create_span("collector1_span2", {"source": "collector1"})
        print(f"Collector 1 has {collector1.get_span_count()} spans")
        collector1.export_spans()
    
    # Collector 2 (separate trace)
    print("\nCreating collector 2...")
    with client.get_span_collector("trace-collector-2") as collector2:
        collector2.create_span("collector2_span1", {"source": "collector2"})
        collector2.create_span("collector2_span2", {"source": "collector2"})
        collector2.create_span("collector2_span3", {"source": "collector2"})
        print(f"Collector 2 has {collector2.get_span_count()} spans")
        collector2.export_spans()
    
    print("✅ Example 5 complete\n")


if __name__ == "__main__":
    print("=" * 60)
    print("SpanCollector Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        example_1_basic_usage()
        example_2_async_span_creation()
        example_3_span_inspection()
        example_4_isolation()
        example_5_multiple_collectors()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
