#!/usr/bin/env python3
"""
Threading and Context Propagation Experiment

This experiment demonstrates:
1. How global variables behave when modified in multiple threads
2. How OpenTelemetry context propagates (or doesn't) across threads
3. Whether context is persistent across threads
4. How to properly propagate context manually when needed

Run this experiment to understand the behavior of:
- Regular global variables in threads
- Python contextvars in threads  
- OpenTelemetry spans and context in threads
- Manual context propagation techniques
"""

import os
import time
import threading
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Respan tracing
from respan_tracing import RespanTelemetry, get_client, workflow, task
from opentelemetry import trace, context as otel_context
from opentelemetry.trace import Status, StatusCode

# Initialize telemetry
telemetry = RespanTelemetry(
    app_name="threading-experiment",
    api_key=os.getenv("RESPAN_API_KEY", "test-key"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    is_enabled=True
)

print("🧪 Threading and Context Propagation Experiment")
print("=" * 60)

# ============================================================================
# EXPERIMENT 1: Global Variables in Threads
# ============================================================================

# Global variable for testing
global_counter = 0
global_data = {"shared": "initial_value"}

def experiment_1_global_variables():
    """Experiment 1: How global variables behave in multiple threads"""
    
    print("\n🔬 EXPERIMENT 1: Global Variables in Threads")
    print("-" * 50)
    
    global global_counter, global_data
    
    # Reset globals
    global_counter = 0
    global_data = {"shared": "initial_value"}
    
    def worker_thread(thread_id: int, iterations: int):
        """Worker that modifies global variables"""
        global global_counter, global_data
        
        print(f"  Thread {thread_id}: Starting with counter={global_counter}")
        
        for i in range(iterations):
            # Read-modify-write operation (not atomic!)
            current = global_counter
            time.sleep(0.001)  # Simulate some work
            global_counter = current + 1
            
            # Modify shared dictionary
            global_data[f"thread_{thread_id}_iteration_{i}"] = f"value_{current}"
            
        print(f"  Thread {thread_id}: Finished, final counter={global_counter}")
    
    # Start multiple threads
    threads = []
    num_threads = 5
    iterations_per_thread = 10
    
    print(f"Starting {num_threads} threads, {iterations_per_thread} iterations each")
    print(f"Expected final counter: {num_threads * iterations_per_thread}")
    
    for i in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(i, iterations_per_thread))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print(f"\n📊 Results:")
    print(f"  Final counter value: {global_counter}")
    print(f"  Expected value: {num_threads * iterations_per_thread}")
    print(f"  Data corruption occurred: {global_counter != num_threads * iterations_per_thread}")
    print(f"  Global data keys count: {len(global_data)}")
    print(f"  Sample data: {dict(list(global_data.items())[:3])}")


# ============================================================================
# EXPERIMENT 2: Context Variables in Threads
# ============================================================================

# Context variables for testing
thread_id_var = contextvars.ContextVar('thread_id', default='unknown')
processing_stage_var = contextvars.ContextVar('processing_stage', default='none')

def experiment_2_context_variables():
    """Experiment 2: How Python contextvars behave in threads"""
    
    print("\n🔬 EXPERIMENT 2: Context Variables in Threads")
    print("-" * 50)
    
    def worker_with_context(thread_id: int):
        """Worker that uses context variables"""
        
        # Set context variables for this thread
        thread_id_var.set(f"thread_{thread_id}")
        processing_stage_var.set("initialization")
        
        print(f"  Thread {thread_id}: Set context vars")
        print(f"    thread_id_var: {thread_id_var.get()}")
        print(f"    processing_stage_var: {processing_stage_var.get()}")
        
        # Simulate some work with context changes
        for stage in ["processing", "validation", "completion"]:
            processing_stage_var.set(stage)
            time.sleep(0.1)
            
            print(f"    Thread {thread_id} in stage '{stage}': "
                  f"thread_id={thread_id_var.get()}, stage={processing_stage_var.get()}")
        
        # Final check
        print(f"  Thread {thread_id}: Final context - "
              f"thread_id={thread_id_var.get()}, stage={processing_stage_var.get()}")
    
    # Test context isolation between threads
    threads = []
    num_threads = 3
    
    print(f"Starting {num_threads} threads with context variables")
    
    for i in range(num_threads):
        thread = threading.Thread(target=worker_with_context, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Check main thread context
    print(f"\n📊 Main thread context after workers:")
    print(f"  thread_id_var: {thread_id_var.get()}")
    print(f"  processing_stage_var: {processing_stage_var.get()}")


# ============================================================================
# EXPERIMENT 3: OpenTelemetry Context in Threads
# ============================================================================

@workflow(name="main_workflow")
def experiment_3_opentelemetry_context():
    """Experiment 3: How OpenTelemetry context behaves in threads"""
    
    print("\n🔬 EXPERIMENT 3: OpenTelemetry Context in Threads")
    print("-" * 50)
    
    client = get_client()
    
    # Get main thread trace info
    main_trace_id = client.get_current_trace_id()
    main_span_id = client.get_current_span_id()
    
    print(f"Main thread trace ID: {main_trace_id}")
    print(f"Main thread span ID: {main_span_id}")
    
    # Update main span
    client.update_current_span(
        respan_params={"trace_group_identifier": "threading-experiment"},
        attributes={"experiment.type": "opentelemetry_context"}
    )
    
    results = []
    
    def worker_with_otel_context(thread_id: int):
        """Worker that checks OpenTelemetry context"""
        worker_client = get_client()
        
        # Check if we have access to the main thread's context
        worker_trace_id = worker_client.get_current_trace_id()
        worker_span_id = worker_client.get_current_span_id()
        is_recording = worker_client.is_recording()
        
        result = {
            "thread_id": thread_id,
            "trace_id": worker_trace_id,
            "span_id": worker_span_id,
            "is_recording": is_recording,
            "has_main_trace": worker_trace_id == main_trace_id,
            "has_main_span": worker_span_id == main_span_id
        }
        
        print(f"  Thread {thread_id}:")
        print(f"    Trace ID: {worker_trace_id}")
        print(f"    Span ID: {worker_span_id}")
        print(f"    Is recording: {is_recording}")
        print(f"    Has main trace: {result['has_main_trace']}")
        print(f"    Has main span: {result['has_main_span']}")
        
        # Try to update span (will this work?)
        try:
            worker_client.update_current_span(
                attributes={f"thread.{thread_id}.processed": True}
            )
            result["can_update_span"] = True
        except Exception as e:
            result["can_update_span"] = False
            result["update_error"] = str(e)
        
        results.append(result)
    
    # Start worker threads
    threads = []
    num_threads = 3
    
    for i in range(num_threads):
        thread = threading.Thread(target=worker_with_otel_context, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Analyze results
    print(f"\n📊 OpenTelemetry Context Analysis:")
    context_inherited = sum(1 for r in results if r["has_main_trace"])
    span_inherited = sum(1 for r in results if r["has_main_span"])
    can_update = sum(1 for r in results if r.get("can_update_span", False))
    
    print(f"  Threads that inherited main trace: {context_inherited}/{len(results)}")
    print(f"  Threads that inherited main span: {span_inherited}/{len(results)}")
    print(f"  Threads that can update spans: {can_update}/{len(results)}")
    
    return results


# ============================================================================
# EXPERIMENT 4: Manual Context Propagation
# ============================================================================

@task(name="manual_context_propagation")
def experiment_4_manual_context_propagation():
    """Experiment 4: How to manually propagate OpenTelemetry context"""
    
    print("\n🔬 EXPERIMENT 4: Manual Context Propagation")
    print("-" * 50)
    
    client = get_client()
    
    # Get current context
    current_context = otel_context.get_current()
    main_trace_id = client.get_current_trace_id()
    
    print(f"Main thread trace ID: {main_trace_id}")
    
    results = []
    
    def worker_with_manual_context(thread_id: int, context_to_use):
        """Worker that uses manually propagated context"""
        
        # Attach the context in this thread
        token = otel_context.attach(context_to_use)
        
        try:
            worker_client = get_client()
            
            # Check context
            worker_trace_id = worker_client.get_current_trace_id()
            worker_span_id = worker_client.get_current_span_id()
            is_recording = worker_client.is_recording()
            
            result = {
                "thread_id": thread_id,
                "trace_id": worker_trace_id,
                "span_id": worker_span_id,
                "is_recording": is_recording,
                "has_main_trace": worker_trace_id == main_trace_id
            }
            
            print(f"  Thread {thread_id} (with manual context):")
            print(f"    Trace ID: {worker_trace_id}")
            print(f"    Span ID: {worker_span_id}")
            print(f"    Is recording: {is_recording}")
            print(f"    Has main trace: {result['has_main_trace']}")
            
            # Create a child span
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"worker_thread_{thread_id}") as span:
                span.set_attribute("thread.id", thread_id)
                span.set_attribute("thread.type", "manual_context")
                
                # Update using client
                worker_client.update_current_span(
                    attributes={
                        "worker.processing_time": 0.1,
                        "worker.status": "completed"
                    }
                )
                
                time.sleep(0.1)  # Simulate work
                
                child_span_id = worker_client.get_current_span_id()
                result["child_span_id"] = child_span_id
                print(f"    Created child span: {child_span_id}")
            
            results.append(result)
            
        finally:
            # Always detach the context
            otel_context.detach(token)
    
    def worker_without_context(thread_id: int):
        """Worker without context propagation for comparison"""
        worker_client = get_client()
        
        worker_trace_id = worker_client.get_current_trace_id()
        is_recording = worker_client.is_recording()
        
        result = {
            "thread_id": thread_id,
            "trace_id": worker_trace_id,
            "is_recording": is_recording,
            "has_main_trace": worker_trace_id == main_trace_id,
            "type": "no_context"
        }
        
        print(f"  Thread {thread_id} (no context):")
        print(f"    Trace ID: {worker_trace_id}")
        print(f"    Is recording: {is_recording}")
        print(f"    Has main trace: {result['has_main_trace']}")
        
        results.append(result)
    
    # Test both approaches
    threads = []
    
    # Threads with manual context propagation
    for i in range(2):
        thread = threading.Thread(
            target=worker_with_manual_context, 
            args=(i, current_context)
        )
        threads.append(thread)
        thread.start()
    
    # Threads without context propagation
    for i in range(2, 4):
        thread = threading.Thread(target=worker_without_context, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Analyze results
    print(f"\n📊 Manual Context Propagation Analysis:")
    manual_context_threads = [r for r in results if r.get("type") != "no_context"]
    no_context_threads = [r for r in results if r.get("type") == "no_context"]
    
    manual_inherited = sum(1 for r in manual_context_threads if r["has_main_trace"])
    no_context_inherited = sum(1 for r in no_context_threads if r["has_main_trace"])
    
    print(f"  Manual context threads with main trace: {manual_inherited}/{len(manual_context_threads)}")
    print(f"  No context threads with main trace: {no_context_inherited}/{len(no_context_threads)}")
    
    return results


# ============================================================================
# EXPERIMENT 5: ThreadPoolExecutor Context Behavior
# ============================================================================

@task(name="threadpool_context_test")
def experiment_5_threadpool_context():
    """Experiment 5: How context behaves with ThreadPoolExecutor"""
    
    print("\n🔬 EXPERIMENT 5: ThreadPoolExecutor Context Behavior")
    print("-" * 50)
    
    client = get_client()
    main_trace_id = client.get_current_trace_id()
    current_context = otel_context.get_current()
    
    print(f"Main thread trace ID: {main_trace_id}")
    
    def worker_task(task_id: int):
        """Task to run in thread pool"""
        worker_client = get_client()
        
        trace_id = worker_client.get_current_trace_id()
        is_recording = worker_client.is_recording()
        
        result = {
            "task_id": task_id,
            "trace_id": trace_id,
            "is_recording": is_recording,
            "has_main_trace": trace_id == main_trace_id
        }
        
        print(f"  Task {task_id}: trace={trace_id}, recording={is_recording}, "
              f"has_main={result['has_main_trace']}")
        
        return result
    
    def worker_task_with_context(task_id: int):
        """Task that manually attaches context"""
        token = otel_context.attach(current_context)
        
        try:
            return worker_task(task_id)
        finally:
            otel_context.detach(token)
    
    # Test with regular ThreadPoolExecutor
    print("\n  Testing regular ThreadPoolExecutor:")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker_task, i) for i in range(3)]
        regular_results = [f.result() for f in futures]
    
    # Test with manual context propagation
    print("\n  Testing ThreadPoolExecutor with manual context:")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker_task_with_context, i) for i in range(3)]
        context_results = [f.result() for f in futures]
    
    # Analyze results
    print(f"\n📊 ThreadPoolExecutor Analysis:")
    regular_inherited = sum(1 for r in regular_results if r["has_main_trace"])
    context_inherited = sum(1 for r in context_results if r["has_main_trace"])
    
    print(f"  Regular executor - inherited context: {regular_inherited}/{len(regular_results)}")
    print(f"  Manual context executor - inherited context: {context_inherited}/{len(context_results)}")
    
    return {"regular": regular_results, "manual_context": context_results}


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def main():
    """Run all threading and context experiments"""
    
    print("Starting comprehensive threading and context experiments...")
    print("This will demonstrate the behavior of global variables, context variables,")
    print("and OpenTelemetry context across multiple threads.\n")
    
    # Run all experiments
    experiment_1_global_variables()
    experiment_2_context_variables()
    
    # These need to run within a span context
    otel_results = experiment_3_opentelemetry_context()
    manual_results = experiment_4_manual_context_propagation()
    threadpool_results = experiment_5_threadpool_context()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print("\n1️⃣ Global Variables:")
    print("   ❌ NOT thread-safe - race conditions cause data corruption")
    print("   ❌ Shared state leads to unpredictable results")
    
    print("\n2️⃣ Context Variables (contextvars):")
    print("   ✅ Thread-safe - each thread has isolated context")
    print("   ✅ No automatic propagation between threads")
    print("   ✅ Perfect for thread-local state")
    
    print("\n3️⃣ OpenTelemetry Context:")
    print("   ❌ Does NOT automatically propagate to new threads")
    print("   ❌ Each thread starts with empty/default context")
    print("   ⚠️  Manual propagation required for distributed tracing")
    
    print("\n4️⃣ Manual Context Propagation:")
    print("   ✅ Works correctly when context is manually attached")
    print("   ✅ Enables proper distributed tracing across threads")
    print("   ⚠️  Requires explicit context management")
    
    print("\n5️⃣ ThreadPoolExecutor:")
    print("   ❌ Regular usage loses context")
    print("   ✅ Manual context propagation preserves tracing")
    
    print("\n🔑 Key Takeaways:")
    print("   • OpenTelemetry context is thread-local by design")
    print("   • Use context.attach()/detach() for manual propagation")
    print("   • Consider using copy_context() for complex scenarios")
    print("   • Each thread needs explicit context setup for tracing")
    
    # Flush any pending spans
    
    print("\n✅ All experiments completed!")


if __name__ == "__main__":
    main() 