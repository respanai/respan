#!/usr/bin/env python3
"""
Example demonstrating the new Respan client API for trace operations.

This example shows how to use the get_client() function and RespanClient
to interact with the current trace/span context.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the new client API
from respan_tracing import RespanTelemetry, get_client, workflow, task
from opentelemetry.trace import StatusCode

# Initialize telemetry
telemetry = RespanTelemetry(
    app_name="client-example",
    api_key=os.getenv("RESPAN_API_KEY", "test-key"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    is_enabled=True
)

print("🚀 Respan Client API Example")
print("=" * 50)


@workflow(name="data_processing_workflow")
def data_processing_workflow(data):
    """Workflow demonstrating client API usage"""
    
    # Get the global client - no need to pass around instances!
    client = get_client()
    
    print(f"📊 Processing data: {data}")
    
    # Get current trace information
    trace_id = client.get_current_trace_id()
    span_id = client.get_current_span_id()
    
    print(f"🔍 Current trace ID: {trace_id}")
    print(f"🔍 Current span ID: {span_id}")
    
    # Update the current span with Respan parameters
    client.update_current_span(
        respan_params={
            "trace_group_identifier": "data-processing-group",
            "metadata": {
                "data_size": len(str(data)),
                "processing_type": "batch"
            }
        },
        attributes={
            "custom.input_data": str(data),
            "custom.processing_stage": "validation"
        }
    )
    
    # Add an event to track progress
    client.add_event("validation_started", {
        "input_length": len(str(data))
    })
    
    # Simulate some processing
    validated_data = validate_data(data)
    
    # Update span name and add more attributes
    client.update_current_span(
        name="data_processing_workflow.completed",
        attributes={
            "custom.processing_stage": "completed",
            "custom.output_size": len(str(validated_data))
        }
    )
    
    return validated_data


@task(name="validate_data")
def validate_data(data):
    """Task demonstrating error handling with client API"""
    
    client = get_client()
    
    try:
        # Add event for processing start
        client.add_event("validation_process_started")
        
        # Simulate validation logic
        if not data or len(str(data)) < 3:
            raise ValueError("Data is too short for processing")
        
        # Update span with success status
        client.update_current_span(
            status=StatusCode.OK,
            attributes={
                "validation.result": "success",
                "validation.data_length": len(str(data))
            }
        )
        
        client.add_event("validation_completed", {
            "result": "success"
        })
        
        return f"validated_{data}"
        
    except Exception as e:
        # Record the exception and set error status
        client.record_exception(e)
        
        # Add additional context
        client.update_current_span(
            attributes={
                "validation.result": "failed",
                "validation.error_type": type(e).__name__
            }
        )
        
        client.add_event("validation_failed", {
            "error": str(e)
        })
        
        # Re-raise the exception
        raise


def demonstrate_context_operations():
    """Demonstrate context value operations"""
    
    client = get_client()
    
    print("\n🔧 Context Operations Demo")
    print("-" * 30)
    
    # Set some context values
    client.set_context_value("custom_session_id", "session_123")
    client.set_context_value("user_id", "user_456")
    
    # Retrieve context values
    session_id = client.get_context_value("custom_session_id")
    user_id = client.get_context_value("user_id")
    
    print(f"Session ID from context: {session_id}")
    print(f"User ID from context: {user_id}")
    
    # Check if we're currently recording
    is_recording = client.is_recording()
    print(f"Currently recording spans: {is_recording}")


@workflow(name="error_demo_workflow")
def error_demo_workflow():
    """Demonstrate error handling"""
    
    client = get_client()
    
    try:
        # This will trigger an error in validation
        return validate_data("x")  # Too short
        
    except ValueError as e:
        # The exception was already recorded in validate_data
        # But we can add workflow-level context
        client.update_current_span(
            attributes={
                "workflow.error_handled": True,
                "workflow.recovery_attempted": False
            }
        )
        
        return f"Error handled: {str(e)}"


def main():
    """Run all examples"""
    
    print("\n1️⃣ Successful Processing Example")
    print("-" * 40)
    result1 = data_processing_workflow("hello world")
    print(f"✅ Result: {result1}")
    
    print("\n2️⃣ Error Handling Example")
    print("-" * 40)
    result2 = error_demo_workflow()
    print(f"⚠️  Result: {result2}")
    
    # Demonstrate context operations outside of spans
    demonstrate_context_operations()
    
    print("\n✅ All examples completed!")
    print("\n📊 Key Client API Features Demonstrated:")
    print("  • get_client() - Global access without instance management")
    print("  • get_current_trace_id() / get_current_span_id() - Trace info")
    print("  • update_current_span() - Flexible span updates")
    print("  • add_event() - Span events for tracking")
    print("  • record_exception() - Automatic error recording")
    print("  • Context value get/set operations")
    print("  • Automatic Respan parameter handling")
    
    # Flush any remaining spans


if __name__ == "__main__":
    main() 