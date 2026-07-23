"""
Example: Using Custom Exporter with Respan Telemetry

This example demonstrates how to create and use a custom exporter
instead of the default HTTP OTLP exporter. This is useful for:
- Internal integrations that need custom export logic
- Sending spans to custom backends
- Logging spans to files or databases
- Testing and debugging
"""

import json
from respan_tracing import RespanTelemetry, Instruments
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class CustomFileExporter(SpanExporter):
    """
    Custom exporter that writes spans to a JSON file
    instead of sending them to Respan API
    """
    
    def __init__(self, filename: str = "spans.jsonl"):
        self.filename = filename
        print(f"CustomFileExporter initialized, writing to {filename}")
    
    def export(self, spans):
        """Export spans to a JSON lines file"""
        try:
            with open(self.filename, 'a') as f:
                for span in spans:
                    # Convert span to a dictionary with relevant information
                    span_dict = {
                        "name": span.name,
                        "trace_id": format(span.context.trace_id, '032x'),
                        "span_id": format(span.context.span_id, '016x'),
                        "parent_span_id": format(span.parent.span_id, '016x') if span.parent else None,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "attributes": dict(span.attributes) if span.attributes else {},
                        "status": {
                            "status_code": span.status.status_code.name,
                            "description": span.status.description
                        },
                        "events": [
                            {
                                "name": event.name,
                                "timestamp": event.timestamp,
                                "attributes": dict(event.attributes) if event.attributes else {}
                            }
                            for event in span.events
                        ]
                    }
                    
                    # Write as JSON line
                    f.write(json.dumps(span_dict) + '\n')
            
            print(f"Exported {len(spans)} span(s) to {self.filename}")
            return SpanExportResult.SUCCESS
            
        except Exception as e:
            print(f"Error exporting spans: {e}")
            return SpanExportResult.FAILURE
    
    def shutdown(self):
        """Cleanup when exporter is shut down"""
        print(f"CustomFileExporter shutdown")
    
    def force_flush(self, timeout_millis=30000):
        """Force flush (not needed for file exporter)"""
        return True


class CustomConsoleExporter(SpanExporter):
    """
    Simple exporter that prints spans to console
    """
    
    def export(self, spans):
        """Print spans to console"""
        print(f"\n{'='*60}")
        print(f"Exporting {len(spans)} span(s)")
        print(f"{'='*60}")
        
        for span in spans:
            print(f"\nSpan: {span.name}")
            print(f"  Trace ID: {format(span.context.trace_id, '032x')}")
            print(f"  Span ID: {format(span.context.span_id, '016x')}")
            if span.parent:
                print(f"  Parent Span ID: {format(span.parent.span_id, '016x')}")
            print(f"  Duration: {(span.end_time - span.start_time) / 1e9:.3f}s")
            
            if span.attributes:
                print(f"  Attributes:")
                for key, value in span.attributes.items():
                    print(f"    {key}: {value}")
            
            if span.events:
                print(f"  Events:")
                for event in span.events:
                    print(f"    {event.name}")
        
        print(f"\n{'='*60}\n")
        return SpanExportResult.SUCCESS
    
    def shutdown(self):
        pass
    
    def force_flush(self, timeout_millis=30000):
        return True


def example_file_exporter():
    """Example: Export spans to a file"""
    print("\n--- Example 1: File Exporter ---")
    
    # Create custom file exporter
    file_exporter = CustomFileExporter("my_traces.jsonl")
    
    # Initialize telemetry with custom exporter
    telemetry = RespanTelemetry(
        app_name="file-export-example",
        custom_exporter=file_exporter,
        block_instruments={Instruments.THREADING},  # Simplify for example  # Simplify for example
    )
    
    # Create some spans
    tracer = telemetry.tracer.get_tracer()
    
    with tracer.start_as_current_span("parent_operation") as parent_span:
        parent_span.set_attribute("operation_type", "data_processing")
        parent_span.add_event("Starting data processing")
        
        with tracer.start_as_current_span("child_operation_1") as child1:
            child1.set_attribute("step", 1)
            child1.set_attribute("description", "Load data")
        
        with tracer.start_as_current_span("child_operation_2") as child2:
            child2.set_attribute("step", 2)
            child2.set_attribute("description", "Process data")
    
    # Flush to ensure all spans are exported
    print(f"\nCheck 'my_traces.jsonl' for exported spans\n")


def example_console_exporter():
    """Example: Print spans to console"""
    print("\n--- Example 2: Console Exporter ---")
    
    # Create custom console exporter
    console_exporter = CustomConsoleExporter()
    
    # Initialize telemetry with custom exporter
    # Note: is_batching_enabled=False ensures immediate export for demonstration
    telemetry = RespanTelemetry(
        app_name="console-export-example",
        custom_exporter=console_exporter,
        is_batching_enabled=False,  # Export immediately for demo
        block_instruments={Instruments.THREADING},  # Simplify for example
    )
    
    # Create some spans
    tracer = telemetry.tracer.get_tracer()
    
    with tracer.start_as_current_span("api_request") as span:
        span.set_attribute("method", "POST")
        span.set_attribute("endpoint", "/api/data")
        span.add_event("Request started")
        
        # Simulate some work
        import time
        time.sleep(0.1)
        
        span.add_event("Request completed")
    
    # Flush to ensure export


def example_custom_backend():
    """Example: Send to custom backend (simulated)"""
    print("\n--- Example 3: Custom Backend Exporter ---")
    
    class CustomBackendExporter(SpanExporter):
        """Simulates sending to a custom backend"""
        
        def export(self, spans):
            print(f"Sending {len(spans)} span(s) to custom backend...")
            
            for span in spans:
                # In real implementation, you would send to your backend
                # e.g., requests.post(your_backend_url, json=span_data)
                print(f"  - Sent span: {span.name}")
            
            return SpanExportResult.SUCCESS
        
        def shutdown(self):
            pass
        
        def force_flush(self, timeout_millis=30000):
            return True
    
    # Initialize with custom backend exporter
    telemetry = RespanTelemetry(
        app_name="custom-backend-example",
        custom_exporter=CustomBackendExporter(),
        block_instruments={Instruments.THREADING},  # Simplify for example
    )
    
    # Create spans
    tracer = telemetry.tracer.get_tracer()
    
    with tracer.start_as_current_span("database_query") as span:
        span.set_attribute("query", "SELECT * FROM users")
        span.set_attribute("database", "postgres")
    


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Custom Exporter Examples")
    print("="*60)
    
    # Run examples
    example_file_exporter()
    example_console_exporter()
    example_custom_backend()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")

