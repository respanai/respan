"""
Standard OpenTelemetry Multi-Exporter Example

This example demonstrates the STANDARD OTEL way to use multiple exporters:
- Multiple processors added via add_span_processor (OTEL standard)
- Filtering via span attributes (OTEL standard)
- No custom routing processor needed!

Use case: Send some spans to production, some to local debug file
"""

import json
import time
from typing import Sequence
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from respan_tracing import RespanTelemetry
from respan_tracing.exporters import RespanSpanExporter


class FileExporter(SpanExporter):
    """Simple file exporter for debugging"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to a JSON file"""
        try:
            # Read existing data
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = []
            
            # Append new spans
            for span in spans:
                span_dict = {
                    "name": span.name,
                    "trace_id": format(span.context.trace_id, '032x'),
                    "span_id": format(span.context.span_id, '016x'),
                    "parent_id": format(span.parent.span_id, '016x') if span.parent else None,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "attributes": dict(span.attributes) if span.attributes else {},
                }
                data.append(span_dict)
            
            # Write back
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Exported {len(spans)} spans to {self.filepath}")
            return SpanExportResult.SUCCESS
            
        except Exception as e:
            print(f"❌ Error exporting to file: {e}")
            return SpanExportResult.FAILURE
    
    def shutdown(self):
        """Shutdown the exporter"""
        pass


def main():
    print("=" * 60)
    print("STANDARD OTEL MULTI-EXPORTER EXAMPLE")
    print("=" * 60)
    
    # Step 1: Initialize telemetry WITHOUT exporters
    print("\n📦 Initializing telemetry...")
    kai = RespanTelemetry(
        app_name="multi-exporter-example",
        api_key="test-key-123",
        log_level="DEBUG"
    )
    
    # Step 2: Add production exporter (all spans)
    print("\n🌐 Adding production exporter (all spans)...")
    kai.add_processor(
        exporter=RespanSpanExporter(
            endpoint="https://api.respan.ai/api",
            api_key="test-key-123"
        ),
        name="production"
        # No filter_fn = all spans go here
    )
    
    # Step 3: Add debug file exporter (only debug spans)
    print("📝 Adding debug file exporter (only spans with processor='debug')...")
    kai.add_processor(
        exporter=FileExporter("./debug_traces.json"),
        name="debug",
        filter_fn=lambda span: span.attributes.get("processor") == "debug"
    )
    
    # Step 4: Add analytics exporter (only analytics spans)
    print("📊 Adding analytics file exporter (only spans with processor='analytics')...")
    kai.add_processor(
        exporter=FileExporter("./analytics_traces.json"),
        name="analytics",
        filter_fn=lambda span: span.attributes.get("processor") == "analytics"
    )
    
    print("\n" + "=" * 60)
    print("RUNNING EXAMPLE TASKS")
    print("=" * 60)
    
    # Example 1: Production task (goes to production only)
    @kai.task(name="production_task")
    def production_task():
        print("\n🏭 Running production task...")
        time.sleep(0.1)
        return "production result"
    
    # Example 2: Debug task (goes to production AND debug file)
    @kai.task(name="debug_task", processor="debug")
    def debug_task():
        print("🐛 Running debug task...")
        time.sleep(0.1)
        return "debug result"
    
    # Example 3: Analytics task (goes to production AND analytics file)
    @kai.task(name="analytics_task", processor="analytics")
    def analytics_task():
        print("📊 Running analytics task...")
        time.sleep(0.1)
        return "analytics result"
    
    # Example 4: Workflow with mixed tasks
    @kai.workflow(name="main_workflow")
    def main_workflow():
        print("\n🔄 Running main workflow...")
        
        # This goes to production only
        prod_result = production_task()
        print(f"   Production result: {prod_result}")
        
        # This goes to production AND debug file
        debug_result = debug_task()
        print(f"   Debug result: {debug_result}")
        
        # This goes to production AND analytics file
        analytics_result = analytics_task()
        print(f"   Analytics result: {analytics_result}")
        
        return {
            "prod": prod_result,
            "debug": debug_result,
            "analytics": analytics_result
        }
    
    # Run the workflow
    result = main_workflow()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n✅ Workflow completed: {result}")
    
    # Flush to ensure all spans are exported
    print("\n💾 Flushing spans...")
    time.sleep(1)  # Give time for async export
    
    print("\n" + "=" * 60)
    print("CHECK THE FILES")
    print("=" * 60)
    print("""
Expected results:
1. Production exporter: Received ALL 4 spans (workflow + 3 tasks)
2. debug_traces.json: Contains ONLY debug_task span
3. analytics_traces.json: Contains ONLY analytics_task span

This is the STANDARD OTEL pattern - much simpler than custom routing!
    """)


if __name__ == "__main__":
    main()

