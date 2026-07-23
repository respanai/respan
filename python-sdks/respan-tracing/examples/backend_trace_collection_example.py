"""
Backend Custom Exporter Examples

Two production-ready patterns for backend systems that need custom logging:

1. DirectLoggingExporter (RECOMMENDED)
   - No memory storage, no OOM risk
   - Logs each span immediately to your backend
   - Simple and safe

2. SafeTraceCollector (ADVANCED)
   - Collects complete traces with automatic cleanup
   - Bounded memory (max_traces limit)
   - Use only if you need complete traces together
"""

from typing import List, Dict, Any, Sequence
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from respan_tracing import RespanTelemetry, workflow, task
import threading
import time


# ============================================================================
# Pattern 1: DirectLoggingExporter (RECOMMENDED)
# ============================================================================

class DirectLoggingExporter(SpanExporter):
    """
    Logs spans DIRECTLY without storing in memory.
    
    ✅ RECOMMENDED for most backends!
    
    Safety:
    - No memory leaks (spans never stored)
    - No manual cleanup needed
    - Simple and production-ready
    
    Usage:
        exporter = DirectLoggingExporter(log_function=your_logger)
        telemetry = RespanTelemetry(
            custom_exporter=exporter,
            is_batching_enabled=False,
        )
    """
    
    def __init__(self, log_function=None):
        """
        Args:
            log_function: Function to call for each span.
                         Signature: log_function(span_data: Dict)
        """
        self.log_function = log_function or self._default_log
        self._lock = threading.Lock()
    
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans by logging them directly (no storage)"""
        with self._lock:
            for span in spans:
                span_data = {
                    "name": span.name,
                    "trace_id": format(span.context.trace_id, '032x'),
                    "span_id": format(span.context.span_id, '016x'),
                    "parent_span_id": format(span.parent.span_id, '016x') if span.parent else None,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "duration_ms": (span.end_time - span.start_time) / 1_000_000,
                    "attributes": dict(span.attributes) if span.attributes else {},
                    "status": {
                        "status_code": span.status.status_code.name,
                        "description": span.status.description
                    },
                }
                
                # Log directly - no storage!
                self.log_function(span_data)
        
        return SpanExportResult.SUCCESS
    
    def _default_log(self, span_data: Dict[str, Any]):
        """Default logging (replace with your implementation)"""
        print(f"[SPAN] {span_data['name']} ({span_data['duration_ms']:.2f}ms)")
    
    def shutdown(self):
        pass
    
    def force_flush(self, timeout_millis=30000):
        return True


# ============================================================================
# Pattern 2: SafeTraceCollector (ADVANCED - Only if needed)
# ============================================================================

class SafeTraceCollector(SpanExporter):
    """
    Collects complete traces with automatic cleanup.
    
    ⚠️  Use ONLY if you need complete traces together!
    
    Safety features:
    - Automatic cleanup when root span ends
    - Bounded memory (max_traces limit)
    - LRU eviction
    
    Usage:
        collector = SafeTraceCollector(
            log_function=your_trace_logger,
            max_traces=100
        )
        telemetry = RespanTelemetry(
            custom_exporter=collector,
            is_batching_enabled=False,
        )
    """
    
    def __init__(self, log_function=None, max_traces: int = 100):
        """
        Args:
            log_function: Function to call with complete trace.
                         Signature: log_function(trace_data: List[Dict])
            max_traces: Maximum traces in memory (default: 100)
        """
        self.traces: Dict[str, List[ReadableSpan]] = {}
        self.trace_timestamps: Dict[str, float] = {}
        self.log_function = log_function or self._default_log
        self.max_traces = max_traces
        self._lock = threading.Lock()
    
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Store spans and auto-cleanup when trace completes"""
        with self._lock:
            for span in spans:
                trace_id = format(span.context.trace_id, '032x')
                
                # Store span
                if trace_id not in self.traces:
                    self.traces[trace_id] = []
                    self.trace_timestamps[trace_id] = time.time()
                
                self.traces[trace_id].append(span)
                
                # Auto-cleanup: root span ends = trace complete
                if span.parent is None:
                    self._log_and_clear_trace(trace_id)
            
            # Enforce size limit
            self._evict_old_traces()
        
        return SpanExportResult.SUCCESS
    
    def _log_and_clear_trace(self, trace_id: str):
        """Log complete trace and remove from memory"""
        if trace_id not in self.traces:
            return
        
        trace_data = [
            {
                "name": span.name,
                "trace_id": format(span.context.trace_id, '032x'),
                "span_id": format(span.context.span_id, '016x'),
                "parent_span_id": format(span.parent.span_id, '016x') if span.parent else None,
                "start_time": span.start_time,
                "end_time": span.end_time,
                "duration_ms": (span.end_time - span.start_time) / 1_000_000,
                "attributes": dict(span.attributes) if span.attributes else {},
            }
            for span in self.traces[trace_id]
        ]
        
        self.log_function(trace_data)
        
        # Automatic cleanup
        del self.traces[trace_id]
        del self.trace_timestamps[trace_id]
    
    def _evict_old_traces(self):
        """Evict oldest traces if over limit"""
        if len(self.traces) <= self.max_traces:
            return
        
        sorted_traces = sorted(self.trace_timestamps.items(), key=lambda x: x[1])
        traces_to_evict = len(self.traces) - self.max_traces
        
        for trace_id, _ in sorted_traces[:traces_to_evict]:
            self._log_and_clear_trace(trace_id)
    
    def _default_log(self, trace_data: List[Dict[str, Any]]):
        """Default logging (replace with your implementation)"""
        print(f"[TRACE] Complete trace with {len(trace_data)} spans")
    
    def shutdown(self):
        with self._lock:
            for trace_id in list(self.traces.keys()):
                self._log_and_clear_trace(trace_id)
    
    def force_flush(self, timeout_millis=30000):
        return True


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("\nBackend Custom Exporter Examples\n")
    
    # Example 1: DirectLoggingExporter (RECOMMENDED)
    print("="*60)
    print("Example 1: DirectLoggingExporter (RECOMMENDED)")
    print("="*60)
    
    def my_span_logger(span_data):
        """Your backend logging function"""
        # Replace with: database.insert_span(span_data)
        # Or: queue.send(span_data)
        print(f"  Logged: {span_data['name']}")
    
    exporter = DirectLoggingExporter(log_function=my_span_logger)
    telemetry = RespanTelemetry(
        app_name="backend-direct",
        custom_exporter=exporter,
        is_batching_enabled=False,
        instruments=set(),
    )
    
    @workflow(name="example_workflow")
    def example_workflow():
        @task(name="task_1")
        def task_1():
            return "result"
        return task_1()
    
    result = example_workflow()
    
    print("\n✅ Each span logged immediately (no memory storage!)\n")
    
    # Example 2: SafeTraceCollector (ADVANCED)
    print("="*60)
    print("Example 2: SafeTraceCollector (ADVANCED)")
    print("="*60)
    
    def my_trace_logger(trace_data):
        """Your backend trace logging function"""
        # Replace with: database.insert_trace(trace_data)
        print(f"  Logged complete trace: {len(trace_data)} spans")
    
    collector = SafeTraceCollector(
        log_function=my_trace_logger,
        max_traces=100
    )
    
    telemetry2 = RespanTelemetry(
        app_name="backend-collector",
        custom_exporter=collector,
        is_batching_enabled=False,
        instruments=set(),
    )
    
    @workflow(name="example_workflow_2")
    def example_workflow_2():
        @task(name="task_2")
        def task_2():
            return "result"
        return task_2()
    
    result = example_workflow_2()
    
    print("\n✅ Complete trace logged with automatic cleanup!\n")
    
    print("="*60)
    print("Choose the right pattern for your backend:")
    print("  - DirectLoggingExporter: 99% of backends")
    print("  - SafeTraceCollector: Only if you need complete traces")
    print("="*60)
