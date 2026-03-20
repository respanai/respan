"""Respan span handler for LlamaIndex instrumentation system."""
import inspect
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import PrivateAttr

from llama_index.core.instrumentation.span import SimpleSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler

from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

logger = logging.getLogger(__name__)

# Args to skip when falling back to first bound argument
_SKIP_ARG_NAMES = frozenset({"self", "kwargs", "extra_info", "callback_manager", "args"})


class RespanSpanHandler(BaseSpanHandler[SimpleSpan]):
    """
    LlamaIndex span handler that captures trace spans and exports them to Respan.

    Usage::

        from respan_exporter_llamaindex import RespanSpanHandler
        import llama_index.core.instrumentation as instrument

        handler = RespanSpanHandler(api_key="your-respan-api-key")
        dispatcher = instrument.get_dispatcher()
        dispatcher.add_span_handler(handler)
    """

    _exporter: RespanLlamaIndexExporter = PrivateAttr()
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _spans: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    _root_span_ids: set = PrivateAttr(default_factory=set)
    _trace_name: Optional[str] = PrivateAttr(default=None)
    _session_identifier: Optional[Union[str, int]] = PrivateAttr(default=None)

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        environment: Optional[str] = None,
        customer_identifier: Optional[Union[str, int]] = None,
        session_identifier: Optional[Union[str, int]] = None,
        trace_name: Optional[str] = None,
        timeout: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._exporter = RespanLlamaIndexExporter(
            api_key=api_key,
            endpoint=endpoint,
            base_url=base_url,
            environment=environment,
            customer_identifier=customer_identifier,
            timeout=timeout,
        )
        self._lock = threading.Lock()
        self._spans = {}
        self._root_span_ids = set()
        self._trace_name = trace_name
        self._session_identifier = session_identifier

    @classmethod
    def class_name(cls) -> str:
        return "RespanSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[SimpleSpan]:
        """Create a new span record."""
        instance_class = type(instance).__name__ if instance is not None else None
        input_data = self._extract_input(bound_args)

        record: Dict[str, Any] = {
            "span_id": id_,
            "parent_id": parent_span_id,
            "span_name": self._infer_span_name(instance, bound_args),
            "instance_class": instance_class,
            "input": input_data,
            "result": None,
            "start_time": datetime.now(timezone.utc),
            "end_time": None,
            "error": None,
            "metadata": dict(tags) if tags else {},
        }

        with self._lock:
            self._spans[id_] = record
            if parent_span_id is None:
                self._root_span_ids.add(id_)

        return SimpleSpan(id_=id_, parent_id=parent_span_id)

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle span completion."""
        with self._lock:
            record = self._spans.get(id_)
            if record is None:
                return None
            record["result"] = result
            record["end_time"] = datetime.now(timezone.utc)
            is_root = id_ in self._root_span_ids

        if is_root:
            self._flush_trace(id_)

        return None

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle span error/drop."""
        with self._lock:
            record = self._spans.get(id_)
            if record is None:
                return None
            record["error"] = str(err) if err else None
            record["end_time"] = datetime.now(timezone.utc)
            is_root = id_ in self._root_span_ids

        if is_root:
            self._flush_trace(id_)

        return None

    def _flush_trace(self, root_span_id: str) -> None:
        """Collect all spans in the trace and export them."""
        with self._lock:
            trace_span_ids = self._collect_trace_span_ids(root_span_id)
            trace_spans = [self._spans[sid] for sid in trace_span_ids if sid in self._spans]

        if not trace_spans:
            return

        trace_id = root_span_id
        trace_name = self._trace_name or trace_spans[0].get("span_name", root_span_id)

        try:
            self._exporter.export(
                spans=trace_spans,
                trace_id=trace_id,
                trace_name=trace_name,
                session_identifier=self._session_identifier,
            )
        except Exception as exc:
            logger.warning("Failed to export LlamaIndex trace: %s", exc)
            return

        # Only remove spans after successful export to avoid data loss
        with self._lock:
            for sid in trace_span_ids:
                self._spans.pop(sid, None)
                self._root_span_ids.discard(sid)

    def _collect_trace_span_ids(self, root_id: str) -> List[str]:
        """Collect all span IDs belonging to a trace (root + descendants)."""
        result = [root_id]
        visited = {root_id}
        children_map: Dict[str, List[str]] = {}
        for sid, record in self._spans.items():
            pid = record.get("parent_id")
            if pid:
                children_map.setdefault(pid, []).append(sid)

        queue = deque([root_id])
        while queue:
            current = queue.popleft()
            for child in children_map.get(current, []):
                if child not in visited:
                    visited.add(child)
                    result.append(child)
                    queue.append(child)

        return result

    @staticmethod
    def _extract_input(bound_args: inspect.BoundArguments) -> Any:
        """Extract meaningful input from bound arguments."""
        args = bound_args.arguments
        for key in ("str_or_query_bundle", "query_bundle", "query", "prompt", "message", "messages", "input"):
            if key in args:
                value = args[key]
                if value is not None:
                    return str(value) if not isinstance(value, (str, list, dict)) else value
        # Fallback: iterate all args, skip common non-input ones
        for key, value in args.items():
            if key in _SKIP_ARG_NAMES:
                continue
            if value is not None:
                return str(value) if not isinstance(value, (str, list, dict)) else value
        return None

    @staticmethod
    def _infer_span_name(instance: Optional[Any], bound_args: inspect.BoundArguments) -> str:
        """Infer a span name from the instance and bound args."""
        if instance is not None:
            return type(instance).__name__
        # Fall back to first non-skipped argument name
        for name in bound_args.arguments:
            if name not in _SKIP_ARG_NAMES:
                return name
        return "span"
