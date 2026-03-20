"""Respan callback handler for LangChain."""
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from respan_exporter_langchain.types import SpanRecord
from respan_exporter_langchain.utils import langchain_messages_to_dicts

logger = logging.getLogger(__name__)


class RespanCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that captures trace spans and exports them to Respan.

    Usage::

        from respan_exporter_langchain import RespanCallbackHandler

        handler = RespanCallbackHandler(api_key="your-respan-api-key")
        chain.invoke({"input": "hello"}, config={"callbacks": [handler]})
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        environment: Optional[str] = None,
        customer_identifier: Optional[Union[str, int]] = None,
        session_identifier: Optional[Union[str, int]] = None,
        trace_group_identifier: Optional[Union[str, int]] = None,
        trace_name: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        super().__init__()
        # Lazy import to avoid circular dependency
        from respan_exporter_langchain.exporter import RespanLangchainExporter

        self._exporter = RespanLangchainExporter(
            api_key=api_key,
            endpoint=endpoint,
            base_url=base_url,
            environment=environment,
            customer_identifier=customer_identifier,
            timeout=timeout,
        )
        self.session_identifier = session_identifier
        self.trace_group_identifier = trace_group_identifier
        self.trace_name = trace_name

        # Thread-safe storage: run_id → SpanRecord
        self._lock = threading.Lock()
        self._spans: Dict[str, SpanRecord] = {}
        # Track which run_ids are root chains (no parent)
        self._root_run_ids: set = set()

    def _run_id_str(self, run_id: Any) -> str:
        return str(run_id) if run_id else str(uuid.uuid4())

    def _register_span(
        self,
        run_id: Any,
        parent_run_id: Any,
        span_type: str,
        name: Optional[str] = None,
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        model: Optional[str] = None,
        prompt_messages: Any = None,
    ) -> None:
        rid = self._run_id_str(run_id)
        pid = self._run_id_str(parent_run_id) if parent_run_id else None
        span = SpanRecord(
            run_id=rid,
            parent_run_id=pid,
            span_type=span_type,
            name=name,
            input=input_data,
            start_time=datetime.now(timezone.utc),
            metadata=metadata or {},
            tags=tags or [],
            model=model,
            prompt_messages=prompt_messages,
        )
        with self._lock:
            self._spans[rid] = span
            if pid is None:
                self._root_run_ids.add(rid)

    def _finish_span(
        self,
        run_id: Any,
        output: Any = None,
        error: Optional[str] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        completion_message: Any = None,
    ) -> None:
        rid = self._run_id_str(run_id)
        with self._lock:
            span = self._spans.get(rid)
            if span is None:
                return
            span.end_time = datetime.now(timezone.utc)
            if output is not None:
                span.output = output
            if error is not None:
                span.error = error
            if completion_message is not None:
                span.completion_message = completion_message
            if token_usage:
                span.prompt_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
                span.completion_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
                span.total_tokens = token_usage.get("total_tokens") or token_usage.get("total")

            # If this is a root span, flush the entire trace
            if rid in self._root_run_ids:
                self._flush_trace(root_run_id=rid)

    def _flush_trace(self, root_run_id: str) -> None:
        """Flush all spans belonging to this trace. Must be called under self._lock."""
        # Collect all spans in this trace tree
        trace_spans = self._collect_trace_spans(root_run_id=root_run_id)
        # Remove from storage
        for span in trace_spans:
            self._spans.pop(span.run_id, None)
        self._root_run_ids.discard(root_run_id)

        if not trace_spans:
            return

        # Export in a separate thread to not block the chain
        thread = threading.Thread(
            target=self._do_export,
            args=(trace_spans,),
            daemon=True,
        )
        thread.start()

    def _collect_trace_spans(self, root_run_id: str) -> List[SpanRecord]:
        """Collect all spans belonging to a trace tree rooted at root_run_id."""
        result: List[SpanRecord] = []
        root = self._spans.get(root_run_id)
        if root is None:
            return result
        # Build parent→children index for O(n) traversal
        children_map: Dict[str, List[str]] = {}
        for span in self._spans.values():
            if span.parent_run_id is not None:
                children_map.setdefault(span.parent_run_id, []).append(span.run_id)
        result.append(root)
        queue = [root_run_id]
        while queue:
            parent = queue.pop(0)
            for child_id in children_map.get(parent, []):
                child = self._spans.get(child_id)
                if child is not None and child not in result:
                    result.append(child)
                    queue.append(child_id)
        return result

    def _do_export(self, spans: List[SpanRecord]) -> None:
        """Export spans to Respan (runs in background thread)."""
        try:
            span_dicts = [self._span_record_to_dict(span=s) for s in spans]
            self._exporter.export(
                spans=span_dicts,
                trace_name=self.trace_name,
                session_identifier=self.session_identifier,
                trace_group_identifier=self.trace_group_identifier,
            )
        except Exception as exc:
            logger.warning("Respan LangChain export failed: %s", exc)

    @staticmethod
    def _span_record_to_dict(span: SpanRecord) -> Dict[str, Any]:
        """Convert SpanRecord to dict for the exporter."""
        return {
            "span_id": span.run_id,
            "parent_id": span.parent_run_id,
            "name": span.name,
            "span_type": span.span_type,
            "input": span.input,
            "output": span.output,
            "error": span.error,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "model": span.model,
            "prompt_tokens": span.prompt_tokens,
            "completion_tokens": span.completion_tokens,
            "total_tokens": span.total_tokens,
            "prompt_messages": span.prompt_messages,
            "completion_message": span.completion_message,
            "metadata": span.metadata,
            "tags": span.tags,
        }

    # ── Extract model name from serialized dict ─────────────────────────

    @staticmethod
    def _extract_model_name(serialized: Dict[str, Any]) -> Optional[str]:
        """Extract model name from LangChain serialized dict."""
        kwargs = serialized.get("kwargs", {})
        for key in ("model_name", "model", "model_id"):
            val = kwargs.get(key)
            if val:
                return str(val)
        name = serialized.get("name", "")
        id_list = serialized.get("id", [])
        if id_list and isinstance(id_list, list):
            last = id_list[-1]
            if isinstance(last, str) and "chat" in last.lower():
                return name or last
        return None

    @staticmethod
    def _extract_chain_name(serialized: Dict[str, Any]) -> Optional[str]:
        """Extract chain/agent name from serialized dict."""
        name = serialized.get("name")
        if name:
            return str(name)
        id_list = serialized.get("id", [])
        if id_list and isinstance(id_list, list):
            return str(id_list[-1])
        return None

    # ── LangChain callback methods ──────────────────────────────────────

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        name = self._extract_chain_name(serialized)
        # Determine if this is an agent chain
        id_list = serialized.get("id", [])
        is_agent = any(
            "agent" in str(part).lower()
            for part in (id_list if isinstance(id_list, list) else [])
        )
        if parent_run_id is None:
            span_type = "workflow"
        elif is_agent:
            span_type = "agent"
        else:
            span_type = "task"

        self._register_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_type=span_type,
            name=name,
            input_data=inputs,
            metadata=metadata,
            tags=tags,
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._finish_span(run_id=run_id, output=outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._finish_span(run_id=run_id, error=str(error))

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        model = self._extract_model_name(serialized)
        name = self._extract_chain_name(serialized) or model
        prompt_messages = [{"role": "user", "content": p} for p in prompts] if prompts else None
        self._register_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_type="generation",
            name=name,
            input_data=prompts,
            metadata=metadata,
            tags=tags,
            model=model,
            prompt_messages=prompt_messages,
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        model = self._extract_model_name(serialized)
        name = self._extract_chain_name(serialized) or model
        # Flatten message groups and convert to dicts
        flat_messages: List[Any] = []
        for group in messages:
            flat_messages.extend(group)
        prompt_messages = langchain_messages_to_dicts(flat_messages)
        input_text = "\n".join(
            m.get("content", "") for m in prompt_messages if isinstance(m, dict)
        )
        self._register_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_type="generation",
            name=name,
            input_data=input_text or prompt_messages,
            metadata=metadata,
            tags=tags,
            model=model,
            prompt_messages=prompt_messages,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        output_text = None
        completion_message = None
        token_usage: Optional[Dict[str, Any]] = None

        if response.llm_output:
            token_usage = response.llm_output.get("token_usage")
            if token_usage is None:
                token_usage = response.llm_output.get("usage")

        # Extract output from generations
        if response.generations:
            first_gen_list = response.generations[0]
            if first_gen_list:
                gen = first_gen_list[0]
                if hasattr(gen, "message"):
                    msg = gen.message
                    content = getattr(msg, "content", None)
                    role = getattr(msg, "type", "assistant")
                    role_map = {"ai": "assistant", "human": "user"}
                    role = role_map.get(role, role)
                    output_text = content
                    completion_message = {"role": role, "content": content}
                    # Extract token usage from AIMessage usage_metadata
                    usage_meta = getattr(msg, "usage_metadata", None)
                    if usage_meta and token_usage is None:
                        if isinstance(usage_meta, dict):
                            token_usage = {
                                "prompt_tokens": usage_meta.get("input_tokens"),
                                "completion_tokens": usage_meta.get("output_tokens"),
                                "total_tokens": usage_meta.get("total_tokens"),
                            }
                        elif hasattr(usage_meta, "input_tokens"):
                            token_usage = {
                                "prompt_tokens": getattr(usage_meta, "input_tokens", None),
                                "completion_tokens": getattr(usage_meta, "output_tokens", None),
                                "total_tokens": getattr(usage_meta, "total_tokens", None),
                            }
                elif hasattr(gen, "text"):
                    output_text = gen.text
                    completion_message = {"role": "assistant", "content": gen.text}

        self._finish_span(
            run_id=run_id,
            output=output_text,
            token_usage=token_usage,
            completion_message=completion_message,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._finish_span(run_id=run_id, error=str(error))

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name")
        if not name:
            id_list = serialized.get("id")
            if isinstance(id_list, list) and id_list:
                name = str(id_list[-1])
        self._register_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_type="tool",
            name=name,
            input_data=input_str,
            metadata=metadata,
            tags=tags,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        output_str = str(output) if output is not None else None
        self._finish_span(run_id=run_id, output=output_str)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._finish_span(run_id=run_id, error=str(error))

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        name = self._extract_chain_name(serialized) or "retriever"
        self._register_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_type="tool",
            name=name,
            input_data=query,
            metadata=metadata,
            tags=tags,
        )

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        doc_list = []
        for doc in documents:
            if hasattr(doc, "page_content"):
                doc_list.append({
                    "page_content": doc.page_content,
                    "metadata": getattr(doc, "metadata", {}),
                })
            else:
                doc_list.append(str(doc))
        self._finish_span(run_id=run_id, output=doc_list)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._finish_span(run_id=run_id, error=str(error))

    # agent_action and agent_finish are informational; the actual chain
    # start/end already captures agent spans, so we skip these to avoid dupes.
    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        pass

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass

    def flush(self) -> None:
        """Force flush any pending spans (e.g. on shutdown)."""
        with self._lock:
            for root_id in list(self._root_run_ids):
                self._flush_trace(root_run_id=root_id)
