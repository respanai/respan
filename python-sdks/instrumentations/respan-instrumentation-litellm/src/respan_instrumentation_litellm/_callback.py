"""Native LiteLLM CustomLogger observers; application events remain unchanged."""

from __future__ import annotations

from litellm.integrations.custom_logger import CustomLogger

from ._translator import get


class RespanLiteLLMCallback(CustomLogger):
    def __init__(self, *, include_content=True, _runtime=None):
        super().__init__()
        self._include_content = include_content
        self._runtime = _runtime
        self._owned_runtime = _runtime is not None

    def runtime(self):
        from ._instrumentation import _RUNTIME, _Runtime

        if self._owned_runtime:
            return self._runtime
        if _RUNTIME is not None and _RUNTIME.active:
            return _RUNTIME
        if self._runtime is None:
            self._runtime = _Runtime(include_content=self._include_content)
        return self._runtime

    def log_pre_api_call(self, model, messages, kwargs):
        from ._instrumentation import safe

        safe(self.runtime().pre, model, messages, kwargs, self._include_content)

    async def async_log_pre_api_call(self, model, messages, kwargs):
        self.log_pre_api_call(model, messages, kwargs)

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        from ._instrumentation import safe

        safe(self.runtime().post, kwargs)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        from ._instrumentation import safe

        safe(self.runtime().event, kwargs, response_obj, None)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        from ._instrumentation import safe

        error = get(kwargs, "exception")
        safe(
            self.runtime().event,
            kwargs,
            response_obj,
            error if isinstance(error, BaseException) else None,
            True,
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.log_failure_event(kwargs, response_obj, start_time, end_time)
