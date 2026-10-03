"""Native Agno instrumentation plugin for Respan."""

import functools
import importlib
import inspect
import logging
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import replace
from threading import RLock
from types import MethodType
from typing import Any

from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_agno._constants import (
    AGNO_AGENT_CLASS_NAME,
    AGNO_AGENT_MODULE,
    AGNO_INSTRUMENTATION_NAME,
    AGNO_TARGET_AGENT,
    AGNO_TARGET_TEAM,
    AGNO_TEAM_CLASS_NAME,
    AGNO_TEAM_MODULE,
    ARUN_METHOD_NAME,
    EVENT_KEY,
    INPUT_KEY,
    RESPAN_AGNO_ORIGINALS_ATTR,
    RESPAN_AGNO_WRAPPED_ATTR,
    RUN_METHOD_NAME,
    RUN_OUTPUT_MARKER_KEYS,
)
from respan_instrumentation_agno._otel_emitter import (
    create_agno_run_context,
    emit_agno_error,
    emit_agno_run,
    use_agno_run_context,
)
from respan_instrumentation_agno._serialization import MAX_ITEMS

logger = logging.getLogger(__name__)
_PATCH_LOCK = RLock()
_PATCH_OWNERS: dict[int, dict[str, Any]] = {}


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _extract_input_value(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get(INPUT_KEY)


def _is_sync_stream_result(result: Any) -> bool:
    if isinstance(result, (str, bytes, bytearray, list, tuple, dict)):
        return False
    return isinstance(result, Iterator)


def _is_async_stream_result(result: Any) -> bool:
    if inspect.isawaitable(result):
        return False
    return hasattr(result, "__aiter__")


def _is_run_output(item: Any) -> bool:
    if _object_value(value=item, key=EVENT_KEY) is not None:
        return False
    return any(
        _object_value(value=item, key=key) is not None for key in RUN_OUTPUT_MARKER_KEYS
    )


def _last_run_output(items: list[Any]) -> Any | None:
    for item in reversed(items):
        if _is_run_output(item=item):
            return item
    return None


def _emit_completed_run(
    *,
    target: Any,
    target_kind: str,
    input_value: Any,
    output: Any | None,
    events: list[Any] | None,
    started_at_ns: int,
) -> None:
    try:
        emit_agno_run(
            target=target,
            target_kind=target_kind,
            input_value=input_value,
            output=output,
            events=events,
            started_at_ns=started_at_ns,
            ended_at_ns=time.time_ns(),
        )
    except Exception:
        logger.exception("Failed to emit Agno run spans")


def _emit_failed_run(
    *,
    target: Any,
    target_kind: str,
    input_value: Any,
    exception: BaseException,
    started_at_ns: int,
) -> None:
    try:
        emit_agno_error(
            target=target,
            target_kind=target_kind,
            input_value=input_value,
            exception=exception,
            started_at_ns=started_at_ns,
            ended_at_ns=time.time_ns(),
        )
    except Exception:
        logger.exception("Failed to emit failed Agno run span")


def _message_ids(output: Any) -> frozenset[str]:
    return frozenset(
        str(_object_value(message, "id"))
        for message in (_object_value(output, "messages") or [])
        if _object_value(message, "id")
    )


def _prior_message_ids(
    original_method: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any = None,
) -> frozenset[str]:
    if "continue_run" not in getattr(original_method, "__name__", ""):
        return frozenset()
    previous = args[0] if args else kwargs.get("run_response")
    if previous is None and target is not None and kwargs.get("run_id"):
        try:
            previous = target.get_run_output(
                run_id=kwargs["run_id"], session_id=kwargs.get("session_id")
            )
        except Exception:  # noqa: BLE001 - telemetry lookup must not alter the native call
            logger.debug("Could not read prior Agno run messages")
    return _message_ids(previous)


async def _async_prior_context(
    target: Any,
    original_method: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    run_context: Any,
) -> Any:
    if (
        run_context.prior_message_ids
        or "continue_run" not in getattr(original_method, "__name__", "")
        or not kwargs.get("run_id")
    ):
        return run_context
    try:
        previous = await target.aget_run_output(
            run_id=kwargs["run_id"], session_id=kwargs.get("session_id")
        )
    except Exception:  # noqa: BLE001 - telemetry lookup must not alter the native call
        logger.debug("Could not read prior async Agno run messages")
        return run_context
    return replace(run_context, prior_message_ids=_message_ids(previous))


def _stream_options(original_method: Any, target: Any, kwargs: dict[str, Any]):
    options = dict(kwargs)
    hide_final = False
    if kwargs.get("stream", getattr(target, "stream", False)):
        try:
            supports_final = (
                "yield_run_output" in inspect.signature(original_method).parameters
            )
        except (ValueError, TypeError):
            supports_final = False
        if supports_final and not kwargs.get("yield_run_output"):
            options["yield_run_output"] = True
            hide_final = True
    return options, hide_final


def _wrap_sync_stream(
    *,
    iterator: Iterator[Any],
    target: Any,
    target_kind: str,
    input_value: Any,
    started_at_ns: int,
    run_context: Any,
    hide_final: bool = False,
) -> Iterator[Any]:
    items: list[Any] = []
    failed = False
    try:
        while True:
            with use_agno_run_context(run_context=run_context):
                item = next(iterator)
            if len(items) < MAX_ITEMS:
                items.append(item)
            elif _is_run_output(item) or str(
                _object_value(item, EVENT_KEY, "")
            ).endswith("Completed"):
                items[-1] = item
            if not (hide_final and _is_run_output(item)):
                yield item
    except StopIteration:
        pass
    except GeneratorExit:
        raise
    except BaseException as exception:
        failed = True
        with use_agno_run_context(run_context=run_context):
            _emit_failed_run(
                target=target,
                target_kind=target_kind,
                input_value=input_value,
                exception=exception,
                started_at_ns=started_at_ns,
            )
        raise
    finally:
        try:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        except BaseException as exception:
            if not failed:
                failed = True
                with use_agno_run_context(run_context=run_context):
                    _emit_failed_run(
                        target=target,
                        target_kind=target_kind,
                        input_value=input_value,
                        exception=exception,
                        started_at_ns=started_at_ns,
                    )
                raise
        finally:
            if not failed:
                with use_agno_run_context(run_context=run_context):
                    _emit_completed_run(
                        target=target,
                        target_kind=target_kind,
                        input_value=input_value,
                        output=_last_run_output(items),
                        events=items,
                        started_at_ns=started_at_ns,
                    )


async def _wrap_async_stream(
    *,
    async_iterator: Any,
    target: Any,
    target_kind: str,
    input_value: Any,
    started_at_ns: int,
    run_context: Any,
    hide_final: bool = False,
    context_loader: Any = None,
) -> AsyncIterator[Any]:
    if context_loader is not None:
        run_context = await context_loader()
    items: list[Any] = []
    failed = False
    try:
        while True:
            with use_agno_run_context(run_context=run_context):
                item = await anext(async_iterator)
            if len(items) < MAX_ITEMS:
                items.append(item)
            elif _is_run_output(item) or str(
                _object_value(item, EVENT_KEY, "")
            ).endswith("Completed"):
                items[-1] = item
            if not (hide_final and _is_run_output(item)):
                yield item
    except StopAsyncIteration:
        pass
    except GeneratorExit:
        raise
    except BaseException as exception:
        failed = True
        with use_agno_run_context(run_context=run_context):
            _emit_failed_run(
                target=target,
                target_kind=target_kind,
                input_value=input_value,
                exception=exception,
                started_at_ns=started_at_ns,
            )
        raise
    finally:
        try:
            close = getattr(async_iterator, "aclose", None)
            if callable(close):
                await close()
        except BaseException as exception:
            if not failed:
                failed = True
                with use_agno_run_context(run_context=run_context):
                    _emit_failed_run(
                        target=target,
                        target_kind=target_kind,
                        input_value=input_value,
                        exception=exception,
                        started_at_ns=started_at_ns,
                    )
                raise
        finally:
            if not failed:
                with use_agno_run_context(run_context=run_context):
                    _emit_completed_run(
                        target=target,
                        target_kind=target_kind,
                        input_value=input_value,
                        output=_last_run_output(items),
                        events=items,
                        started_at_ns=started_at_ns,
                    )


def _wrap_sync_method(
    *,
    original_method: Any,
    target_kind: str,
    is_bound_method: bool,
    owner_target: Any = None,
) -> Any:
    @functools.wraps(original_method)
    def wrapped_sync_method(*args: Any, **kwargs: Any) -> Any:
        target = args[0]
        call_args = args[1:] if is_bound_method else args
        if owner_target is not None and id(owner_target) not in _PATCH_OWNERS:
            return original_method(*call_args, **kwargs)
        input_args = args[1:]
        input_value = _extract_input_value(args=input_args, kwargs=kwargs)
        started_at_ns = time.time_ns()
        run_context = create_agno_run_context(
            target=target,
            target_kind=target_kind,
            started_at_ns=started_at_ns,
            prior_message_ids=_prior_message_ids(
                original_method, input_args, kwargs, target
            ),
        )

        call_kwargs, hide_final = _stream_options(original_method, target, kwargs)
        with use_agno_run_context(run_context=run_context):
            try:
                result = original_method(*call_args, **call_kwargs)
            except Exception as exception:
                _emit_failed_run(
                    target=target,
                    target_kind=target_kind,
                    input_value=input_value,
                    exception=exception,
                    started_at_ns=started_at_ns,
                )
                raise

            if _is_sync_stream_result(result=result):
                return _wrap_sync_stream(
                    iterator=result,
                    target=target,
                    target_kind=target_kind,
                    input_value=input_value,
                    started_at_ns=started_at_ns,
                    run_context=run_context,
                    hide_final=hide_final,
                )

            _emit_completed_run(
                target=target,
                target_kind=target_kind,
                input_value=input_value,
                output=result,
                events=None,
                started_at_ns=started_at_ns,
            )
            return result

    setattr(wrapped_sync_method, RESPAN_AGNO_WRAPPED_ATTR, True)
    return wrapped_sync_method


def _wrap_async_method(
    *,
    original_method: Any,
    target_kind: str,
    is_bound_method: bool,
    owner_target: Any = None,
) -> Any:
    @functools.wraps(original_method)
    def wrapped_async_method(*args: Any, **kwargs: Any) -> Any:
        target = args[0]
        call_args = args[1:] if is_bound_method else args
        if owner_target is not None and id(owner_target) not in _PATCH_OWNERS:
            return original_method(*call_args, **kwargs)
        input_args = args[1:]
        input_value = _extract_input_value(args=input_args, kwargs=kwargs)
        started_at_ns = time.time_ns()
        run_context = create_agno_run_context(
            target=target,
            target_kind=target_kind,
            started_at_ns=started_at_ns,
            prior_message_ids=_prior_message_ids(original_method, input_args, kwargs),
        )

        call_kwargs, hide_final = _stream_options(original_method, target, kwargs)
        with use_agno_run_context(run_context=run_context):
            try:
                result = original_method(*call_args, **call_kwargs)
            except Exception as exception:
                _emit_failed_run(
                    target=target,
                    target_kind=target_kind,
                    input_value=input_value,
                    exception=exception,
                    started_at_ns=started_at_ns,
                )
                raise

            if _is_async_stream_result(result=result):
                return _wrap_async_stream(
                    async_iterator=result,
                    target=target,
                    target_kind=target_kind,
                    input_value=input_value,
                    started_at_ns=started_at_ns,
                    run_context=run_context,
                    hide_final=hide_final,
                    context_loader=lambda: _async_prior_context(
                        target, original_method, input_args, kwargs, run_context
                    ),
                )

            if inspect.isawaitable(result):

                async def await_and_emit() -> Any:
                    resolved_context = await _async_prior_context(
                        target, original_method, input_args, kwargs, run_context
                    )
                    with use_agno_run_context(run_context=resolved_context):
                        try:
                            output = await result
                        except BaseException as exception:
                            _emit_failed_run(
                                target=target,
                                target_kind=target_kind,
                                input_value=input_value,
                                exception=exception,
                                started_at_ns=started_at_ns,
                            )
                            raise

                        _emit_completed_run(
                            target=target,
                            target_kind=target_kind,
                            input_value=input_value,
                            output=output,
                            events=None,
                            started_at_ns=started_at_ns,
                        )
                        return output

                return await_and_emit()

            _emit_completed_run(
                target=target,
                target_kind=target_kind,
                input_value=input_value,
                output=result,
                events=None,
                started_at_ns=started_at_ns,
            )
            return result

    setattr(wrapped_async_method, RESPAN_AGNO_WRAPPED_ATTR, True)
    return wrapped_async_method


class AgnoInstrumentor:
    """Respan instrumentor for Agno.

    This native integration patches Agno's public run methods and emits
    Respan-compatible OTEL spans directly. It intentionally does not use
    ``openinference-instrumentation-agno``.
    """

    name = AGNO_INSTRUMENTATION_NAME

    def __init__(
        self,
        agent: Any | None = None,
        *,
        include_teams: bool = True,
    ) -> None:
        self._agent = agent
        self._include_teams = include_teams
        self._patches: list[tuple[Any, dict[str, Any]]] = []
        self._is_instrumented = False

    def _patch_target(
        self,
        *,
        target: Any,
        target_kind: str,
        is_bound_method: bool,
    ) -> bool:
        shared = _PATCH_OWNERS.get(id(target))
        if shared is not None:
            shared["owners"] += 1
            self._patches.append((target, shared["originals"]))
            return True
        if vars(target).get(RESPAN_AGNO_WRAPPED_ATTR, False):
            return False

        owned_methods = set(vars(target))
        originals: dict[str, Any] = {}
        for method_name, wrap in (
            (RUN_METHOD_NAME, _wrap_sync_method),
            (ARUN_METHOD_NAME, _wrap_async_method),
            ("continue_run", _wrap_sync_method),
            ("acontinue_run", _wrap_async_method),
        ):
            original = getattr(target, method_name, None)
            if not callable(original):
                continue
            originals[method_name] = original
            call_original = original
            if is_bound_method and getattr(original, RESPAN_AGNO_WRAPPED_ATTR, False):
                call_original = MethodType(inspect.unwrap(original), target)
            replacement = wrap(
                original_method=call_original,
                target_kind=target_kind,
                is_bound_method=is_bound_method,
                owner_target=target,
            )
            if is_bound_method:
                replacement = MethodType(replacement, target)
            setattr(target, method_name, replacement)

        if not originals:
            return False

        setattr(target, RESPAN_AGNO_ORIGINALS_ATTR, originals)
        setattr(target, RESPAN_AGNO_WRAPPED_ATTR, True)
        self._patches.append((target, originals))
        _PATCH_OWNERS[id(target)] = {
            "target": target,
            "owners": 1,
            "originals": originals,
            "owned_methods": owned_methods,
            "is_bound_method": is_bound_method,
            "installed": {name: getattr(target, name) for name in originals},
        }
        return True

    @staticmethod
    def _load_agent_class() -> type:
        agent_module = importlib.import_module(AGNO_AGENT_MODULE)
        return getattr(agent_module, AGNO_AGENT_CLASS_NAME)

    @staticmethod
    def _load_team_class() -> type:
        team_module = importlib.import_module(AGNO_TEAM_MODULE)
        return getattr(team_module, AGNO_TEAM_CLASS_NAME)

    def activate(self) -> None:
        """Activate Agno instrumentation with shared patch ownership."""
        with _PATCH_LOCK:
            self._activate()

    def _activate(self) -> None:
        if self._is_instrumented:
            return

        if not _is_respan_tracing_enabled():
            logger.info(
                "Agno instrumentation skipped because Respan tracing is disabled"
            )
            return

        if self._agent is not None:
            target_kind = (
                AGNO_TARGET_TEAM
                if type(self._agent).__module__.startswith("agno.team")
                else AGNO_TARGET_AGENT
            )
            self._patch_target(
                target=self._agent,
                target_kind=target_kind,
                is_bound_method=True,
            )
            self._is_instrumented = bool(self._patches)
            return

        try:
            agent_class = self._load_agent_class()
        except ImportError as exception:
            logger.warning(
                f"Failed to activate Agno instrumentation - missing dependency: {exception}"
            )
            return

        self._patch_target(
            target=agent_class,
            target_kind=AGNO_TARGET_AGENT,
            is_bound_method=False,
        )

        if self._include_teams:
            try:
                team_class = self._load_team_class()
            except ImportError:
                logger.info(
                    "Agno Team instrumentation skipped because Team is unavailable"
                )
            else:
                self._patch_target(
                    target=team_class,
                    target_kind=AGNO_TARGET_TEAM,
                    is_bound_method=False,
                )

        self._is_instrumented = bool(self._patches)
        if self._is_instrumented:
            logger.info("Agno instrumentation activated")

    def deactivate(self) -> None:
        """Restore patches after the final owner deactivates."""
        with _PATCH_LOCK:
            self._deactivate()

    def _deactivate(self) -> None:
        for target, originals in reversed(self._patches):
            shared = _PATCH_OWNERS.get(id(target))
            if shared is None:
                continue
            shared["owners"] -= 1
            if shared["owners"]:
                continue
            _PATCH_OWNERS.pop(id(target), None)
            for method_name, original_method in originals.items():
                if getattr(target, method_name) is shared["installed"][method_name]:
                    if (
                        shared["is_bound_method"]
                        and method_name not in shared["owned_methods"]
                    ):
                        delattr(target, method_name)
                    else:
                        setattr(target, method_name, original_method)
            if hasattr(target, RESPAN_AGNO_ORIGINALS_ATTR):
                delattr(target, RESPAN_AGNO_ORIGINALS_ATTR)
            if hasattr(target, RESPAN_AGNO_WRAPPED_ATTR):
                delattr(target, RESPAN_AGNO_WRAPPED_ATTR)

        self._patches.clear()
        self._is_instrumented = False
        logger.info("Agno instrumentation deactivated")
