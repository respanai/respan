"""Pytest entry-point module for Respan test tracing."""

from __future__ import annotations

import os
from typing import Any

from respan_instrumentation_pytest._constants import (
    ENV_CAPTURE_CONTENT,
    ENV_ENABLED,
    ENV_WORKFLOW_NAME,
    PYTEST_RUNTIME_PLUGIN_NAME,
)
from respan_instrumentation_pytest._runtime import PytestRuntimePlugin


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("respan")
    group.addoption(
        "--respan-tracing",
        action="store_true",
        default=None,
        help="Emit Respan workflow/task spans for the Pytest session and tests.",
    )
    group.addoption(
        "--respan-capture-content",
        action="store_true",
        dest="respan_capture_content",
        default=None,
        help="Capture test parameters and failure messages in Respan spans.",
    )
    group.addoption(
        "--no-respan-capture-content",
        action="store_false",
        dest="respan_capture_content",
        default=None,
        help="Omit test parameters and failure messages from Respan spans.",
    )
    parser.addini(
        "respan_tracing", "Enable Respan Pytest tracing.", type="bool", default=False
    )
    parser.addini(
        "respan_capture_content",
        "Capture test parameters and failure messages.",
        type="bool",
        default=True,
    )
    parser.addini(
        "respan_workflow_name",
        "Workflow name used for the Pytest session trace.",
        default="",
    )


def _enabled(config: Any) -> bool:
    cli_value = config.getoption("respan_tracing")
    if cli_value is not None:
        return bool(cli_value)
    if _env_bool(ENV_ENABLED, False):
        return True
    return bool(config.getini("respan_tracing"))


def _capture_content(config: Any) -> bool:
    cli_value = config.getoption("respan_capture_content")
    if cli_value is not None:
        return bool(cli_value)
    if os.getenv(ENV_CAPTURE_CONTENT) is not None:
        return _env_bool(ENV_CAPTURE_CONTENT, True)
    return bool(config.getini("respan_capture_content"))


def _workflow_name(config: Any) -> str | None:
    return os.getenv(ENV_WORKFLOW_NAME) or config.getini("respan_workflow_name") or None


def pytest_configure(config: Any) -> None:
    if not _enabled(config):
        return
    pluginmanager = config.pluginmanager
    if pluginmanager.get_plugin(PYTEST_RUNTIME_PLUGIN_NAME) is not None:
        return
    instrumentor = PytestRuntimePlugin(
        capture_content=_capture_content(config),
        workflow_name=_workflow_name(config),
    )
    instrumentor.activate()
    pluginmanager.register(instrumentor, PYTEST_RUNTIME_PLUGIN_NAME)


def pytest_unconfigure(config: Any) -> None:
    pluginmanager = config.pluginmanager
    instrumentor = pluginmanager.get_plugin(PYTEST_RUNTIME_PLUGIN_NAME)
    if instrumentor is None:
        return
    instrumentor.deactivate()
    pluginmanager.unregister(instrumentor)
