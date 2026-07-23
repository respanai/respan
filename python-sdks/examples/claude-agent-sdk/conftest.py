"""Shared pytest configuration for Claude Agent SDK examples.

Creates a single Respan + ClaudeAgentSDKInstrumentor instance shared across
all test modules. This prevents monkey-patch conflicts when running multiple
test files in a single pytest session.

Individual test files can still be run standalone via `python <file>.py`.
"""

import os
import sys
import time

import pytest
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env", override=True)

# Ensure basic/_sdk_runtime is importable from any subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "basic"))

from respan import Respan
from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor


@pytest.fixture(scope="session", autouse=True)
def respan_instance():
    """Single shared Respan instance for the entire test session."""
    instrumentor = ClaudeAgentSDKInstrumentor()
    respan = Respan(instrumentations=[instrumentor])
    yield respan
    time.sleep(2)  # Allow batch export HTTP requests to complete
