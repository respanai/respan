"""Keep SDK tests local and use isolated CrewAI SQLite storage."""

import os
from tempfile import TemporaryDirectory

import pytest

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACKING"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
_STORAGE = TemporaryDirectory(prefix="respan-crewai-tests-")
os.environ["CREWAI_STORAGE_DIR"] = _STORAGE.name


@pytest.fixture(autouse=True)
def isolated_crewai_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crewai"))
