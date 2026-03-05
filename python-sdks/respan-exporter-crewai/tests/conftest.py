"""Pytest configuration: load .env from repo root so tests can use RESPAN_API_KEY etc."""
import os

import pytest

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_dotenv_from_repo_root() -> None:
    if load_dotenv is None:
        return
    # Repo root: tests/ -> respan-exporter-crewai -> python-sdks -> respan
    _tests_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.normpath(os.path.join(_tests_dir, "..", "..", ".."))
    load_dotenv(dotenv_path=os.path.join(_repo_root, ".env"), override=False)
    load_dotenv(override=False)  # cwd .env if present


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    _load_dotenv_from_repo_root()
