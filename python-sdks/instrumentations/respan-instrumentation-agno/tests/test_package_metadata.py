from pathlib import Path

import tomllib


def test_package_does_not_depend_on_openinference_agno():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    package_config = tomllib.loads(pyproject_path.read_text())
    dependencies = package_config["tool"]["poetry"]["dependencies"]

    assert "openinference-instrumentation-agno" not in dependencies
    assert "agno" in dependencies
