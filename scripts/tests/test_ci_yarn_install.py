from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ci_yarn_install  # noqa: E402


class YarnInstallTests(unittest.TestCase):
    def run_install(self, outcomes: list[tuple[int, str]]) -> tuple[int, int, str]:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "outcomes.json").write_text(json.dumps(outcomes))
            yarn = directory / "yarn"
            yarn.write_text(
                f"#!{sys.executable}\n"
                "import json, sys\n"
                "from pathlib import Path\n"
                "assert sys.argv[1:] == ['install']\n"
                "directory = Path(__file__).parent\n"
                "counter = directory / 'attempts'\n"
                "attempt = int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(attempt + 1))\n"
                "outcomes = json.loads((directory / 'outcomes.json').read_text())\n"
                "status, output = outcomes[attempt]\n"
                "print(output, file=sys.stderr, flush=True)\n"
                "sys.exit(status)\n"
            )
            yarn.chmod(0o755)
            output = io.StringIO()
            with (
                mock.patch.dict(os.environ, {"PATH": str(directory)}),
                mock.patch.object(ci_yarn_install.time, "sleep"),
                contextlib.redirect_stdout(output),
            ):
                status = ci_yarn_install.install_workspace()
            attempts = int((directory / "attempts").read_text())
            return status, attempts, output.getvalue()

    def test_success_needs_one_install(self) -> None:
        status, attempts, output = self.run_install([(0, "Installed")])
        self.assertEqual((status, attempts), (0, 1))
        self.assertIn("Installed", output)

    def test_cancellation_crash_retries_and_streams_both_attempts(self) -> None:
        status, attempts, output = self.run_install(
            [(1, ci_yarn_install.CANCELLATION_ERROR), (0, "Installed")]
        )
        self.assertEqual((status, attempts), (0, 2))
        self.assertIn(ci_yarn_install.CANCELLATION_ERROR, output)
        self.assertIn("Installed", output)

    def test_other_install_failure_is_not_retried(self) -> None:
        status, attempts, _ = self.run_install([(42, "YN0028: Lockfile changed")])
        self.assertEqual((status, attempts), (42, 1))

    def test_previous_crash_does_not_retry_a_later_deterministic_error(self) -> None:
        status, attempts, _ = self.run_install(
            [(1, ci_yarn_install.CANCELLATION_ERROR), (23, "YN0018: Cache mismatch")]
        )
        self.assertEqual((status, attempts), (23, 2))

    def test_repeated_crash_exhausts_three_attempts_and_preserves_exit_code(self) -> None:
        status, attempts, _ = self.run_install(
            [(7, ci_yarn_install.CANCELLATION_ERROR)] * 3
        )
        self.assertEqual((status, attempts), (7, 3))

    def test_crash_message_inside_source_excerpt_is_not_retried(self) -> None:
        status, attempts, _ = self.run_install(
            [(1, f'throw new Error("{ci_yarn_install.CANCELLATION_ERROR}")')]
        )
        self.assertEqual((status, attempts), (1, 1))


if __name__ == "__main__":
    unittest.main()
