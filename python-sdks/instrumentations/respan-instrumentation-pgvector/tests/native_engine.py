"""Actual released bundled PostgreSQL binaries; isolated native engine lifecycle."""

import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from pgserver._commands import POSTGRES_BIN_PATH


@contextmanager
def postgres():
    with TemporaryDirectory(prefix="p16pg-", dir="/private/tmp") as folder:
        root = Path(folder)
        data = root / "data"
        sockets = root / "sock"
        sockets.mkdir()
        started = False
        try:
            subprocess.run(
                [
                    str(POSTGRES_BIN_PATH / "initdb"),
                    "-D",
                    str(data),
                    "--auth=trust",
                    "--auth-local=trust",
                    "--encoding=UTF8",
                    "--locale=C",
                    "-U",
                    "postgres",
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    str(POSTGRES_BIN_PATH / "pg_ctl"),
                    "-D",
                    str(data),
                    "-w",
                    "-l",
                    str(root / "postgres.log"),
                    "-o",
                    f'-h "" -k {sockets}',
                    "start",
                ],
                check=True,
                capture_output=True,
            )
            started = True
            yield f"dbname=postgres host={sockets} user=postgres sslmode=disable"
        finally:
            if started:
                subprocess.run(
                    [
                        str(POSTGRES_BIN_PATH / "pg_ctl"),
                        "-D",
                        str(data),
                        "-w",
                        "-m",
                        "fast",
                        "stop",
                    ],
                    check=True,
                    capture_output=True,
                )
