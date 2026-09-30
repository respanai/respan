#!/usr/bin/env python3
"""Run from javascript-sdks; retry only Yarn's observed request-cancellation crash."""

from __future__ import annotations

import subprocess
import sys
import time


CANCELLATION_ERROR = (
    "Error: The `onCancel` handler was attached after the promise settled."
)
MAX_ATTEMPTS = 3


def install_workspace() -> int:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        cancellation_crash = False
        with subprocess.Popen(
            ["yarn", "install"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                # The minified Yarn source in a stack trace also contains this
                # string. Match the actual error line, not a source excerpt.
                if line.strip() == CANCELLATION_ERROR:
                    cancellation_crash = True
            status = process.wait()

        if status == 0 or not cancellation_crash or attempt == MAX_ATTEMPTS:
            return status if status >= 0 else 128 - status

        delay = 5 * attempt
        print(
            f"Yarn request-cancellation crash; retrying install "
            f"({attempt + 1}/{MAX_ATTEMPTS}) in {delay}s.",
            flush=True,
        )
        time.sleep(delay)

    raise AssertionError("unreachable")


if __name__ == "__main__":
    sys.exit(install_workspace())
