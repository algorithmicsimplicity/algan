"""Validate the production bloom gate and compare corrected fallback renders."""

from __future__ import annotations

import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/unit_tests/test_bloom_device_parity.py",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "benchmarks/_mac_early_profile.py",
        "--child",
        "mps",
        "--quality",
        "UHD",
        "--arena-mib",
        "1720",
        "--runs",
        "4",
        "--bloom-ab",
    ],
    check=True,
)
