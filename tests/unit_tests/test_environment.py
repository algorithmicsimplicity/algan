from __future__ import annotations

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from algan.environment import (
    ALGAN_ENVIRONMENT_VARIABLES,
    unknown_algan_environment_variables,
    warn_for_unknown_algan_environment_variables,
)
from algan.errors import AlganWarning


def test_environment_registry_matches_package_environment_reads():
    source_root = Path(__file__).parents[2] / "algan"
    patterns = (
        r"os\.(?:getenv|environ\.get)\(\s*['\"](ALGAN_[A-Z0-9_]+)",
        r"os\.environ\[\s*['\"](ALGAN_[A-Z0-9_]+)",
        r"['\"](ALGAN_[A-Z0-9_]+)['\"]\s+(?:not\s+)?in\s+os\.environ",
        r"_parse_device\(\s*['\"](ALGAN_[A-Z0-9_]+)",
        r"dict\(os\.environ,\s*(ALGAN_[A-Z0-9_]+)\s*=",
    )
    used = set()
    for source_path in source_root.rglob("*.py"):
        if (
            "external_libraries" in source_path.parts
            or source_path.name == "environment.py"
        ):
            continue
        source = source_path.read_text(encoding="utf-8")
        for pattern in patterns:
            used.update(re.findall(pattern, source))

    assert used == ALGAN_ENVIRONMENT_VARIABLES


def test_unknown_algan_environment_variables_are_sorted_and_warned():
    environ = {
        "PATH": "ignored",
        "ALGAN_RENDER_DEVICE": "cpu",
        "ALGAN_Z_UNKNOWN": "1",
        "ALGAN_A_UNKNOWN": "1",
    }

    assert unknown_algan_environment_variables(environ) == (
        "ALGAN_A_UNKNOWN",
        "ALGAN_Z_UNKNOWN",
    )
    with pytest.warns(
        AlganWarning,
        match="Unknown Algan environment variables: ALGAN_A_UNKNOWN, ALGAN_Z_UNKNOWN",
    ):
        warn_for_unknown_algan_environment_variables(environ)


def test_all_registered_algan_environment_variables_are_accepted():
    environ = dict.fromkeys(ALGAN_ENVIRONMENT_VARIABLES, "unused")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_for_unknown_algan_environment_variables(environ)
    assert not caught


@pytest.mark.slow
def test_import_algan_warns_for_an_unknown_environment_variable(tmp_path):
    environ = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("ALGAN_")
    }
    environ.update(
        {
            "ALGAN_ANIMATION_DEVICE": "cpu",
            "ALGAN_HOME": str(tmp_path / "algan_home"),
            "ALGAN_RENDER_DEVIC": "cpu",
            "ALGAN_RENDER_DEVICE": "cpu",
        }
    )

    completed = subprocess.run(
        [sys.executable, "-W", "always", "-c", "import algan"],
        capture_output=True,
        check=False,
        cwd=os.fspath(Path(__file__).parents[2]),
        env=environ,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Unknown Algan environment variable: ALGAN_RENDER_DEVIC" in completed.stderr
