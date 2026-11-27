"""The console script must not import ``algan`` to answer a cheap command.

``algan_cli`` lives outside the ``algan`` package for one reason: Python
imports a submodule's parents first, so an ``algan.cli:main`` entry point runs
``algan/__init__.py`` -- torch, Quadrants and the vendored Manim config, ~8.4 s
-- before argparse sees argv. ``algan --version`` measured 10-13 s that way and
~0.3 s this way.

Nothing about that is enforced by the module's own imports being tidy today: a
single top-level ``from algan.settings import SETTINGS`` added later restores
the whole cost silently, because every command still *works*. Hence a
subprocess check -- in-process assertions cannot see it, since by the time this
file runs pytest has imported ``algan`` many times over.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _algan_is_installed() -> bool:
    try:
        distribution("algan")
    except PackageNotFoundError:
        return False
    return True


#: Two of these read installed metadata -- the version and the console-script
#: entry point are properties of the *distribution*, not of the source tree,
#: and that is exactly what makes them worth asserting. A source-only checkout
#: has neither, so they are skipped rather than failed there.
needs_installed = pytest.mark.skipif(
    not _algan_is_installed(),
    reason="algan is not installed as a distribution; no metadata to read",
)


def _probe(source: str) -> str:
    """Run ``source`` in a clean interpreter and return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"probe failed ({result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result.stdout.strip()


def test_importing_the_cli_does_not_import_algan() -> None:
    loaded = _probe(
        "import algan_cli, sys; "
        "print(sorted(n for n in sys.modules if n == 'algan' "
        "or n.startswith('algan.')))"
    )

    assert loaded == "[]", (
        "importing algan_cli pulled in the algan package, which costs ~8.4s and "
        "is paid by every `algan --version` and `algan --help`. Move the offending "
        f"import inside the subcommand that needs it. Loaded: {loaded}"
    )


def test_the_cli_does_not_import_torch_or_the_kernel_compiler() -> None:
    """The two heavyweights, named individually so a failure says which."""
    loaded = _probe(
        "import algan_cli, sys; "
        "print(sorted({'torch', 'taichi', 'quadrants'} & set(sys.modules)))"
    )

    assert loaded == "[]", f"algan_cli imported a heavyweight dependency: {loaded}"


@needs_installed
def test_version_is_read_from_metadata_not_from_the_package() -> None:
    version = _probe("import algan_cli; print(algan_cli._version())")

    unresolved = "the version came back unresolved; _version() reads installed "
    assert version, unresolved + "metadata and printed nothing"
    assert version != "0+unknown", unresolved + "metadata and found no algan"


@needs_installed
def test_the_entry_point_points_outside_the_package() -> None:
    """The pyproject wiring, not just the module, is what users invoke."""
    from importlib.metadata import entry_points

    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}

    assert scripts.get("algan") == "algan_cli:main", (
        "the `algan` console script must resolve to the top-level algan_cli "
        "module; an `algan.cli:main` entry point runs algan/__init__.py before "
        f"argparse. Found: {scripts.get('algan')!r}"
    )


def test_a_daemon_trigger_does_not_import_torch(tmp_path) -> None:
    """``algan daemon ping`` is what an editor key runs on every press.

    It sends one line over a socket, and used to import the whole package
    first (about 3 s and 600 MB). It still has to read the state file, so the
    check runs it against an empty home, where it reports no daemon.
    """
    loaded = _probe(
        "import os, sys; "
        f"os.environ['ALGAN_HOME'] = {str(tmp_path)!r}; "
        "import algan_cli; "
        "status = algan_cli.main(['daemon', 'ping']); "
        "print(status, sorted({'torch', 'taichi', 'quadrants'} & set(sys.modules)), "
        "'algan' in sys.modules)"
    )

    assert loaded.splitlines()[-1] == "1 [] False", loaded


def test_launching_a_script_does_not_import_torch(tmp_path) -> None:
    """A flagless ``algan render`` only starts the script, which the daemon serves.

    The parent used to import the package to spell one environment variable,
    and held about 600 MB while the child ran.
    """
    script = tmp_path / "scene.py"
    script.write_text("print('ran')\n", encoding="utf-8")
    loaded = _probe(
        "import sys, algan_cli; "
        "seen = []; "
        "algan_cli.subprocess.call = lambda cmd, env=None, **k: seen.append(env) or 0; "
        f"status = algan_cli.main(['render', {str(script)!r}, '--no-daemon']); "
        "print(status, seen[0]['ALGAN_USE_DAEMON'], "
        "sorted({'torch', 'taichi', 'quadrants'} & set(sys.modules)))"
    )

    assert loaded.splitlines()[-1] == "0 0 []", loaded
