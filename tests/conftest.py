"""Shared fixtures and options for both Algan test suites.

Two things live here because they are cross-cutting:

* ``--skip-slow`` / the ``slow`` marker, so the behavioural suite can be run in
  seconds while the GPU-heavy and pixel-comparison tests are opt-in;
* per-test isolation of the two pieces of process-global state Algan owns --
  ``SETTINGS`` and the active-Scene stack. Leaking either between tests makes
  failures depend on test order, which is the hardest kind of flake to chase.
"""

from __future__ import annotations

import pytest

from algan import SETTINGS
from algan.scene_manager import SceneManager


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow",
        "--skip_slow",
        action="store_true",
        default=False,
        help="Skip tests marked 'slow' (GPU renders and pixel comparisons).",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skip-slow"):
        return
    skip = pytest.mark.skip(reason="--skip-slow was passed")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolate_global_settings():
    """Undo any settings a test wrote, however it exited.

    ``SETTINGS`` sections keep stable identity, so ``restore`` writes values
    back into the same objects the engine already holds references to.
    """
    snapshot = SETTINGS.snapshot()
    try:
        yield
    finally:
        SETTINGS.restore(snapshot)


@pytest.fixture
def fresh_scene():
    """A pristine active-Scene stack, torn down again afterwards.

    Prefer ``with Scene() as scene:`` inside a test when you need a handle on
    the Scene; use this when the code under test resolves the *current* Scene
    itself.
    """
    SceneManager.reset()
    try:
        yield SceneManager.instance().current_scene
    finally:
        SceneManager.reset()
