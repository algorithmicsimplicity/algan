"""``clear_cache`` and ``clear_cached_kernels``: emptying Algan's cache directory.

Algan's content caches -- Tex and text glyphs, tessellations, audio -- live
under ``SETTINGS.paths.cache_directory``, and by default so do the compiled
kernels. Those take minutes to rebuild, so ``clear_cache()`` spares their
directory and ``clear_cached_kernels()`` is how to drop them as well.

Every test points the cache directory, and the kernel directory inside it, at
``tmp_path`` first: these functions delete files, and the real cache holds
the compiled kernels the rest of the suite runs on.

A feature test, so it sits outside the fast suite: it only breaks when these
two functions change.
"""

from __future__ import annotations

import pytest

import algan
from algan import SETTINGS
from algan.settings import _startup


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    kernels = root / "kernels"
    (root / "tex").mkdir(parents=True)
    (root / "tex" / "glyph.svg").write_text("<svg/>")
    (root / "tessellation.pt").write_bytes(b"\0")
    (kernels / "llvm").mkdir(parents=True)
    (kernels / "llvm" / "kernel.bin").write_bytes(b"\0")

    SETTINGS.paths.set(cache_directory=str(root))
    monkeypatch.setattr(_startup, "_TAICHI_CACHE_DIRECTORY", kernels)
    assert SETTINGS.paths.cache_directory == str(root)
    return root, kernels


def test_clear_cache_drops_the_content_caches_and_keeps_the_kernels(cache):
    root, kernels = cache
    algan.clear_cache()

    assert root.is_dir()
    assert sorted(path.name for path in root.iterdir()) == ["kernels"]
    assert (kernels / "llvm" / "kernel.bin").is_file()


@pytest.mark.parametrize(
    "clear",
    [
        pytest.param(lambda: algan.clear_cache(include_kernels=True), id="include"),
        pytest.param(algan.clear_cached_kernels, id="clear_cached_kernels"),
    ],
)
def test_clearing_the_kernels_too_empties_the_whole_directory(cache, clear):
    root, _ = cache
    clear()
    assert not root.exists()


def test_clearing_a_cache_that_was_never_written_does_nothing(tmp_path, monkeypatch):
    missing = tmp_path / "never_created"
    SETTINGS.paths.set(cache_directory=str(missing))
    monkeypatch.setattr(_startup, "_TAICHI_CACHE_DIRECTORY", missing / "kernels")

    algan.clear_cache()
    algan.clear_cached_kernels()
    assert not missing.exists()


def test_both_are_public_names():
    assert {"clear_cache", "clear_cached_kernels"} <= set(algan.__all__)


def test_the_inductor_cache_lives_in_the_cache_directory(tmp_path):
    """torch.compile's cache follows ``ALGAN_CACHE_DIR``, not torch's ``/tmp`` default.

    So ``clear_cache()`` empties it and cache isolation covers it. A fresh
    interpreter, since the variable is set once, when ``_startup`` imports.
    """
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "TORCHINDUCTOR_CACHE_DIR"}
    env["ALGAN_CACHE_DIR"] = str(tmp_path / "cache")
    probe = (
        "import algan.settings._startup\n"
        "from torch._inductor.runtime.cache_dir_utils import cache_dir\n"
        "print(cache_dir())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == str(
        tmp_path / "cache" / "torchinductor"
    )
