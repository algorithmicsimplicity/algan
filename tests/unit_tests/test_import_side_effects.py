"""What ``import algan`` is allowed to do to the process that imports it.

A library that is imported into someone else's process may configure itself;
it may not reconfigure them. Two things it used to do are pinned here:

* it entered ``torch.inference_mode()`` process-wide and never exited it, so a
  notebook that imported Algan could never train a model afterwards;
* it printed Taichi's ``[Taichi] version ...`` banner to **stdout** and its own
  render-device line to stderr, so a script whose stdout is data got the banner
  mixed into it.

The subprocess is the point: these are properties of a fresh interpreter, and
the test runner's own process has long since imported everything.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from algan import LD, Scene, Square


def _fresh_python(code, env=None):
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_autograd_still_works_after_the_import():
    """The import must not decide autograd for the process it lands in."""
    result = _fresh_python(
        "import algan, torch\n"
        "assert torch.is_grad_enabled(), 'grad mode was switched off'\n"
        "x = torch.ones(3, requires_grad=True)\n"
        "(x * 2).sum().backward()\n"
        "assert x.grad is not None and float(x.grad[0]) == 2.0\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_import_prints_nothing_to_stdout():
    """A piped script's data must not have a Taichi banner in it."""
    result = _fresh_python("import algan")
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert "[Taichi]" not in result.stdout + result.stderr


def test_the_import_says_nothing_at_info():
    result = _fresh_python("import algan")
    assert result.returncode == 0, result.stderr
    assert "Rendering device set to" not in result.stderr
    assert result.stderr == ""


def test_the_import_is_quiet_without_ffmpeg_on_path():
    """No dependency may probe PATH for ffmpeg while it is being imported.

    pydub did, arriving with the ``manim`` distribution, and warned "Couldn't
    find ffmpeg or avconv" on every ``import algan`` on a machine carrying only
    the build imageio-ffmpeg bundles -- which is every machine that installed
    Algan and nothing else. Algan encodes through that bundled build and worked
    the whole time, so the warning was pure noise, and ``algan/__init__.py``
    carried a filter plus a converter fix-up to undo it.

    Both are gone with the dependency: the vendored Manim subset carries none
    of Manim's audio code (see ``algan/external_libraries/manim/VENDORING.md``).
    So this pins the outcome rather than the workaround -- nothing says
    anything, and pydub is not in the process to say it.
    """
    env = dict(os.environ, PATH="/nonexistent")
    result = _fresh_python(
        "import algan, sys\nprint('pydub' in sys.modules)\n", env=env
    )
    assert result.returncode == 0, result.stderr
    assert "Couldn't find ffmpeg" not in result.stderr
    assert result.stdout.strip() == "False", (
        "pydub is back in the dependency set; it probes PATH for ffmpeg at its "
        "own import, so `import algan` will warn again on a machine that has "
        "only the imageio-ffmpeg build"
    )


def test_a_render_leaves_the_scene_mutable(tmp_path):
    """``no_grad``, not ``inference_mode``: rendering must not freeze state.

    An inference tensor cannot be mutated in place once the mode has exited,
    and a ``reset=False`` render leaves the Scene's tensors behind for the
    authoring that follows -- so a scoped ``inference_mode()`` would turn the
    next ``move`` into a RuntimeError.
    """
    with Scene() as scene:
        square = Square().spawn(animate=False)
        scene.save_frame(str(tmp_path / "frame.png"), video_settings=LD)
        square.move(torch.tensor([1.0, 0.0, 0.0]))
        assert not torch.is_inference(square.location.data)


@pytest.mark.parametrize("attribute", ["location", "color"])
def test_rendered_state_is_not_an_inference_tensor(tmp_path, attribute):
    with Scene() as scene:
        square = Square().spawn(animate=False)
        scene.save_frame(str(tmp_path / "frame.png"), video_settings=LD)
        assert not torch.is_inference(getattr(square, attribute).data)
