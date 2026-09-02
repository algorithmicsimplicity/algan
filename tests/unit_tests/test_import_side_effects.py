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


def test_pydub_is_quiet_and_pointed_at_the_bundled_ffmpeg():
    """pydub probes PATH at its own import and warned on every `import algan`.

    Everything worked anyway -- Algan encodes through imageio-ffmpeg's bundled
    build -- so the warning was pure noise, and pydub's own converter was left
    pointing at a name PATH cannot resolve.
    """
    env = dict(os.environ, PATH="/nonexistent")
    result = _fresh_python(
        "import algan, sys\nprint(sys.modules['pydub'].AudioSegment.converter)\n",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Couldn't find ffmpeg" not in result.stderr
    assert result.stdout.strip() != "ffmpeg", "pydub still has no usable binary"
    assert os.path.isfile(result.stdout.strip())


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
