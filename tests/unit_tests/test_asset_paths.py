"""An environment map is found the way every other image is.

The documentation promises that image paths are tried against the working
directory and then beside the script, for ``set_environment_map`` among
others. It opened the path as given, and took it only as a ``str``: a
``pathlib.Path`` raised ``Could not infer dtype of PosixPath``.
"""

import numpy as np
import pytest
from PIL import Image

from algan import Scene


@pytest.fixture
def script_dir(tmp_path, monkeypatch):
    from algan.settings import path_settings

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    Image.fromarray(pixels).save(scripts / "sky.png")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(path_settings, "_main_script_path", lambda: scripts / "v.py")
    return scripts


@pytest.mark.parametrize("source", ["sky.png", "path"], ids=["by name", "Path"])
def test_an_environment_map_beside_the_script_loads(script_dir, source):
    source = script_dir / "sky.png" if source == "path" else source
    with Scene() as scene:
        scene.set_environment_map(source)
        environment = scene.environment_map
    assert float(environment[..., 0].mean()) == pytest.approx(1.0)
    assert float(environment[..., 1:].abs().max()) == 0
