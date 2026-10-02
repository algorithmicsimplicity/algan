"""Native and vendored SVG/text imports use authored appearance and layout."""

import base64
import io
import json
import os
import subprocess
import sys

import manim as mn
import pytest
import torch
from PIL import Image

import algan
from algan.mobs import text as text_module
from algan.utils import manim_svg_cache


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path):
    algan.SETTINGS.paths.cache_directory = str(tmp_path / "cache")
    manim_svg_cache._MEM_CACHE.clear()
    text_module._TEX_GLYPH_MEMO.clear()
    yield
    manim_svg_cache._MEM_CACHE.clear()
    text_module._TEX_GLYPH_MEMO.clear()


@pytest.mark.parametrize("scope", ["element", "group", "root", "nested", "style"])
def test_svg_opacity_multiplies_fill_and_stroke(tmp_path, scope):
    shape = '<rect width="12" height="12" fill="red" fill-opacity="0.5" stroke="blue" stroke-opacity="0.5" {} />'
    shape = shape.format('opacity="0.25"' if scope == "element" else "")
    if scope in ("group", "nested", "style"):
        attr = 'style="opacity:0.25"' if scope == "style" else 'opacity="0.25"'
        shape = f"<g {attr}>{shape}</g>"
    root_attr = 'opacity="0.25"' if scope in ("root", "nested") else ""
    path = tmp_path / f"{scope}.svg"
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" {root_attr}>{shape}</svg>'
    )
    expected = 0.5 * 0.25 * (0.25 if scope == "nested" else 1)
    imported = mn.SVGMobject(str(path))
    rectangle = imported.submobjects[0]
    assert rectangle.get_fill_opacity() == pytest.approx(expected, abs=1 / 255)
    assert rectangle.get_stroke_opacity() == pytest.approx(expected, abs=1 / 255)
    with algan.Scene(), algan.Off():
        native = algan.SVGMob(str(path))
        leaves = [
            mob
            for mob in native.get_descendants()
            if hasattr(mob, "control_points") and not getattr(mob, "empty", False)
        ]
        assert leaves
        assert all(
            float(mob.color[..., -1].reshape(-1)[0])
            == pytest.approx(expected, abs=1 / 255)
            for mob in leaves
        )


@pytest.mark.parametrize("height", [None, 2])
def test_svg_embedded_raster_matches_authored_vector_dimensions(tmp_path, height):
    buffer = io.BytesIO()
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    path = tmp_path / "image.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
        '<rect width="12" height="12"/>'
        f'<image x="20" y="0" width="12" height="12" xlink:href="data:image/png;base64,{encoded}"/>'
        "</svg>"
    )
    imported = mn.SVGMobject(str(path), height=height)
    rectangle, raster = imported.submobjects
    assert raster.width == pytest.approx(rectangle.width)
    assert raster.height == pytest.approx(rectangle.height)
    with algan.Scene(), algan.Off():
        native = algan.SVGMob(str(path), height=height)
        rasters = [
            mob for mob in native.get_descendants() if isinstance(mob, algan.ImageMob)
        ]
        vectors = [
            mob
            for mob in native.get_descendants()
            if hasattr(mob, "control_points") and not getattr(mob, "empty", False)
        ]
        assert len(rasters) == len(vectors) == 1
        torch.testing.assert_close(
            rasters[0].get_length_in_direction(algan.RIGHT),
            vectors[0].get_length_in_direction(algan.RIGHT),
        )
        torch.testing.assert_close(
            rasters[0].get_length_in_direction(algan.UP),
            vectors[0].get_length_in_direction(algan.UP),
        )


def test_text_gradient_change_matches_isolated_fresh_process(tmp_path):
    algan.SETTINGS.paths.cache_directory = str(tmp_path / "warm")
    text_module._TEX_GLYPH_MEMO.clear()
    manim_svg_cache._MEM_CACHE.clear()
    with algan.Scene(), algan.Off():
        first = algan.Text("AB", gradient_map={"AB": (algan.RED, algan.BLUE)})
        second = algan.Text("AB", gradient_map={"AB": (algan.GREEN, algan.YELLOW)})
        colors = torch.cat(
            [glyph.color.reshape(-1, 5) for glyph in second.character_mobs]
        ).cpu()
        old = torch.cat(
            [glyph.color.reshape(-1, 5) for glyph in first.character_mobs]
        ).cpu()
        assert not torch.equal(colors, old)
    script = (
        "import json\nfrom algan import *\n"
        "with Scene(), Off():\n"
        " t = Text('AB', gradient_map={'AB': (GREEN, YELLOW)})\n"
        " print(json.dumps([row for glyph in t.character_mobs for row in glyph.color.reshape(-1,5).tolist()]))\n"
    )
    env = {**os.environ, "ALGAN_HOME": str(tmp_path / "fresh"), "ALGAN_USE_DAEMON": "0"}
    env.pop("ALGAN_CACHE_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    expected = torch.tensor(json.loads(result.stdout.strip().splitlines()[-1]))
    torch.testing.assert_close(colors, expected)
