"""PN-soup conversion contracts used by cross-primitive ``become``."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from algan import BLACK, SMOKE_TEST, Off, Scene, Sphere
from algan.animatable_base.morph_conversions import convert_to_pn_soup
from algan.mobs.pn_mesh import PNMesh
from algan.scene_manager import SceneManager


def test_surface_conversion_reproduces_its_logical_pn_primitive():
    with Scene() as scene:
        sphere = Sphere(radius=0.6, scene=scene, add_to_scene=False)
        soup = convert_to_pn_soup(sphere, add_to_scene=False)

        expected = sphere.get_render_primitives()
        actual = soup.get_render_primitives()

        assert isinstance(soup, PNMesh)
        assert torch.equal(actual.corners, expected.corners)
        assert torch.allclose(actual.normals, expected.normals, atol=2e-7)
        assert torch.equal(actual.colors, expected.colors)
        assert torch.equal(actual.glow, expected.glow)
        assert actual.render_tolerance == expected.render_tolerance
        assert actual.render_tolerance_pixels == expected.render_tolerance_pixels
        assert actual.shader is expected.shader


def test_triangulated_circuit_conversion_is_planar_and_fills_the_circle():
    from algan import Circle

    with Scene() as scene:
        circle = Circle(radius=1.0, scene=scene, add_to_scene=False)
        soup = convert_to_pn_soup(circle, add_to_scene=False)

        corners = soup.location[0].reshape(-1, 3, 3)
        centered = corners - circle.get_center().reshape(1, 1, 3)
        normal = soup.normals[0, 0]
        distances = (centered * normal).sum(-1).abs()
        centroids = centered.mean(-2)

        assert distances.max().item() < 1e-5
        assert torch.allclose(
            soup.normals,
            normal.reshape(1, 1, 3).expand_as(soup.normals),
            atol=1e-6,
        )
        # An inverted polygon mask fills the bounding-box complement.  Valid
        # circle tiles stay inside the radius while reaching its broad extent.
        radial = centroids.norm(dim=-1)
        assert radial.max().item() < 1.05
        assert centroids[:, 0].amax().item() > 0.75
        assert centroids[:, 0].amin().item() < -0.75


def test_surface_and_pn_conversion_render_pixel_identically(tmp_path):
    settings = SMOKE_TEST.set(resolution=(96, 54))

    def render(name, convert):
        SceneManager.reset()
        with Scene(video_settings=settings) as scene:
            scene.set_background_color(BLACK)
            with Off():
                sphere = Sphere(
                    radius=0.8,
                    scene=scene,
                    add_to_scene=not convert,
                )
                mob = (
                    convert_to_pn_soup(sphere, add_to_scene=True) if convert else sphere
                )
                mob.spawn(animate=False)
            result = scene.save_frame(
                tmp_path / name,
                video_settings=settings,
                overwrite=True,
            )
        with Image.open(result.output_path) as image:
            return np.asarray(image.convert("RGB"))

    try:
        surface = render("surface", False)
        soup = render("soup", True)
    finally:
        SceneManager.reset()

    assert np.array_equal(surface, soup)
