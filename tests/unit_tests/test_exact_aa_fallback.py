from __future__ import annotations

import cv2
import numpy as np
import pytest
import taichi as ti

from algan import (
    BLUE,
    GREEN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    UP,
    YELLOW,
    Cube,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    Off,
    Scene,
    Square,
)
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing import tracer
from algan.rendering.raytracing.raster_pipeline import (
    get_exact_aa_fallback_counts,
    reset_exact_aa_fallback_counts,
)
from algan.scene_manager import SceneManager

pytestmark = pytest.mark.slow


@ti.func
def _subpixel_background(x, y, time):
    return ti.Vector([x, y, 0.2 + 0.0 * time, 0.0])


@ti.func
def _blue_background(x, y, time):
    return ti.Vector([0.0 * x, 0.0 * y, 1.0 + 0.0 * time, 0.0])


def test_ambiguous_pixels_launch_exact_requested_fallback_grid(tmp_path):
    """Coincident translucent regions cannot be ordered from scalar areas."""
    settings = SMOKE_TEST.set(resolution=(24, 24), anti_alias_level=2)
    snapshot = SETTINGS.snapshot()
    exact_before = rt_settings.ANALYTIC_AA_EXACT_COVERAGE
    force_before = rt_settings.ANALYTIC_AA_FORCE_FALLBACK
    SceneManager.reset()
    reset_exact_aa_fallback_counts()
    tracer._EXACT_AA_FALLBACK_PIXELS[0] = 0
    tracer._EXACT_AA_FALLBACK_PRIMARY_PATHS[0] = 0
    try:
        rt_settings.ANALYTIC_AA_EXACT_COVERAGE = True
        rt_settings.ANALYTIC_AA_FORCE_FALLBACK = False
        SETTINGS.raytracing.set(shadows=False, tonemapping=False)
        with Scene(video_settings=settings) as scene:
            with Off():
                first = Square(side_length=2.2, color=RED, opacity=0.6).rotate(24, OUT)
                first.spawn(animate=False)
                second = Square(side_length=2.0, color=BLUE, opacity=0.55).rotate(
                    -19, OUT
                )
                second.spawn(animate=False)
            scene.save_frame(
                tmp_path / "fallback.png",
                video_settings=settings,
                overwrite=True,
            )
    finally:
        rt_settings.ANALYTIC_AA_EXACT_COVERAGE = exact_before
        rt_settings.ANALYTIC_AA_FORCE_FALLBACK = force_before
        SETTINGS.restore(snapshot)
        SceneManager.reset()

    fallback_pixels = tracer._EXACT_AA_FALLBACK_PIXELS[0]
    assert fallback_pixels > 0
    assert tracer._EXACT_AA_FALLBACK_PRIMARY_PATHS[0] == fallback_pixels * 4
    assert get_exact_aa_fallback_counts()["depth_uncertainty"] > 0


@pytest.mark.parametrize(
    "aa_level",
    [pytest.param(1), pytest.param(2), pytest.param(4)],
)
def test_forced_fallback_matches_in_place_ssaa(tmp_path, aa_level):
    """Every fallback sample is the matching classic deterministic primary."""
    settings = SMOKE_TEST.set(resolution=(24, 24), anti_alias_level=aa_level)
    snapshot = SETTINGS.snapshot()
    analytic_before = rt_settings.ANALYTIC_AA
    exact_before = rt_settings.ANALYTIC_AA_EXACT_COVERAGE
    force_before = rt_settings.ANALYTIC_AA_FORCE_FALLBACK

    def render(path, *, analytic, exact, force):
        SceneManager.reset()
        rt_settings.ANALYTIC_AA = analytic
        rt_settings.ANALYTIC_AA_EXACT_COVERAGE = exact
        rt_settings.ANALYTIC_AA_FORCE_FALLBACK = force
        with Scene(video_settings=settings) as scene:
            with Off():
                square = Square(side_length=2.3, color=RED).rotate(23, OUT)
                square.spawn(animate=False)
            scene.save_frame(path, video_settings=settings, overwrite=True)

    try:
        SETTINGS.raytracing.set(shadows=False, tonemapping=False)
        reference_path = tmp_path / f"ssaa_{aa_level}.png"
        exact_path = tmp_path / f"fallback_{aa_level}.png"
        render(reference_path, analytic=False, exact=False, force=False)
        reset_exact_aa_fallback_counts()
        tracer._EXACT_AA_FALLBACK_PIXELS[0] = 0
        tracer._EXACT_AA_FALLBACK_PRIMARY_PATHS[0] = 0
        render(exact_path, analytic=True, exact=True, force=True)
    finally:
        rt_settings.ANALYTIC_AA = analytic_before
        rt_settings.ANALYTIC_AA_EXACT_COVERAGE = exact_before
        rt_settings.ANALYTIC_AA_FORCE_FALLBACK = force_before
        SETTINGS.restore(snapshot)
        SceneManager.reset()

    expected = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
    actual = cv2.imread(str(exact_path), cv2.IMREAD_UNCHANGED)
    assert expected is not None
    assert actual is not None
    assert expected.shape == actual.shape
    assert int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()) <= 2
    fallback_pixels = tracer._EXACT_AA_FALLBACK_PIXELS[0]
    assert fallback_pixels == settings.resolution[0] * settings.resolution[1]
    assert (
        tracer._EXACT_AA_FALLBACK_PRIMARY_PATHS[0]
        == fallback_pixels * aa_level * aa_level
    )


def test_forced_fallback_preserves_subpixel_background_correlation(tmp_path):
    """Visibility is composited with each supersampled background value."""
    settings = SMOKE_TEST.set(resolution=(16, 16), anti_alias_level=2)
    snapshot = SETTINGS.snapshot()
    analytic_before = rt_settings.ANALYTIC_AA
    exact_before = rt_settings.ANALYTIC_AA_EXACT_COVERAGE
    force_before = rt_settings.ANALYTIC_AA_FORCE_FALLBACK

    def render(path, *, analytic, exact, force):
        SceneManager.reset()
        rt_settings.ANALYTIC_AA = analytic
        rt_settings.ANALYTIC_AA_EXACT_COVERAGE = exact
        rt_settings.ANALYTIC_AA_FORCE_FALLBACK = force
        with Scene(
            video_settings=settings,
            background_frame=_subpixel_background,
        ) as scene:
            with Off():
                square = Square(side_length=2.3, color=RED).rotate(23, OUT)
                square.spawn(animate=False)
            scene.save_frame(path, video_settings=settings, overwrite=True)

    try:
        SETTINGS.raytracing.set(shadows=False, tonemapping=False)
        reference_path = tmp_path / "background_ssaa.png"
        exact_path = tmp_path / "background_fallback.png"
        render(reference_path, analytic=False, exact=False, force=False)
        reset_exact_aa_fallback_counts()
        tracer._EXACT_AA_FALLBACK_PIXELS[0] = 0
        tracer._EXACT_AA_FALLBACK_PRIMARY_PATHS[0] = 0
        render(exact_path, analytic=True, exact=True, force=True)
    finally:
        rt_settings.ANALYTIC_AA = analytic_before
        rt_settings.ANALYTIC_AA_EXACT_COVERAGE = exact_before
        rt_settings.ANALYTIC_AA_FORCE_FALLBACK = force_before
        SETTINGS.restore(snapshot)
        SceneManager.reset()

    expected = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
    actual = cv2.imread(str(exact_path), cv2.IMREAD_UNCHANGED)
    assert expected is not None
    assert actual is not None
    assert expected.shape == actual.shape
    assert int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()) <= 2
    fallback_pixels = tracer._EXACT_AA_FALLBACK_PIXELS[0]
    assert fallback_pixels == settings.resolution[0] * settings.resolution[1]
    assert tracer._EXACT_AA_FALLBACK_PRIMARY_PATHS[0] == fallback_pixels * 4


def test_reflective_circuit_transport_matches_classic_ray_path(tmp_path):
    """Circuit metalness uses the same packed channel on both primary paths."""
    settings = SMOKE_TEST.set(resolution=(16, 16), anti_alias_level=1)
    snapshot = SETTINGS.snapshot()
    globals_before = (
        rt_settings.ANALYTIC_AA,
        rt_settings.ANALYTIC_AA_EXACT_COVERAGE,
        rt_settings.ANALYTIC_AA_FORCE_FALLBACK,
        rt_settings.HYBRID_RASTER,
    )

    def render(path, *, analytic):
        SceneManager.reset()
        rt_settings.ANALYTIC_AA = analytic
        rt_settings.ANALYTIC_AA_EXACT_COVERAGE = analytic
        rt_settings.ANALYTIC_AA_FORCE_FALLBACK = False
        rt_settings.HYBRID_RASTER = analytic
        with Scene(
            video_settings=settings,
            background_frame=_blue_background,
        ) as scene:
            with Off():
                mirror = Square(side_length=20.0, color=RED)
                mirror.set_material(
                    MeshStandardMaterial(metalness=1.0, roughness=0.0)
                ).spawn(animate=False)
            scene.save_frame(path, video_settings=settings, overwrite=True)

    try:
        SETTINGS.raytracing.set(shadows=False, tonemapping=False, max_bounces=8)
        classic_path = tmp_path / "reflective_circuit_classic.png"
        exact_path = tmp_path / "reflective_circuit_exact.png"
        render(classic_path, analytic=False)
        render(exact_path, analytic=True)
    finally:
        (
            rt_settings.ANALYTIC_AA,
            rt_settings.ANALYTIC_AA_EXACT_COVERAGE,
            rt_settings.ANALYTIC_AA_FORCE_FALLBACK,
            rt_settings.HYBRID_RASTER,
        ) = globals_before
        SETTINGS.restore(snapshot)
        SceneManager.reset()

    classic = cv2.imread(str(classic_path), cv2.IMREAD_UNCHANGED)
    exact = cv2.imread(str(exact_path), cv2.IMREAD_UNCHANGED)
    assert classic is not None
    assert exact is not None
    centre = (settings.resolution[1] // 2, settings.resolution[0] // 2)
    assert int(np.abs(classic[centre].astype(np.int16) - exact[centre]).max()) <= 2
    # A red, fully metallic mirror over a blue background has no red diffuse
    # lane. Reading roughness (zero) as metalness would incorrectly leave it red.
    assert int(classic[centre][2]) < 8


def test_forced_fallback_matches_complete_deterministic_transport(tmp_path):
    """Fallback primaries retain transparency, shadows and continuations."""
    settings = SMOKE_TEST.set(resolution=(20, 20), anti_alias_level=2)
    snapshot = SETTINGS.snapshot()
    analytic_before = rt_settings.ANALYTIC_AA
    exact_before = rt_settings.ANALYTIC_AA_EXACT_COVERAGE
    force_before = rt_settings.ANALYTIC_AA_FORCE_FALLBACK
    hybrid_before = rt_settings.HYBRID_RASTER

    def render(path, *, analytic, exact, force):
        SceneManager.reset()
        rt_settings.ANALYTIC_AA = analytic
        rt_settings.ANALYTIC_AA_EXACT_COVERAGE = exact
        rt_settings.ANALYTIC_AA_FORCE_FALLBACK = force
        rt_settings.HYBRID_RASTER = analytic
        with Scene(video_settings=settings) as scene:
            with Off():
                # Coloured geometry behind the refractor makes its continuation
                # visible; the tilted metal circuit exercises reflection.
                Square(color=YELLOW).scale(0.55).move(
                    LEFT * 0.55 + UP * 0.2 - OUT * 1.6
                ).rotate(17, OUT).spawn(animate=False)
                glass = Cube(side_length=1.15).move(LEFT * 0.35).rotate(21, UP)
                glass.set_color(BLUE).set_material(
                    MeshPhysicalMaterial(
                        transmission=0.9,
                        roughness=0.02,
                        ior=1.45,
                    )
                )
                glass.spawn(animate=False)
                mirror = Square(color=RED).scale(0.7).move(RIGHT * 0.75)
                mirror.rotate(-48, UP).set_material(
                    MeshStandardMaterial(metalness=1.0, roughness=0.0)
                ).spawn(animate=False)
                overlay = Square(color=GREEN, opacity=0.45).scale(0.75)
                overlay.move(RIGHT * 0.15 + UP * 0.55 + OUT * 0.35)
                overlay.rotate(-19, OUT).spawn(animate=False)
            scene.save_frame(path, video_settings=settings, overwrite=True)

    try:
        SETTINGS.raytracing.set(shadows=True, tonemapping=False)
        reference_path = tmp_path / "transport_ssaa.png"
        exact_path = tmp_path / "transport_fallback.png"
        render(reference_path, analytic=False, exact=False, force=False)
        reset_exact_aa_fallback_counts()
        tracer._EXACT_AA_FALLBACK_PIXELS[0] = 0
        tracer._EXACT_AA_FALLBACK_PRIMARY_PATHS[0] = 0
        render(exact_path, analytic=True, exact=True, force=True)
    finally:
        rt_settings.ANALYTIC_AA = analytic_before
        rt_settings.ANALYTIC_AA_EXACT_COVERAGE = exact_before
        rt_settings.ANALYTIC_AA_FORCE_FALLBACK = force_before
        rt_settings.HYBRID_RASTER = hybrid_before
        SETTINGS.restore(snapshot)
        SceneManager.reset()

    expected = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
    actual = cv2.imread(str(exact_path), cv2.IMREAD_UNCHANGED)
    assert expected is not None
    assert actual is not None
    assert expected.shape == actual.shape
    assert int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()) <= 2
    fallback_pixels = tracer._EXACT_AA_FALLBACK_PIXELS[0]
    assert fallback_pixels == settings.resolution[0] * settings.resolution[1]
    assert tracer._EXACT_AA_FALLBACK_PRIMARY_PATHS[0] == fallback_pixels * 4
