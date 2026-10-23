"""Density grids: heterogeneous interiors tracked with null collisions."""

from __future__ import annotations

import math

import pytest
import torch

from algan import (
    BLACK,
    ORIGIN,
    OUT,
    RIGHT,
    SMOKE_TEST,
    UP,
    WHITE,
    DensityGrid,
    DirectionalLight,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    Prism,
    Scene,
    Sphere,
)
from algan.errors import AlganConfigurationError
from algan.rendering.raytracing.pt_volume import build_volume_tables
from algan.rendering.raytracing.settings import _MAT_DEFAULTS
from algan.rendering.raytracing.shading_taichi import (
    _MAT_DENSITY_VOLUME,
    _MAT_MEDIUM_CLOSED,
    _MID_PHYSICAL,
)
from algan.rendering.shaders.density_grid import (
    _density_key,
    _world_to_box,
)
from algan.rendering.taichi_runtime import init_taichi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDENTITY = torch.tensor([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])


def _samples(n=65536, seed=97):
    return torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(n)


def _tables(grid, xforms=None, key=7):
    """``(vol_i, vol_f)`` for one interior that names ``grid``.

    ``xforms`` is ``[T, 12]`` world-to-box rows; the identity puts the box on
    the unit cube.
    """
    mat = torch.tensor(_MAT_DEFAULTS).reshape(1, 1, -1).clone()
    mat[..., _MAT_MEDIUM_CLOSED] = 1
    mat[..., _MAT_DENSITY_VOLUME] = key
    merged = {
        "tri_mat": mat,
        "tri_mat_id": torch.full((1, 1), _MID_PHYSICAL, dtype=torch.int32),
    }
    xforms = IDENTITY.reshape(1, 12) if xforms is None else xforms
    snapshot = (((key, grid),), xforms.reshape(-1, 1, 12))
    vol_i, vol_f = build_volume_tables(merged, snapshot, xforms.shape[0], 1)
    return vol_i.to(DEVICE), vol_f.to(DEVICE)


def _track(
    grid,
    sigma_a,
    sigma_s,
    *,
    scatter=0,
    ray=(-0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 2.0),
    xforms=None,
    frame=0,
    n=65536,
):
    from pt_volume_probe_taichi import track_probe

    init_taichi()
    vol_i, vol_f = _tables(grid, xforms)
    out = torch.zeros((n, 6), device=DEVICE)
    track_probe(
        vol_i,
        vol_f,
        0,
        frame,
        torch.tensor(ray, dtype=torch.float32, device=DEVICE),
        torch.tensor(list(sigma_a) + list(sigma_s), dtype=torch.float32, device=DEVICE),
        scatter,
        _samples(n).to(DEVICE),
        out,
    )
    return out.cpu().double()


def _ramp(n=16, power=2.0):
    """Density ``3 x^power`` along x, constant across y and z."""
    x = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    return (3 * x**power).reshape(n, 1, 1).expand(n, 4, 4).clone()


def _optical_depth(values):
    """The exact integral of the interpolated density along x across the box.

    Clamped at the half cells beyond the outer samples and linear between
    samples, the trapezoid sum is exactly ``h * sum(samples)``.
    """
    return float(values[:, 0, 0].double().mean())


# --- the grid ---------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        torch.ones(4, 4),
        torch.ones(0, 4, 4),
        -torch.ones(2, 2, 2),
        torch.full((2, 2, 2), float("nan")),
        torch.full((2, 2, 2), float("inf")),
    ],
)
def test_invalid_density_values_are_rejected(values):
    with pytest.raises(ValueError, match="values"):
        DensityGrid(values)


@pytest.mark.parametrize(
    "bounds",
    [((0, 0, 0), (1, 1)), ((0, 0, 0), (1, 1, 0)), ((0, 0, 0), (1, float("nan"), 1))],
)
def test_invalid_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="bounds"):
        DensityGrid(torch.ones(2, 2, 2), bounds=bounds)


def test_material_density_must_be_a_grid():
    with pytest.raises(TypeError, match="DensityGrid"):
        MeshPhysicalMaterial(sigma_s=1, density=torch.ones(2, 2, 2))
    grid = DensityGrid(torch.ones(2, 2, 2))
    assert MeshPhysicalMaterial(sigma_s=1, density=grid).density is grid
    assert "density" not in MeshPhysicalMaterial(density=grid).get_shader_param_values()


def test_from_function_samples_cell_centres_in_box_coordinates():
    grid = DensityGrid.from_function(lambda p: p[..., 0] + 1, (4, 2, 3))
    assert grid.resolution == (4, 2, 3)
    expected = torch.tensor([-0.75, -0.25, 0.25, 0.75]) + 1
    assert torch.allclose(grid.values[:, 1, 2], expected)
    with pytest.raises(ValueError, match="shape"):
        DensityGrid.from_function(lambda p: p, 4)


def test_noise_is_deterministic_bounded_and_seeded():
    points = torch.rand(4096, 3, generator=torch.Generator().manual_seed(3)) * 4 - 2
    a = DensityGrid.noise(points, seed=5)
    assert torch.equal(a, DensityGrid.noise(points, seed=5))
    assert not torch.equal(a, DensityGrid.noise(points, seed=6))
    assert a.shape == (4096,)
    assert float(a.min()) >= 0
    assert float(a.max()) <= 1
    assert float(a.mean()) == pytest.approx(0.5, abs=0.05)
    # Smooth: nearby points take nearby values.
    b = DensityGrid.noise(points + 1e-4, seed=5)
    assert float((a - b).abs().max()) < 1e-2


@pytest.mark.parametrize("seed", [11, -1, 2**40])
def test_noise_takes_any_integer_seed(seed):
    # The hash works in int64: an unreduced seed of 11 already overflowed it.
    values = DensityGrid.noise(torch.zeros(2, 3) + 0.3, seed=seed, octaves=6)
    assert bool(torch.isfinite(values).all())


def test_majorant_bounds_the_interpolated_density_everywhere():
    """Including on and beside the coarse cells' own boundaries, where the
    tracker's cell walk may assign a point to either neighbour by rounding.
    """
    from pt_volume_probe_taichi import density_probe

    init_taichi()
    gen = torch.Generator().manual_seed(11)
    values = torch.rand(13, 7, 5, generator=gen) ** 4 * 10
    grid = DensityGrid(values)
    _density, majorant, _n, m = grid._packed_grid()
    majorant = majorant.reshape(m)
    points = torch.rand(200000, 3, generator=gen)
    # Snap a third of the points onto a coarse boundary along one axis.
    for axis in range(3):
        rows = slice(axis * 20000, (axis + 1) * 20000)
        cells = torch.randint(0, m[axis] + 1, (20000,), generator=gen)
        points[rows, axis] = (cells / m[axis]).clamp(0, 1)
    vol_i, vol_f = _tables(grid)
    out = torch.zeros(points.shape[0], device=DEVICE)
    density_probe(vol_i, vol_f, points.to(DEVICE), out)
    density = out.cpu()
    for nudge in (-1e-6, 0.0, 1e-6):
        u = (points + nudge).clamp(0, 1)
        cell = [(u[:, k] * m[k]).floor().long().clamp(0, m[k] - 1) for k in range(3)]
        bound = majorant[cell[0], cell[1], cell[2]]
        assert bool((density <= bound + 1e-5).all())


def test_kernel_density_is_cell_centred_trilinear_clamped_to_the_edges():
    from pt_volume_probe_taichi import density_probe

    init_taichi()
    gen = torch.Generator().manual_seed(5)
    values = torch.rand(6, 5, 4, generator=gen)
    vol_i, vol_f = _tables(DensityGrid(values))
    points = torch.rand(4096, 3, generator=gen)
    out = torch.zeros(points.shape[0], device=DEVICE)
    density_probe(vol_i, vol_f, points.to(DEVICE), out)
    # grid_sample's align_corners=False puts samples at cell centres, and its
    # border padding is the clamp: the same interpolant, from torch.
    grid = (points.flip(-1) * 2 - 1).reshape(1, -1, 1, 1, 3)
    reference = torch.nn.functional.grid_sample(
        values.reshape(1, 1, 6, 5, 4),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).reshape(-1)
    assert torch.allclose(out.cpu(), reference, atol=1e-5)


# --- tracking estimators ------------------------------------------------------


@pytest.mark.parametrize("values", [torch.ones(4, 4, 4), _ramp()], ids=["flat", "ramp"])
def test_ratio_tracking_matches_the_optical_depth_per_channel(values):
    sigma_t = (0.4, 1.1, 2.5)
    out = _track(DensityGrid(values), (0.1, 0.1, 0.5), (0.3, 1.0, 2.0))
    depth = _optical_depth(values)
    assert not out[:, 0].any(), "transmittance mode never collides"
    assert not out[:, 5].any()
    for c in range(3):
        expected = math.exp(-sigma_t[c] * depth)
        assert float(out[:, 2 + c].mean()) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("values", [torch.ones(4, 4, 4), _ramp()], ids=["flat", "ramp"])
def test_spectral_tracking_splits_into_transmittance_and_scattering(values):
    """Escaping weight estimates the transmittance; collision weight the
    scattered fraction, per channel, so a colored medium stays unbiased in
    every channel although each flight samples one distance.
    """
    sigma_a = (0.1, 0.1, 0.5)
    sigma_s = (0.3, 1.0, 2.0)
    out = _track(DensityGrid(values), sigma_a, sigma_s, scatter=1)
    depth = _optical_depth(values)
    hit = out[:, 0] > 0.5
    assert hit.any()
    assert (~hit).any()
    for c in range(3):
        sigma_t = sigma_a[c] + sigma_s[c]
        transmittance = math.exp(-sigma_t * depth)
        escaped = float(torch.where(~hit, out[:, 2 + c], 0).mean())
        scattered = float(torch.where(hit, out[:, 2 + c], 0).mean())
        assert escaped == pytest.approx(transmittance, abs=0.01)
        assert scattered == pytest.approx(
            sigma_s[c] / sigma_t * (1 - transmittance), abs=0.01
        )


def test_spectral_tracking_places_collisions_by_density():
    """Collision weight in each quarter of the ramp is the closed-form
    single-scatter integral there, ``int sigma_s rho(x) T(x) dx``.
    """
    values = _ramp(n=64, power=1.0)
    sigma_s = 1.2
    out = _track(DensityGrid(values), (0, 0, 0), (sigma_s,) * 3, scatter=1, n=131072)
    hit = out[:, 0] > 0.5
    x = out[:, 1] - 0.5  # the box starts half a unit along the ray
    xs = torch.linspace(0, 1, 20001, dtype=torch.float64)
    rho = torch.from_numpy(
        __import__("numpy").interp(
            xs.numpy(), ((torch.arange(64) + 0.5) / 64).numpy(), values[:, 0, 0].numpy()
        )
    )
    tau = torch.cumulative_trapezoid(sigma_s * rho, xs, dim=0)
    tau = torch.cat((torch.zeros(1, dtype=torch.float64), tau))
    density = sigma_s * rho * torch.exp(-tau)
    for lo in (0.0, 0.25, 0.5, 0.75):
        inside = (xs >= lo) & (xs <= lo + 0.25)
        expected = float(torch.trapezoid(density[inside], xs[inside]))
        in_bin = hit & (x >= lo) & (x < lo + 0.25)
        actual = float(torch.where(in_bin, out[:, 2], 0).mean())
        assert actual == pytest.approx(expected, abs=0.006), lo


def test_the_box_clips_the_segment_and_the_outside_is_vacuum():
    grid = DensityGrid(torch.full((2, 2, 2), 3.0))
    # Misses the unit box entirely.
    miss = _track(grid, (1, 1, 1), (1, 1, 1), ray=(-0.5, 2.0, 0.5, 1, 0, 0, 0, 2))
    assert (miss[:, 2:5] == 1).all()
    # Stops at the box's middle: half the optical depth.
    half = _track(grid, (0, 0, 0), (1, 1, 1), ray=(-0.5, 0.5, 0.5, 1, 0, 0, 0, 1))
    assert float(half[:, 2].mean()) == pytest.approx(math.exp(-1.5), abs=0.01)


def test_each_frame_reads_its_own_transform():
    grid = DensityGrid(torch.ones(2, 2, 2))
    away = IDENTITY.clone()
    away[9:12] = torch.tensor([-50.0, 0, 0])  # the box sits 50 units along +x
    xforms = torch.stack((IDENTITY, away))
    here = _track(grid, (0, 0, 0), (1, 1, 1), xforms=xforms, frame=0)
    there = _track(grid, (0, 0, 0), (1, 1, 1), xforms=xforms, frame=1)
    assert float(here[:, 2].mean()) == pytest.approx(math.exp(-1), abs=0.01)
    assert (there[:, 2:5] == 1).all()


def test_a_segment_that_exhausts_the_step_budget_is_absorbed_and_flagged():
    """One hot voxel sets the majorant of a region that is otherwise empty,
    so every tentative collision there is null: far past the budget.
    """
    values = torch.zeros(8, 8, 8)
    values[0, 0, 0] = 1e5
    out = _track(
        DensityGrid(values),
        (0, 0, 0),
        (1, 1, 1),
        ray=(-0.5, 0.2, 0.2, 1, 0, 0, 0, 2),
        n=64,
    )
    assert (out[:, 5] == 1).all()
    assert (out[:, 2:5] == 0).all()


# --- the Mob side -------------------------------------------------------------


def test_world_to_box_inverts_the_mob_frame():
    gen = torch.Generator().manual_seed(2)
    basis = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))[0]
    basis = basis * torch.tensor([[2.0], [0.5], [1.5]], dtype=torch.float64)
    location = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    lower = torch.tensor([-1.0, -2.0, 0.0], dtype=torch.float64)
    upper = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    scale = 1 / (upper - lower)
    local_to_box = torch.cat((torch.diag(scale), (-lower * scale)[:, None]), 1)
    xform = _world_to_box(local_to_box, location[None], basis[None])[0].double()
    local = (
        torch.rand(64, 3, generator=gen, dtype=torch.float64) * (upper - lower) + lower
    )
    world = location + local @ basis
    u = world @ xform[:9].reshape(3, 3).T + xform[9:]
    assert torch.allclose(u, (local - lower) * scale, atol=1e-5)
    # A frame scaled to nothing puts every point outside the box.
    empty = _world_to_box(
        local_to_box, location[None], torch.zeros(1, 3, 3, dtype=torch.float64)
    )
    assert torch.equal(empty[0, 9:], torch.full((3,), -1.0))
    assert not empty[0, :9].any()


def test_set_material_binds_one_key_to_the_whole_subtree():
    grid = DensityGrid(torch.ones(2, 2, 2))
    prism = Prism(width=2, height=1, depth=4)
    prism.set_material(MeshPhysicalMaterial(sigma_s=1, density=grid))
    key = _density_key(prism)
    assert key > 0
    assert {_density_key(d) for d in prism.get_descendants()} == {key}
    binding = prism._density_binding
    assert binding.owner is prism
    # The default box is the Prism's bounding box in its local frame: it
    # holds the solid, give or take the beads on its corners.
    a, b = binding.local_to_box[:, :3], binding.local_to_box[:, 3]
    for sign in (-1.0, 1.0):
        corner = a @ (sign * torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)) + b
        inset = corner if sign < 0 else 1 - corner
        assert bool(((inset >= 0) & (inset < 0.08)).all()), corner
    # Another material clears it.
    prism.set_material(MeshStandardMaterial())
    assert {_density_key(d) for d in prism.get_descendants()} == {0}


def test_explicit_bounds_are_in_the_mobs_own_unrotated_frame():
    grid = DensityGrid(torch.ones(2, 2, 2), bounds=((-1, -1, -1), (1, 1, 1)))
    sphere = Sphere().move_to(RIGHT * 3)
    sphere.rotate(90, UP)
    sphere.set_material(MeshPhysicalMaterial(sigma_s=1, density=grid))
    l2b = sphere._density_binding.local_to_box
    assert torch.allclose(l2b[:, :3], torch.eye(3, dtype=torch.float64) / 2)
    assert torch.allclose(l2b[:, 3], torch.full((3,), 0.5, dtype=torch.float64))


def test_a_clone_owns_its_own_binding_and_key():
    grid = DensityGrid(torch.ones(2, 2, 2))
    prism = Prism()
    prism.set_material(MeshPhysicalMaterial(sigma_s=1, density=grid))
    twin = prism.clone()
    assert _density_key(twin) not in (0, _density_key(prism))
    assert {_density_key(d) for d in twin.get_descendants()} == {_density_key(twin)}
    assert twin._density_binding.owner is twin
    assert twin._density_binding.grid is grid


def test_a_part_cloned_alone_carries_the_field_in_its_own_frame():
    grid = DensityGrid(torch.ones(2, 2, 2))
    prism = Prism(width=2, height=2, depth=2).move_to(RIGHT * 2)
    prism.set_material(MeshPhysicalMaterial(sigma_s=1, density=grid))
    part = prism.children[0]
    copy_ = part.clone()
    binding = copy_._density_binding
    assert binding.owner is copy_
    # The part's box, mapped back to the world, is the prism's box.
    src = prism._density_binding
    from algan.rendering.shaders.density_grid import _frame

    for mob, b in ((prism, src), (copy_, binding)):
        location, basis = _frame(mob)
        xform = _world_to_box(b.local_to_box, location[None], basis[None])[0].double()
        world = torch.tensor([[2.5, 0.3, -0.4]], dtype=torch.float64)
        u = world @ xform[:9].reshape(3, 3).T + xform[9:]
        if mob is prism:
            first = u
    assert torch.allclose(u, first, atol=1e-5)


def test_a_missing_owner_is_refused_rather_than_rendered_homogeneous():
    mat = torch.tensor(_MAT_DEFAULTS).reshape(1, 1, -1).clone()
    mat[..., _MAT_MEDIUM_CLOSED] = 1
    mat[..., _MAT_DENSITY_VOLUME] = 3
    merged = {
        "tri_mat": mat,
        "tri_mat_id": torch.full((1, 1), _MID_PHYSICAL, dtype=torch.int32),
    }
    with pytest.raises(AlganConfigurationError, match="not spawned"):
        build_volume_tables(merged, None, 1, 1)
    vol_i, _vol_f = build_volume_tables(merged, None, 1, 0)
    assert vol_i.tolist() == [0, 1], "no medium walk, nothing to look up"


def test_an_absorbing_density_interior_is_tracked_not_chorded():
    from contextlib import nullcontext
    from types import SimpleNamespace

    from algan.rendering.raytracing import tracer
    from algan.rendering.raytracing.pt_media import _prepare_media

    memory = SimpleNamespace(scope=lambda *_a, **_k: nullcontext())
    mat = torch.tensor(_MAT_DEFAULTS).reshape(1, 1, -1).repeat(1, 3, 1)
    mat[..., _MAT_MEDIUM_CLOSED] = 1
    mat[..., 27:30] = 0.5  # absorbing, not scattering
    mat[0, 0, _MAT_DENSITY_VOLUME] = 4  # only the first names a grid
    merged = {
        "tri_mat": mat,
        "tri_mat_id": torch.full((1, 3), _MID_PHYSICAL, dtype=torch.int32),
        "tri_obj": torch.tensor([[10, 20, 30]], dtype=torch.int32),
        "tri_extra": torch.ones((1, 3, 15)),
    }
    original = tracer._arena_copy
    tracer._arena_copy = lambda _memory, value: value.clone()
    try:
        _shells, extra, offset = _prepare_media(
            memory, merged, torch.full((1, 1), -1, dtype=torch.int32)
        )
    finally:
        tracer._arena_copy = original
    assert offset == 3
    assert not extra[0, 0, 12:15].any(), (
        "the grid's interior integrates its own absorption"
    )
    assert extra[0, 1:, 12:15].eq(1).all(), "plain absorbing glass keeps its chord"


def test_render_plan_names_density_media_and_the_deterministic_gap():
    from algan.rendering.raytracing.tracer import _build_render_plan

    merged = {
        "has_scattering_media": True,
        "has_density_media": True,
        "has_homogeneous_media": False,
    }
    deterministic = _build_render_plan(1, None, merged)
    assert deterministic.unsupported_features == (
        "heterogeneous media (density grids)",
    )
    path = _build_render_plan(2, None, merged)
    assert path.is_supported
    assert path.requested_features[0] == "heterogeneous media (density grids)"


# --- renders ------------------------------------------------------------------


_LINEAR = {
    "linear_color_space": False,
    "tonemapping": False,
}
_EXACT = {"post_process_tonemap": False, "pt_error_target": 0, "pt_firefly_clamp": 0}


@pytest.mark.parametrize("profile", ["flat", "ramp"])
def test_slab_single_scatter_matches_closed_form_for_any_profile(tmp_path, profile):
    """The homogeneous slab reference (test_pt_media), with density varying
    along the beam. Camera and light segments together cross the whole slab
    once, so single scattering depends on the profile only through its
    integral: ``L sigma_s D m exp(-sigma_t D m) / 4 pi`` with ``m`` the mean
    density. A flat grid of ones is exactly the homogeneous slab.
    """
    from test_path_tracer import _render_scene_exp

    ss, sa, depth, radiance = 0.8, 0.2, 2.0, 4.0
    z = (torch.arange(8, dtype=torch.float32) + 0.5) / 8
    profiles = {"flat": torch.ones(8), "ramp": 0.3 + 1.2 * z * z}
    values = profiles[profile].reshape(1, 1, 8).expand(2, 2, 8).clone()
    grid = DensityGrid(values, bounds=((-5, -5, -1), (5, 5, 1)))

    def build(scene):
        Scene.clear_lights()
        scene.set_background(BLACK)
        scene.camera.set_fov(1)
        slab = Prism(width=10, height=10, depth=depth)
        slab.set_material(
            MeshPhysicalMaterial(
                color=WHITE,
                transmission=1,
                ior=1,
                roughness=0,
                sigma_s=ss,
                attenuation_color=(math.exp(-sa),) * 3,
                attenuation_distance=1,
                density=grid,
            )
        )
        slab.spawn(animate=False)
        DirectionalLight(
            location=-OUT * 5, target=ORIGIN, color=WHITE, intensity=radiance
        ).spawn(animate=False)

    image = _render_scene_exp(
        tmp_path,
        f"slab_{profile}.png",
        build,
        512,
        video=SMOKE_TEST,
        max_bounces=1,
        experimental=_EXACT,
        **_LINEAR,
    )
    mean = float(values.mean())
    expected = (
        radiance
        * ss
        * depth
        * mean
        * math.exp(-(sa + ss) * depth * mean)
        / (4 * math.pi)
    )
    actual = float(image[..., :3].float().mean()) / 255
    assert actual == pytest.approx(expected, abs=0.005), (actual, expected)


def test_absorbing_density_dims_the_environment_by_its_optical_depth(tmp_path):
    from test_path_tracer import _render_scene_exp

    sa, depth = 0.6, 2.0
    z = (torch.arange(8, dtype=torch.float32) + 0.5) / 8
    values = (2 * z).reshape(1, 1, 8).expand(2, 2, 8).clone()
    grid = DensityGrid(values, bounds=((-5, -5, -1), (5, 5, 1)))

    def build(scene):
        Scene.clear_lights()
        scene.set_background(BLACK)
        scene.set_environment_map(torch.full((4, 8, 3), 0.5), ambient=False)
        scene.camera.set_fov(1)
        slab = Prism(width=10, height=10, depth=depth)
        slab.set_material(
            MeshPhysicalMaterial(
                color=WHITE,
                transmission=1,
                ior=1,
                roughness=0,
                attenuation_color=(math.exp(-sa),) * 3,
                attenuation_distance=1,
                density=grid,
            )
        )
        slab.spawn(animate=False)

    image = _render_scene_exp(
        tmp_path,
        "absorb.png",
        build,
        64,
        video=SMOKE_TEST,
        max_bounces=2,
        experimental=_EXACT,
        **_LINEAR,
    )
    expected = 0.5 * math.exp(-sa * depth * float(values.mean()))
    actual = float(image[..., :3].float().mean()) / 255
    assert actual == pytest.approx(expected, abs=0.01), (actual, expected)


def _half_fog(size):
    """A grid filling only the local -x half of a cube of side ``size``."""
    values = torch.zeros(2, 1, 1)
    values[0] = 1.0
    half = size / 2
    return DensityGrid(values, bounds=((-half,) * 3, (half,) * 3))


def _zoom(scene, factor):
    """Narrow the camera's view to ``factor`` of its height at any distance."""
    half = math.tan(math.radians(scene.camera.get_fov()) / 2) * factor
    scene.camera.set_fov(2 * math.degrees(math.atan(half)))


def _key_light():
    DirectionalLight(
        location=OUT * 5 + UP * 5, target=ORIGIN, color=WHITE, intensity=3
    ).spawn(animate=False)


def _half_cloud(tmp_path, name, turn):
    """A cube whose density fills only its local -x half, lit from the front.

    Against a black background the fog glows and the clear half shows
    nothing, so the bright half on screen is wherever the fog went.
    """
    from test_path_tracer import _render_scene_exp

    def build(scene):
        Scene.clear_lights()
        scene.set_background(BLACK)
        _zoom(scene, 0.3)
        _key_light()
        cube = Prism(width=2, height=2, depth=2)
        cube.set_material(
            MeshPhysicalMaterial(
                color=WHITE,
                transmission=1,
                ior=1,
                roughness=0,
                sigma_s=3,
                density=_half_fog(2),
            )
        )
        cube.rotate(turn, UP)
        cube.spawn(animate=False)

    return _render_scene_exp(
        tmp_path,
        name,
        build,
        32,
        video=SMOKE_TEST.set(resolution=(48, 48)),
        max_bounces=4,
        experimental=_EXACT,
        **_LINEAR,
    ).float()


def test_the_field_turns_with_its_mob(tmp_path):
    """Half a cube full of fog. Turned half a revolution after the material
    is set, the fog's half must swap sides on screen.
    """
    straight = _half_cloud(tmp_path, "straight.png", 0)
    turned = _half_cloud(tmp_path, "turned.png", 180)
    centre = (straight.shape[1] - 1) / 2

    def centroid(img):
        """The brightness-weighted mean column: where the glowing fog is."""
        column = img[..., :3].mean(-1).sum(0)
        assert float(column.sum()) > 0, "the fog must scatter light"
        return float((column * torch.arange(len(column))).sum() / column.sum())

    assert centroid(straight) < centre - 1
    assert centroid(turned) > centre + 1


def test_packed_owners_are_refused(tmp_path):
    from algan import SceneManager
    from algan.rendering.shaders.density_grid import _density_volume_snapshot
    from algan.utils.mob_utils import batch_mobs

    SceneManager.reset()
    try:
        with Scene() as scene:
            members = [Sphere(add_to_scene=False).move(RIGHT * x) for x in (-2, 2)]
            for m in members:
                m.set_material(
                    MeshPhysicalMaterial(
                        sigma_s=1, density=DensityGrid(torch.ones(2, 2, 2))
                    )
                )
            pack = batch_mobs(members)
            pack.spawn(animate=False)
            with pytest.raises(AlganConfigurationError, match="packs 2"):
                _density_volume_snapshot(scene, 0.0, 1.0, 1)
    finally:
        SceneManager.reset()


def _twins(tmp_path, turn):
    """A cube with fog in its local -x half at ``RIGHT``, and its clone, turned
    by ``turn`` about ``UP``, at ``LEFT``. Returns the brightness-weighted
    fog column on each side of the frame, as a fraction of its width.
    """
    from test_path_tracer import _render_scene_exp

    def build(scene):
        Scene.clear_lights()
        scene.set_background(BLACK)
        _zoom(scene, 0.2)
        _key_light()
        cube = Prism(width=1, height=1, depth=1)
        cube.set_material(
            MeshPhysicalMaterial(
                color=WHITE,
                transmission=1,
                ior=1,
                roughness=0,
                sigma_s=6,
                density=_half_fog(1),
            )
        )
        cube.move(RIGHT * 1)
        twin = cube.clone()
        twin.rotate(turn, UP)
        twin.move_to(RIGHT * -1)
        cube.spawn(animate=False)
        twin.spawn(animate=False)

    image = _render_scene_exp(
        tmp_path,
        f"twins_{turn}.png",
        build,
        32,
        video=SMOKE_TEST.set(resolution=(96, 48)),
        max_bounces=4,
        experimental=_EXACT,
        **_LINEAR,
    ).float()
    column = image[..., :3].mean(-1).sum(0)
    w = len(column)
    sides = []
    for half in (slice(0, w // 2), slice(w // 2, w)):
        mass = column[half]
        assert float(mass.sum()) > 0, "each cube's fog must scatter light"
        index = torch.arange(w)[half]
        sides.append(float((mass * index).sum() / mass.sum()) / w)
    return sides


def test_a_clone_renders_its_own_field_in_its_own_frame(tmp_path):
    """Each cube must show fog -- the clone's field follows the clone, not
    the original, whose box is empty where the clone stands -- and turning
    the clone moves its fog to its other half without moving the
    original's.
    """
    twin_still, original_still = _twins(tmp_path, 0)
    twin_turned, original_turned = _twins(tmp_path, 180)
    # Unturned, the twin's fog is its outer (left) half; turned, its inner.
    assert twin_turned > twin_still + 0.04
    assert original_turned == pytest.approx(original_still, abs=0.01)
