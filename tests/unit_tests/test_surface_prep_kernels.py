"""The CPU batch-prep kernels: dispatch, parity, and mesh watertightness.

Three prep rows run through Taichi when the arch is the CPU
(``surface_kernels_taichi``, ``triangle_primitive_kernels_taichi``). Two of them
are exact copies of what torch did; the normals kernel is deliberately allowed
to differ in the last ulp or two, so watertightness has to be asserted rather
than inherited from byte-identity.

Watertight here means what the renderer needs it to mean: grid points that are
geometrically the same point must come out of prep carrying the same normal.
Logical PN patches build their curvature from corner normals, so two sides of a
closed seam that disagreed would crack the *geometry*, not merely the shading.

Every test skips on a GPU arch, where the kernels are not dispatched at all.
"""

import pytest
import torch

import algan.rendering.taichi_runtime as taichi_runtime
from algan.mobs.surfaces.surface import (
    compute_grid_vertex_normals,
    get_grid_to_triangle_indices,
    grid_to_triangle_vertices,
)
from algan.rendering.primitives.triangle_primitive import _bake_glow_and_opacity
from algan.rendering.taichi_runtime import cpu_prep_kernel_enabled, taichi_arch_is_cpu

pytestmark = pytest.mark.skipif(
    not taichi_arch_is_cpu(), reason="the prep kernels only dispatch on a CPU arch"
)


@pytest.fixture
def all_kernels_on(monkeypatch):
    """Opt the two off-by-default kernels in for the parity tests.

    Only ``cpunormals`` ships enabled -- the gather and the colour bake measured
    slower than torch (see ``_CPU_PREP_KERNELS_ON_BY_DEFAULT``). Their
    correctness still has to be asserted, so the tests that exercise them turn
    them on explicitly.
    """
    monkeypatch.setattr(
        taichi_runtime, "_OPT_ENABLED", frozenset(("cpugather", "cpucolors"))
    )


def _sphere_grid(w=24, h=12, frames=2):
    """A closed-in-x sphere with collapsed poles: both weld cases at once."""
    u = torch.linspace(0, 2 * torch.pi, w)
    v = torch.linspace(0, torch.pi, h)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    g = torch.stack(
        (vv.sin() * uu.cos(), vv.sin() * uu.sin(), vv.cos().expand_as(uu)), -1
    )
    return g.unsqueeze(0).expand(frames, -1, -1, -1).contiguous()


def _cylinder_grid(w=16, h=8, frames=2):
    """Closed in x, open in y -- the seam case without the poles."""
    u = torch.linspace(0, 2 * torch.pi, w)
    v = torch.linspace(-1, 1, h)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    g = torch.stack((uu.cos(), uu.sin(), vv), -1)
    return g.unsqueeze(0).expand(frames, -1, -1, -1).contiguous()


def _torch_arm(monkeypatch):
    """Force every prep kernel to decline, leaving the shipped torch path."""
    import algan.animation_timeline.timeline as tl

    monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset(("cpukernels",)))


def test_the_kernels_are_actually_dispatched(all_kernels_on):
    """Guard against the whole suite passing because nothing ever ran.

    The gather declined silently once already: its index table is flattened to
    ``[triangles * 3]``, not ``[triangles, 3]``, so a shape guard written for
    the latter fell through to torch on every call.
    """
    assert cpu_prep_kernel_enabled("cpunormals")
    assert cpu_prep_kernel_enabled("cpugather")
    assert cpu_prep_kernel_enabled("cpucolors")

    from algan.mobs.surfaces.surface import (
        _gather_triangles_on_cpu,
        _sides_and_crosses_on_cpu,
    )

    grid = _sphere_grid()
    assert _sides_and_crosses_on_cpu(grid) is not None
    flat = grid.reshape(*grid.shape[:-3], -1, 3)
    indices = get_grid_to_triangle_indices(grid.shape[-3], grid.shape[-2], grid.device)
    assert _gather_triangles_on_cpu(flat, indices) is not None


def test_opt_disable_returns_the_torch_path(monkeypatch, all_kernels_on):
    _torch_arm(monkeypatch)
    assert not cpu_prep_kernel_enabled("cpunormals")
    assert not cpu_prep_kernel_enabled("cpugather")
    assert not cpu_prep_kernel_enabled("cpucolors")


def test_only_the_normals_kernel_is_on_by_default(monkeypatch):
    """The two that measured slower than torch must stay opt-in."""
    monkeypatch.setattr(taichi_runtime, "_OPT_ENABLED", frozenset())
    assert cpu_prep_kernel_enabled("cpunormals")
    assert not cpu_prep_kernel_enabled("cpugather")
    assert not cpu_prep_kernel_enabled("cpucolors")


@pytest.mark.parametrize(
    "make_grid", [_sphere_grid, _cylinder_grid], ids=["sphere", "cylinder"]
)
def test_normals_match_the_torch_path_closely(make_grid, monkeypatch):
    """Not bit-identical by design, but the deviation must stay at the last ulp."""
    grid = make_grid()
    kernel = compute_grid_vertex_normals(grid)
    with monkeypatch.context() as patch:
        _torch_arm(patch)
        reference = compute_grid_vertex_normals(grid)

    assert kernel.shape == reference.shape
    assert torch.isfinite(kernel).all()
    # Normals are unit length after the shared normalize, so an absolute
    # tolerance is meaningful and a couple of ulps is ~2.4e-7.
    torch.testing.assert_close(kernel, reference, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    "make_grid", [_sphere_grid, _cylinder_grid], ids=["sphere", "cylinder"]
)
def test_closed_seam_normals_are_identical_on_both_sides(make_grid):
    """The x seam of a closed grid must carry one normal, not two close ones.

    Bitwise, not approximately: the seam merge assigns one computed value to
    both columns, so anything else means the merge stopped seeing the columns
    as closed.
    """
    normals = compute_grid_vertex_normals(make_grid())
    assert torch.equal(normals[..., 0, :, :], normals[..., -1, :, :])


def test_pole_normals_collapse_to_one_vector():
    """Every column of a collapsed pole row is one vertex, so one normal."""
    normals = compute_grid_vertex_normals(_sphere_grid())
    for row in (0, -1):
        pole = normals[..., :, row, :]
        assert torch.equal(pole, pole[..., :1, :].expand_as(pole))


def test_welded_gather_shares_vertices_so_the_mesh_stays_closed(all_kernels_on):
    """A welded seam gathers column 0 twice, not two columns 1.7e-7 apart."""
    grid = _cylinder_grid()
    welded = grid_to_triangle_vertices(grid, weld=(True, False, False))
    plain = grid_to_triangle_vertices(grid, weld=(False, False, False))
    assert welded.shape == plain.shape

    W, H = grid.shape[-3], grid.shape[-2]
    flat = grid.reshape(*grid.shape[:-3], W * H, 3)
    seam = get_grid_to_triangle_indices(W, H, grid.device, (True, False, False))
    # No triangle references the duplicate column under the wrap weld.
    assert int(seam.max()) < (W - 1) * H
    torch.testing.assert_close(welded, flat[..., seam, :], rtol=0, atol=0)


@pytest.mark.parametrize(
    "weld",
    [(False, False, False), (True, False, False), (True, True, True)],
    ids=["open", "wrap_x", "wrap_and_poles"],
)
def test_gather_is_byte_identical_to_the_advanced_index(
    weld, monkeypatch, all_kernels_on
):
    grid = _sphere_grid()
    kernel = grid_to_triangle_vertices(grid, weld=weld)
    with monkeypatch.context() as patch:
        _torch_arm(patch)
        reference = grid_to_triangle_vertices(grid, weld=weld)
    assert torch.equal(kernel, reference)


@pytest.mark.parametrize(
    ("opacity_shape", "glow_shape"),
    [((1, 1, 1), (1, 1, 1)), ((1, 1, 1), (1, 7, 1)), ((3, 7, 1), (3, 7, 1))],
    ids=["both scalar", "per-row glow", "both per-row"],
)
def test_colour_bake_is_byte_identical(opacity_shape, glow_shape, all_kernels_on):
    """Broadcast from one element or carried per row, the kernel must match."""
    from algan.utils.tensor_utils import broadcast_all

    generator = torch.Generator().manual_seed(3)
    colors = torch.rand(3, 7, 5, generator=generator)
    opacity = torch.rand(opacity_shape, generator=generator)
    glow = torch.rand(glow_shape, generator=generator)
    colors, opacity, glow = broadcast_all([colors, opacity, glow], ignored_dims=[-1])

    kernel = _bake_glow_and_opacity(colors, opacity, glow)
    reference = colors.clone()
    reference[..., -2:-1] += glow
    reference[..., -1:] *= opacity

    assert torch.equal(kernel, reference)
    # The input must survive: shader_param_values reads the unmodified colours
    # after the bake, which a clone guaranteed and an in-place kernel would not.
    assert not torch.equal(kernel, colors)


def test_empty_inputs_fall_back_instead_of_reaching_a_kernel(all_kernels_on):
    """Empty inputs must never reach a kernel.

    A zero-extent tensor has nothing for Taichi to bind, and the compact form of
    a broadcast channel would index element 0 of an empty buffer.
    """
    from algan.mobs.surfaces.surface import (
        _gather_triangles_on_cpu,
        _sides_and_crosses_on_cpu,
    )
    from algan.utils.tensor_utils import broadcast_all

    assert _sides_and_crosses_on_cpu(torch.zeros(0, 4, 4, 3)) is None
    assert (
        _gather_triangles_on_cpu(
            torch.zeros(0, 16, 3), torch.zeros(6, dtype=torch.int64)
        )
        is None
    )
    assert (
        _gather_triangles_on_cpu(
            torch.zeros(2, 16, 3), torch.zeros(0, dtype=torch.int64)
        )
        is None
    )

    # Matching empty leading dims: torch will not broadcast 0 rows against 1.
    colors, opacity, glow = broadcast_all(
        [torch.zeros(0, 5), torch.zeros(0, 1), torch.zeros(0, 1)], ignored_dims=[-1]
    )
    baked = _bake_glow_and_opacity(colors, opacity, glow)
    assert baked.shape == colors.shape
    assert baked.numel() == 0


def test_colour_bake_leaves_a_non_float32_input_to_torch():
    from algan.utils.tensor_utils import broadcast_all

    colors = torch.rand(2, 4, 5, dtype=torch.float64)
    opacity = torch.full((1, 1, 1), 0.5, dtype=torch.float64)
    glow = torch.full((1, 1, 1), 0.25, dtype=torch.float64)
    colors, opacity, glow = broadcast_all([colors, opacity, glow], ignored_dims=[-1])

    baked = _bake_glow_and_opacity(colors, opacity, glow)
    reference = colors.clone()
    reference[..., -2:-1] += glow
    reference[..., -1:] *= opacity
    assert baked.dtype == torch.float64
    assert torch.equal(baked, reference)
