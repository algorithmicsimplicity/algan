"""Watertight ray/triangle intersection (``ALGAN_WATERTIGHT_TRI``).

The shipped test is Moller-Trumbore with two matched epsilons:
``BARYCENTRIC_EPSILON`` dilates every triangle so a ray on a shared edge cannot
miss *both* neighbours and leave a crack, and ``TRIANGLE_EDGE_EPSILON`` removes
the duplicate hit the dilation then manufactures. Woop-Benthin-Wald needs
neither: it transforms the ray into a space where it is the +z axis, so a shared
edge's edge function is computed from the same two projected vertices in both
triangles and comes out as the exact negative, and a canonical-edge tie-break
picks one owner when the ray lands exactly on the edge.

The property that matters is therefore **exactly one hit per shared edge**: not
zero (a crack), not two (a duplicate the seam rule has to clean up).

``WATERTIGHT_TRI`` (``SETTINGS.raytracing.experimental.watertight_tri``, seeded
from ``ALGAN_WATERTIGHT_TRI``) is a ``ti.static`` gate over the kernel body, so
it is fixed for every kernel already compiled and one process can only exercise
one arm. These tests assert whichever arm the environment selected, and say so.
The
end-to-end evidence is separate and lives in the commit: with the hybrid raster
disabled so all visibility goes through the ray path, a Sphere/Cube/plane scene
moves 11 of 419904 pixels by at most 1 channel value across the flag.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _hit_counts(origins):
    """Hits on each of two triangles sharing the edge (0,0,0)-(1,0,0).

    Rays travel along -z from above, so the shared edge lies exactly under the
    rays whose y is 0.
    """
    import taichi as ti

    from algan.rendering.raytracing import raytrace_kernels_taichi as k

    n = origins.shape[0]
    out = ti.field(ti.i32, shape=(n, 2))
    src = ti.field(ti.f32, shape=(n, 3))
    src.from_torch(origins.contiguous())

    # Two triangles sharing edge A-B, consistently wound (as a welded grid or an
    # oriented Polyhedron produces).
    a = ti.math.vec3(0.0, 0.0, 0.0)
    b = ti.math.vec3(1.0, 0.0, 0.0)
    c = ti.math.vec3(0.0, 1.0, 0.0)
    d = ti.math.vec3(1.0, -1.0, 0.0)

    @ti.kernel
    def probe():
        for i in range(n):
            ro = ti.math.vec3(src[i, 0], src[i, 1], src[i, 2])
            rd = ti.math.vec3(0.0, 0.0, -1.0)
            ok0, _u0, _v0, _t0 = k._tri_hit(ro, rd, a, b, c)
            # Neighbour traverses the shared edge the other way: B then A.
            ok1, _u1, _v1, _t1 = k._tri_hit(ro, rd, b, a, d)
            out[i, 0] = ok0
            out[i, 1] = ok1

    probe()
    return out.to_torch()


def _watertight():
    from algan.rendering.raytracing import raytrace_kernels_taichi as k

    return k.watertight_tri()


def test_a_ray_exactly_on_a_shared_edge_hits_exactly_one_neighbour():
    """The property the whole epsilon pair exists to fake."""
    xs = torch.linspace(0.05, 0.95, 37)
    origins = torch.stack([xs, torch.zeros_like(xs), torch.full_like(xs, 5.0)], dim=-1)
    counts = _hit_counts(origins).sum(-1)
    if _watertight():
        assert torch.equal(counts, torch.ones_like(counts)), (
            "a watertight test must return exactly one hit on a shared edge, "
            f"got {sorted(set(counts.tolist()))}"
        )
    else:
        # The shipped arm: dilation makes both neighbours accept, which is
        # exactly the duplicate TRIANGLE_EDGE_EPSILON exists to discard.
        assert bool((counts == 2).all()), (
            "the dilated Moller-Trumbore arm is expected to double-hit a "
            f"shared edge, got {sorted(set(counts.tolist()))}"
        )


def test_rays_well_inside_one_triangle_hit_only_that_one():
    """A sanity floor: whatever the edge rule, the interior must be unambiguous.

    The second triangle is B-A-D with A=(0,0), B=(1,0), D=(1,-1), so its
    interior is the wedge below y=0 and above the line y=-x.
    """
    origins = torch.tensor(
        [
            [0.2, 0.3, 5.0],
            [0.1, 0.6, 5.0],
            [0.6, -0.3, 5.0],
            [0.7, -0.2, 5.0],
        ]
    )
    counts = _hit_counts(origins)
    assert counts[0].tolist() == [1, 0]
    assert counts[1].tolist() == [1, 0]
    assert counts[2].tolist() == [0, 1]
    assert counts[3].tolist() == [0, 1]


def test_a_ray_missing_both_triangles_hits_neither():
    origins = torch.tensor([[3.0, 3.0, 5.0], [-2.0, 0.5, 5.0]])
    counts = _hit_counts(origins)
    assert int(counts.sum()) == 0


def test_the_gate_is_on_by_default():
    """It changes the compiled kernel body, so it must not creep either way.

    This pinned OFF until the watertight test's correctness was qualified on CUDA
    and its cost was shown to sit under the measuring machine's noise floor
    (``DESIGN_mesh_identity.md`` ss3.2). It pins ON for the same reason it pinned
    OFF: a default that decides which intersection routine ships should move only
    when somebody means it, and flipping it moves rendered output on any scene
    with a secondary ray.
    """
    import os

    if "ALGAN_WATERTIGHT_TRI" in os.environ:
        pytest.skip("ALGAN_WATERTIGHT_TRI is set in this environment")
    assert _watertight()


def test_the_raycast_fallback_asks_tri_hit_rather_than_its_own_epsilon():
    """REGRESSION, source-level: the straddler path must not re-inline the test.

    ``_raycast_pixel`` in ``raster_taichi`` is the fallback for triangles that
    straddle the camera plane, where screen-space projection is invalid and the
    exact fixed-point fill rule ``_ss_pixel`` uses is therefore unavailable. It
    answers set membership by casting one ray per sub-pixel sample, and it used
    to do that with its own inline Moller-Trumbore dilated by
    ``BARYCENTRIC_EPSILON`` -- which double-claims a sample lying on an edge two
    straddling triangles share.

    It asks ``_tri_hit`` now, so it inherits whichever intersection the ray path
    ships, watertight included. The property that buys
    (exactly one hit per shared edge) is pinned by the tests above; what this
    pins is the WIRING, because that is what regresses: an inline copy of an
    intersection test drifts from the shared one silently, which is the same
    defect ``DESIGN_mesh_identity.md`` ss6.6.1 records for the analytic-AA gates
    and ss3.2 records for the three call sites ``_tri_hit`` replaced.

    Source-level for the reason ``test_environment`` audits ``os.environ`` use:
    the property is "this function does not do it itself", which no amount of
    calling it can demonstrate.
    """
    import ast
    import inspect

    from algan.rendering.raytracing import raster_taichi

    tree = ast.parse(inspect.getsource(raster_taichi))
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_raycast_pixel"
        ),
        None,
    )
    assert fn is not None, "_raycast_pixel not found; this test needs rewiring"

    calls = {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_tri_hit" in calls, (
        "_raycast_pixel must route its per-sample membership test through "
        "_tri_hit, so the straddler path inherits the shipped intersection "
        "instead of carrying a second copy of it"
    )
