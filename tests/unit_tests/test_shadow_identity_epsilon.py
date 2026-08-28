"""The shadow acceptance floor scales with the scene, not with a constant.

Identity-aware shadow rejection (``shadow_identity_reject``,
DESIGN_mesh_identity_open.md ssI) replaces the absolute ``MIN_HIT_DISTANCE``
= 1e-4 on the shadow path with a floor proportional to the batch's own scene
scale. That constant is only ever right for a scene about ten units across:
smaller geometry loses contact shadows it should keep, larger geometry gets
acne. ``_shadow_identity_epsilons`` is where the policy lives, deliberately on
the host in plain torch rather than inside a kernel, so it can be asserted
directly.

These are pure tensor assertions -- no render, no Taichi. They are a feature
test of the shadow policy rather than of anything the timeline or the Scene
can break, so they stay out of the fast suite.
"""

import pytest
import torch

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.raster_pipeline import (
    _shadow_identity_epsilons,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import MIN_HIT_DISTANCE
from algan.settings import SETTINGS

rt = SETTINGS.raytracing


def _merged(scale):
    """One triangle spanning a cube of side ``scale`` at the origin."""
    tri = torch.tensor(
        [[0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, scale, scale]],
        dtype=torch.float32,
    )
    return {"tri_pos": tri.unsqueeze(0)}


@pytest.fixture
def restore_settings():
    before = (
        rt.experimental.shadow_eps_relative,
        rt.experimental.shadow_near_fraction,
    )
    yield
    rt.experimental.set(shadow_eps_relative=before[0], shadow_near_fraction=before[1])


def test_floor_is_proportional_to_scene_scale(restore_settings):
    """Ten times the geometry, ten times the floor -- that is the whole point."""
    rt.experimental.set(shadow_eps_relative=1e-5, shadow_near_fraction=0.0)
    small, _ = _shadow_identity_epsilons(_merged(1.0))
    big, _ = _shadow_identity_epsilons(_merged(10.0))
    assert big == pytest.approx(small * 10.0, rel=1e-5)


def test_default_reproduces_the_constant_it_replaces(restore_settings):
    """At the scale the old constant was chosen for, the new floor matches it.

    ``MIN_HIT_DISTANCE`` = 1e-4 is 1e-5 of a ten-unit scene, so a scene of
    roughly that size must land on the same number the renderer used before.
    Without this the change would silently re-tune every existing scene.
    """
    rt.experimental.set(shadow_eps_relative=1e-5, shadow_near_fraction=0.0)
    # A triangle spanning ten units has a bounding-box diagonal near ten.
    eps_self, _ = _shadow_identity_epsilons(_merged(10.0 / (3.0**0.5)))
    assert eps_self == pytest.approx(MIN_HIT_DISTANCE, rel=0.05)


def test_near_fraction_scales_the_same_mesh_tier(restore_settings):
    """The same-mesh floor is exactly its fraction of the self floor."""
    rt.experimental.set(shadow_eps_relative=1e-4, shadow_near_fraction=0.25)
    eps_self, eps_near = _shadow_identity_epsilons(_merged(4.0))
    assert eps_near == pytest.approx(eps_self * 0.25, rel=1e-6)


def test_primitive_precise_is_the_default(restore_settings):
    """Default 0 means only the ray's own triangle keeps a floor."""
    rt.experimental.set(shadow_eps_relative=1e-5, shadow_near_fraction=0.0)
    _, eps_near = _shadow_identity_epsilons(_merged(7.0))
    assert eps_near == 0.0


@pytest.mark.parametrize("fraction", [-0.5, float("nan")])
def test_same_mesh_floor_is_never_negative(fraction, monkeypatch, restore_settings):
    """A negative floor would let geometry BEHIND the origin cast a shadow.

    ``t > eps_near`` is the acceptance test, so a negative ``eps_near`` admits
    hits at ``t <= 0`` on every non-source triangle of the source mesh. The
    self floor has always been guarded; this pins the same guarantee on the
    same-mesh tier, which is reachable from a single bad env var.

    Written straight to the module global, because the env var is: ``env_float``
    parses ``-0.5`` and ``nan`` happily -- it only falls back on a value it
    cannot parse at all -- so ``ALGAN_SHADOW_NEAR_FRACTION=-0.5`` lands here
    with nothing in between. ``SETTINGS.raytracing.experimental`` now refuses
    both, which is why this no longer goes through it, and is also why the
    clamp below is still load-bearing rather than dead: the settings API is not
    the only way in.
    """
    rt.experimental.set(shadow_eps_relative=1e-5, shadow_near_fraction=0.0)
    monkeypatch.setattr(rt_settings, "shadow_near_fraction", fraction)
    eps_self, eps_near = _shadow_identity_epsilons(_merged(7.0))
    assert eps_self > 0.0
    assert eps_near == 0.0


@pytest.mark.parametrize(
    "tri_pos",
    [
        torch.zeros((1, 0, 9), dtype=torch.float32),  # no triangles at all
        torch.zeros((1, 1, 9), dtype=torch.float32),  # every vertex coincident
        torch.full((1, 1, 9), float("nan"), dtype=torch.float32),
    ],
    ids=["empty", "degenerate", "nan"],
)
def test_degenerate_geometry_falls_back_to_the_constant(tri_pos, restore_settings):
    """A zero or NaN floor would accept every self-hit and shade garbage.

    A batch can legitimately carry no triangles (a 2-D-only scene still runs
    the shadow path for its circuits), and NaN geometry reaches the renderer
    from a malformed import. Neither may produce a floor of zero.
    """
    rt.experimental.set(shadow_eps_relative=1e-5, shadow_near_fraction=0.0)
    eps_self, eps_near = _shadow_identity_epsilons({"tri_pos": tri_pos})
    assert eps_self == pytest.approx(MIN_HIT_DISTANCE)
    assert eps_near == 0.0
