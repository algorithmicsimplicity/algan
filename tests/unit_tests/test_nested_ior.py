"""Nested-IOR refraction: the stack arithmetic (``ALGAN_NESTED_IOR``, §H).

``_refract_ray`` reads its ``ior`` argument as n_inside/n_outside and picks the
side from ``sign(rd . n_out)``, so the whole of §H is a question of handing it
the RELATIVE index instead of the material's own. These tests drive the
``@ti.func``s that compute it directly from a tiny kernel -- the same shape as
``test_watertight_triangle.py`` -- because the arithmetic is what is easy to get
backwards, and a rendered frame cannot say which of the two directions was
wrong.

The pixel-level evidence that the mechanism engages at all is separate and lives
in ``benchmarks/_nested_ior_ab.py``, which renders a nested pair with the gate
off and on and requires the frame to move (and two control scenes not to).

Unmarked, so outside ``--fast``: nothing outside this feature can break it.
"""

# No ``from __future__ import annotations`` here, deliberately: the probes
# below define real ``@ti.kernel``s, and stringised annotations turn
# ``ti.types.ndarray()`` into text that Taichi rejects at compile time.
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _kernels():
    from algan.rendering.raytracing import wavefront_kernels_taichi as k

    return k


def _f32(values):
    return np.asarray(values, dtype=np.float32).tolist()


def _rows(stacks):
    """An ``rs_sca``-shaped array whose IOR columns hold ``stacks``.

    Columns 0-6 are filled with a value that is neither a valid depth nor a
    valid IOR, so a reader that indexes the wrong column produces an obviously
    wrong answer rather than a plausible one.
    """
    k = _kernels()
    rows = np.full((len(stacks), k.SCA_WIDTH_NESTED), -7.0, dtype=np.float32)
    for i, stack in enumerate(stacks):
        rows[i, k._SCA_IOR_DEPTH] = float(len(stack))
        for j, ior in enumerate(stack[: k.IOR_STACK_DEPTH]):
            rows[i, k._SCA_IOR_BASE + j] = float(ior)
    return rows


def _relative(stacks, iors, entering, nested=1):
    """``_relative_ior`` for each (stack, material ior, side) triple."""
    import taichi as ti

    k = _kernels()
    n = len(iors)
    rs = ti.ndarray(ti.f32, shape=(n, k.SCA_WIDTH_NESTED))
    rs.from_numpy(_rows(stacks))
    mat = ti.ndarray(ti.f32, shape=(n,))
    mat.from_numpy(np.asarray(iors, dtype=np.float32))
    ent = ti.ndarray(ti.i32, shape=(n,))
    ent.from_numpy(np.asarray(entering, dtype=np.int32))
    out = ti.ndarray(ti.f32, shape=(n,))

    @ti.kernel
    def run(
        rs_sca: ti.types.ndarray(),
        mio: ti.types.ndarray(),
        e: ti.types.ndarray(),
        o: ti.types.ndarray(),
        gate: ti.template(),
    ):
        for i in range(n):
            o[i] = k._relative_ior(rs_sca, i, mio[i], e[i] != 0, gate)

    run(rs, mat, ent, out, nested)
    return out.to_numpy()


def _written(stacks, iors, entering, refracting):
    """The (depth, entries) ``_write_ior_stack`` gives each row's child."""
    import taichi as ti

    k = _kernels()
    n = len(iors)
    both = np.zeros((2 * n, k.SCA_WIDTH_NESTED), dtype=np.float32)
    both[:n] = _rows(stacks)
    rs = ti.ndarray(ti.f32, shape=(2 * n, k.SCA_WIDTH_NESTED))
    rs.from_numpy(both)
    mat = ti.ndarray(ti.f32, shape=(n,))
    mat.from_numpy(np.asarray(iors, dtype=np.float32))
    ent = ti.ndarray(ti.i32, shape=(n,))
    ent.from_numpy(np.asarray(entering, dtype=np.int32))
    ref = ti.ndarray(ti.i32, shape=(n,))
    ref.from_numpy(np.asarray(refracting, dtype=np.int32))

    @ti.kernel
    def run(
        rs_sca: ti.types.ndarray(),
        mio: ti.types.ndarray(),
        e: ti.types.ndarray(),
        rf: ti.types.ndarray(),
        gate: ti.template(),
    ):
        for i in range(n):
            k._write_ior_stack(rs_sca, i, n + i, mio[i], e[i] != 0, rf[i], gate)

    run(rs, mat, ent, ref, 1)
    got = rs.to_numpy()[n:]
    depths = [int(round(v)) for v in got[:, k._SCA_IOR_DEPTH]]
    entries = got[:, k._SCA_IOR_BASE : k._SCA_IOR_BASE + k.IOR_STACK_DEPTH]
    return depths, entries


def test_the_state_width_grows_only_when_the_feature_is_on():
    k = _kernels()
    assert k.sca_width(False) == 7
    assert k.sca_width(True) == k._SCA_IOR_BASE + k.IOR_STACK_DEPTH


def test_an_empty_stack_reproduces_todays_behaviour_exactly():
    # Air on both sides: the relative index IS the material's, bit for bit,
    # which is what makes every un-nested scene byte-identical with the gate on.
    got = _relative([(), (), (), ()], [1.5, 1.5, 1.33, 1.33], [1, 0, 1, 0])
    assert got.tolist() == _f32([1.5, 1.5, 1.33, 1.33])


def test_the_gate_off_returns_the_material_index_untouched():
    got = _relative([(1.5,), (1.5, 1.2)], [1.2, 1.2], [1, 0], nested=0)
    assert got.tolist() == _f32([1.2, 1.2])


def test_entering_a_lighter_medium_from_a_denser_one_inverts_the_bend():
    # A sphere of ior 1.2 inside a box of ior 1.5: 1.2/1.5 = 0.8 < 1, so the
    # transmitted ray bends the OTHER WAY from the air-outside assumption. This
    # is the case the whole feature exists for.
    got = _relative([(1.5,)], [1.2], [1])
    assert got[0] == pytest.approx(1.2 / 1.5, rel=1e-6)


def test_exiting_takes_its_outside_from_the_enclosing_medium():
    # Leaving the inner sphere back into the outer glass: the outside is the
    # enclosing medium, not air.
    assert _relative([(1.5, 1.2)], [1.2], [0])[0] == pytest.approx(0.8, rel=1e-6)
    # Leaving the outer sphere, whose outside really is air.
    assert _relative([(1.5,)], [1.5], [0])[0] == pytest.approx(1.5, rel=1e-6)


def test_the_inside_index_always_comes_from_the_hit_never_from_the_stack():
    # The stack supplies the OUTSIDE only. Reading the inside off the stack's
    # top entry instead looks equivalent -- it is the value pushed at the
    # matching entry -- but ``ior`` is barycentrically interpolated per hit, so
    # a constant 1.5 comes back as 1.5*(w0+w1+w2) and the entry and exit hits
    # disagree in their last bits. That moved 660 channels of a SINGLE glass
    # sphere in benchmarks/_nested_ior_ab.py, a scene with no nesting at all.
    #
    # Pinned here by making the stack's top entry disagree with the hit
    # outright: the answer must follow the hit.
    assert _relative([(9.0,)], [1.5], [0])[0] == pytest.approx(1.5, rel=1e-6)
    assert _relative([(1.5, 9.0)], [1.2], [0])[0] == pytest.approx(0.8, rel=1e-6)


def test_exiting_a_medium_never_entered_falls_back_to_the_material_index():
    # A camera inside glass, or a ray whose entry interface was culled: depth 0
    # on an exit is exactly today's assumption, and must stay that way.
    assert _relative([()], [1.5], [0])[0] == pytest.approx(1.5, rel=1e-6)


def test_a_push_records_the_medium_and_a_pop_removes_it():
    depths, entries = _written(
        [(), (1.5,), (1.5, 1.2), (1.5,)],
        [1.5, 1.2, 1.2, 1.5],
        [1, 1, 0, 0],  # enter, enter, exit, exit
        [1, 1, 1, 1],
    )
    assert depths == [1, 2, 1, 0]
    assert entries[0][0] == pytest.approx(1.5)
    assert entries[1][:2].tolist() == pytest.approx([1.5, 1.2])
    assert entries[2][0] == pytest.approx(1.5)


def test_a_reflection_inherits_the_stack_unchanged():
    # A reflection and a coverage pass-through stay in the medium they were in,
    # whatever the surface they bounced off is made of.
    depths, entries = _written([(1.5, 1.2)], [1.7], [1], [0])
    assert depths == [2]
    assert entries[0][:2].tolist() == pytest.approx([1.5, 1.2])


def test_nesting_deeper_than_the_stack_stays_symmetric():
    # Past IOR_STACK_DEPTH the entries stop being recorded but the depth keeps
    # counting, so the matching exits pop back onto a stack that is still right.
    # Refusing to count instead -- the design doc's letter -- would shift every
    # later pop by one and corrupt the interfaces OUTSIDE the overflow too.
    k = _kernels()
    full = tuple(1.5 - 0.1 * i for i in range(k.IOR_STACK_DEPTH))
    depths, entries = _written([full], [1.05], [1], [1])
    assert depths == [len(full) + 1]
    assert entries[0].tolist() == pytest.approx(list(full))
    depths, entries = _written([full + (1.05,)], [1.05], [0], [1])
    assert depths == [len(full)]
    assert entries[0].tolist() == pytest.approx(list(full))


def test_a_full_nested_round_trip_returns_to_air():
    # enter A(1.5) -> enter B(1.2) -> exit B -> exit A. B's two interfaces agree
    # (the medium behind its outward normal is B either way, which is what lets
    # _refract_ray pick the direction from the normal's sign); A's outside
    # really is air, so its two agree at the material index.
    assert _relative([()], [1.5], [1])[0] == pytest.approx(1.5)
    assert _relative([(1.5,)], [1.2], [1])[0] == pytest.approx(0.8, rel=1e-6)
    assert _relative([(1.5, 1.2)], [1.2], [0])[0] == pytest.approx(0.8, rel=1e-6)
    assert _relative([(1.5,)], [1.5], [0])[0] == pytest.approx(1.5)


def test_the_setting_is_reachable_from_settings_and_round_trips():
    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt_settings

    before = rt_settings.nested_ior
    try:
        SETTINGS.raytracing.experimental.set(nested_ior=True)
        assert rt_settings.nested_ior is True
        assert rt_settings.nested_ior_mode() == 1
        SETTINGS.raytracing.experimental.set(nested_ior=False)
        assert rt_settings.nested_ior is False
        assert rt_settings.nested_ior_mode() == 0
    finally:
        rt_settings.set_nested_ior(before)


def test_the_experimental_toggle_is_not_settable_on_the_public_section():
    from algan import SETTINGS
    from algan.errors import AlganConfigurationError

    with pytest.raises(AlganConfigurationError, match="experimental"):
        SETTINGS.raytracing.set(nested_ior=True)
