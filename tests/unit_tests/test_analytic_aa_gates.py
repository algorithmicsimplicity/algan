"""The analytic-AA group value, and the host agreement it encodes.

``aa_grp`` is the emission's grouping value. Its readers have to agree about
it, and historically they were not all in the same language: kernels tested it
through ``_aa_run_full``, and the host's **emission truncation** has to apply
the relaxed gate's mitigation whenever a reader takes the relaxed gate.

They drifted, which is why this file exists. ``analytic_aa_one_mesh`` sets
``aa_grp = 3``, and ``_aa_run_full`` treats 3 as the relaxed gate (the one-mesh
rule needs it: the near sheet's exact area is only worth reading once the gate
lets it be read). But the truncation tested ``analytic_aa_run_full`` alone, so
with ONE_MESH on and RUN_FULL off the relaxed semantics ran over fragment
lists whose empty-mask **area donors had already been discarded** -- the exact
configuration ``DESIGN_mesh_identity.md`` ss6.3.2 measured as an interior notch,
and the one ss6.6 documents as the shipping shape of the rule. Measured on CUDA
before the fix: a flat quad's ink wobble improved 8% where wiring both gates gave
63%, and a default ``Cylinder`` 47% where both gave 78%.

The fragment walk's higher ladder rungs (occlusion-write scaling, run caps,
exact run lanes) are deleted with the walk (DESIGN_sheet_resolve.md ss7); the
sheet resolve's per-sheet claim arithmetic subsumes them, and its only
emission-side dependency is the truncation gate pinned here.

Pure settings/predicate assertions -- no render, no Taichi kernel launch -- so
this is cheap. It is NOT marked ``fast``: it breaks when the analytic-AA gating
is worked on, which is its own subsystem (``tests/README.md``'s rule).
"""

from __future__ import annotations

import pytest

from algan.rendering.raytracing import raster_pipeline as rp
from algan.rendering.raytracing import settings as rt_settings

# A batch with triangles under the run representation, which is what the gates
# are about; the bezier arm rides the same value.
AA_BEZ, AA_TRI = 0, 4


@pytest.fixture
def restore_aa():
    """Undo the analytic-AA settings this module writes."""
    before = (
        rt_settings.analytic_aa,
        rt_settings.analytic_aa_seam,
        rt_settings.analytic_aa_run_full,
        rt_settings.analytic_aa_one_mesh,
    )
    try:
        yield
    finally:
        rt_settings.set_analytic_aa(
            before[0],
            seam=before[1],
            run_full=before[2],
            one_mesh=before[3],
        )


def _grp(run_full, one_mesh):
    rt_settings.set_analytic_aa(
        True,
        seam=True,
        run_full=run_full,
        one_mesh=one_mesh,
    )
    return rp._aa_group(AA_BEZ, AA_TRI)


@pytest.mark.parametrize(
    ("run_full", "one_mesh", "expected"),
    [
        (False, False, 1),  # seam grouping only
        (True, False, 2),  # + the relaxed truncation gate
        (False, True, 3),  # + the one-mesh ceiling, which IMPLIES the relaxed gate
        (True, True, 3),  # 3 subsumes 2
    ],
)
def test_aa_group_encodes_the_gate_combination(
    run_full, one_mesh, expected, restore_aa
):
    assert _grp(run_full, one_mesh) == expected


@pytest.mark.parametrize(
    ("run_full", "one_mesh"),
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_every_gate_that_relaxes_the_semantics_also_relaxes_the_truncation(
    run_full, one_mesh, restore_aa
):
    """REGRESSION. The relaxed semantics require the truncation's mitigation.

    ``_aa_run_full`` is what the readers test, so it is also what the emission
    truncation must test. The failure this pins is silent: output is produced,
    looks plausible, and carries interior notches the coverage harness cannot see
    because it scores silhouette pixels only.
    """
    grp = _grp(run_full, one_mesh)
    from algan.rendering.raytracing.raster_taichi import _aa_run_full

    assert _aa_run_full(grp), (
        f"run_full={run_full} one_mesh={one_mesh} "
        f"gives aa_grp={grp}, which readers do not treat as the relaxed gate"
    )


def test_only_aa_group_reads_the_run_full_setting():
    """REGRESSION, and the one that actually pins the wiring.

    The invariant test above passed *before* the fix too: ``_aa_run_full(3)`` was
    always true: the bug was that the truncation site never asked it, testing
    ``analytic_aa_run_full`` directly instead and so missing the ONE_MESH-implies-2
    case. What has to hold is therefore structural -- ``aa_grp`` is the single
    answer, so exactly one function may read the raw setting, and every other
    reader must go through ``_aa_group`` / ``_aa_run_full``.

    Source-level for the same reason ``test_environment`` audits ``os.environ``
    access: the property is "nothing else reaches this", which no amount of
    calling the code can demonstrate.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(rp))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Attribute)
                and inner.attr == "analytic_aa_run_full"
                and node.name != "_aa_group"
            ):
                offenders.append(f"{node.name}:{inner.lineno}")
    assert not offenders, (
        "ANALYTIC_AA_RUN_FULL must only be read by _aa_group, so the emission "
        "truncation and every other reader cannot disagree about whether the "
        f"relaxed gate is active; also read by {offenders}"
    )


def test_seam_grouping_off_disables_every_group_rule(restore_aa):
    """Both rules are subordinate to seam grouping, so 0 must stay 0."""
    rt_settings.set_analytic_aa(True, seam=False, run_full=True, one_mesh=True)
    assert rp._aa_group(AA_BEZ, AA_TRI) == 0


def test_no_grouping_without_analytic_aa_geometry(restore_aa):
    """Neither geometry arm analytic means there is nothing to group."""
    rt_settings.set_analytic_aa(True, seam=True, run_full=True, one_mesh=True)
    assert rp._aa_group(0, 0) == 0


@pytest.mark.parametrize("time_start", [0, 1, 7, 15])
def test_the_tri_obj_row_does_not_depend_on_where_the_chunk_starts(time_start):
    """The one-mesh flag must read the frame the RESOLVE reads.

    The same host/kernel-agreement failure as the rest of this module, one level
    down. ``prepare_sparse_raster_coverage`` decides which pixels are
    single-surface by grouping ``tri_obj`` ids, and ``tri_obj`` is per frame for
    a diced logical-PN primitive. A fragment's compact pixel index is
    CHUNK-relative; every kernel converts it back with ``f = time_start +
    g // ppf`` before indexing. So the row is a property of the ABSOLUTE frame,
    and asking for it must give the same answer however the render loop happened
    to split the batch into chunks -- which is exactly what the reduction got
    wrong by dropping ``time_start``.

    Stated as that invariant rather than as a transcription of the kernel's
    arithmetic, so it cannot pass by restating the code it guards.
    """
    ppf, rows = 100, 16
    for f_abs in range(time_start, rows):
        pix = (f_abs - time_start) * ppf + 7  # what the emission writes
        assert rp._tri_obj_row(pix, ppf, time_start, rows) == f_abs % rows
