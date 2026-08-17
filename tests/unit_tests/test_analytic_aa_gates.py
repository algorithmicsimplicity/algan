"""The analytic-AA group value, and the host/kernel agreement it encodes.

``aa_grp`` is the resolve's grouping template. Three readers have to agree about
it, and they are not all in the same language: the two kernel-launch sites in
``raster_pipeline`` pass it to Taichi, the kernels test it through
``_aa_run_full`` / ``_aa_one_mesh``, and the host's **emission truncation** has
to apply the relaxed gate's mitigation whenever the kernels take the relaxed
gate.

They drifted, which is why this file exists. ``ANALYTIC_AA_ONE_MESH`` sets
``aa_grp = 3``, and ``_aa_run_full`` treats 3 as the relaxed gate (the one-mesh
rule needs it: the near sheet's exact area is only worth reading once the gate
lets it be read). But the truncation tested ``ANALYTIC_AA_RUN_FULL`` alone, so
with ONE_MESH on and RUN_FULL off the kernel ran the relaxed scan over fragment
lists whose empty-mask **area donors had already been discarded** -- the exact
configuration ``DESIGN_mesh_identity.md`` ss6.3.2 measured as an interior notch,
and the one ss6.6 documents as the shipping shape of the rule. Measured on CUDA
before the fix: a flat quad's ink wobble improved 8% where wiring both gates gave
63%, and a default ``Cylinder`` 47% where both gave 78%.

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
        rt_settings.ANALYTIC_AA,
        rt_settings.ANALYTIC_AA_SEAM,
        rt_settings.ANALYTIC_AA_RUN_FULL,
        rt_settings.ANALYTIC_AA_ONE_MESH,
    )
    try:
        yield
    finally:
        rt_settings.set_analytic_aa(
            before[0], seam=before[1], run_full=before[2], one_mesh=before[3]
        )


def _grp(run_full, one_mesh):
    rt_settings.set_analytic_aa(True, seam=True, run_full=run_full, one_mesh=one_mesh)
    return rp._aa_group(AA_BEZ, AA_TRI)


@pytest.mark.parametrize(
    ("run_full", "one_mesh", "expected"),
    [
        (False, False, 1),  # seam grouping only
        (True, False, 2),  # + the relaxed run-scan gate
        (False, True, 3),  # + the one-mesh cap, which IMPLIES the relaxed gate
        (True, True, 3),  # 3 subsumes 2
    ],
)
def test_aa_group_encodes_the_gate_combination(
    run_full, one_mesh, expected, restore_aa
):
    assert _grp(run_full, one_mesh) == expected


@pytest.mark.parametrize(
    ("run_full", "one_mesh"),
    [(True, False), (False, True), (True, True)],
)
def test_every_gate_that_relaxes_the_scan_also_relaxes_the_truncation(
    run_full, one_mesh, restore_aa
):
    """REGRESSION. The kernel's relaxed scan requires the host's mitigation.

    ``_aa_run_full`` is what the kernels test, so it is also what the emission
    truncation must test. The failure this pins is silent: output is produced,
    looks plausible, and carries interior notches the coverage harness cannot see
    because it scores silhouette pixels only.
    """
    grp = _grp(run_full, one_mesh)
    from algan.rendering.raytracing.raster_taichi import _aa_run_full

    assert _aa_run_full(grp), (
        f"run_full={run_full} one_mesh={one_mesh} gives aa_grp={grp}, which the "
        "kernels do not treat as the relaxed gate"
    )


def test_only_aa_group_reads_the_run_full_setting():
    """REGRESSION, and the one that actually pins the wiring.

    The invariant test above passed *before* the fix too: ``_aa_run_full(3)`` was
    always true: the bug was that the truncation site never asked it, testing
    ``ANALYTIC_AA_RUN_FULL`` directly instead and so missing the ONE_MESH-implies-2
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
                and inner.attr == "ANALYTIC_AA_RUN_FULL"
                and node.name != "_aa_group"
            ):
                offenders.append(f"{node.name}:{inner.lineno}")
    assert not offenders, (
        "ANALYTIC_AA_RUN_FULL must only be read by _aa_group, so the host and "
        "the kernels cannot disagree about whether the relaxed run gate is "
        f"active; also read by {offenders}"
    )


def test_seam_grouping_off_disables_every_group_rule(restore_aa):
    """Both rules are subordinate to seam grouping, so 0 must stay 0."""
    rt_settings.set_analytic_aa(True, seam=False, run_full=True, one_mesh=True)
    assert rp._aa_group(AA_BEZ, AA_TRI) == 0


def test_no_grouping_without_analytic_aa_geometry(restore_aa):
    """Neither geometry arm analytic means there is nothing to group."""
    rt_settings.set_analytic_aa(True, seam=True, run_full=True, one_mesh=True)
    assert rp._aa_group(0, 0) == 0
