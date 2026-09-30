"""Deep same-surface stacks keep one sheet per crossing.

The sheet compaction used to clamp a fragment's conflict rank to 15 so it fit
four bits of a ``parent * 16 + rank`` key. The 17th and later overlapping
layers of one surface in one pixel then merged into the 16th sub-band and
attenuated once between them instead of once each, so the stack rendered too
light. The rank is no longer clamped, and the two torch packings take a radix
past the chunk's deepest rank (``sheets._rank_key_base``); a chunk no deeper
than 16 layers packs exactly the keys it always did.
"""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.truncation import (
    reset_truncations,
    snapshot_truncations,
)
from tests.unit_tests.test_sheet_compaction import _coverage


def test_the_packing_radix_is_unchanged_up_to_sixteen_layers():
    # Ranks 0..15 are what the clamp let through: the radix those chunks pack
    # with is the old 16, so their keys -- and every group -- are unchanged.
    assert [sheets._rank_key_base(d) for d in range(16)] == [16] * 16
    assert [sheets._rank_key_base(d) for d in (16, 17, 64, 256)] == [17, 18, 65, 257]
    # The widest a signed-int32 stream can ask for still packs under 2**62.
    count = 2**31 - 1
    assert (count - 1) * sheets._rank_key_base(count - 1) + count - 1 < 2**62


@pytest.mark.parametrize("layers", [16, 17, 65, 257])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("pass_deepest", [False, True])
def test_deep_rank_pooling_never_collides_between_parents(
    monkeypatch, layers, reuse, pass_deepest
):
    """Pooling keys must not spill one parent's ranks into the next's.

    Three dense parents of ``layers`` single-fragment sub-bands each, ranks
    ``0..layers-1``. Only the first -- full sample union, exact area under a
    pixel -- is eligible to pool. At a radix of 16, rank 16 of parent 1 would
    share parent 2's rank-0 key, and ``reuse`` (the consecutive grouping)
    would also see the keys stop ascending.
    """
    monkeypatch.setattr(sheets, "sheet_pixel_sort", reuse)
    monkeypatch.setattr(sheets, "sheet_group_reuse", reuse)
    parents = [0] * layers + [1] * layers + [2] * layers
    ranks = list(range(layers)) * 3
    n = len(parents)
    cid_band = torch.tensor(parents, dtype=torch.int64)
    rank_of_cid = torch.tensor(ranks, dtype=torch.int64)
    band_of_frag = torch.arange(n)
    area = torch.tensor([0.5 / layers] * layers + [0.25] * layers + [1.0] * layers)
    mask = torch.tensor(
        [sheets.AA_MASK_ALL] * layers + [1] * layers + [sheets.AA_MASK_ALL] * layers,
        dtype=torch.int32,
    )
    count, groups = sheets._rank_pool_groups(
        cid_band,
        rank_of_cid,
        band_of_frag,
        area,
        mask,
        n,
        3,
        layers - 1 if pass_deepest else None,
    )
    assert count == 1 + 2 * layers
    expected = [0] * layers + [1 + r for r in range(layers)]
    expected += [1 + layers + r for r in range(layers)]
    assert groups.tolist() == expected


@pytest.mark.parametrize("layers", [16, 17, 65, 257])
@pytest.mark.parametrize("kernels", [False, True])
@pytest.mark.parametrize("shade_split", [False, True])
def test_compaction_keeps_each_deep_transparent_crossing(layers, kernels, shade_split):
    """``layers`` full-pixel crossings of ONE surface give ``layers`` sheets.

    ``kernels`` runs both the conflict-rank scan and the rank grouping on their
    kernel arms or both on their torch arms. The surface is declared a closed
    shell with every crossing facing the same way, so the shell ceiling's
    allowance must keep real same-facing self-overlap too.
    """
    frags = [(0, 1.0 + 1e-5 * i, i % 4, 1.0, sheets.AA_MASK_ALL) for i in range(layers)]
    coverage, merged, cam, pws = _coverage(frags)
    merged["tri_closed"] = torch.ones((1, 8))
    reset_truncations()
    try:
        with SETTINGS.raytracing.experimental.override(
            sheet_rank_kernel=kernels, sheet_rank_groups=kernels
        ):
            out = sheets.compact_sheets(
                coverage,
                merged,
                cam,
                pws,
                0,
                4,
                4,
                band_rule="facing",
                shade_split=shade_split,
                sample_depth=True,
            )
        assert out["num_sheets"] == layers
        assert snapshot_truncations().sheet_layers == 0
    finally:
        reset_truncations()
    assert out["sheet_offsets"].tolist() == [0, layers]
    assert torch.all(out["sheet_cov"] == 1.0)
    assert torch.all((out["sheet_msk"] & sheets.AA_MASK_ALL) == sheets.AA_MASK_ALL)
    claims, transmission = sheets.resolve_pixel_reference(
        out["sheet_cov"].tolist(),
        out["sheet_msk"].tolist(),
        [False] * layers,
        alphas=[0.01] * layers,
    )
    expected = 0.99**layers
    assert transmission == pytest.approx([expected] * sheets.AA_NUM_SAMPLES, rel=1e-12)
    assert sum(claims) == pytest.approx(1.0 - expected, rel=1e-12)


@pytest.mark.parametrize("layers", [17, 65])
def test_rendered_same_surface_stack_matches_independent_surfaces(
    tmp_path, monkeypatch, layers
):
    """Removing the clamp must fix the transport, not just the report.

    ``layers`` coplanar-ish translucent quads rendered once as the faces of ONE
    Polyhedron (one surface, so the conflict rank separates them) and once as
    ``layers`` separate Polyhedra (separate surfaces, never ranked). Each
    crossing attenuates once either way, so the two frames must be identical.
    """
    import numpy as np
    from PIL import Image

    from algan import BLUE, MeshBasicMaterial, Polyhedron, Scene, SceneManager
    from algan.settings.video_settings import SMOKE_TEST

    compactions = []
    original = sheets.compact_sheets

    def counted(*args, **kwargs):
        compactions.append(True)
        return original(*args, **kwargs)

    # The render must reach the sheet route, or the comparison says nothing.
    monkeypatch.setattr(sheets, "compact_sheets", counted)
    vertices, faces = [], []
    for i in range(layers):
        z = i * 1e-4
        first = len(vertices)
        vertices.extend([[-2, -2, z], [2, -2, z], [2, 2, z], [-2, 2, z]])
        faces.append([first, first + 1, first + 2, first + 3])
    images = []
    with SETTINGS.raytracing.override(shadows=False, samples_per_pixel=1):
        for shared in (True, False):
            SceneManager.reset()
            try:
                with Scene(video_settings=SMOKE_TEST) as scene:
                    material = MeshBasicMaterial(
                        color=BLUE, opacity=0.03, transparent=True
                    )
                    if shared:
                        Polyhedron(vertices, faces).set_material(material).spawn(
                            animate=False
                        )
                    else:
                        for i in range(layers):
                            Polyhedron(
                                vertices[4 * i : 4 * i + 4], [[0, 1, 2, 3]]
                            ).set_material(material).spawn(animate=False)
                    path = tmp_path / f"layers-{layers}-shared-{shared}.png"
                    compactions.clear()
                    result = scene.save_frame(
                        str(path), video_settings=SMOKE_TEST, overwrite=True
                    )
                    assert compactions
                    assert result.render_plan.truncations.sheet_layers == 0
                    with Image.open(path) as image:
                        images.append(np.array(image.convert("RGB")))
            finally:
                SceneManager.reset()
    assert np.any(images[0])  # not a vacuous comparison of two black frames
    assert np.array_equal(images[0], images[1])
