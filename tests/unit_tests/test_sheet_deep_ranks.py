"""Count-bounded grouping retains deep same-surface transparent stacks."""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.sheet_grouping import rank_key_base
from algan.rendering.raytracing.truncation import (
    reset_truncations,
    snapshot_truncations,
)
from tests.unit_tests.test_sheet_compaction import _coverage
from tests.unit_tests.test_sheet_fragment_workspace import _poison, _workspace
from tests.unit_tests.test_sheet_rank_groups import ranks_for_masks


def _pairs(parents, ranks, device):
    pairs = list(zip(parents, ranks))
    labels = sorted(set(pairs))
    lookup = {pair: i for i, pair in enumerate(labels)}
    return (
        torch.tensor(
            [lookup[pair] for pair in pairs], dtype=torch.int64, device=device
        ),
        torch.tensor([pair[0] for pair in labels], dtype=torch.int64, device=device),
        torch.tensor([pair[1] for pair in labels], dtype=torch.int64, device=device),
    )


@pytest.mark.parametrize("layers", [16, 17, 64, 257])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("friendly", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_deep_conflict_ranks_and_groups_match_independent_pairs(
    monkeypatch, layers, native, friendly, owned
):
    memory, ws = _workspace()
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", "1" if friendly else "0")
    monkeypatch.setattr(sheets.rt_settings, "sheet_rank_kernel", native)
    monkeypatch.setattr(sheets, "sheet_rank_groups", native)
    # After the long repeated lane, ranks drop; donors never consume a lane.
    bands = [[1] * layers + [2, 0, 2, 255], [0, 1, 1, 2, 2, 1], [255] * 2]
    parents, ranks = ranks_for_masks(bands)
    n = len(parents)
    parent = torch.tensor(parents, device=ws.device)
    expected_rank = torch.tensor(ranks, dtype=torch.int32, device=ws.device)
    masks = torch.tensor(
        [m for b in bands for m in b], dtype=torch.int32, device=ws.device
    )
    positions = torch.arange(n, device=ws.device)
    order = positions.flip(0)
    raw_mask = masks.flip(0)
    starts = torch.ones(n, dtype=torch.bool, device=ws.device)
    starts[1:] = parent[1:] != parent[:-1]
    wanted = _pairs(parents, ranks, ws.device)
    with ws.stage():
        rank = ws.tensor((n,), torch.int32)
        out = ws.tensor((n,), torch.int64) if owned else None
        floor = memory.get_pointers()
        assert (
            sheets._conflict_rank(
                starts, order, raw_mask, positions, out=rank, workspace=ws
            )
            is rank
        )
        assert torch.equal(rank, expected_rank)
        result = sheets._sheet_rank_groups(parent, rank, out=out, workspace=ws)
        assert not owned or result.ids is out
        assert memory.get_pointers() == floor
        _poison(memory)
        assert all(torch.equal(a, b) for a, b in zip(result, wanted))
    assert memory.current_pointer == ws._depth == ws._live_bytes == 0
    if not owned:
        _poison(memory)
        assert all(torch.equal(a, b) for a, b in zip(result, wanted))


@pytest.mark.parametrize("layers", [16, 17, 65, 257])
@pytest.mark.parametrize("friendly", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_deep_rank_pooling_never_collides_between_parents(
    monkeypatch, layers, friendly, native, reuse
):
    memory, ws = _workspace()
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", "1" if friendly else "0")
    monkeypatch.setattr(sheets.rt_settings, "sheet_mask_kernel", native)
    monkeypatch.setattr(sheets, "sheet_pixel_sort", reuse)
    monkeypatch.setattr(sheets, "sheet_group_reuse", reuse)
    parents = [0] * layers + [1] * layers + [3] * layers
    ranks = list(range(layers)) * 3
    n = len(parents)
    parent = torch.tensor(parents, device=ws.device)
    rank = torch.tensor(ranks, device=ws.device)
    bands = torch.arange(n, device=ws.device)
    area = torch.tensor(
        [0.5 / layers] * layers + [0.25] * layers + [1.0] * layers, device=ws.device
    )
    mask = torch.tensor(
        [255] * layers + [1] * layers + [255] * layers,
        dtype=torch.int32,
        device=ws.device,
    )
    # Only the first parent's full-union, sub-unit area is eligible to pool.
    expected, _, _ = _pairs(parents, [0] * layers + list(range(layers)) * 2, ws.device)
    with ws.stage():
        out = ws.tensor((n,), torch.int64)
        floor = memory.get_pointers()
        count, result = sheets._rank_pool_groups(
            parent, rank, bands, area, mask, n, out=out, workspace=ws
        )
        assert count == 1 + 2 * layers
        assert result is out
        assert memory.get_pointers() == floor
        _poison(memory)
        assert torch.equal(result, expected)
    assert memory.current_pointer == ws._depth == ws._live_bytes == 0


@pytest.mark.parametrize("layers", [16, 17, 65, 257])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("friendly", [False, True])
@pytest.mark.parametrize("shade_split", [False, True])
def test_compaction_keeps_each_deep_transparent_crossing(
    monkeypatch, layers, native, friendly, shade_split
):
    memory, ws = _workspace()
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", "1" if friendly else "0")
    monkeypatch.setattr(sheets.rt_settings, "sheet_rank_kernel", native)
    monkeypatch.setattr(sheets.rt_settings, "sheet_mask_kernel", native)
    monkeypatch.setattr(sheets, "sheet_rank_groups", native)
    monkeypatch.setattr(sheets, "sheet_rank_pool", True)
    frags = [(0, 1.0 + 1e-5 * i, i % 4, 1.0, 255) for i in range(layers)]
    coverage, merged, cam, pws = _coverage(frags)
    coverage = {
        k: v.to(ws.device) if torch.is_tensor(v) else v for k, v in coverage.items()
    }
    merged = {
        k: v.to(ws.device) if torch.is_tensor(v) else v for k, v in merged.items()
    }
    # All crossings face the same way: the existing shell allowance must also
    # preserve real same-facing self-overlap rather than imposing unit area.
    merged["tri_closed"] = torch.ones((1, 8), device=ws.device)
    reset_truncations()
    try:
        result = sheets.compact_sheets(
            coverage,
            merged,
            cam.to(ws.device),
            pws.to(ws.device),
            0,
            4,
            4,
            band_rule="facing",
            shade_split=shade_split,
            sample_depth=True,
            diagnostics=False,
            resolver_memory=memory,
            workspace=ws,
        )
        assert result.num_sheets == layers
        assert snapshot_truncations().sheet_layers == 0
        _poison(memory)
        assert result.sheet_offsets.tolist() == [0, layers]
        assert torch.all(result.sheet_cov == 1.0)
        assert torch.all((result.sheet_msk & sheets.AA_MASK_ALL) == sheets.AA_MASK_ALL)
        claims, transmission = sheets.resolve_pixel_reference(
            result.sheet_cov.tolist(),
            result.sheet_msk.tolist(),
            [False] * layers,
            alphas=[0.01] * layers,
        )
        expected = 0.99**layers
        assert transmission == pytest.approx(
            [expected] * sheets.AA_NUM_SAMPLES, rel=1e-12
        )
        assert sum(claims) == pytest.approx(1.0 - expected, rel=1e-12)
    finally:
        reset_truncations()
    assert memory.current_pointer == ws._depth == ws._live_bytes == 0


@pytest.mark.parametrize("count", [0, 1, 16, 17, 65536, 2**31 - 1])
def test_count_radix_has_no_collision_or_int64_overflow(count):
    radix = rank_key_base(count)
    assert radix == max(1, count)
    if count:
        last = (count - 1) * radix + count - 1
        assert last == count**2 - 1
        assert last < 2**62
    if count > 1:
        assert (count - 2) * radix + count - 1 < (count - 1) * radix


@pytest.mark.parametrize("count", [-1, 2**31, 2**40])
def test_unrepresentable_row_counts_fail_before_compaction_allocates(count):
    memory, ws = _workspace()
    with pytest.raises(ValueError, match="signed-int32 capacity"):
        rank_key_base(count)
    with pytest.raises(ValueError, match="signed-int32 capacity"):
        sheets.compact_sheets(
            {
                "num_fragments": count,
                "frag_key": torch.empty(0, dtype=torch.int64, device=ws.device),
            },
            {},
            None,
            None,
            0,
            1,
            1,
            diagnostics=False,
            resolver_memory=memory,
            workspace=ws,
        )
    assert memory.current_pointer == ws._depth == ws._live_bytes == 0


@pytest.mark.parametrize("entry", ["rank", "pool"])
def test_mps_pair_group_failure_rewinds_its_workspace(monkeypatch, entry):
    memory, ws = _workspace()
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", "1")
    monkeypatch.setattr(sheets, "sheet_rank_groups", False)
    name = "class_groups" if entry == "rank" else "consecutive_pair_ids"
    original = getattr(sheets, name)

    def fail_after(*args, **kwargs):
        original(*args, **kwargs)
        raise LookupError("pair grouping failure")

    monkeypatch.setattr(sheets, name, fail_after)
    parent = torch.tensor([0] * 17 + [1] * 17, device=ws.device)
    rank = torch.tensor(list(range(17)) * 2, device=ws.device)
    with ws.stage():
        out = ws.tensor((34,), torch.int64)
        sentinel = ws.tensor((5,), torch.int64, 107)
        floor = memory.get_pointers()

        def run():
            if entry == "rank":
                sheets._sheet_rank_groups(parent, rank, out=out, workspace=ws)
            else:
                sheets._rank_pool_groups(
                    parent,
                    rank,
                    torch.arange(34, device=ws.device),
                    torch.ones(34, device=ws.device),
                    torch.full((34,), 255, dtype=torch.int32, device=ws.device),
                    34,
                    out=out,
                    workspace=ws,
                )

        with pytest.raises(LookupError, match="pair grouping failure"):
            run()
        assert memory.get_pointers() == floor
        _poison(memory)
        assert torch.all(sentinel == 107)
    assert memory.current_pointer == ws._depth == ws._live_bytes == 0


@pytest.mark.parametrize("layers", [17, 65])
def test_rendered_same_surface_stack_matches_independent_surfaces(tmp_path, layers):
    """The removed cap must fix transport, not just stop reporting truncation."""
    import numpy as np
    from PIL import Image

    from algan import BLUE, SETTINGS, MeshBasicMaterial, Polyhedron, Scene, SceneManager
    from algan.settings.video_settings import SMOKE_TEST

    vertices, faces = [], []
    for i in range(layers):
        z = i * 1e-4
        first = len(vertices)
        vertices.extend([[-2, -2, z], [2, -2, z], [2, 2, z], [-2, 2, z]])
        faces.append([first, first + 1, first + 2, first + 3])
    images = []
    with SETTINGS.raytracing.override(shadows=False):
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
                    result = scene.save_frame(
                        str(path), video_settings=SMOKE_TEST, overwrite=True
                    )
                    assert result.render_plan.truncations.sheet_layers == 0
                    with Image.open(path) as image:
                        images.append(np.array(image.convert("RGB")))
            finally:
                SceneManager.reset()
    assert np.any(images[0])  # neither comparison image may be a vacuous black frame
    assert np.array_equal(images[0], images[1])
