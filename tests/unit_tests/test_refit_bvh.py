"""Focused tests for the shared-topology refit BVH builder."""

import warnings

import pytest
import torch

from algan.rendering.raytracing.refit_bvh import build_refit_bvh


def test_refit_bvh_does_not_expose_index_reduce_beta_warning():
    """The internal PyTorch beta warning must not reach Algan users."""
    lo = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]]])
    hi = lo + 0.5

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"index_reduce\(\) is in beta and the API may change at any time\.",
            category=UserWarning,
        )
        build_refit_bvh(lo, hi, num_frames=1)


def _block_bits(tree):
    """Bit pattern of a tree's blocks (f16 blocks carry link words as NaN
    payloads, which ``torch.equal`` would never call equal).
    """
    blocks = tree.blocks
    return blocks.view(torch.int16 if blocks.dtype == torch.float16 else torch.int32)


def _static_bounds(n=6):
    torch.manual_seed(0)
    lo = torch.rand(1, n, 3)
    return lo, lo + 0.5


def test_refit_bvh_reduces_per_frame_opacity_over_a_static_tree():
    """Static bounds with a still-per-frame opacity mask must build.

    The merge collapses each temporally-constant table on its own, so a batch
    whose geometry holds still while a mob fades reaches the builder as
    ``Tc == 1, To == T``. One tree covers every frame, so the flag holds only
    where it holds on all of them -- the reduction build_stbvh applies to its
    static instances.
    """
    lo, hi = _static_bounds()
    n = lo.shape[1]
    opaque = torch.zeros((4, n), dtype=torch.bool)
    opaque[:2] = True  # opaque early, translucent later

    tree = build_refit_bvh(lo, hi, num_frames=4, opaque=opaque)
    conservative = build_refit_bvh(
        lo, hi, num_frames=4, opaque=opaque.all(0, keepdim=True)
    )
    assert torch.equal(_block_bits(tree), _block_bits(conservative))

    # A primitive opaque on every frame keeps its flag.
    always = build_refit_bvh(
        lo, hi, num_frames=4, opaque=torch.ones((4, n), dtype=torch.bool)
    )
    flagged = build_refit_bvh(
        lo, hi, num_frames=4, opaque=torch.ones((1, n), dtype=torch.bool)
    )
    assert torch.equal(_block_bits(always), _block_bits(flagged))
    assert not torch.equal(_block_bits(always), _block_bits(tree))


def test_refit_bvh_rejects_an_opacity_mask_of_a_foreign_frame_count():
    lo, hi = _static_bounds()
    lo = lo.expand(4, -1, -1).contiguous()
    hi = hi.expand(4, -1, -1).contiguous()
    with pytest.raises(ValueError, match="opacity mask has 3 frames"):
        build_refit_bvh(
            lo, hi, num_frames=4, opaque=torch.zeros((3, lo.shape[1]), dtype=torch.bool)
        )


def _local_refit_device():
    from algan.rendering.taichi_runtime import (
        init_taichi,
        taichi_arch_is_cuda,
        taichi_launch_is_local,
    )

    init_taichi()
    device = torch.device("cuda" if taichi_arch_is_cuda() else "cpu")
    if not taichi_launch_is_local(device):
        pytest.skip("packed refit requires a local CPU or CUDA tensor/backend pair")
    return device


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize(("frames", "primitives"), [(1, 1), (1, 41), (4, 257)])
def test_packed_refit_matches_torch_bits(half, frames, primitives, monkeypatch):
    """Pack the same topology with moving visibility, flags and duplicate leaves."""
    from algan import SETTINGS
    from algan.rendering.raytracing import refit_bvh

    device = _local_refit_device()
    generator = torch.Generator().manual_seed(832)
    lo = torch.randn(frames, primitives, 3, generator=generator).to(device)
    hi = lo + torch.rand(frames, primitives, 3, generator=generator).to(device)
    if primitives > 1:
        lo[0, ::3], hi[0, ::3] = 1e17, -1e17
    opaque = (torch.rand(frames, primitives, generator=generator) > 0.5).to(device)
    casts = (torch.rand(primitives, generator=generator) > 0.5).to(device)
    leaf_prim = torch.arange(primitives, device=device) // 2
    # Exercise only the builder: no traversal compiled with the startup-fixed
    # layout reads these trees. The new pack kernel takes layout as a template.
    monkeypatch.setattr(refit_bvh, "bvh_block_f16", half)
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=False):
        reference = build_refit_bvh(
            lo, hi, opaque=opaque, casts=casts, leaf_prim=leaf_prim
        )
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=True):
        candidate = build_refit_bvh(
            lo, hi, opaque=opaque, casts=casts, leaf_prim=leaf_prim
        )
    assert candidate.first_leaf == reference.first_leaf
    assert torch.equal(_block_bits(candidate), _block_bits(reference))


def test_packed_refit_half_rounding_boundaries(monkeypatch):
    """Outward rounding must preserve tiny and clamped boxes on either side of zero."""
    from algan import SETTINGS
    from algan.rendering.raytracing import refit_bvh

    device = _local_refit_device()
    values = torch.tensor(
        [
            -1e17,
            -65504.0,
            -1.001,
            -(2**-14),
            -(2**-20),
            -(2**-25),
            -(2**-30),
            -0.0,
            0.0,
            2**-30,
            2**-25,
            2**-20,
            2**-14,
            1.001,
            65504.0,
            1e17,
        ],
        device=device,
    )
    lo = values.reshape(1, -1, 1).expand(-1, -1, 3).contiguous()
    hi = lo.clone()
    monkeypatch.setattr(refit_bvh, "bvh_block_f16", True)
    original = refit_bvh.build_refit_bvh
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=False):
        reference = original(lo, hi)
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=True):
        candidate = original(lo, hi)
    actual, expected = _block_bits(candidate), _block_bits(reference)
    different = actual != expected
    assert not bool(different.any()), (
        different.nonzero().tolist(),
        actual[different].tolist(),
        expected[different].tolist(),
    )


@pytest.mark.parametrize("hidden", [False, True])
def test_packed_refit_static_opacity_and_empty_tree(hidden, monkeypatch):
    from algan import SETTINGS
    from algan.rendering.raytracing import refit_bvh_taichi

    device = _local_refit_device()
    lo, hi = (bound.to(device) for bound in _static_bounds())
    if hidden:
        lo.fill_(1e17)
        hi.fill_(-1e17)
    opacity = torch.ones((4, lo.shape[1]), dtype=torch.bool, device=device)
    opacity[1, ::2] = False
    original = refit_bvh_taichi.refit_pack_level
    launches = []

    def tracked(*args):
        launches.append(1)
        return original(*args)

    monkeypatch.setattr(refit_bvh_taichi, "refit_pack_level", tracked)
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=False):
        reference = build_refit_bvh(lo, hi, num_frames=4, opaque=opacity)
    assert not launches
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=True):
        candidate = build_refit_bvh(lo, hi, num_frames=4, opaque=opacity)
    assert launches, "the packed kernel must actually run in the parity test"
    assert torch.equal(_block_bits(candidate), _block_bits(reference))


def _topology_inputs(n, kind, seed):
    """``[n, 3]`` union boxes: scattered, clustered with exact duplicates (the
    median-split fallbacks), or a long line (the depth budget's forced splits).
    """
    generator = torch.Generator().manual_seed(seed)
    if kind == "scattered":
        lo = torch.randn(n, 3, generator=generator) * 10
    elif kind == "duplicates":
        lo = torch.randint(0, 4, (n, 3), generator=generator).float()
    else:
        lo = torch.zeros(n, 3)
        lo[:, 0] = torch.arange(n, dtype=torch.float32) ** 2
    hi = lo + torch.rand(n, 3, generator=generator)
    return lo, hi


@pytest.mark.parametrize("kind", ["scattered", "duplicates", "line"])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 37, 1000])
def test_host_topology_is_the_device_topology(n, kind):
    """The NumPy build is the torch build, bit for bit, on every device."""
    import numpy as np

    from algan.rendering.raytracing import refit_bvh

    lo, hi = _topology_inputs(n, kind, seed=n)
    lo_np, hi_np = lo.numpy(), hi.numpy()
    kind_h, ref_h, levels_h = refit_bvh._build_topology_host(
        n, (lo_np + hi_np) * np.float32(0.5), lo_np, hi_np, refit_bvh.bvh_arity
    )
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        dlo, dhi = lo.to(device), hi.to(device)
        kind_d, ref_d, levels_d = refit_bvh._build_topology(
            n, (dlo + dhi) * 0.5, dlo, dhi, refit_bvh.bvh_arity
        )
        assert levels_d == levels_h, device
        assert np.array_equal(kind_d.cpu().numpy(), kind_h), device
        assert np.array_equal(ref_d.cpu().numpy(), ref_h), device


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_host_topology_builds_the_device_tree(monkeypatch, device):
    """Through ``build_refit_bvh``: moving visibility, flags and slivers."""
    from algan.rendering.raytracing import refit_bvh

    generator = torch.Generator().manual_seed(5)
    lo = (torch.randn(6, 700, 3, generator=generator) * 4).to(device)
    hi = lo + torch.rand(6, 700, 3, generator=generator).to(device)
    lo[2, ::5], hi[2, ::5] = 1e17, -1e17
    lo[:, ::97], hi[:, ::97] = 1e17, -1e17  # never visible
    opaque = (torch.rand(6, 700, generator=generator) > 0.3).to(device)
    casts = (torch.rand(700, generator=generator) > 0.2).to(device)
    leaf_prim = (torch.arange(700) // 2).to(device)
    trees = []
    # 0 keeps the topology on the tensors' own device; the second arm takes
    # the NumPy build (a CPU render takes it at any bound above 0).
    for bound in (0, 1 << 20):
        monkeypatch.setattr(refit_bvh, "refit_host_topology_max_columns", bound)
        trees.append(
            build_refit_bvh(lo, hi, opaque=opaque, casts=casts, leaf_prim=leaf_prim)
        )
    assert trees[0].num_blocks == trees[1].num_blocks
    assert torch.equal(_block_bits(trees[0]), _block_bits(trees[1]))


def test_host_topology_is_quiet_about_huge_coordinates():
    """NumPy warns where a product of ~1e19 coordinates overflows; torch is
    silent there, and the overflow is masked out of the costs either way.
    """
    import warnings

    from algan.rendering.raytracing import refit_bvh

    torch.manual_seed(3)
    lo = torch.rand(1, 64, 3) * 1e20
    hi = lo + torch.rand(1, 64, 3) * 1e20
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        host = build_refit_bvh(lo, hi)
    device = refit_bvh.refit_host_topology_max_columns
    try:
        refit_bvh.refit_host_topology_max_columns = 0
        reference = build_refit_bvh(lo, hi)
    finally:
        refit_bvh.refit_host_topology_max_columns = device
    assert torch.equal(_block_bits(host), _block_bits(reference))


# --- time split under large in-batch motion ----------------------------------


def _shuffle_bounds(n=512, frames=16, seed=5):
    """Every box travels from one grid cell to a random other one over the
    batch: the motion that leaves one union topology spanning the scene.
    """
    generator = torch.Generator().manual_seed(seed)
    start = torch.rand(n, 3, generator=generator)
    end = start[torch.randperm(n, generator=generator)]
    t = torch.linspace(0, 1, frames).view(frames, 1, 1)
    centre = start * (1 - t) + end * t
    return centre - 0.01, centre + 0.01


def _frame_walk(tree, frame):
    """Leaves reachable from ``frame``'s root, with the box each leaf's parent
    slot gives it, by following that frame's own link words.
    """
    from algan.rendering.raytracing.refit_bvh import LINK_PRIM_MASK, bvh_arity

    words = _block_bits(tree)
    if tree.blocks.dtype == torch.float16:
        link = (words[:, 6].to(torch.int32) & 0xFFFF) | (
            words[:, 7].to(torch.int32) << 16
        )
    else:
        link = words[:, 6]
    bounds = tree.blocks.float()
    row0 = (frame % tree.num_time) * tree.num_blocks
    leaves, stack = {}, [0]
    while stack:
        block = stack.pop()
        for child in range(bvh_arity):
            word = int(link[row0 + block, child])
            if word == -1:
                continue
            if word >= 0:
                stack.append(word)
                continue
            box = bounds[row0 + block, 0:6, child]
            leaves[word & LINK_PRIM_MASK] = (box[:3], box[3:])
    return leaves


@pytest.mark.parametrize("half", [False, True])
def test_large_motion_splits_the_batch_and_every_frame_still_finds_every_box(
    half, monkeypatch
):
    from algan import SETTINGS
    from algan.rendering.raytracing import refit_bvh

    monkeypatch.setattr(refit_bvh, "bvh_block_f16", half)
    lo, hi = _shuffle_bounds()
    hidden = torch.zeros(lo.shape[:2], dtype=torch.bool)
    hidden[:5, ::7] = True  # appears late
    lo[hidden], hi[hidden] = 1e17, -1e17
    with SETTINGS.raytracing.experimental.override(refit_time_split=1):
        shared = build_refit_bvh(lo, hi)
    tree = build_refit_bvh(lo, hi)

    assert shared.time_segments == 1
    assert 1 < tree.time_segments <= refit_bvh.refit_time_split
    assert tree.num_time == lo.shape[0]
    assert tree.blocks.shape[0] == tree.num_time * tree.num_blocks
    for frame in range(lo.shape[0]):
        leaves = _frame_walk(tree, frame)
        visible = (~hidden[frame]).nonzero().flatten().tolist()
        assert sorted(leaves) == visible
        for prim, (box_lo, box_hi) in leaves.items():
            assert (box_lo <= lo[frame, prim]).all()
            assert (box_hi >= hi[frame, prim]).all()


def test_the_split_cuts_the_traversal_proxy_against_one_shared_topology():
    """What the split is for: summed over the batch, the per-segment trees
    fetch fewer sibling blocks per ray than the one shared topology did.
    """
    from algan.rendering.raytracing import refit_bvh

    lo, hi = _shuffle_bounds()
    shared, shared_cost = refit_bvh._build_segment(lo, hi, None, None, None)
    segments = refit_bvh._split_in_time(
        lo,
        hi,
        None,
        None,
        None,
        0,
        lo.shape[0],
        shared,
        shared_cost,
        refit_bvh.refit_time_split,
    )
    assert len(segments) > 1
    split_cost, start = 0.0, 0
    for segment in segments:
        stop = start + segment.num_time
        _, cost = refit_bvh._build_segment(
            lo[start:stop], hi[start:stop], None, None, None
        )
        split_cost += float(cost.sum())
        start = stop
    assert start == lo.shape[0]
    assert float(shared_cost.sum()) > refit_bvh.refit_time_split_gain * split_cost


@pytest.mark.parametrize("motion", ["static", "drift"])
def test_still_or_drifting_geometry_keeps_one_topology(motion):
    from algan import SETTINGS

    generator = torch.Generator().manual_seed(9)
    centre = torch.rand(1, 300, 3, generator=generator).expand(12, -1, -1)
    if motion == "drift":
        centre = centre + torch.linspace(0, 0.02, 12).view(12, 1, 1)
    lo, hi = (centre - 0.01).contiguous(), (centre + 0.01).contiguous()
    with SETTINGS.raytracing.experimental.override(refit_time_split=1):
        shared = build_refit_bvh(lo, hi)
    tree = build_refit_bvh(lo, hi)
    assert tree.time_segments == 1
    assert torch.equal(_block_bits(tree), _block_bits(shared))
