"""Proof and queue regressions for the opt-in tiled primary frontend.

These feature tests intentionally launch the real kernels. Wide-index ordering
uses a CPU oracle even when the tested backend is Metal. No timing assertions.
"""

import numpy as np
import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import tile_raster as tiled
from algan.rendering.raytracing import tile_raster_taichi as kernels
from algan.rendering.raytracing.raster_pipeline import _exact_fragment_order
from algan.rendering.raytracing.raster_taichi import _AA_MASK_ALL
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti


@pytest.fixture(autouse=True)
def runtime():
    init_taichi()


def _device(*values):
    return tuple(v.to(SETTINGS.computing.render_device) for v in values)


def _keys(pixels, depths):
    return (pixels.to(torch.int64) << 32) | (depths.float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF)


def _active_tiles(pixels, width, height):
    fr, local = pixels // (width * height), pixels % (width * height)
    x, y = local % width, local // width
    tw, th = (width + 15) // 16, (height + 15) // 16
    return torch.unique(((fr * th + y // 16) * tw + x // 16).to(torch.int32), sorted=True)


@ti.kernel
def _proof_kernel(screen: ti.types.ndarray(), pos: ti.types.ndarray(), camera: ti.types.ndarray(),
                  rectangles: ti.types.ndarray(), result: ti.types.ndarray()):
    for p in range(result.shape[0]):
        full, outside, near, far = kernels._rect_proof(
            p, 0, rectangles[p, 0], rectangles[p, 1], rectangles[p, 2], rectangles[p, 3],
            screen, pos, camera,
        )
        result[p, 0], result[p, 1], result[p, 2], result[p, 3] = full, outside, near, far


def _projected(triangles, depths=None, width=4, height=4):
    xy = torch.tensor(triangles, dtype=torch.float32).reshape(-1, 3, 2)
    n = xy.shape[0]
    z = torch.ones(n, 3) if depths is None else torch.tensor(depths, dtype=torch.float32).expand(n, 3).clone()
    pos = torch.stack(((xy[..., 0] - width / 2) / (height / 2) * z,
                       (xy[..., 1] - height / 2) / (height / 2) * z, z), dim=-1).reshape(1, n, 9)
    screen = torch.zeros(1, n, 13)
    screen[0, :, :3], screen[0, :, 3:6], screen[0, :, 6:9] = xy[..., 0], xy[..., 1], 1 / z
    screen[..., 9] = 1
    return screen, pos, torch.zeros(1, 3)


def _proofs(triangles, rectangles, depths=None):
    screen, pos, camera = _projected(triangles, depths)
    rect = torch.tensor(rectangles, dtype=torch.float32)
    out = torch.empty(len(triangles), 4)
    screen, pos, camera, rect, out = _device(screen, pos, camera, rect, out)
    _proof_kernel(screen, pos, camera, rect, out)
    return out.cpu(), screen.cpu(), pos.cpu()


def test_containment_is_a_footprint_proof_not_a_sample_mask():
    broad = [[-10, -10], [20, -10], [-10, 20]]
    clipped_corner = [[-100, -100], [101.9999, -100], [-100, 101.9999]]
    outside = [[5, 5], [6, 5], [5, 6]]
    degenerate = [[0, 0], [1, 1], [2, 2]]
    out, _, _ = _proofs([broad, broad[::-1], clipped_corner, outside, degenerate], [[0, 0, 1, 1]] * 5)
    assert out[:, 0].tolist() == [1, 1, 0, 0, 0]
    assert out[3, 1] == 1
    assert bool(torch.isfinite(out[:2, 2:]).all())
    assert bool((out[:2, 2] <= out[:2, 3]).all())


@pytest.mark.parametrize('invalid', ['straddler', 'nan', 'huge'])
def test_invalid_projection_never_certifies_a_drop(invalid):
    screen, pos, camera = _projected([[[-10, -10], [20, -10], [-10, 20]]])
    if invalid == 'straddler':
        screen[..., 9] = 0
    elif invalid == 'nan':
        screen[..., 0] = float('nan')
    else:
        screen[..., 0] = 1e9
    rect, out = torch.tensor([[0., 0., 1., 1.]]), torch.empty(1, 4)
    screen, pos, camera, rect, out = _device(screen, pos, camera, rect, out)
    _proof_kernel(screen, pos, camera, rect, out)
    assert out[0, :2].cpu().tolist() == [0, 0]
    assert out[0, 3].cpu().isinf()


def test_distance_intervals_enclose_continuous_perspective_intersections():
    rng = np.random.default_rng(654)
    triangles = np.tile(np.array([[-10., -10.], [20., -10.], [-10., 20.]]), (12, 1, 1))
    depths = rng.uniform(0.2, 12, (12, 3))
    out, screen, pos = _proofs(triangles, [[0, 0, 1, 1]] * 12, depths)
    points = rng.uniform(0, 1, (1000, 2))
    points = np.concatenate([points, [[0, 0], [1, 0], [0, 1], [1, 1]]])
    for i in range(12):
        sx, sy, iw = screen[0, i, :3].numpy(), screen[0, i, 3:6].numpy(), screen[0, i, 6:9].numpy()
        e = []
        for k in range(3):
            a, b = (k + 1) % 3, (k + 2) % 3
            e.append(((float(sx[b]) - float(sx[a])) * (points[:, 1] - float(sy[a]))
                      - (float(sy[b]) - float(sy[a])) * (points[:, 0] - float(sx[a]))) * float(iw[k]))
        weights = np.stack(e, -1)
        weights /= weights.sum(-1, keepdims=True)
        hp = weights @ pos[0, i].numpy().reshape(3, 3).astype(np.float64)
        distance = np.linalg.norm(hp, axis=-1)
        assert out[i, 0] == 1
        assert out[i, 2] <= distance.min()
        assert out[i, 3] >= distance.max()


@pytest.mark.parametrize('lengths', [[], [1], [16, 17, 257, 4097]])
@pytest.mark.parametrize('wide', [False, True])
def test_tiled_primary_order_matches_reference(lengths, wide):
    gen = torch.Generator().manual_seed(187)
    n = sum(lengths)
    pixels = torch.repeat_interleave(torch.tensor([0, 51, 117, 899][:len(lengths)]), torch.tensor(lengths, dtype=torch.int64))
    if wide:
        pixels += 1 << 25
    width, height = 67, 35
    # Repeated depth bins and mixed circuit/triangle tie identities.
    depths = torch.randint(1, 16, (n,), generator=gen).float() * 0.0001
    depths[::11] = 1e20  # saturated bins still order by signed layer
    refs = torch.randint(-900, 1000, (n,), generator=gen, dtype=torch.int32)
    if wide:
        refs[::7] = 2**30 + 37
    shuffle = torch.randperm(n, generator=gen)
    keys, refs = _keys(pixels, depths)[shuffle], refs[shuffle]
    expected = _exact_fragment_order(keys, refs, 7)
    tiles = _active_tiles(pixels, width, height)
    keys, refs, tiles = _device(keys, refs, tiles)
    actual = tiled.tile_fragment_order(keys, refs, 7, tiles, width, height)
    assert torch.equal(actual.cpu(), expected)


def test_scans_fail_before_narrowing_overflow():
    counts, = _device(torch.tensor([2**31 - 1, 1], dtype=torch.int64))
    with pytest.raises(OverflowError, match='int32 capacity'):
        tiled._prefix(counts, 'test')


@pytest.mark.parametrize('n', [1, 17, 257, 1001])
def test_simple_classifier_has_no_local_layer_capacity(n):
    offsets = torch.tensor([0, n], dtype=torch.int32)
    eligible = torch.ones(n, dtype=torch.int32)
    distance = torch.stack((torch.arange(n).float() + 1, torch.arange(n).float() + 1.1), -1)
    surfaces = torch.randperm(n, generator=torch.Generator().manual_seed(178)).to(torch.int32)
    scratch, simple = torch.empty(n, dtype=torch.int32), torch.empty(1, dtype=torch.int32)
    args = _device(offsets, eligible, distance, surfaces, scratch, simple)
    kernels.interior_pixels(*args, 1)
    assert args[-1].item() == 1
    if n > 1:
        args[3][n - 1] = args[3][0]
        kernels.interior_pixels(*args, 1)
        assert args[-1].item() == 0


@pytest.mark.parametrize('reason', ['overlap', 'coplanar', 'ineligible', 'saturation'])
def test_simple_classifier_rejects_ambiguous_stacks(reason):
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    eligible = torch.ones(2, dtype=torch.int32)
    distance = torch.tensor([[1., 2.], [3., 4.]])
    if reason == 'overlap':
        distance[1, 0] = 1.9
    elif reason == 'coplanar':
        distance[:] = 1.
    elif reason == 'ineligible':
        eligible[1] = 0
    else:
        distance *= 1e12
    surfaces = torch.tensor([5, 7], dtype=torch.int32)
    scratch, simple = torch.empty(2, dtype=torch.int32), torch.empty(1, dtype=torch.int32)
    args = _device(offsets, eligible, distance, surfaces, scratch, simple)
    kernels.interior_pixels(*args, 1)
    assert args[-1].item() == 0


def _scene_data(triangles, depths, alphas, width, height):
    from algan.rendering.raytracing.raytrace_kernels_taichi import _M_WIDTH

    screen, pos, camera = _projected(triangles, depths, width, height)
    n = len(triangles)
    colors = torch.zeros(1, n, 3, 5)
    colors[..., :3] = torch.tensor([0.8, 0.2, 0.1])
    colors[..., 4] = torch.tensor(alphas).view(1, n, 1)
    extra = torch.zeros(1, n, 15)
    extra[..., 6:9] = 1.
    merged = {
        'num_triangles': n, 'num_colored_triangles': n, 'num_circuits': 0,
        'tri_pos': pos, 'tri_colors': colors, 'tri_extra': extra,
        'tri_frame_valid': torch.ones(1, n, dtype=torch.bool),
        'tri_frame_opaque': torch.tensor(alphas).view(1, n) == 1.,
        'tri_alpha_uncertain': torch.zeros(1, n, dtype=torch.bool),
        'tri_closed': torch.zeros(1, n),
        'tri_obj': torch.arange(n, dtype=torch.int32).view(1, n),
        'tri_norm': torch.tensor([0., 0., -1.] * 3).expand(1, n, 9).contiguous(),
        'tri_uvs': torch.zeros(1, 1, 6),
        'tri_tex_meta': torch.full((1, 21), -1, dtype=torch.int32),
        'textures': torch.zeros(1, 1, 5),
        'tri_mat_id': torch.zeros(1, n, dtype=torch.int32),
        'tri_mat': torch.zeros(1, n, _M_WIDTH),
        'circuit_meta': torch.zeros(1, 1, 32),
        'circuit_colors': torch.zeros(1, 1, 1, 5),
        'circuit_border_colors': torch.zeros(1, 1, 1, 5),
        'edges_2d': torch.zeros(1, 1, 4),
        'edge_accel': torch.zeros(1, 1),
    }
    device = SETTINGS.computing.render_device
    merged = {k: v.to(device) if torch.is_tensor(v) else v for k, v in merged.items()}
    return merged, screen.to(device), camera.to(device)


def _discover(merged, camera, width, height, bins, simple, start=0, frames=1):
    from algan.rendering.raytracing.raster_pipeline import (
        precompute_triangle_projection,
        precompute_triangle_screen_bounds,
        prepare_sparse_raster_coverage,
    )
    from algan.utils.memory_utils import ManualMemory

    device = merged['tri_pos'].device
    memory = ManualMemory(0, device=device, num_bytes=64 << 20)
    camera = camera.expand(start + frames, 3).contiguous()
    screen_point = torch.tensor([[0., 0., 1.]], device=device).expand(start + frames, 3).contiguous()
    pbx = torch.tensor([[1., 0., 0.]], device=device).expand_as(screen_point).contiguous()
    pby = torch.tensor([[0., 1., 0.]], device=device).expand_as(screen_point).contiguous()
    pws = torch.full((start + frames,), 2. / height, device=device)
    col_row = torch.zeros(start + frames, dtype=torch.int32, device=device)
    with SETTINGS.raytracing.experimental.override(raster_tile_binning=bins, raster_simple_interiors=simple):
        screen = precompute_triangle_projection(merged, camera, screen_point, pbx, pby, width / 2, height / 2, memory)
        bounds = precompute_triangle_screen_bounds(merged, screen, camera, screen_point, pbx, pby, width / 2, height / 2, width, memory)
        result = prepare_sparse_raster_coverage(
            merged, screen, bounds, None, memory, camera, screen_point, pbx, pby, pws, col_row,
            start, start + frames, width, height, width / 2, height / 2, 0,
        )
        return None if result is None else {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in result.items()}


@pytest.mark.parametrize('simple', [False, True])
@pytest.mark.parametrize(('start', 'frames'), [(0, 1), (3, 2)])
def test_tiled_discovery_preserves_records_and_culls_only_certified_suffix(simple, start, frames):
    width, height = 67, 35  # partial fine/coarse bins on both axes
    broad = [[-500., -500.], [1500., -500.], [-500., 1500.]]
    diagonal = [[0., 1.], [65., 33.], [65., 33.1]]
    # Transparent front layer, full opaque middle, and a far invisible layer.
    merged, _, camera = _scene_data([broad, broad, broad, diagonal],
                                    [[1.] * 3, [20.] * 3, [1000.] * 3, [0.5] * 3],
                                    [0.4, 1., 0.5, 0.8], width, height)
    expected = _discover(merged, camera, width, height, False, False, start, frames)
    actual = _discover(merged, camera, width, height, True, simple, start, frames)
    assert expected is not None
    assert actual is not None
    assert actual['tile_candidates'] > 0
    assert actual['tile_occluded'] > 0
    assert actual['tile_bbox_rejected'] > 0
    for name in ('covered_idx', 'run_offsets', 'frag_key', 'frag_ref', 'frag_ab', 'frag_cov', 'frag_msk', 'frag_cap',
                 'sheet_key', 'sheet_ref', 'sheet_ab', 'sheet_cov', 'sheet_cap', 'sheet_offsets'):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
    assert torch.equal(actual['sheet_msk'] & ~kernels.SIMPLE_INTERIOR_BIT, expected['sheet_msk'])
    if simple:
        assert actual['num_simple_pixels'] > 0
        assert actual['num_general_pixels'] > 0  # sliver/diagonal geometry remains general


@pytest.mark.parametrize('simple', [False, True])
def test_offscreen_scene_returns_no_coverage(simple):
    merged, _, camera = _scene_data([[[100., 100.], [101., 100.], [100., 101.]]], [[1.] * 3], [0.5], 8, 8)
    assert _discover(merged, camera, 8, 8, True, simple) is None


def _interior_input():
    broad = [[-500., -500.], [1500., -500.], [-500., 1500.]]
    merged, screen, cam = _scene_data([broad, broad], [[1.] * 3, [20.] * 3], [0.5, 0.5], 4, 4)
    pixels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    depths = torch.tensor([1., 20., 1., 20.])
    coverage = {
        'frag_key': _keys(pixels, depths), 'frag_ref': torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        'frag_ab': torch.full((4, 2), 0.25), 'frag_cov': torch.ones(4),
        'frag_msk': torch.full((4,), _AA_MASK_ALL, dtype=torch.int32), 'frag_cap': torch.full((4,), 2.),
        'covered_idx': torch.tensor([0, 1], dtype=torch.int32), 'run_offsets': torch.tensor([0, 2, 4], dtype=torch.int32),
        'num_fragments': 4, 'num_covered': 2,
    }
    device = SETTINGS.computing.render_device
    coverage = {k: v.to(device) if torch.is_tensor(v) else v for k, v in coverage.items()}
    return coverage, merged, cam, torch.full((1,), 0.5, device=device), screen


@pytest.mark.parametrize('general_pixel', [False, True])
def test_simple_construction_bypasses_general_metadata(monkeypatch, general_pixel):
    from algan.rendering.raytracing import sheets

    cov, merged, cam, pws, screen = _interior_input()
    if general_pixel:
        cov['frag_cov'][2] = 0.5
        cov['frag_msk'][2] = 0  # an analytic-area donor between sample locations
    options = dict(tri_screen=screen, positioned_depth=False, sample_depth=False)
    expected = sheets.compact_sheets(cov, merged, cam, pws, 0, 4, 4, **options)
    original = sheets.compact_sheets
    seen = []

    def counted(coverage, *args, **kwargs):
        seen.append(coverage['num_covered'])
        return original(coverage, *args, **kwargs)

    monkeypatch.setattr(sheets, 'compact_sheets', counted)
    actual = tiled.compact_interior_sheets(cov, merged, cam, pws, 0, 4, 4, **options)
    assert seen == ([1] if general_pixel else [])
    assert actual['num_simple_pixels'] == (1 if general_pixel else 2)
    for name in ('sheet_key', 'sheet_ref', 'sheet_ab', 'sheet_wgt', 'sheet_cap', 'sheet_offsets'):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, check_dtype=False, msg=name)
    assert torch.equal(actual['sheet_wmsk'] & ~kernels.SIMPLE_INTERIOR_BIT, expected['sheet_wmsk'])


@pytest.mark.parametrize('reason', ['closed', 'uncertain', 'custom', 'same_surface', 'mask', 'coplanar'])
def test_simple_metadata_fallback_is_whole_pixel_and_preserves_general_results(reason):
    from algan.rendering.raytracing.sheets import compact_sheets

    cov, merged, cam, pws, screen = _interior_input()
    if reason == 'closed':
        merged['tri_closed'][:] = 1
    elif reason == 'uncertain':
        merged['tri_alpha_uncertain'][:] = True
    elif reason == 'custom':
        merged['has_user_pipeline'] = True
    elif reason == 'same_surface':
        merged['tri_obj'][:] = 0
    elif reason == 'mask':
        cov['frag_msk'][:] = 0
    else:
        merged['tri_pos'][:, 1] = merged['tri_pos'][:, 0]
        screen[:, 1] = screen[:, 0]
    options = dict(tri_screen=screen, positioned_depth=False, sample_depth=False)
    expected = compact_sheets(cov, merged, cam, pws, 0, 4, 4, **options)
    actual = tiled.compact_interior_sheets(cov, merged, cam, pws, 0, 4, 4, **options)
    assert actual['num_simple_pixels'] == 0
    for name in ('sheet_key', 'sheet_ref', 'sheet_ab', 'sheet_wgt', 'sheet_wmsk', 'sheet_cap', 'sheet_offsets'):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)


@ti.kernel
def _nonlinear_proofs(values: ti.types.ndarray(), results: ti.types.ndarray()):
    for i in range(values.shape[0]):
        r = kernels._reciprocal_interval(values[i])
        s = kernels._sqrt_interval(ti.abs(values[i]))
        results[i, 0], results[i, 1], results[i, 2], results[i, 3] = r[0], r[1], s[0], s[1]


def test_nonlinear_bounds_verify_the_estimate_instead_of_assuming_fast_math_accuracy():
    rng = np.random.default_rng(18)
    x = np.exp2(rng.uniform(-120, 120, 2000)).astype(np.float32)
    x[::2] *= -1
    values = torch.from_numpy(x)
    results = torch.empty(len(x), 4)
    values, results = _device(values, results)
    _nonlinear_proofs(values, results)
    out = results.cpu().numpy().astype(np.float64)
    reciprocal = 1. / x.astype(np.float64)
    square_root = np.sqrt(np.abs(x.astype(np.float64)))
    assert np.all(out[:, 0] <= reciprocal)
    assert np.all(out[:, 1] >= reciprocal)
    assert np.all(out[:, 2] <= square_root)
    assert np.all(out[:, 3] >= square_root)
    assert np.isfinite(out).all()  # ordinary finite ranges should certify


@pytest.mark.parametrize('case', ['alpha', 'transmission', 'alpha_texture', 'custom', 'legacy_custom'])
def test_opaque_rejection_requires_strict_material_proof(case):
    broad = [[-500., -500.], [1500., -500.], [-500., 1500.]]
    merged, screen, camera = _scene_data([broad, broad], [[1.] * 3, [100.] * 3], [1., 0.5], 16, 16)
    if case == 'alpha':
        merged['tri_colors'][:, 0, :, 4] = 1. - 1e-7
    elif case == 'transmission':
        merged['tri_extra'][:, 0, 9:12] = 1e-7
    elif case == 'custom':
        merged['has_user_pipeline'] = True
    elif case == 'legacy_custom':
        from algan.rendering.raytracing.settings import _USER_PIPELINE_BASE
        merged['tri_mat_id'][:, 0] = _USER_PIPELINE_BASE
    else:
        # Raw strict material test: an alpha map is uncertain even if the
        # per-vertex fallback or coarse opaque-class flag looks opaque.
        merged['num_colored_triangles'] = 0
        merged['tri_uvs'] = torch.zeros(1, 2, 6, device=screen.device)
        merged['tri_tex_meta'] = torch.full((2, 21), -1, dtype=torch.int32, device=screen.device)
        merged['tri_tex_meta'][0, :3] = torch.tensor([0, 2, 1], device=screen.device)
        merged['textures'] = torch.ones(1, 2, 5, device=screen.device)
    from algan.rendering.raytracing.raster_pipeline import precompute_triangle_screen_bounds
    from algan.utils.memory_utils import ManualMemory
    memory = ManualMemory(0, device=screen.device, num_bytes=1 << 20)
    sp, pbx, pby = _device(torch.tensor([[0., 0., 1.]]), torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 1., 0.]]))
    bounds = precompute_triangle_screen_bounds(merged, screen, camera, sp, pbx, pby, 8., 8., 16, memory)
    col_row = torch.zeros(1, dtype=torch.int32, device=screen.device)
    specs, _, stats = tiled.tiled_specs(merged, screen, bounds, None, camera, col_row, 0, 1, 16, 16)
    assert specs
    assert stats['tile_occluded'] == 0


def test_candidate_depth_saturation_and_integer_overflow_do_not_authorize_culling():
    candidates = torch.zeros(2, 9, dtype=torch.int32)
    candidates[:, 4:6] = 15
    candidates[:, 6] = 1
    flags = torch.zeros(2, dtype=torch.int32)
    intervals = torch.tensor([[1e10, 2e10], [1., 1e20]])
    bound = torch.tensor([0.01])
    counts, stats = torch.empty(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int32)
    args = _device(candidates, flags, intervals, bound, counts, stats)
    kernels.tile_pair_counts(*args, 2)
    assert args[-1].cpu().tolist() == [0, 0]
    assert bool((args[-2] > 0).all())
