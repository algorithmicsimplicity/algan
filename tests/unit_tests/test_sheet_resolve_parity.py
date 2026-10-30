"""The sheet resolve kernel against its sequential oracle, one sheet list at a time.

``sheets.resolve_pixel_reference`` states, plainly and in order, what a pixel's
depth-sorted sheet list composites to (DESIGN_sheet_resolve.md §2 and §4), and
the kernel's docstring promises to stay in lockstep with it. Nothing checked
that promise: every other oracle test compares the oracle with the
compaction's output, and the harness that does run the kernel
(``benchmarks/_sheet_resolve_parity.py --verify``) is in no group, and had
drifted onto stale argument positions and passed over zero rows.

So one 32x32 frame of six translucent unlit triangles, stacked over almost
every pixel, is rendered with its resolve launch intercepted. Before the real
kernel runs, the hook overwrites chosen pixels' sheet records in place with
hand-built lists -- the named cases below, then a seeded batch of random ones
-- and gives the six triangles the colours and alphas the lists name. What the
kernel commits lands in each pixel's accumulator row: RGB is the claims
weighted by colour, and the leftover weight (with its geometric share,
``ACC_GEO``) is the transmittance the walk left for the background. The oracle
computes both from the same list.

Pure unlit colours keep shading out of the comparison: what is compared is
coverage, claims and occlusion, the arithmetic the two are meant to share.
Transmission and circuits stay out (they need the refraction variant and a
circuit in the scene); circuits take the areal rule the sliver cases exercise.

Not marked fast: only a change to the resolve, its launch or the oracle can
break it.
"""

from __future__ import annotations

import random
from typing import NamedTuple

import pytest
import torch

from algan import IN, SMOKE_TEST, WHITE, MeshBasicMaterial, Off, Scene
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing import sheet_resolve_taichi
from algan.rendering.raytracing.glossy_prefilter_taichi import ACC_GEO
from algan.rendering.raytracing.raster_taichi import (
    _AA_BAND_CONT_BIT as BAND_CONT,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_LOSE_SHIFT as LOSE_SHIFT,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL as FULL,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_NUM_SAMPLES as N,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_ONE_MESH_BIT as ONE_MESH,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_SLIVER_BIT as SLIVER,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import min_alpha
from algan.rendering.raytracing.sheets import resolve_pixel_reference
from algan.settings import SETTINGS

PARAMS = sheet_resolve_taichi._SHEET_RESOLVE_SHADE_PARAMS

#: The materials a list's sheets name: (linear RGB, alpha). One per triangle of
#: the stack, so the stack's six refs are the six roles.
ROLES = {
    "R": ((1.0, 0.0, 0.0), 1.0),
    "G": ((0.0, 1.0, 0.0), 1.0),
    "B": ((0.0, 0.0, 1.0), 1.0),
    "g": ((0.0, 1.0, 0.0), 0.2),
    "veil": ((0.0, 0.0, 0.0), 0.9),
    "w": ((1.0, 1.0, 1.0), 0.5),
}

#: Sheets per stacked pixel, the longest list a case can have.
SLOTS = len(ROLES)

#: f32 kernel against the f64 oracle; measured agreement is ~2e-7.
TOLERANCE = 2e-5


class S(NamedTuple):
    """One sheet of a list, front to back."""

    role: str
    area: float
    msk: int = FULL
    #: The one-mesh ceiling; 2.0 is the host's "none" sentinel.
    cap: float = 2.0
    #: Its §4.4 band continues at the next sheet.
    continues: bool = False


def _record(sheet):
    """(area, mask word) as the compaction writes them: a band that continues
    carries ``_AA_BAND_CONT_BIT`` (``sheets._sibling_weights``).
    """
    return sheet.area, sheet.msk | (BAND_CONT if sheet.continues else 0)


def _ceded(samples):
    """Mask-word bits for samples ceded to a nearer surface (sheet_sample_depth)."""
    return samples << LOSE_SHIFT


#: Fills a pixel's unused slots: a zero-area sheet claims and occludes nothing.
_PAD = S("R", 0.0)

CASES = {
    "opaque": [S("R", 1.0)],
    "translucent stack": [S("w", 1.0), S("g", 1.0), S("B", 1.0)],
    "partial union over the background": [S("R", 0.3, 0b00000111)],
    "partial union over a floor": [S("R", 0.3, 0b00000111), S("B", 1.0)],
    "full union inside the dust band": [S("R", 1.0 - 5e-4), S("G", 1.0)],
    "full union outside the dust band": [S("R", 0.7), S("G", 1.0)],
    "areal sliver": [S("R", 0.2, SLIVER), S("B", 1.0)],
    "below min_alpha": [S("R", 5e-4), S("B", 1.0)],
    "rod behind an occluder (corr > 1)": [
        S("R", 0.875, 0b11111110),
        S("G", 0.4, 0b00000001),
    ],
    "two silhouettes (corr 1.2)": [
        S("R", 0.625, 0b11111000),
        S("G", 0.45, 0b00000111),
    ],
    # Where the samples the sheet does not own are still lit, so its clamped
    # residue has somewhere to go.
    "corr 1.2 over the background": [S("G", 0.45, 0b00000111)],
    "translucent corr 2.8 over a floor": [S("w", 0.7, 0b00000011), S("B", 1.0)],
    "band closed by a faint sibling": [
        S("R", 0.5, continues=True),
        S("g", 0.5),
        S("B", 1.0),
    ],
    "band closed by an opaque sibling": [
        S("g", 0.5, continues=True),
        S("R", 0.5),
        S("B", 1.0),
    ],
    "band behind a veil": [
        S("veil", 1.0),
        S("R", 0.997, continues=True),
        S("G", 0.003),
        S("B", 1.0),
    ],
    "band behind a partial occluder": [
        S("R", 0.875, 0b11111110),
        S("G", 0.994, continues=True),
        S("B", 0.006),
        S("G", 1.0),
    ],
    "a zero-area sibling inside a band": [
        S("R", 0.5, continues=True),
        S("G", 0.0, continues=True),
        S("B", 0.5),
        S("R", 1.0),
    ],
    "one-mesh ceiling spent": [
        S("R", 0.25, ONE_MESH | 0b00000111, cap=0.25),
        S("G", 0.25, ONE_MESH | 0b00000111, cap=0.25),
    ],
    "one-mesh ceiling splits a sheet": [
        S("R", 0.3, ONE_MESH | 0b00001111, cap=0.4),
        S("G", 0.3, ONE_MESH | 0b11110000, cap=0.4),
    ],
    "ceded samples": [
        S("R", 0.4, 0b00001111 | _ceded(0b00001100)),
        S("G", 1.0),
    ],
    "ceded samples on a corr > 1 sheet": [
        S("R", 0.6, 0b00001111 | _ceded(0b00001100)),
        S("G", 0.5, 0b00111100),
    ],
}

#: Cases where the two are known to disagree, with why.
KNOWN_DIVERGENCES = {}


def _random_list(rng):
    """One to six sheets drawn over every rule the walk has: full and partial
    unions, the dust band, corr above 1, areal slivers, bands, the one-mesh
    ceiling and ceded samples.

    Except W5's case: samples are only ceded on a sheet that redistributes no
    residue -- outside a band, where the summed factor can pass 1, and at
    corr <= 1. Drop that condition with the xfail above.
    """
    n = rng.randint(1, SLOTS)
    sheets = []
    for i in range(n):
        role = rng.choice(list(ROLES))
        kind = rng.random()
        if kind < 0.35:
            low = FULL
            area = rng.choice(
                (
                    1.0,
                    1.0 - rng.uniform(0.0, 2e-3),
                    1.0 + rng.uniform(0.0, 2e-3),
                    rng.random(),
                )
            )
        elif kind < 0.85:
            low = rng.randint(1, FULL - 1)
            area = rng.uniform(0.0, min(1.0, 1.6 * low.bit_count() / N))
        else:
            low = 0
            area = rng.uniform(0.0, 0.4)
        msk = low or SLIVER
        cap = 2.0
        if rng.random() < 0.3:
            msk |= ONE_MESH
            cap = rng.uniform(0.2, 1.0)
        continues = i < n - 1 and rng.random() < 0.25
        in_band = continues or (i > 0 and sheets[-1].continues)
        corr_at_most_1 = area <= low.bit_count() / N
        if low and not in_band and corr_at_most_1 and rng.random() < 0.3:
            msk |= _ceded(low & rng.randint(1, FULL))
        sheets.append(S(role, area, msk, cap, continues))
    return sheets


RANDOM = [_random_list(random.Random(seed)) for seed in range(800)]


def _oracle(sheets):
    """What ``resolve_pixel_reference`` says the pixel holds: RGB and the
    background's share.
    """
    sheets = sheets + [_PAD] * (SLOTS - len(sheets))
    covs, msks = zip(*map(_record, sheets))
    claims, transmittance = resolve_pixel_reference(
        list(covs),
        list(msks),
        [False] * len(sheets),
        [ROLES[s.role][1] for s in sheets],
        caps=[s.cap for s in sheets],
        min_alpha=min_alpha,
    )
    rgb = [
        sum(c * ROLES[s.role][0][k] for c, s in zip(claims, sheets)) for k in range(3)
    ]
    return rgb, sum(transmittance) / N


def _patch(launch, lists):
    """Write ``lists`` into the launch's sheet records, one stacked pixel each,
    and the roles into the stack's colours; return each list's covered index.
    """
    offsets = launch["sheet_offsets"].cpu()
    stacked = torch.nonzero(offsets[1:] - offsets[:-1] == SLOTS).flatten().tolist()
    assert len(stacked) >= len(lists), (
        f"{len(stacked)} pixels hold all {SLOTS} triangles; the cases need {len(lists)}"
    )
    ref = launch["sheet_ref"]
    refs = sorted(set(ref[: int(offsets[-1])].cpu().tolist()))
    assert len(refs) == SLOTS, f"the stack resolved to triangles {refs}"
    colors = launch["tri_colors"]
    assert colors.shape[1] == SLOTS, "per-vertex colours were promoted away"
    role_ref = dict(zip(ROLES, refs))
    for role, r in role_ref.items():
        rgb, alpha = ROLES[role]
        colors[:, r, :, :3] = torch.tensor(
            rgb, dtype=colors.dtype, device=colors.device
        )
        colors[:, r, :, 3] = 0.0
        colors[:, r, :, 4] = alpha
    slots, values = [], []
    for t, sheets in zip(stacked, lists):
        first = int(offsets[t])
        for j, sheet in enumerate(sheets + [_PAD] * (SLOTS - len(sheets))):
            slots.append(first + j)
            values.append((role_ref[sheet.role], *_record(sheet), sheet.cap))
    index = torch.tensor(slots, dtype=torch.int64, device=ref.device)
    for column, name in enumerate(("sheet_ref", "sheet_cov", "sheet_msk", "sheet_cap")):
        target = launch[name]
        target[index] = torch.tensor(
            [v[column] for v in values], dtype=target.dtype, device=target.device
        )
    return stacked[: len(lists)]


@pytest.fixture(scope="module")
def committed(tmp_path_factory):
    """Every list's accumulator row, keyed like ``CASES`` and by index into
    ``RANDOM``.
    """
    lists = {**CASES, **dict(enumerate(RANDOM))}
    rows = {}
    real = sheet_resolve_taichi.sheet_resolve_shade

    def patched(*args):
        launch = dict(zip(PARAMS, args))
        if int(launch["mode"]) != 0 or rows:
            return real(*args)
        pixels = _patch(launch, list(lists.values()))
        launch["pix_accum"].zero_()
        out = real(*args)
        accum = launch["pix_accum"].cpu()
        rows.update((key, accum[t].tolist()) for key, t in zip(lists, pixels))
        return out

    tmp_path = tmp_path_factory.mktemp("resolve_parity")
    snapshot = SETTINGS.snapshot()
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(sheet_resolve_taichi, "sheet_resolve_shade", patched)
            # Per-vertex colours, which the hook can rewrite; promotion would
            # move them into shared 1x1 maps holding the same values.
            monkeypatch.setattr(rt_settings, "promote_constants", False)
            SETTINGS.raytracing.set(shadows=False)
            with Scene() as scene:
                with Off():
                    for i in range(SLOTS):
                        corners = torch.tensor(
                            [[-9.0, -6.0, 0.0], [9.0, -6.0, 0.0], [0.0, 9.0, 0.0]]
                        ) + IN.view(1, 3) * (0.3 * i)
                        tri = TriangleTriangulated(corners, color=WHITE)
                        tri.set_material(MeshBasicMaterial(color=WHITE))
                        # After the material, which resets it.
                        tri.set_opacity(0.5)
                        tri.spawn(animate=False)
                scene.save_frame(str(tmp_path / "stack.png"), video_settings=SMOKE_TEST)
    finally:
        SETTINGS.restore(snapshot)
    assert rows, "the render made no shading launch to patch"
    return rows


def _deviation(row, sheets):
    """Largest gap between the kernel's row and the oracle, over RGB, the
    leftover weight's three channels and its geometric share.
    """
    rgb, background = _oracle(sheets)
    gaps = [abs(row[k] - rgb[k]) for k in range(3)]
    gaps += [abs(row[4 + k] - background) for k in range(3)]
    gaps.append(abs(row[ACC_GEO] - background))
    return max(gaps)


def _case(name):
    why = KNOWN_DIVERGENCES.get(name)
    if why is None:
        return name
    return pytest.param(name, marks=pytest.mark.xfail(reason=why, strict=True))


@pytest.mark.parametrize("name", [_case(name) for name in CASES])
def test_the_kernel_commits_what_the_oracle_says(committed, name):
    row, (rgb, background) = committed[name], _oracle(CASES[name])
    assert _deviation(row, CASES[name]) <= TOLERANCE, (
        f"kernel RGB {row[:3]}, leftover {row[4:7]}, geometric {row[ACC_GEO]}; "
        f"oracle RGB {rgb}, leftover {background}"
    )


def test_random_sheet_lists_match_the_oracle(committed):
    worst = sorted(
        ((_deviation(committed[i], sheets), i) for i, sheets in enumerate(RANDOM)),
        reverse=True,
    )
    off = [(gap, i) for gap, i in worst if gap > TOLERANCE]
    assert not off, f"{len(off)} of {len(RANDOM)} random lists disagree; " + "; ".join(
        f"list {i} by {gap:.3g}: {RANDOM[i]}" for gap, i in off[:3]
    )
