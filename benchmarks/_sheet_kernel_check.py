"""A/B parity for the compaction kernels (RASTER_FUSED_GATHER, SHEET_MASK_KERNEL).

The kernels replace multi-pass torch loops with one pass, and all are meant to
be BIT-IDENTICAL to the arm they replace: the gather copies bits, the mask
reductions are integer, and the exact-area sum keeps the f64 accumulator the
torch ``scatter_add_`` had -- that last one by measurement rather than by
construction, since an f64 atomic add reassociates and only the f32 cast makes
it agree. This checks two ways, because the two catch different mistakes:

* **unit** -- the kernels against the exact torch expressions they replaced,
  on random inputs at a 4K frame's shapes, including the ones the render never
  produces (empty bands, every-sample-shared bands, sliver flags, and 4096
  addends in a single area sum), plus a repeat-run check on the area;
* **end to end** -- four rendered frames of a scene carrying PN surfaces, flat
  polyhedra, transparency, bezier circuits and text, hashed with both toggles
  ON and both OFF.

    <venv-python> benchmarks/_sheet_kernel_check.py
"""

import hashlib
import os

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from algan import *  # noqa: E402,F403
from algan.constants.math import GIGABYTES  # noqa: E402
from algan.rendering.raytracing import sheets  # noqa: E402
from algan.rendering.raytracing.raster_pipeline import (  # noqa: E402
    _gather_fragment_arrays,
)
from algan.rendering.raytracing.raster_taichi import (  # noqa: E402
    _AA_MASK_ALL,
    _AA_NUM_SAMPLES,
    _AA_SLIVER_BIT,
)
from algan.settings import SETTINGS  # noqa: E402

EXPERIMENTAL = SETTINGS.raytracing.experimental
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
failures = []


def check(name, ok):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


def bits_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.is_floating_point:
        # Bitwise, so a NaN payload or a signed zero cannot pass as equal.
        width = torch.int32 if a.dtype == torch.float32 else torch.int64
        return bool((a.view(width) == b.view(width)).all())
    return bool((a == b).all())


# --------------------------------------------------------------- unit: gather
print("\ngather_fragment_arrays vs six index_selects")
g = torch.Generator(device=DEV).manual_seed(7)
for n, m in ((3_661_824, 3_290_404), (1, 1), (5000, 0), (17, 17)):
    key = torch.randint(-(1 << 45), 1 << 45, (n,), generator=g, device=DEV)
    ref = torch.randint(-9, 1 << 20, (n,), generator=g, device=DEV, dtype=torch.int32)
    ab = torch.randn(n, 2, generator=g, device=DEV)
    cov = torch.randn(n, generator=g, device=DEV)
    msk = torch.randint(0, 1 << 22, (n,), generator=g, device=DEV, dtype=torch.int32)
    opq = torch.randint(0, 2, (n,), generator=g, device=DEV) > 0
    idx = torch.randint(0, n, (m,), generator=g, device=DEV)
    args = (idx, key, ref, ab, cov, msk, opq)
    EXPERIMENTAL.set(raster_fused_gather=False)
    want = _gather_fragment_arrays(*args)
    EXPERIMENTAL.set(raster_fused_gather=True)
    got = _gather_fragment_arrays(*args)
    check(
        f"n={n} m={m}",
        all(bits_equal(a, b) for a, b in zip(want, got)) and len(want) == len(got),
    )

# ----------------------------------------------------------- unit: band reduce
print("\n_band_reduce / _popcount_lanes vs the torch passes they replace")


def mask_case(label, band, msk, cov, nb, want_sliver):
    EXPERIMENTAL.set(sheet_mask_kernel=False)
    w_area, w_union, w_fused, w_sliver = sheets._band_reduce(
        band, msk, cov, nb, want_sliver=want_sliver
    )
    w_pop = sheets._popcount_lanes(w_union)
    EXPERIMENTAL.set(sheet_mask_kernel=True)
    g_area, g_union, g_fused, g_sliver = sheets._band_reduce(
        band, msk, cov, nb, want_sliver=want_sliver
    )
    g_pop = sheets._popcount_lanes(g_union)
    ok = (
        bits_equal(w_area, g_area)
        and bits_equal(w_union, g_union)
        and bits_equal(w_fused, g_fused)
        and bits_equal(w_pop, g_pop)
        and ((w_sliver is None and g_sliver is None) or bits_equal(w_sliver, g_sliver))
    )
    check(f"{label} (fused bands: {int(w_fused.sum())})", ok)
    # The area sum is the one float reduction, and an f64 atomic add is not
    # order-independent in principle -- only far enough below an f32 ulp that
    # the cast absorbs it. That is the claim; this is the check.
    repeats = [
        sheets._band_reduce(band, msk, cov, nb, want_sliver=want_sliver)[0]
        for _ in range(4)
    ]
    check(
        f"{label}: area reproducible over 4 kernel runs",
        all(bits_equal(r, g_area) for r in repeats),
    )


n, nb = 3_661_824, 3_290_404
band = torch.randint(0, nb, (n,), generator=g, device=DEV)
msk = torch.randint(0, 1 << 22, (n,), generator=g, device=DEV, dtype=torch.int32)
# Exact areas look like the real thing: mostly whole pixels, a partial tail.
area = torch.rand(n, generator=g, device=DEV)
area = torch.where(area > 0.4, torch.ones_like(area), area)
mask_case("4K shapes, random masks, sliver", band, msk, area, nb, True)
mask_case("4K shapes, random masks, no sliver", band, msk, area, nb, False)

# Every fragment in one band, so every lane is claimed many times over: the
# fusion detector's saturated case, which random wide bands never reach, and
# the DEEPEST possible area sum -- 4096 addends into one f64 accumulator.
small = torch.zeros(4096, dtype=torch.int64, device=DEV)
dense = torch.full((4096,), _AA_MASK_ALL, dtype=torch.int32, device=DEV)
deep = torch.rand(4096, generator=g, device=DEV)
mask_case("one band, every lane shared, 4096 addends", small, dense, deep, 1, True)

# Masks that PARTITION their band -- the healthy case, which must NOT fuse.
part = torch.arange(8, dtype=torch.int64, device=DEV) % 4
lanes = (1 << (torch.arange(8, device=DEV) % _AA_NUM_SAMPLES)).to(torch.int32)
mask_case(
    "partitioning masks (must not fuse)",
    part,
    lanes,
    torch.rand(8, generator=g, device=DEV),
    4,
    True,
)

# Empty sample words plus sliver flags: the areal/donor sheets.
donor_band = torch.zeros(64, dtype=torch.int64, device=DEV)
donor_msk = torch.full((64,), _AA_SLIVER_BIT, dtype=torch.int32, device=DEV)
mask_case(
    "donors only (empty union, sliver set)",
    donor_band,
    donor_msk,
    torch.rand(64, generator=g, device=DEV),
    1,
    True,
)

# ------------------------------------------------------------- end to end
print("\nrendered frames, both toggles ON vs both OFF")
SETTINGS.computing.set(available_memory_override=2 * GIGABYTES)
sphere = Sphere().scale(1.4).move(LEFT * 3).set_color(GREEN).spawn()
cube = Cube().scale(1.1).move(RIGHT * 3).set_color(BLUE).spawn()
glass = Sphere().scale(0.9).move(UP * 1.2).spawn()
glass.opacity = 0.45
circle = Circle().scale(0.8).move(DOWN * 1.8 + LEFT * 1.5).set_color(RED).spawn()
label = Text("sheets").scale(0.6).move(DOWN * 2.4 + RIGHT * 1.5).spawn()
with Sync():
    sphere.rotate(70, UP)
    cube.rotate(55, OUT + RIGHT)
    glass.move(RIGHT * 1.3)
    circle.rotate(40, OUT)
    label.move(UP * 0.3)


def render_hashes(arm):
    out = []
    for i, at in enumerate((0.0, 0.35, 0.7, 1.0)):
        path = Scene.save_frame(f"_sheet_kernel_{arm}_{i}.png", MD, at=at).output_path
        arr = np.asarray(Image.open(path).convert("RGB"))
        out.append(hashlib.sha256(arr.tobytes()).hexdigest())
    return out


EXPERIMENTAL.set(raster_fused_gather=False, sheet_mask_kernel=False)
torch_arm = render_hashes("torch")
EXPERIMENTAL.set(raster_fused_gather=True, sheet_mask_kernel=True)
kernel_arm = render_hashes("kernel")
for i, (a, b) in enumerate(zip(torch_arm, kernel_arm)):
    check(f"frame {i}  {a[:16]}", a == b)

print("\nFAILURES:", failures if failures else "none -- bit-identical")
raise SystemExit(1 if failures else 0)
