# Read-only audit probes: batch-window dependence of per-batch decisions.
# CPU tensors only; no rendering. Run:
#   ALGAN_USE_DAEMON=0 ALGAN_RENDER_DEVICE=cpu uv run python scratch_perf/ox/probe_batchwide_audit.py
import os

assert os.environ.get("ALGAN_RENDER_DEVICE") == "cpu", "set ALGAN_RENDER_DEVICE=cpu"
assert os.environ.get("ALGAN_USE_DAEMON") == "0"

import torch

from algan.rendering.raytracing.primitives import RayTracedBezierCircuitPrimitive
from algan.rendering.raytracing.scene_builder import _dedup_time, _split_promotable
from algan.rendering.raytracing.stbvh import segment_primitives_in_time


def sec(name):
    print(f"\n=== {name} ===")


# ------------------------------------------------------------------
# Probe 1: _compute_samples_per_segment -- one chord count per segment,
# chosen from max error over ALL frames of the window; a longer window
# can only pick a finer (larger) count.
# ------------------------------------------------------------------
sec("Q1 bezier chord counts vs window")

stub = object.__new__(RayTracedBezierCircuitPrimitive)
stub.num_pixels_per_sample = 0.5
stub.max_samples_per_segment = 512

T_full = 19
S = 1
t = torch.arange(T_full, dtype=torch.float32)
bow = 0.02 + 6.0 * (t / (T_full - 1)) ** 2  # screen bow grows with time
corners = torch.zeros(T_full, S, 4, 3)
corners[:, :, 0, 0] = -1.0
corners[:, :, 1, 1] = bow.view(-1, 1)
corners[:, :, 2, 1] = bow.view(-1, 1)
corners[:, :, 3, 0] = 1.0

cam_o = torch.tensor([[0.0, 0.0, 50.0]]).repeat(T_full, 1)
sp = torch.tensor([[0.0, 0.0, 0.0]]).repeat(T_full, 1)
sb = torch.zeros(T_full, 3, 3)
sb[:, 0, 0] = 1.0
sb[:, 1, 1] = 1.0
sb[:, 2, 2] = -1.0
screen_h = 1080.0

full_count = int(
    stub._compute_samples_per_segment(corners, cam_o, sp, sb, screen_h)[0]
)
sub_counts = {}
for lo in range(0, T_full, 3):
    hi = min(lo + 3, T_full)
    c = stub._compute_samples_per_segment(
        corners[lo:hi], cam_o[lo:hi], sp[lo:hi], sb[lo:hi], screen_h
    )
    sub_counts[(lo, hi)] = int(c[0])

print("count over full 19-frame window:", full_count)
print("per-3-frame-window counts:", sorted(set(sub_counts.values())))
print("monotone (every short-window count <= long-window count):",
      all(v <= full_count for v in sub_counts.values()))

# ------------------------------------------------------------------
# Probe 2: _split_promotable -- promotability is judged over the WINDOW's
# frames ("corner-uniform in every frame of the batch").
# ------------------------------------------------------------------
sec("Q3 _split_promotable vs window")


class P:
    pass


def make_p(T):
    # Corner-uniform colour while static; a corner GRADIENT (non-uniform)
    # once the animation starts at frame 3.
    colors = torch.full((T, 1, 3, 5), 0.5)
    for f in range(3, T):
        colors[f, :, 0, 0] = 0.5
        colors[f, :, 1, 0] = 0.5 + 0.02 * f
        colors[f, :, 2, 0] = 0.5 + 0.04 * f
    extra = torch.zeros(T, 1, 15)
    p = P()
    p._rt_tri_colors = colors
    p._rt_tri_extra = extra
    return p


scene = {}
tex_off = [0]


def append_texture(tex, is_color=False):
    o = tex_off[0]
    tex_off[0] += 1
    return (o, 1, 1)


for T in (3, 6, 19):
    tex_off[0] = 0
    scene.clear()
    keep, promo, meta = _split_promotable(
        make_p(T), append_texture, torch.device("cpu"), scene
    )
    print(f"T={T}: kept={keep.numel()} promoted={promo.numel()}")

# ------------------------------------------------------------------
# Probe 3: _dedup_time fires window-dependently but is value-preserving.
# ------------------------------------------------------------------
sec("Q3 _dedup_time")
x_var = torch.arange(3.0).view(3, 1, 1).repeat(1, 4, 2)
x_const = torch.ones(3, 4, 2)
print("varying table rows kept:", _dedup_time(x_var).shape[0], "(expect 3)")
print("constant table rows kept:", _dedup_time(x_const).shape[0], "(expect 1)")

# ------------------------------------------------------------------
# Probe 4: STBVH temporal segmentation -- instance set depends on the window.
# ------------------------------------------------------------------
sec("Q4 segment_primitives_in_time vs window")


def moving_bounds(T, origin=0.0):
    lo = torch.zeros(T, 2, 3)
    hi = torch.zeros(T, 2, 3)
    for f in range(T):
        d = 0.05 * (f + origin)
        lo[f, 0] = torch.tensor([-1 + d, -1, -1])
        hi[f, 0] = torch.tensor([1 + d, 1, 1])
        lo[f, 1] = torch.tensor([-1, -1, -2.0])
        hi[f, 1] = torch.tensor([1, 1, -2.0])
    return lo, hi


lo_full, hi_full = moving_bounds(19)
inst = segment_primitives_in_time(lo_full, hi_full, tightness=2.0)
print(f"19-frame window: {inst[0].numel()} instances",
      sorted(zip(inst[1].tolist(), inst[2].tolist())))
for a, b in ((0, 3), (8, 11)):
    lo, hi = moving_bounds(b - a, origin=a)
    inst = segment_primitives_in_time(lo, hi, tightness=2.0)
    print(f"same global motion, window [{a},{b}): {inst[0].numel()} instances",
          sorted(zip(inst[1].tolist(), inst[2].tolist())))

# ------------------------------------------------------------------
# Probe 5: surface_weld_flags / surface_closed_axes reduce over ALL frames
# of the materialized grid, including time.
# ------------------------------------------------------------------
sec("Q5 weld flags / closed axes vs window")
from algan.mobs.surfaces.surface import surface_closed_axes, surface_weld_flags

W, H = 12, 6
gap_per_frame = 2e-5  # frames 0..2 stay under the 1e-4 weld tolerance
grids = []
for f in range(19):
    g = gap_per_frame * f
    col0 = torch.zeros(H, 3)
    colN = torch.zeros(H, 3)
    colN[:, 0] = g
    grids.append(torch.stack([torch.lerp(col0, colN, w) for w in torch.linspace(0, 1, W)], 0))
grid_full = torch.stack(grids)  # [19, W, H, 3]
print("seam gaps frames 0..4:", [round(gap_per_frame * f, 7) for f in range(5)])
print("wrap_x  full-19:", surface_weld_flags(grid_full)[0],
      " first-3:", surface_weld_flags(grid_full[:3])[0])
print("closed_u full-19:", surface_closed_axes(grid_full)[0],
      " first-3:", surface_closed_axes(grid_full[:3])[0])

print("\ndone.")
