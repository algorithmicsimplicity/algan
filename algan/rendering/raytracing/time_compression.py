"""Per-primitive temporal compression of animated geometry.

Algan renders a chunk of ``T`` frames at once. Today every animated geometry
array is materialized densely as ``[T, N, D]`` (T frames, N primitives, D
feature floats -- e.g. ``D = 9`` for a triangle's three corners) and uploaded to
the GPU in full. Crucially the time axis is a *single shared dimension*: the
trace kernels fetch geometry with ``tp = f % array.shape[0]`` (see
``ray_trace_taichi.py``), so the moment a single primitive in a merged array
moves, the array's length becomes ``T`` and **every** static primitive in it is
replicated across all ``T`` frames -- in memory and across PCIe.

This module replaces that shared dense time axis with a per-primitive
piecewise-linear *knot* representation that the trace kernel reconstructs on the
fly. Each primitive is classified independently:

* **static** -- its geometry is constant over the chunk: one knot.
* **linear** -- its geometry sweeps a single straight line in ``D``-space (pure
  translation, uniform scale, colour/opacity fades, with *any* easing or pauses,
  including motion that continues in one direction across several animations):
  two knots (the segment endpoints) plus a per-frame eased fraction ``z`` so the
  exact eased motion is reproduced by ``lerp(knot0, knot1, z[f])``. Primitives
  that share the same ``z`` trajectory (the common case: one animation driving
  many triangles) share one *schedule*, so ``z`` is stored once per distinct
  timeline rather than per primitive.
* **dense** -- anything else (rotation, multi-direction paths): one knot per
  frame, i.e. the original dense data, so correctness is never sacrificed.

The eased fraction is *recovered numerically* from the dense geometry (the
orthogonal projection of each frame onto the segment chord), not threaded out of
the animation system: a frame is accepted as "linear" only when its residual to
the chord is below ``tol_linear``, so anything classified linear is
reconstructed to within that tolerance and everything else falls back to dense.
With a tight tolerance the reconstruction stays far below one pixel, within the
renderer's frame-comparison budget.

Layout (a :class:`TimeCompressed`): knot *values* are stored CSR-style in a flat
``knot_val[total_knots, D]`` indexed by per-primitive ``knot_base[N]`` /
``knot_count[N]``; the per-frame *schedule* (which knot interval each frame sits
in, and the blend fraction) lives in small ``sched_seg[S, T]`` / ``sched_z[S,
T]`` tables addressed by ``sched_id[N]``. Reconstructing primitive ``p`` at frame
``f`` is::

    s   = sched_id[p]
    k   = sched_seg[s, f]                 # local knot interval
    khi = min(k + 1, sched_nknots[s] - 1)
    v   = lerp(knot_val[knot_base[p] + k],
               knot_val[knot_base[p] + khi], sched_z[s, f])

Everything here is vectorized PyTorch and device-agnostic; the per-ray
reconstruction lives in the Taichi kernels.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# A frame counts as lying on a primitive's segment chord when its max
# coordinate deviation is below this (world units for positions). Kept tight so
# a "linear" classification is numerically near-exact -- well under a pixel --
# and anything genuinely curved (rotation) falls through to the dense path.
DEFAULT_TOL_LINEAR = 1e-5
# A primitive is static when its geometry varies by less than this across the
# whole chunk.
DEFAULT_TOL_STATIC = 1e-6
# Two linear primitives are merged into one shared schedule when their recovered
# ``z`` trajectories agree to within this (max abs difference over frames). Copies
# of the same animation agree to float noise (~1e-7) so they group; genuinely
# different animations differ by far more somewhere and stay separate. The stored
# schedule ``z`` is an actual group member, so a merge adds at most this much
# error (times the chord length) to the other members' reconstruction.
DEFAULT_Z_GROUP_TOL = 1e-4


@dataclass
class TimeCompressed:
    """Compressed temporal representation of a dense ``[T, N, D]`` array.

    Attributes
    ----------
    knot_val : Tensor[total_knots, D] (float32)
        Flat CSR store of every primitive's knot values, in primitive order.
    knot_base : Tensor[N] (int32)
        First row in ``knot_val`` belonging to each primitive.
    knot_count : Tensor[N] (int32)
        Number of knots each primitive owns (== ``sched_nknots[sched_id]``).
    sched_id : Tensor[N] (int32)
        Schedule each primitive follows.
    sched_seg : Tensor[S, T] (int32)
        For each schedule and frame, the local knot interval index in
        ``[0, sched_nknots - 1]``.
    sched_z : Tensor[S, T] (float32)
        For each schedule and frame, the blend fraction within that interval.
    sched_nknots : Tensor[S] (int32)
        Knot count of each schedule.
    T, N, D : int
        Frame count, primitive count, feature width of the source array.
    """

    knot_val: torch.Tensor
    knot_base: torch.Tensor
    knot_count: torch.Tensor
    sched_id: torch.Tensor
    sched_seg: torch.Tensor
    sched_z: torch.Tensor
    sched_nknots: torch.Tensor
    T: int
    N: int
    D: int

    @property
    def total_knots(self) -> int:
        return int(self.knot_val.shape[0])

    @property
    def num_schedules(self) -> int:
        return int(self.sched_nknots.shape[0])

    def to(self, device) -> "TimeCompressed":
        return TimeCompressed(
            knot_val=self.knot_val.to(device),
            knot_base=self.knot_base.to(device),
            knot_count=self.knot_count.to(device),
            sched_id=self.sched_id.to(device),
            sched_seg=self.sched_seg.to(device),
            sched_z=self.sched_z.to(device),
            sched_nknots=self.sched_nknots.to(device),
            T=self.T, N=self.N, D=self.D,
        )


def _make_static(X: torch.Tensor) -> TimeCompressed:
    """Single-frame source: every primitive is one static knot."""
    _, N, D = X.shape
    device = X.device
    return TimeCompressed(
        knot_val=X[0].to(torch.float32).contiguous(),
        knot_base=torch.arange(N, dtype=torch.int32, device=device),
        knot_count=torch.ones(N, dtype=torch.int32, device=device),
        sched_id=torch.zeros(N, dtype=torch.int32, device=device),
        sched_seg=torch.zeros((1, 1), dtype=torch.int32, device=device),
        sched_z=torch.zeros((1, 1), dtype=torch.float32, device=device),
        sched_nknots=torch.ones(1, dtype=torch.int32, device=device),
        T=1, N=N, D=D,
    )


def compress_time(
    X: torch.Tensor,
    tol_linear: float = DEFAULT_TOL_LINEAR,
    tol_static: float = DEFAULT_TOL_STATIC,
    z_group_tol: float = DEFAULT_Z_GROUP_TOL,
) -> TimeCompressed:
    """Compress a dense ``[T, N, D]`` geometry array into knots + schedules.

    Each primitive is classified independently as static (1 knot), linear (2
    knots + a recovered eased ``z``) or dense (``T`` knots). See the module
    docstring for the representation. ``expand_time`` is the exact inverse up to
    ``tol_linear`` for primitives accepted as linear (and bit-exact for static /
    dense ones).
    """
    if X.dim() != 3:
        raise ValueError(f"expected [T, N, D], got {tuple(X.shape)}")
    X = X.to(torch.float32)
    T, N, D = X.shape
    device = X.device
    if T <= 1:
        return _make_static(X)

    X0 = X[0]                               # [N, D]  start knot
    X1 = X[-1]                              # [N, D]  end knot
    chord = X1 - X0                         # [N, D]
    chord_sq = (chord * chord).sum(-1)      # [N]

    # Static: max deviation from the first frame is negligible.
    dev_static = (X - X0.unsqueeze(0)).abs().amax(dim=0).amax(dim=-1)   # [N]
    is_static = dev_static < tol_static

    # Linear: every frame projects onto the chord with a tiny residual. The
    # least-squares scalar for frame f is z = <X[f]-X0, chord>/|chord|^2 (the
    # same z for all D, so the residual measures departure from a shared-z
    # line); reconstruct and check the worst coordinate error.
    rel = X - X0.unsqueeze(0)                                 # [T, N, D]
    safe_chord_sq = chord_sq.clamp_min(1e-30)
    z = (rel * chord.unsqueeze(0)).sum(-1) / safe_chord_sq.unsqueeze(0)   # [T, N]
    recon = X0.unsqueeze(0) + z.unsqueeze(-1) * chord.unsqueeze(0)        # [T,N,D]
    resid = (X - recon).abs().amax(dim=0).amax(dim=-1)                    # [N]
    moves = chord_sq > (tol_static * tol_static)
    is_linear = (~is_static) & moves & (resid < tol_linear)

    is_dense = ~is_static & ~is_linear

    # ----- schedule assignment -------------------------------------------
    # Schedule 0: static (1 knot).  Schedule 1: dense (T knots) -- shared by all
    # dense primitives (seg = frame index, z = 0).  Schedules 2..: one per
    # distinct linear z trajectory.
    sched_id = torch.empty(N, dtype=torch.int64, device=device)
    sched_id[is_static] = 0
    sched_id[is_dense] = 1

    sched_seg_rows = [
        torch.zeros(T, dtype=torch.int64, device=device),            # static
        torch.arange(T, dtype=torch.int64, device=device),           # dense
    ]
    sched_z_rows = [
        torch.zeros(T, dtype=torch.float32, device=device),          # static
        torch.zeros(T, dtype=torch.float32, device=device),          # dense
    ]
    sched_nknots_list = [1, T]

    lin_idx = is_linear.nonzero(as_tuple=True)[0]                     # [n_lin]
    if lin_idx.numel() > 0:
        zT = z[:, lin_idx].transpose(0, 1).contiguous()              # [n_lin, T]
        # Group primitives that share an animation: greedy tolerance clustering.
        # Round-based keys fragment near rounding boundaries, so instead cluster
        # by max-abs-difference. One vectorized pass per distinct animation (a
        # small number in practice), so this is cheap unless every primitive
        # genuinely animates differently.
        n_lin = lin_idx.numel()
        remaining = torch.ones(n_lin, dtype=torch.bool, device=device)
        cluster_of = torch.empty(n_lin, dtype=torch.int64, device=device)
        cid = 0
        while bool(remaining.any()):
            seed = int(remaining.nonzero(as_tuple=True)[0][0].item())
            rep = zT[seed]
            match = remaining & ((zT - rep).abs().amax(dim=1) <= z_group_tol)
            cluster_of[match] = cid
            sched_seg_rows.append(
                torch.zeros(T, dtype=torch.int64, device=device))
            sched_z_rows.append(rep)
            sched_nknots_list.append(2)
            remaining = remaining & ~match
            cid += 1
        sched_id[lin_idx] = 2 + cluster_of

    sched_seg = torch.stack(sched_seg_rows, 0).to(torch.int32)        # [S, T]
    sched_z = torch.stack(sched_z_rows, 0).to(torch.float32)         # [S, T]
    sched_nknots = torch.tensor(sched_nknots_list, dtype=torch.int32,
                                device=device)                       # [S]

    # ----- knot values (CSR) ---------------------------------------------
    knot_count = sched_nknots.to(torch.int64)[sched_id]              # [N]
    knot_base = torch.zeros(N, dtype=torch.int64, device=device)
    knot_base[1:] = torch.cumsum(knot_count, 0)[:-1]
    total = int(knot_count.sum().item())

    prim_of_row = torch.repeat_interleave(
        torch.arange(N, device=device), knot_count)                  # [total]
    local_k = torch.arange(total, device=device) - knot_base[prim_of_row]
    # Source frame of each knot row: static -> 0; linear -> {0, T-1}; dense -> k.
    cls = torch.zeros(N, dtype=torch.int64, device=device)
    cls[is_linear] = 1
    cls[is_dense] = 2
    cls_row = cls[prim_of_row]
    frame_of_row = torch.where(
        cls_row == 0, torch.zeros_like(local_k),
        torch.where(cls_row == 1, local_k * (T - 1), local_k))
    knot_val = X[frame_of_row, prim_of_row].contiguous()             # [total, D]

    return TimeCompressed(
        knot_val=knot_val,
        knot_base=knot_base.to(torch.int32),
        knot_count=knot_count.to(torch.int32),
        sched_id=sched_id.to(torch.int32),
        sched_seg=sched_seg,
        sched_z=sched_z,
        sched_nknots=sched_nknots,
        T=T, N=N, D=D,
    )


def extract_global_linear_z(X: torch.Tensor, tol: float = DEFAULT_TOL_LINEAR):
    """If a dense ``[T, N, D]`` attribute is a single straight-line segment shared
    by *every* element -- ``X[f] = X[0] + z[f] * (X[-1] - X[0])`` for one scalar
    ``z[f]`` per frame, with any easing -- return that per-frame fraction ``z``
    (shape ``[T]``); otherwise return ``None``.

    This is the cheap timeline probe behind mode-3 compression: run it on a small
    attribute (e.g. a mob's location) to decide whether the whole mob is moving
    along one eased line, in which case its geometry need only be *built* at the
    two segment endpoints and linearly expanded with ``z`` (geometry is affine in
    location). ``None`` means "not globally linear" -> fall back to dense.
    """
    if X.dim() != 3:
        raise ValueError(f"expected [T, N, D], got {tuple(X.shape)}")
    X = X.to(torch.float32)
    T = X.shape[0]
    if T <= 1:
        return torch.zeros(T, device=X.device)
    X0, X1 = X[0], X[-1]
    chord = X1 - X0                                  # [N, D]
    chord_sq = float((chord * chord).sum())
    if chord_sq < tol * tol:                         # no motion -> static
        moved = (X - X0.unsqueeze(0)).abs().amax()
        return torch.zeros(T, device=X.device) if moved < tol else None
    rel = X - X0.unsqueeze(0)                         # [T, N, D]
    z = (rel * chord.unsqueeze(0)).sum((-1, -2)) / chord_sq          # [T]
    recon = X0.unsqueeze(0) + z.view(T, 1, 1) * chord.unsqueeze(0)
    if float((X - recon).abs().amax()) < tol:
        return z
    return None


def expand_linear(knot_lo: torch.Tensor, knot_hi: torch.Tensor,
                  z: torch.Tensor) -> torch.Tensor:
    """Expand a 2-knot linear segment to dense frames: returns ``[T, *shape]``
    with ``out[f] = knot_lo + z[f] * (knot_hi - knot_lo)``. ``knot_lo``/``knot_hi``
    are a single frame's values (any trailing shape); ``z`` is ``[T]``."""
    T = z.shape[0]
    zv = z.view(T, *([1] * knot_lo.dim()))
    return knot_lo.unsqueeze(0) + zv * (knot_hi - knot_lo).unsqueeze(0)


def expand_time(tc: TimeCompressed) -> torch.Tensor:
    """Reconstruct the dense ``[T, N, D]`` array from a :class:`TimeCompressed`.

    The exact inverse of :func:`compress_time` (up to ``tol_linear`` for
    primitives that were accepted as linear). Used to feed the geometry to
    kernels that have not been ported to in-kernel reconstruction, and for
    validation.
    """
    T, N, D = tc.T, tc.N, tc.D
    device = tc.knot_val.device
    sched_id = tc.sched_id.to(torch.int64)
    base = tc.knot_base.to(torch.int64)

    seg = tc.sched_seg.to(torch.int64)[sched_id]                     # [N, T]
    zf = tc.sched_z[sched_id]                                        # [N, T]
    nk = tc.sched_nknots.to(torch.int64)[sched_id]                  # [N]
    seg_hi = torch.minimum(seg + 1, (nk - 1).unsqueeze(-1))         # [N, T]

    lo_idx = base.unsqueeze(-1) + seg                               # [N, T]
    hi_idx = base.unsqueeze(-1) + seg_hi                            # [N, T]
    v_lo = tc.knot_val[lo_idx]                                      # [N, T, D]
    v_hi = tc.knot_val[hi_idx]                                      # [N, T, D]
    zf = zf.unsqueeze(-1)                                           # [N, T, 1]
    out = v_lo + zf * (v_hi - v_lo)                                 # [N, T, D]
    return out.transpose(0, 1).contiguous()                        # [T, N, D]
