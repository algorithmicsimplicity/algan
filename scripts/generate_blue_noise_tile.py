"""Generate the path tracer's blue-noise sampler tile.

What this produces
------------------

``algan/rendering/raytracing/data/blue_noise_tile_64.npy``: a ``uint16``
``64 x 64`` array holding a **permutation** of ``0 .. 4095``. Each entry is a
per-pixel sampler key -- the whole per-pixel half of a Sobol-Owen draw's seed
(``path_seed = hash_combine(_PT_BN_SALT, tile[y, x])`` in
``path_tracer_taichi``) -- and the permutation is optimised so that the Monte
Carlo error of a low-sample-count render is distributed as **blue noise** in
screen space (Heitz et al., "A Low-Discrepancy Sampler that Distributes Monte
Carlo Errors as a Blue Noise in Screen Space", SIGGRAPH 2019).

The optimisation is Heitz's, adapted to a sampler that derives every dimension
pair from ONE per-pixel key. Their tiles are per-dimension (a scrambling and a
ranking value per pixel per dimension); Algan's sampler hashes the dimension
pair on top of a single per-pixel seed, so there is one layer and the energy
has to see all the dimensions that matter at once. Concretely, for each
candidate key ``v`` we build the **sample vector**

    s(v) = [ pt_sample_2d(path_seed(v), pair, i)  for pair in PAIRS
                                                  for i in range(SAMPLES) ]

-- the sampler's own first ``SAMPLES`` draws in the dimension pairs a low-spp
render spends most of its variance on -- and anneal the assignment of keys to
pixels to minimise

    E = sum over pixel pairs (p, q)  exp( -|p - q|^2 / sigma_i^2
                                          -|s_p - s_q|^2 / (sigma_s^2 * D) )

with toroidal pixel distances (the tile is tiled across the screen, so its
metric must wrap) over a 7x7 neighbourhood. That is the paper's energy with one
deviation, noted because it matters: the paper raises the sample-space distance
to the power ``d/2``, which is calibrated for a per-dimension tile (``d = 2``).
At our ``D = 12`` that exponent makes ``exp(-|ds|^6)`` either 0 or 1 for every
pair and the landscape goes flat, so the distance is used squared and
normalised per component instead.

Optimising the *sample vectors* rather than the *error* of chosen integrands is
also the paper's trick, and it is what makes the tile scene-independent: error
at a pixel is a functional of that pixel's sample set, so sample sets that are
far apart in sample space have errors that are (for anything smooth enough)
anti-correlated. It also measured better here than an error-vector energy over
representative integrands (1.14x against 1.10x on the held-out metric below).

The assignment is a **permutation**, and that is load-bearing rather than
traditional: over every tile period the multiset of keys is the whole key set,
so no bias can be introduced by preferring keys whose samples sit anywhere in
particular. Picking keys freely from a larger pool optimises better and is
wrong -- a fixed key is a quadrature rule, not an unbiased estimator; only the
randomness of the assignment makes it one.

Which dimension pairs
---------------------

``PAIRS`` is ``(0, 3, 54)`` in the module's table
(``path_tracer_taichi``'s docstring):

* ``0`` -- sub-pixel jitter. Every render, every pixel; drives edge AA.
* ``3`` -- bounce 0's BSDF direction: the indirect-light draw. A fixed index
  (``2 + 6b + 1``), so it is pair 3 in every render whatever ``max_bounces``
  is.
* ``54`` -- the first surface crossing's light point, where a lit scene's
  direct-lighting variance lives, at the SHIPPED defaults
  (``2 + 6B + (2L+1)c + 1`` with ``B = max_bounces = 8``,
  ``L = pt_light_samples = 1``, ``c = 1``). A render at a different
  ``max_bounces`` moves it, and keeps the benefit of the other two pairs only
  -- the tile stays a valid seed assignment either way (same estimator, same
  convergence), it is only its screen-space optimisation that degrades.

More pairs is not better, measured: the energy is one assignment serving every
component of ``s``, so widening ``s`` spends the same optimisation over more
dimensions and each gets less. Six pairs by two samples scored 1.13x on the
held-out metric where three by two scored 1.37x, and one pair alone scored
2.4x on ITS pair and nothing on the others.

Parameters of record
--------------------

``TILE = 64``, ``PAIRS = (0, 3, 54)``, ``SAMPLES = 2`` (so ``D = 12``),
``RADIUS = 3`` (a 7x7 toroidal neighbourhood), ``SIGMA_I = 2.1``,
``SIGMA_S = 0.35``, ``SWEEPS = 300`` (300 * 4096 = 1.2 M proposed swaps),
geometric temperature schedule from ``T0 = 0.30`` to ``T1 = 0.002``, seeded
``numpy.random.default_rng(20260905)``. Reproducible: rerunning writes a
byte-identical file. ``SIGMA_S`` and ``SWEEPS`` were scanned; 0.2-0.35 are
equivalent and 0.55 upward is worse, and the energy has converged by ~120
sweeps (400 measured no better).

Held-out result of the shipped tile (integrands the energy never saw, against
a random permutation of the same keys): the low-frequency energy of the error
field is **1.46-1.56x lower at 1 sample, 1.28-1.59x at 2 and 1.01-1.06x at
4**, over the three optimised pairs, with the total error power unchanged. The decay with sample count is inherent -- only the
first ``SAMPLES`` draws of each pair are in the energy -- and it is why the
render-level payoff (``benchmarks/_pt_blue_noise_check.py``) is a low-spp
effect.

Run::

    uv run python scripts/generate_blue_noise_tile.py            # write the tile
    uv run python scripts/generate_blue_noise_tile.py --verify   # check the
        # committed tile against the kernel's own sampler and report its
        # held-out spectral numbers, writing nothing

Numpy only, no Taichi, no GPU: about two minutes for the 64x64 tile on one CPU
core.
``--verify`` additionally launches ``pt_sampler_probe`` to prove the numpy
replica of the sampler in here still agrees with the kernel bit for bit; that
is the only step that imports algan, which is why the run ends with the
runner's "finished without rendering anything" note. It is not an error --
this script produces a table, not a frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
OUT_PATH = (
    REPO / "algan" / "rendering" / "raytracing" / "data" / "blue_noise_tile_64.npy"
)

TILE = 64
PAIRS = (0, 3, 54)
SAMPLES = 2
RADIUS = 3
SIGMA_I = 2.1
SIGMA_S = 0.35
SWEEPS = 300
T0 = 0.30
T1 = 0.002
RNG_SEED = 20260905

#: Must match ``path_tracer_taichi._PT_BN_SALT``: the fixed hash seed the
#: kernel combines a tile value with to get the path seed. Fixed (not
#: ``pt_seed``) on purpose -- the map from tile value to sample sequence is
#: what this script optimises, so nothing per-render may enter it. The render
#: seed and the frame enter as a toroidal SHIFT of the tile lookup instead,
#: which is an isometry of the torus and so preserves the optimisation.
PT_BN_SALT = 0x9E3779B1


# ---------------------------------------------------------------------------
# The sampler, in numpy (a replica of path_tracer_taichi's; --verify checks it)
# ---------------------------------------------------------------------------

U32 = np.uint32


def _u32(x):
    return np.asarray(x, dtype=np.uint64).astype(U32)


def pt_hash(x):
    x = np.array(x, dtype=U32, copy=True)
    x ^= x >> U32(16)
    x *= U32(0x7FEB352D)
    x ^= x >> U32(15)
    x *= U32(0x846CA68B)
    x ^= x >> U32(16)
    return x


def pt_hash_combine(seed, value):
    seed = np.asarray(seed, dtype=U32)
    value = np.asarray(value, dtype=U32)
    return pt_hash(
        seed ^ (value + U32(0x9E3779B9) + (seed << U32(6)) + (seed >> U32(2)))
    )


def pt_reverse_bits(x):
    x = np.array(x, dtype=U32, copy=True)
    x = ((x >> U32(1)) & U32(0x55555555)) | ((x & U32(0x55555555)) << U32(1))
    x = ((x >> U32(2)) & U32(0x33333333)) | ((x & U32(0x33333333)) << U32(2))
    x = ((x >> U32(4)) & U32(0x0F0F0F0F)) | ((x & U32(0x0F0F0F0F)) << U32(4))
    x = ((x >> U32(8)) & U32(0x00FF00FF)) | ((x & U32(0x00FF00FF)) << U32(8))
    return (x >> U32(16)) | (x << U32(16))


def pt_laine_karras(x, seed):
    x = np.array(x, dtype=U32, copy=True) + np.asarray(seed, dtype=U32)
    x ^= x * U32(0x6C50B47C)
    x ^= x * U32(0xB82F1E52)
    x ^= x * U32(0xC7AFE638)
    x ^= x * U32(0x8D22F6E6)
    return x


def pt_owen_scramble(x, seed):
    return pt_reverse_bits(pt_laine_karras(pt_reverse_bits(x), seed))


def _sobol_dim1_directions():
    dirs = []
    v = 0x80000000
    for _ in range(32):
        dirs.append(v)
        v = v ^ (v >> 1)
    return np.array(dirs, dtype=U32)


_SOBOL_DIM1 = _sobol_dim1_directions()


def pt_sobol_dim1(index):
    index = np.asarray(index, dtype=U32)
    out = np.zeros(index.shape, dtype=U32)
    for j in range(32):
        bit = ((index >> U32(j)) & U32(1)).astype(bool)
        out[bit] ^= _SOBOL_DIM1[j]
    return out


def pt_sample_2d_seeded(path_seed, pair, sample_index):
    """``[..., 2]`` of the sampler's draw, vectorised over ``path_seed``."""
    pair_seed = pt_hash_combine(path_seed, np.full_like(path_seed, U32(pair)))
    shuffle_seed = pt_hash(pair_seed ^ U32(0x51633E2D))
    seed_x = pt_hash(pair_seed ^ U32(0x68BC21EB))
    seed_y = pt_hash(pair_seed ^ U32(0x02E5BE93))
    index = pt_owen_scramble(np.full_like(pair_seed, U32(sample_index)), shuffle_seed)
    vx = pt_reverse_bits(pt_laine_karras(index, seed_x))
    vy = pt_owen_scramble(pt_sobol_dim1(index), seed_y)
    scale = np.float64(1.0 / 16777216.0)
    return np.stack(
        [
            (vx >> U32(8)).astype(np.float64) * scale,
            (vy >> U32(8)).astype(np.float64) * scale,
        ],
        axis=-1,
    )


def bn_path_seed(values):
    """The path seed the kernel derives from a tile value."""
    values = np.asarray(values, dtype=U32)
    return pt_hash_combine(np.full_like(values, U32(PT_BN_SALT)), values)


def sample_vectors(keys, pairs=PAIRS, samples=SAMPLES):
    """``[len(keys), 2 * len(pairs) * samples]`` sample vectors."""
    seeds = bn_path_seed(keys)
    cols = []
    for pair in pairs:
        for i in range(samples):
            cols.append(pt_sample_2d_seeded(seeds, pair, i))
    return np.concatenate(cols, axis=-1)


# ---------------------------------------------------------------------------
# Annealing
# ---------------------------------------------------------------------------


def _neighbourhood(tile, radius, sigma_i):
    """``(offsets [K], weights [K])`` of a toroidal ``(2r+1)^2`` window."""
    offs = []
    ws = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            offs.append((dy, dx))
            ws.append(np.exp(-(dx * dx + dy * dy) / (sigma_i * sigma_i)))
    return np.array(offs, dtype=np.int64), np.array(ws, dtype=np.float64)


def _neighbour_index(tile, offsets):
    """``[tile*tile, K]`` flat indices of every pixel's neighbourhood."""
    ys, xs = np.divmod(np.arange(tile * tile), tile)
    ny = (ys[:, None] + offsets[None, :, 0]) % tile
    nx = (xs[:, None] + offsets[None, :, 1]) % tile
    return (ny * tile + nx).astype(np.int64)


def anneal(
    tile=TILE,
    pairs=PAIRS,
    samples=SAMPLES,
    sweeps=SWEEPS,
    seed=RNG_SEED,
    radius=RADIUS,
    sigma_i=SIGMA_I,
    sigma_s=SIGMA_S,
    t0=T0,
    t1=T1,
    verbose=True,
):
    n = tile * tile
    rng = np.random.default_rng(seed)
    keys = np.arange(n, dtype=np.uint32)
    vecs = sample_vectors(keys, pairs, samples)
    dim = vecs.shape[1]
    inv = 1.0 / (sigma_s * sigma_s * dim)

    offsets, weights = _neighbourhood(tile, radius, sigma_i)
    nbr = _neighbour_index(tile, offsets)
    assign = rng.permutation(n)  # assign[pixel] = key

    proposals = sweeps * n
    energy = _energy(vecs, assign, nbr, weights, inv)
    accepted = 0
    for step in range(proposals):
        p = int(rng.integers(n))
        q = int(rng.integers(n))
        if p == q:
            continue
        kp, kq = assign[p], assign[q]
        sp, sq = vecs[kp], vecs[kq]
        rp = assign[nbr[p]]
        rq = assign[nbr[q]]
        vp = vecs[rp]
        vq = vecs[rq]
        # exp(-|ds|^2 / (sigma_s^2 D)) against each neighbourhood, before and
        # after. The p-q term (if they are neighbours) is invariant under the
        # swap and cancels, so it is left in both sums.
        d_before = (weights * np.exp(-((vp - sp) ** 2).sum(-1) * inv)).sum() + (
            weights * np.exp(-((vq - sq) ** 2).sum(-1) * inv)
        ).sum()
        d_after = (weights * np.exp(-((vp - sq) ** 2).sum(-1) * inv)).sum() + (
            weights * np.exp(-((vq - sp) ** 2).sum(-1) * inv)
        ).sum()
        delta = d_after - d_before
        temp = t0 * (t1 / t0) ** (step / max(proposals - 1, 1))
        if delta < 0.0 or rng.random() < np.exp(-delta / temp):
            assign[p], assign[q] = kq, kp
            energy += delta
            accepted += 1
        if verbose and (step + 1) % (proposals // 10) == 0:
            print(
                f"  {100 * (step + 1) // proposals:3d}%  energy "
                f"{_energy(vecs, assign, nbr, weights, inv):.2f}  "
                f"accepted {accepted}",
                flush=True,
            )
    return assign.reshape(tile, tile).astype(np.uint16), vecs


def _energy(vecs, assign, nbr, weights, inv):
    v = vecs[assign]
    nv = vecs[assign[nbr]]
    return float(
        (weights[None, :] * np.exp(-((nv - v[:, None, :]) ** 2).sum(-1) * inv)).sum()
    )


# ---------------------------------------------------------------------------
# Held-out evaluation: integrands the energy never saw
# ---------------------------------------------------------------------------


def _test_integrands(keys, pair, n_samples, rng):
    """Estimates and exact values of a few 2D integrands over one pair.

    Deliberately NOT what the energy optimises (that is the raw sample
    vector): a smooth Gaussian bump, a disk indicator (the shadow-edge case)
    and a half-plane at an angle (the geometry-edge case).
    """
    seeds = bn_path_seed(np.asarray(keys, dtype=np.uint32))
    u = np.stack(
        [pt_sample_2d_seeded(seeds, pair, i) for i in range(n_samples)], axis=1
    )  # [K, n, 2]
    x, y = u[..., 0], u[..., 1]
    out = []
    cx, cy, r = 0.37, 0.62, 0.31
    disk = ((x - cx) ** 2 + (y - cy) ** 2 < r * r).astype(np.float64)
    out.append((disk.mean(1), np.pi * r * r))
    ang = 0.7
    half = ((np.cos(ang) * x + np.sin(ang) * y) < 0.6).astype(np.float64)
    ref_half = _half_plane_area(np.cos(ang), np.sin(ang), 0.6)
    out.append((half.mean(1), ref_half))
    bump = np.exp(-8.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
    ref_bump = _mc_reference(
        lambda a, b: np.exp(-8.0 * ((a - 0.5) ** 2 + (b - 0.5) ** 2)), rng
    )
    out.append((bump.mean(1), ref_bump))
    return out


def _half_plane_area(a, b, c, n=4096):
    g = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(g, g)
    return float((a * xx + b * yy < c).mean())


def _mc_reference(f, rng, n=1 << 22):
    a = rng.random(n)
    b = rng.random(n)
    return float(f(a, b).mean())


def _low_frequency_energy(err, box=4):
    """Mean square of the box-filtered error image: what blue noise lowers."""
    k = np.ones((box, box)) / (box * box)
    pad = np.pad(err, ((0, box - 1), (0, box - 1)), mode="wrap")
    acc = np.zeros_like(err)
    for dy in range(box):
        for dx in range(box):
            acc += k[dy, dx] * pad[dy : dy + err.shape[0], dx : dx + err.shape[1]]
    return float((acc**2).mean())


def evaluate(tile_values, rng_seed=7):
    """Held-out spectral report for a tile, against a white-noise assignment."""
    rng = np.random.default_rng(rng_seed)
    tile = tile_values.shape[0]
    flat = tile_values.reshape(-1).astype(np.uint32)
    white = (
        np.random.default_rng(rng_seed + 1).permutation(tile * tile).astype(np.uint32)
    )
    report = []
    for pair in (0, 3, 54):
        for n_samples in (1, 2, 4):
            lo_b = lo_w = 0.0
            for est, ref in _test_integrands(flat, pair, n_samples, rng):
                lo_b += _low_frequency_energy((est - ref).reshape(tile, tile))
            for est, ref in _test_integrands(white, pair, n_samples, rng):
                lo_w += _low_frequency_energy((est - ref).reshape(tile, tile))
            report.append((pair, n_samples, lo_b, lo_w))
    return report


def verify_against_kernel(tile_values):
    """The numpy sampler replica must agree with the Taichi kernel exactly."""
    import torch

    from algan.rendering.raytracing.path_tracer_taichi import pt_sampler_probe
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    n = 64
    out = torch.zeros((n, 2), dtype=torch.float32)
    worst = 0.0
    for pixel in (0, 1, 4095):
        for pair in (0, 3, 54):
            pt_sampler_probe(0, 0, int(pixel), int(pair), out)
            key = pt_hash_combine(np.uint32(0), np.uint32(pixel))
            path_seed = pt_hash_combine(np.uint32(0), key)
            mine = np.stack(
                [
                    pt_sample_2d_seeded(np.full(1, path_seed, dtype=U32), pair, i)[0]
                    for i in range(n)
                ]
            )
            worst = max(worst, float(np.abs(mine - out.numpy()).max()))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verify", action="store_true", help="check the committed tile, write nothing"
    )
    ap.add_argument("--sweeps", type=int, default=SWEEPS)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if args.verify:
        values = np.load(args.out)
        print(f"{args.out.relative_to(REPO)}: {values.dtype} {values.shape}")
        flat = np.sort(values.reshape(-1).astype(np.int64))
        assert np.array_equal(flat, np.arange(values.size)), "not a permutation"
        print("permutation of 0..N-1: ok")
        for pair, n_samples, lo_b, lo_w in evaluate(values):
            print(
                f"  pair {pair:>3}, {n_samples} spp: low-frequency error energy "
                f"{lo_b:.3e} vs white {lo_w:.3e}  ({lo_w / max(lo_b, 1e-30):.2f}x lower)"
            )
        # Last, because it is the one step that needs the compiler: a tile is
        # only optimised for the sampler the kernel actually runs, so the numpy
        # replica up top has to still be that sampler.
        worst = verify_against_kernel(values)
        print(f"numpy sampler replica vs pt_sampler_probe: worst |diff| {worst:g}")
        assert worst == 0.0, "the numpy replica has drifted from the kernel"
        return

    print(f"annealing a {args.tile}x{args.tile} tile, {args.sweeps} sweeps ...")
    values, _vecs = anneal(tile=args.tile, sweeps=args.sweeps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, values)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    for pair, n_samples, lo_b, lo_w in evaluate(values):
        print(
            f"  pair {pair:>3}, {n_samples} spp: low-frequency error energy "
            f"{lo_b:.3e} vs white {lo_w:.3e}  ({lo_w / max(lo_b, 1e-30):.2f}x lower)"
        )


if __name__ == "__main__":
    main()
