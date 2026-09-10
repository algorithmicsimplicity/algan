"""Generate the reciprocal, coupled GGX dielectric escape-energy lookup.

Run from the repository root with the development interpreter. Only NumPy is
needed; this deliberately does not import Algan or initialise a GPU. No random
seed, training, fitting service, renderer invocation, or external data is used.
The default has 17 x 33 x 65 x 2 directional f32 values plus cosine means,
integrated using 8192
Hammersley visible-normal samples per row. Use --samples for convergence checks.

The generator integrates the single-scatter POWER transport operator, i.e.
transmission does not include radiance's eta-squared factor. Both reflected and
refracted outcomes are integrated analytically over the same sampled normal.
The table stores missing directional power 1-E on BOTH hemispheres and its
cosine averages. The renderer closes this energy budget with a reciprocal
coupled two-port compensation lobe; it never rescales Fresnel independently.
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONSTANTS = runpy.run_path(str(ROOT / "algan/rendering/raytracing/glass_energy.py"))
NR = CONSTANTS["GLASS_ROUGHNESS_SIZE"]
NE = CONSTANTS["GLASS_IOR_SIZE"]
NM = CONSTANTS["GLASS_COSINE_SIZE"]


def hammersley(count):
    """Midpoint-shifted 2D Hammersley quadrature, bounded to [0,1)."""
    bits = np.arange(count, dtype=np.uint32)
    reversed_bits = np.zeros_like(bits)
    for _ in range(32):
        reversed_bits = (reversed_bits << 1) | (bits & 1)
        bits >>= 1
    return np.stack(
        (
            (np.arange(count) + 0.5) / count,
            (reversed_bits.astype(np.float64) + 0.5) / 2**32,
        ),
        axis=-1,
    )


def smith_lambda(mu, alpha):
    c2 = np.clip(mu * mu, 1e-8, 1.0)
    return 0.5 * (np.sqrt(1 + alpha * alpha * (1 - c2) / c2) - 1)


def fresnel(ci, eta):
    st2 = eta * eta * np.maximum(1 - ci * ci, 0)
    ct = np.sqrt(np.maximum(1 - st2, 0))
    rs = (eta * ci - ct) / np.maximum(eta * ci + ct, 1e-12)
    rp = (ci - eta * ct) / np.maximum(ci + eta * ct, 1e-12)
    return np.where(st2 >= 1, 1.0, 0.5 * (rs * rs + rp * rp)), ct


def transport_samples(roughness, eta, mu, u):
    """Outgoing cosines and POWER weights, independently of branch sampling.

    Matches the spherical-cap VNDF and height-correlated Smith G2 used by the
    renderer. Shapes are [incident directions, quadrature samples].
    """
    alpha = max(roughness * roughness, 1e-4)
    mu = np.maximum(np.asarray(mu), 1e-4)[:, None]
    vx = np.sqrt(np.maximum(1 - mu * mu, 0))
    sx, sz = alpha * vx, mu
    norm = np.sqrt(sx * sx + sz * sz)
    sx, sz = sx / norm, sz / norm
    phi = 2 * np.pi * u[:, 0]
    z = (1 - u[:, 1]) * (1 + sz) - sz
    radius = np.sqrt(np.maximum(1 - z * z, 0))
    hx = alpha * (radius * np.cos(phi) + sx)
    hy = alpha * radius * np.sin(phi)
    hz = np.maximum(z + sz, 1e-6)
    norm = np.sqrt(hx * hx + hy * hy + hz * hz)
    hx, hz = hx / norm, hz / norm
    ci = np.clip(vx * hx + mu * hz, 0, 1)
    F, ct = fresnel(ci, eta)
    reflected_mu = -mu + 2 * ci * hz
    transmitted_mu = -eta * mu + (eta * ci - ct) * hz
    lv = smith_lambda(mu, alpha)
    r = F * (1 + lv) / (1 + lv + smith_lambda(reflected_mu, alpha))
    t = (1 - F) * (1 + lv) / (1 + lv + smith_lambda(transmitted_mu, alpha))
    r = np.where(reflected_mu > 1e-6, r, 0)
    t = np.where(transmitted_mu < -1e-6, t, 0)
    return reflected_mu, transmitted_mu, r, t


def cosine_average(values):
    """Exact cosine average of a function piecewise linear in sqrt(mu)."""
    u = np.linspace(0, 1, NM)
    a, b = u[:-1], u[1:]
    fourth = b**4 - a**4
    fifth = 0.8 * (b**5 - a**5)
    left = (b * fourth - fifth) / (b - a)
    right = (fifth - a * fourth) / (b - a)
    return np.sum(values[:-1] * left + values[1:] * right)


def generate(samples):
    u = hammersley(samples)
    mu = np.linspace(0, 1, NM) ** 2
    table = np.zeros((NR, NE, NM + 1, 2), dtype=np.float32)
    for r in range(1, NR):
        roughness = r / (NR - 1)
        for e in range(NE):
            x = (e / (NE - 1)) ** 2
            ratio = (1 + x) / max(1 - x, 1e-6)
            for side, eta in enumerate((1 / ratio, ratio)):
                _mr, _mt, wr, wt = transport_samples(roughness, eta, mu, u)
                lost = np.clip(1 - np.mean(wr + wt, axis=1), 0, 1)
                table[r, e, :NM, side] = lost
                table[r, e, NM, side] = cosine_average(lost)
        print(f"roughness {r}/{NR - 1}", flush=True)
    return table.reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "algan/rendering/raytracing/_glass_energy_lut.npz",
    )
    args = parser.parse_args()
    if not 1024 <= args.samples <= 65536:
        parser.error("--samples must lie between 1024 and 65536 (bounded memory)")
    table = generate(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directional = table.reshape(NR, NE, NM + 1, 2)[:, :, :NM, :]
    packed = np.rint(directional * 255).astype(np.uint8)
    # Modulo-256 differencing is reversible and compresses the smooth grid.
    packed = np.diff(packed, axis=2, prepend=np.zeros_like(packed[:, :, :1, :]))
    packed = np.diff(packed, axis=1, prepend=np.zeros_like(packed[:, :1, :, :]))
    np.savez_compressed(args.output, loss_delta=packed)
    print(f"Wrote {table.shape}, {table.nbytes} uncompressed bytes to {args.output}")


if __name__ == "__main__":
    main()
