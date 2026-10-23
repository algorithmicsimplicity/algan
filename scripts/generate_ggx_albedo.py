"""Generate the path tracer's opaque GGX directional-albedo table.

Run from the repository root with the development interpreter. Only NumPy is
needed. For every (roughness, sqrt(n.v)) grid point it integrates the lobe
``D * G2 / (4 n.v)`` (height-correlated Smith, ``alpha = max(r^2, 1e-4)``) with
Schlick's ``Fc = (1 - v.h)^5`` split out, using the same visible-normal
quadrature as ``generate_glass_energy.py``:

    A = integral (1 - Fc) f cos dw,    B = integral Fc f cos dw

so a lobe with Schlick Fresnel ``f0 + (1 - f0) Fc`` reflects ``f0 A + B``.
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GLASS = runpy.run_path(str(ROOT / "scripts/generate_glass_energy.py"))
hammersley = GLASS["hammersley"]
smith_lambda = GLASS["smith_lambda"]
CONSTANTS = runpy.run_path(str(ROOT / "algan/rendering/raytracing/ggx_albedo.py"))
NR = CONSTANTS["GGX_ROUGHNESS_SIZE"]
NM = CONSTANTS["GGX_COSINE_SIZE"]


def scale_bias(roughness, mu, u):
    """``(A, B)`` for each cosine in ``mu`` at one roughness.

    With visible normals sampled, the estimator of ``integral f cos dw`` is
    ``G2 / G1(v)`` for a reflected direction above the horizon, else 0.
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
    v_dot_h = np.clip(vx * hx + mu * hz, 0, 1)
    reflected_mu = -mu + 2 * v_dot_h * hz
    lv = smith_lambda(mu, alpha)
    w = (1 + lv) / (1 + lv + smith_lambda(reflected_mu, alpha))
    w = np.where(reflected_mu > 1e-6, w, 0)
    fc = (1 - v_dot_h) ** 5
    return np.mean((1 - fc) * w, axis=1), np.mean(fc * w, axis=1)


def generate(samples):
    u = hammersley(samples)
    mu = np.linspace(0, 1, NM) ** 2
    table = np.zeros((NR, NM, 2), dtype=np.float32)
    for r in range(NR):
        a, b = scale_bias(r / (NR - 1), mu, u)
        table[r, :, 0] = a
        table[r, :, 1] = b
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "algan/rendering/raytracing/_ggx_albedo_lut.npz",
    )
    args = parser.parse_args()
    if not 1024 <= args.samples <= 262144:
        parser.error("--samples must lie between 1024 and 262144 (bounded memory)")
    table = generate(args.samples)
    np.savez_compressed(args.output, scale_bias=table)
    print(f"Wrote {table.shape} to {args.output}")


if __name__ == "__main__":
    main()
