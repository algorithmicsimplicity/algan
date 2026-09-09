"""Regenerate the coupled rough-glass loss table (deterministic VNDF quadrature).

Run from the repository root: ``python scripts/generate_glass_energy_table.py``.
Only NumPy is needed; no renderer, GPU, fitted opaque albedo, or random walk is
involved. See DESIGN_path_tracer_roadmap.md, "Coupled rough-glass compensation".
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from _memory_cap import cap_process_memory

# These bounded numerical harnesses do not need BLAS worker stacks.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

cap_process_memory(2)

import numpy as np  # noqa: E402

# Keep in step with glass_energy.py. The last cosine row is the cosine average.
ETA_SIZE, ROUGH_SIZE, MU_SIZE = 65, 33, 65


def smith_lambda(mu: np.ndarray | float, alpha: float) -> np.ndarray:
    c2 = np.clip(np.asarray(mu) ** 2, 1e-8, 1.0)
    return 0.5 * (np.sqrt(1.0 + alpha * alpha * (1.0 - c2) / c2) - 1.0)


def hammersley(count: int) -> np.ndarray:
    """Midpoint Hammersley points; count must be a power of two."""
    if count < 2 or count & (count - 1):
        raise ValueError("samples must be a power of two, at least two")
    indices = np.arange(count, dtype=np.uint32)
    bits = indices.copy()
    reversed_bits = np.zeros(count, dtype=np.uint32)
    for _ in range(count.bit_length() - 1):
        reversed_bits = (reversed_bits << 1) | (bits & 1)
        bits >>= 1
    return np.stack(((indices + 0.5) / count, (reversed_bits + 0.5) / count), axis=-1)


def directional_loss(
    rough: float, mu: float, etas: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Integrate both facet outcomes, counting rejected hemispheres as zero.

    Transmission is in power (radiance divided by eta**2). VNDF importance
    sampling reduces f*cos/pdf to G2/G1, avoiding a narrow BTDF quadrature.
    Returns one loss for every incident/transmitted index ratio in ``etas``.
    """
    if rough == 0.0:
        return np.zeros(len(etas))
    mu = max(mu, 1e-5)
    alpha = max(rough * rough, 1e-4)
    v = np.array([np.sqrt(1.0 - mu * mu), 0.0, mu])
    stretched = v * [alpha, alpha, 1.0]
    stretched /= np.linalg.norm(stretched)
    phi = 2.0 * np.pi * u[:, 0]
    z = (1.0 - u[:, 1]) * (1.0 + stretched[2]) - stretched[2]
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    h = np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=-1) + stretched
    h[:, :2] *= alpha
    h[:, 2] = np.maximum(h[:, 2], 1e-6)
    h /= np.linalg.norm(h, axis=-1)[:, None]
    vh = np.clip(h @ v, 0.0, 1.0)
    reflected_mu = -mu + 2.0 * vh * h[:, 2]
    lv = smith_lambda(mu, alpha)
    reflected_escape = np.where(
        reflected_mu > 1e-6,
        (1.0 + lv) / (1.0 + lv + smith_lambda(reflected_mu, alpha)),
        0.0,
    )
    result = np.empty(len(etas))
    # Bounded temporaries rather than a table-size x sample-size allocation.
    for i, eta in enumerate(etas):
        sin2 = eta * eta * (1.0 - vh * vh)
        ct = np.sqrt(np.maximum(0.0, 1.0 - sin2))
        rs = (eta * vh - ct) / np.maximum(eta * vh + ct, 1e-12)
        rp = (vh - eta * ct) / np.maximum(vh + eta * ct, 1e-12)
        fresnel = np.where(sin2 >= 1.0, 1.0, 0.5 * (rs * rs + rp * rp))
        transmitted_mu = -eta * mu + (eta * vh - ct) * h[:, 2]
        transmitted_escape = np.where(
            (sin2 < 1.0) & (transmitted_mu < -1e-6),
            (1.0 + lv) / (1.0 + lv + smith_lambda(transmitted_mu, alpha)),
            0.0,
        )
        result[i] = np.clip(
            1.0
            - np.mean(
                fresnel * reflected_escape + (1.0 - fresnel) * transmitted_escape
            ),
            0.0,
            1.0,
        )
    return result


def cosine_average(values: np.ndarray) -> np.ndarray:
    """Exact 2*integral(mu*d(mu)) for our piecewise-linear cosine lookup."""
    mu = np.linspace(0.0, 1.0, values.shape[-1])
    a, b = mu[:-1], mu[1:]
    # Integrals of 2*mu times the two linear basis functions of each interval.
    left = (b - a) * (2.0 * a + b) / 3.0
    right = (b - a) * (a + 2.0 * b) / 3.0
    return np.sum(values[..., :-1] * left + values[..., 1:] * right, axis=-1)


def generate(samples: int) -> np.ndarray:
    u = hammersley(samples)
    x = np.linspace(0.0, 1.0, ETA_SIZE)
    # x=1 is the perfectly reflecting limit, computed separately below.
    q = (1.0 + x[:-1]) / (1.0 - x[:-1])
    etas = np.concatenate((1.0 / q, q))
    table = np.zeros((ETA_SIZE, ROUGH_SIZE, 2, MU_SIZE + 1), dtype=np.float32)
    for r, rough in enumerate(np.linspace(0.0, 1.0, ROUGH_SIZE)):
        for m, mu in enumerate(np.linspace(0.0, 1.0, MU_SIZE)):
            losses = directional_loss(rough, mu, etas, u)
            table[:-1, r, 0, m] = losses[: len(q)]
            table[:-1, r, 1, m] = losses[len(q) :]
            # eta -> infinity, all visible facets reflect. eta -> zero has
            # the same Fresnel-one limit, so the reciprocal endpoint is shared.
            limit = directional_loss(rough, mu, np.array([1e12]), u)[0]
            table[-1, r, :, m] = limit
        table[:, r, :, -1] = cosine_average(table[:, r, :, :-1])
        print(f"roughness {rough:.5f}: complete", flush=True)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "algan/rendering/raytracing/data/glass_energy.npy",
    )
    args = parser.parse_args()
    table = generate(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, table, allow_pickle=False)
    print(f"Wrote {args.output}: {table.shape}, {table.nbytes} bytes")


if __name__ == "__main__":
    main()
