"""Print unclamped power-furnace sweeps for the coupled rough-glass closure.

Run from the repository root with the environment's Python. Writes a CSV; use
``--output`` to retain it. Each configuration is integrated twice through the
shipped sampler and evaluator: once with the shipped energy table, and once
with a zeroed table, which disables the compensation lobe and leaves plain
single scatter. Transmission is converted to power by dividing radiance by the
relative eta squared, and is never clipped to display white -- the whole point
is to see energy above and below one.

``compensated`` should sit at 1.0 for ideal white glass. ``single_scatter`` is
how far short the uncompensated GGX dielectric falls, which is what the
compensation exists to recover.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._memory_cap import cap_process_memory

# These bounded numerical harnesses do not need BLAS worker stacks.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

cap_process_memory(8)

import torch  # noqa: E402

from algan.rendering.raytracing.glass_energy import glass_energy_table  # noqa: E402
from tests.unit_tests.test_rough_dielectric import _sample, init_taichi  # noqa: E402

#: Roughness, relative IOR and incidence angle grid. Both sides of each
#: interface appear, so the reciprocal pairs can be read against each other.
ROUGHNESSES = (0.08, 0.35, 0.65, 1.0)
RELATIVE_IORS = (1 / 2.4, 1 / 1.5, 1 / 1.1, 1.1, 1.5 / 1.33, 1.5, 2.4)
ANGLES = (0.0, 35.0, 60.0, 85.0)


def _powers(table, rough, eta, angle, count, seed):
    """Per-sample throughput in POWER units for one configuration."""
    u = torch.rand((count, 3), generator=torch.Generator().manual_seed(seed))
    out = torch.zeros((count, 6))
    _sample(table, rough, eta, math.radians(angle), u, out)
    transmitted = out[:, 3] > 0
    scale = torch.where(transmitted, torch.tensor(eta * eta), torch.tensor(1.0))
    return out[:, 5] / scale, transmitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=9187)
    args = parser.parse_args()
    if not 1024 <= args.samples <= 1048576:
        parser.error("--samples must be between 1024 and 1048576")

    init_taichi()
    shipped = torch.from_numpy(glass_energy_table())
    # A zeroed table costs the lookup nothing and turns the mixture weight to
    # zero, which is exactly the pre-compensation sampler.
    zeroed = torch.zeros_like(shipped)

    stream = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "roughness",
                "relative_ior",
                "angle_degrees",
                "single_scatter",
                "compensated",
                "reflected",
                "transmitted_power",
                "iid_standard_error_proxy",
            )
        )
        # One bounded configuration at a time, even at the maximum sample count.
        for rough, eta, angle in itertools.product(ROUGHNESSES, RELATIVE_IORS, ANGLES):
            power, transmitted = _powers(
                shipped, rough, eta, angle, args.samples, args.seed
            )
            single, _ = _powers(zeroed, rough, eta, angle, args.samples, args.seed)
            writer.writerow(
                (
                    rough,
                    eta,
                    angle,
                    float(single.mean()),
                    float(power.mean()),
                    float(power[~transmitted].sum() / args.samples),
                    float(power[transmitted].sum() / args.samples),
                    float(power.std() / math.sqrt(args.samples)),
                )
            )
            stream.flush()
    finally:
        if stream is not sys.stdout:
            stream.close()


if __name__ == "__main__":
    main()
