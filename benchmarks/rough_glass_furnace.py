"""Print unclamped power-furnace sweeps for the coupled glass closure.

Run from the repository root with the environment's Python. Outputs a CSV;
use --output to retain it. Both old single scatter and the complete new
sampler/evaluator are integrated using scrambled Sobol points. Transmission
radiance is divided by relative eta squared, never clipped to display white.
"""

from __future__ import annotations

import argparse
import os
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._memory_cap import cap_process_memory

# These bounded numerical harnesses do not need BLAS worker stacks.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

cap_process_memory(8)

from tests.unit_tests.test_glass_energy_compensation import _run_furnaces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=131072)
    args = parser.parse_args()
    if not 1024 <= args.samples <= 1048576:
        parser.error("--samples must be between 1024 and 1048576")
    rows = list(itertools.product(
        (0.08, 0.35, 0.65, 1.0),
        (1 / 2.4, 1 / 1.5, 1 / 1.1, 1.1, 1.5 / 1.33, 1.5, 2.4),
        (0.0, 35.0, 60.0, 85.0),
    ))
    # One bounded case at a time, even at the maximum allowed sample count.
    stream = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.writer(stream)
        writer.writerow(("roughness", "relative_ior", "angle_degrees", "single_scatter",
                         "compensated", "reflected", "transmitted_power", "iid_standard_error_proxy"))
        for row in rows:
            samples = _run_furnaces([row], count=args.samples)[0]
            mean = samples.mean(dim=0)
            stderr = float(samples[:, 0].std() / args.samples**0.5)
            writer.writerow((*row, float(mean[1]), float(mean[0]), float(mean[2]), float(mean[3]), stderr))
            stream.flush()
    finally:
        if stream is not sys.stdout:
            stream.close()


if __name__ == "__main__":
    main()
