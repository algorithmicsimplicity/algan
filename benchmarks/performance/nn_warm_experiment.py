"""Run the exact UHD benchmark with repeated warm measurements.

Examples (use the venv interpreter):
    nn_warm_experiment.py --tag control --runs 4
    nn_warm_experiment.py --tag reg64 --max-reg 64 --runs 4

Compiler variants require separate processes. Compare warm samples only and
bracket experiments with another control to expose thermal drift. The scene,
duration, quality and encoder arguments come from nn_scene_UHD.py itself.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path

os.environ["ALGAN_USE_DAEMON"] = "0"

import algan.utils.profiling_utils as profiling
from algan.rendering import taichi_runtime
from algan.taichi_compat import program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--max-reg", type=int, default=0)
    parser.add_argument("--readonly", action="store_true")
    parser.add_argument("--kernel-profiler", action="store_true")
    parser.add_argument("--legacy-refit-links", action="store_true")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("at least two runs are required to exclude the cold run")

    original_kwargs = taichi_runtime.taichi_init_kwargs

    def experiment_kwargs():
        kwargs = original_kwargs()
        if args.max_reg:
            kwargs["gpu_max_reg"] = args.max_reg
        if args.readonly:
            kwargs["readonly_ndarray_ldg"] = True
        return kwargs

    taichi_runtime.taichi_init_kwargs = experiment_kwargs
    if args.legacy_refit_links:
        from benchmarks.performance.refit_link_control_taichi import legacy_refit_link
        from algan.rendering.raytracing import raytrace_kernels_taichi

        raytrace_kernels_taichi._refit_link = legacy_refit_link
    original_profile = profiling.profile_scene

    def experiment_profile(scene, quality, tag, **kwargs):
        kwargs.update(runs=args.runs, kernel_profiler=args.kernel_profiler)
        result = original_profile(scene, quality, tag + "_" + args.tag, **kwargs)
        cfg = program().config()
        summary = {
            "tag": args.tag,
            "seconds": [r["total"] for r in result],
            "warm_seconds": [r["total"] for r in result[1:]],
            "legacy_refit_links": args.legacy_refit_links,
        }
        summary_path = Path(__file__).with_name(args.tag + "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("MEASUREMENTS " + json.dumps(summary), flush=True)
        print("LIVE_CONFIG " + json.dumps({
            "gpu_max_reg": cfg.gpu_max_reg,
            "readonly_ndarray_ldg": getattr(cfg, "readonly_ndarray_ldg", None),
            "invariant_arg_loads": getattr(cfg, "invariant_arg_loads", None),
        }), flush=True)
        return result

    profiling.profile_scene = experiment_profile
    runpy.run_path(str(Path(__file__).with_name("nn_scene_UHD.py")), run_name="__main__")


if __name__ == "__main__":
    main()
