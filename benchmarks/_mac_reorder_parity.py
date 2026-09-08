"""Compare repaired sorted MPS rays with unsorted MPS and CPU rendering."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "algan_outputs" / "reorder_parity"


def child(arm):
    sys.path.insert(0, str(ROOT))
    import random
    import runpy

    import numpy as np
    import torch

    sys.argv = [str(ROOT / "benchmarks/_mps_warm_regression.py"), "1", "PREVIEW"]
    ns = runpy.run_path(sys.argv[0], run_name="parity_scene")
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if arm == "mps-sorted":
        from _mps_reorder_probe_taichi import install

        install()
    if arm == "mps-unsorted":
        ns["SETTINGS"].raytracing.experimental.wf_ray_sort = False
    ns["SceneManager"].reset()
    ns["Scene"].set_video_settings(ns["PRESET"])
    ns["scene"]()
    start = time.perf_counter()
    result = ns["Scene"].save_frame(
        str(OUT / (arm + ".png")), video_settings=ns["PRESET"], at=0.15, overwrite=True
    )
    print(
        "PARITY_FRAME",
        arm,
        str(result.output_path),
        time.perf_counter() - start,
        flush=True,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ("mps-sorted", "mps-unsorted", "cpu"):
        env = dict(
            os.environ,
            ALGAN_RENDER_DEVICE="cpu" if arm == "cpu" else "mps",
            ALGAN_TORCH_COMPILE="0",
            ALGAN_MPS_HOST_SHARE="0",
            ALGAN_USE_DAEMON="0",
            PYTHONUNBUFFERED="1",
        )
        with (OUT / (arm + ".txt")).open("w") as log:
            result = subprocess.run(
                [sys.executable, __file__, "--child", arm],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=450,
            )
        print((OUT / (arm + ".txt")).read_text(), flush=True)
        if result.returncode:
            return result.returncode
    import numpy as np
    from PIL import Image

    images = {
        arm: np.asarray(Image.open(OUT / (arm + ".png"))).astype(np.int16)
        for arm in ("mps-sorted", "mps-unsorted", "cpu")
    }
    comparisons = {}
    for arm in ("mps-unsorted", "cpu"):
        d = np.abs(images["mps-sorted"] - images[arm])
        comparisons[arm] = {
            "max": int(d.max()),
            "mean": float(d.mean()),
            "p999": float(np.quantile(d, 0.999)),
            "fraction_over_2": float((d > 2).mean()),
            "shape": list(d.shape),
        }
    print("PARITY_RESULT", json.dumps(comparisons), flush=True)
    (OUT / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    assert comparisons["mps-unsorted"]["max"] <= 2, comparisons
    return 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        child(sys.argv[-1])
    else:
        raise SystemExit(main())
