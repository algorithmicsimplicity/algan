"""T4 acceptance driver for the preflight overlap (REPORT_preflight_overlap_impl.md).

Runs the lossless A/B the report specifies, one process per arm with
``ALGAN_PREFETCH_GPU_PREP`` flipped in the child env, at two memory pins:
a roomy one for the headline byte-identity check and a tight one that
forces the OOM window-shrink retry to fire under the overlap. Exits 0 only
if every pair diffs to zero and every render completes.

Meant to run as a notebook arm (any CWD): paths derive from this file.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[3]
AB = HERE.parent / "ab_preflight_overlap.py"
OUT = HERE.parent / "out"
DIFF = REPO / "benchmarks" / "_video_diff.py"

CONFIGS = [
    ("main", 4096),
    ("oomforce", 1024),
]

failures = []
for label, mb in CONFIGS:
    videos = []
    for arm, flag in (("off", "0"), ("on", "1")):
        tag = f"{label}_{arm}"
        env = dict(os.environ)
        env["ALGAN_PREFETCH_GPU_PREP"] = flag
        env["ALGAN_USE_DAEMON"] = "0"
        env["MPLBACKEND"] = "Agg"
        cmd = [sys.executable, str(AB), "PREVIEW", tag, str(mb)]
        print(f"=== {label}/{arm}: override {mb} MB", flush=True)
        r = subprocess.run(cmd, cwd=str(REPO), env=env)
        if r.returncode:
            failures.append(f"{label}/{arm} render exit {r.returncode}")
            continue
        videos.append(OUT / f"{tag}_PREVIEW.mp4")
    if len(videos) == 2:
        d = subprocess.run(
            [sys.executable, str(DIFF), str(videos[0]), str(videos[1])],
            cwd=str(REPO), capture_output=True, text=True,
        )
        print(d.stdout, d.stderr, flush=True)
        if "worst channel diff: 0" not in d.stdout:
            failures.append(f"{label}: pixel diff nonzero (see above)")
    else:
        failures.append(f"{label}: missing an arm, no diff run")

for js in sorted(OUT.glob("*_PREVIEW_summary.json")):
    print(f"--- {js.name}: {js.read_text()}", flush=True)

if failures:
    print("FAILURES: " + "; ".join(failures), flush=True)
    sys.exit(1)
print("PFO T4 acceptance: all pairs byte-identical, all renders completed", flush=True)
