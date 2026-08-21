"""Attribute the CUDA allocation PEAK of a render by replaying the allocator trace.

Unlike a snapshot at an arbitrary moment, this walks torch's own
alloc/free trace, finds the instant of maximum live bytes, and reports what
was alive then and which call site owned it.

    <venv-python> benchmarks/_uhd_mem_peak.py [width] [height]
"""

import collections
import os
import sys

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import torch  # noqa: E402

from algan import *  # noqa: E402,F403

GB = 1 << 30
MB = 1 << 20
W = int(sys.argv[1]) if len(sys.argv) > 2 else 3840
H = int(sys.argv[2]) if len(sys.argv) > 2 else 2160


def site(frames):
    for f in frames or ():
        fn = os.path.basename(f.get("filename", "?"))
        if fn == "memory_utils.py":
            return "ARENA (ManualMemory buffer)"
        if fn in ("functional.py", "_tensor.py", "_jit_internal.py", "overrides.py"):
            continue
        return f"{fn}:{f.get('line')} {f.get('name')}"
    return "<unknown>"


torch.cuda.memory._record_memory_history(max_entries=400000)
Sphere().scale(3).spawn()
Scene.save_frame("_uhd_mem_peak.png", UHD.set(resolution=(W, H)))

snap = torch.cuda.memory._snapshot()
live, cur, peak, peak_live = {}, 0, 0, None
for trace in snap.get("device_traces", []):
    for ev in trace:
        act = ev.get("action")
        if act == "alloc":
            live[ev["addr"]] = ev
            cur += ev["size"]
            if cur > peak:
                peak, peak_live = cur, dict(live)
        elif act in ("free_completed", "free_requested"):
            ev0 = live.pop(ev["addr"], None)
            if ev0 is not None:
                cur -= ev0["size"]

print(f"\n=== {W}x{H} allocation peak: {peak / GB:.3f} GB ===")
by = collections.Counter()
cnt = collections.Counter()
for ev in (peak_live or {}).values():
    key = site(ev.get("frames"))
    by[key] += ev["size"]
    cnt[key] += 1
mod = collections.Counter()
for key, b in by.items():
    mod[key.split(":")[0] if ":" in key else key] += b
print("-- by module --")
for m, b in mod.most_common():
    print(f"  {b / MB:9.1f} MB  {b / peak * 100:5.1f}%  {m}")
print("-- top sites --")
for key, b in by.most_common(20):
    print(f"  {b / MB:9.1f} MB  x{cnt[key]:3d}  {key}")
