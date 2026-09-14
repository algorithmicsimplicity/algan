"""Where the sheet frame-table stages spend their time, per tree.

``sheets._shade_class`` and ``sheets._prim_split_after`` build the two
per-``(frame, triangle)`` tables the ``prim`` band rule and ``shade_split``
need, and they are the two stages the arena-ownership refactor changed most.
This runs each of them on realistic shapes through whichever production path
the checked-out tree has -- master takes no destination, the ownership branch
takes ``out=``/``workspace=`` -- and reports, per call:

* the number of torch operations dispatched (Python-level, so tensor metadata
  reads such as ``data_ptr`` and ``stride`` count alongside kernel launches),
  and
* warm wall time.

**Run it at two sizes.** At a tiny size the tensors are irrelevant and the
number is host time: Python, validation, arena bookkeeping, launch overhead.
At the real size it is host time plus memory traffic. The difference between
the two is what says which of those a regression lives in -- and that question
has a different answer on each backend, because a stage that is
bandwidth-bound on a CPU can be launch-bound on a GPU.

The two arms must run in ONE session on the same host (see
``agent_guidance/gpu_harnesses.md``): the stage costs single-digit
milliseconds, which is inside the session-to-session drift.

Tensor sizes come from parameters rather than from a scene, so the footprint
is checked before anything is allocated (``benchmarks/_memory_cap.py`` says
why). The host-RAM ceiling is installed only for a CPU run: ``RLIMIT_AS``
counts the virtual address space CUDA reserves, so capping a GPU run fails
inside the driver rather than raising here.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

#: Refuse a parameter set whose arrays would exceed this, before allocating.
DEFAULT_MAX_GB = 8.0


def estimate_bytes(frames, tris, frags):
    """Roughly what the inputs plus one block of scratch will occupy."""
    # tri_norm + tri_pos (9 floats each) + tri_screen (10) + cam/scale.
    tables = frames * tris * 28 * 4
    # The block stages hold about a dozen [frames, tris, 3, 3] float arrays.
    scratch = 12 * frames * tris * 9 * 4
    # frame_rel/safe_ref/order int64, t/t_o float32, is_tri bool, out int64.
    per_fragment = frags * 38
    return tables + scratch + per_fragment


def _supports_workspace(fn):
    return "workspace" in inspect.signature(fn).parameters


def build_inputs(frames, tris, frags, device, seed=0):
    """Inputs with both classification branches and both split arms live."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=generator).to(device)

    # Half the triangles carry three equal vertex normals (declared flat), half
    # do not, so neither the flat nor the smooth path is measured alone.
    normal = torch.nn.functional.normalize(rand(frames, tris, 3), dim=-1)
    tri_norm = normal.repeat_interleave(3, dim=-1).contiguous()
    smooth = torch.arange(tris, device=device) % 2 == 0
    tri_norm[:, smooth, 3:6] += 0.25
    tri_norm[:, smooth, 6:9] -= 0.25
    tri_pos = rand(frames, tris, 9) * 4.0 - 2.0
    tri_screen = rand(frames, tris, 10) * 100.0
    # Column 9 is the projection-valid flag; keep a tenth of it invalid so the
    # conservative raw-extent arm is exercised too.
    tri_screen[..., 9] = (rand(frames, tris) > 0.1).float()

    return {
        "merged": {"tri_norm": tri_norm, "tri_pos": tri_pos},
        "cam_origin": rand(frames, 3) * 8.0,
        "pixel_world_scale": rand(frames) * 1e-3 + 1e-4,
        "tri_screen": tri_screen,
        "frame_rel": torch.randint(0, frames, (frags,), device=device),
        "safe_ref": torch.randint(0, tris, (frags,), device=device),
        "is_tri": torch.rand(frags, generator=generator).to(device) > 0.05,
        "t": rand(frags) * 10.0,
        "t_o": torch.sort(rand(frags) * 10.0).values,
        "order": torch.randperm(frags, generator=generator).to(device),
        "frames": frames,
        "frags": frags,
    }


def make_calls(data, memory):
    """``{name: callable}``, each taking the tree's own production path."""
    import torch

    from algan.rendering.raytracing import sheets

    n = data["frags"]
    owned = _supports_workspace(sheets._shade_class)
    if owned:
        from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace

    def shade_class():
        args = (
            data["merged"],
            data["frame_rel"],
            0,
            data["safe_ref"],
            data["is_tri"],
            True,
            data["frames"],
        )
        if not owned:
            return sheets._shade_class(*args)
        workspace = CompactionWorkspace(memory)
        # ``temp`` returns the caller-owned destination's bytes too, so repeated
        # calls measure one allocation pattern rather than a growing arena.
        with memory.temp():
            out = memory.get_tensor((n,), torch.int64)
            with workspace.stage():
                return sheets._shade_class(*args, out=out, workspace=workspace).clone()

    def prim_split():
        args = (
            data["merged"],
            data["cam_origin"],
            data["pixel_world_scale"],
            data["tri_screen"],
            data["frame_rel"],
            0,
            data["safe_ref"],
            data["is_tri"],
            data["t"],
            data["t_o"],
            data["order"],
            0.5,
            data["frames"],
        )
        if not _supports_workspace(sheets._prim_split_after):
            return sheets._prim_split_after(*args)
        workspace = CompactionWorkspace(memory)
        with memory.temp():
            out = memory.get_tensor((max(0, n - 1),), torch.bool)
            with workspace.stage():
                return sheets._prim_split_after(
                    *args, out=out, workspace=workspace
                ).clone()

    return {"shade_class": shade_class, "prim_split": prim_split}


def measure(fn, reps, device):
    """Warm, then count one call's operations and time ``reps`` of them."""
    import torch
    from torch.overrides import TorchFunctionMode

    class CountOps(TorchFunctionMode):
        def __init__(self):
            self.count = 0
            self.names = {}

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.count += 1
            name = getattr(func, "__name__", str(func))
            self.names[name] = self.names.get(name, 0) + 1
            return func(*args, **(kwargs or {}))

    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    counter = CountOps()
    with counter:
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps, counter.count, counter.names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=23)
    parser.add_argument("--tris", type=int, default=44444)
    parser.add_argument("--frags", type=int, default=2_500_000)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--arena-mb", type=int, default=3072)
    parser.add_argument(
        "--max-gb",
        type=float,
        default=DEFAULT_MAX_GB,
        help="refuse a parameter set whose arrays would exceed this",
    )
    parser.add_argument("--top", type=int, default=8, help="operations to name")
    parser.add_argument("--label", default="", help="names the arm in the output")
    args = parser.parse_args(argv)

    if min(args.frames, args.tris, args.frags, args.reps) < 1:
        parser.error("frames, tris, frags and reps must all be positive")
    wanted = estimate_bytes(args.frames, args.tris, args.frags)
    if wanted > args.max_gb * (1 << 30):
        parser.error(
            f"these sizes would allocate about {wanted / (1 << 30):.1f} GB, past "
            f"the {args.max_gb} GB ceiling; raise --max-gb deliberately"
        )

    from algan import SETTINGS

    device = SETTINGS.computing.render_device
    if device.type == "cpu":
        # Only on the CPU: RLIMIT_AS counts the address space CUDA reserves.
        from _memory_cap import cap_process_memory

        cap_process_memory(max(2.0, args.max_gb + (args.arena_mb / 1024) + 2.0))

    import torch

    from algan.utils.memory_utils import ManualMemory

    memory = ManualMemory(0, device=device, num_bytes=args.arena_mb << 20)
    data = build_inputs(args.frames, args.tris, args.frags, device)
    calls = make_calls(data, memory)

    from algan.rendering.raytracing import sheets

    path = (
        "destination/workspace"
        if _supports_workspace(sheets._shade_class)
        else "allocator"
    )
    print(
        f"# {args.label or '(unlabelled)'}: device={device} path={path} "
        f"frames={args.frames} tris={args.tris} frags={args.frags} "
        f"reps={args.reps} torch={torch.__version__}",
        flush=True,
    )
    payload = {
        "benchmark": "frame_table_ab",
        "label": args.label,
        "device": str(device),
        "path": path,
        "frames": args.frames,
        "tris": args.tris,
        "frags": args.frags,
        "stages": {},
    }
    for name, call in calls.items():
        seconds, count, names = measure(call, args.reps, device)
        top = sorted(names.items(), key=lambda kv: -kv[1])[: args.top]
        print(
            f"{name:12} {seconds * 1e3:9.3f} ms/call {count:5d} torch ops  "
            + " ".join(f"{key}={value}" for key, value in top),
            flush=True,
        )
        payload["stages"][name] = {
            "ms": round(seconds * 1e3, 4),
            "ops": count,
            "by_op": names,
        }
    print("")
    print("RESULTS " + json.dumps(payload, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
