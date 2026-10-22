"""Dump the CUDA PTX of the BVH traversal and count its global loads by width.

The question it answers: what does a sibling-block (node) fetch compile to?
``stbvh.kernel_blocks`` passes the block bytes as 64-bit words
(``raytrace_kernels_taichi._node_row``); ``ALGAN_BVH_NODE_WORDS=0`` (read at
import) restores the original ``vector(4, f16)`` lanes and their fetch. On the
RTX 5080 build both came out as 32/64-bit loads only, so the words' win is not
a load-count one -- compare the two arms' dumps instruction by instruction.

Two parts, both with ``print_kernel_asm`` on, the offline cache off and the
source-key shortcut off, so every kernel really goes through the backend:

1. **Probe kernels** (always): one sibling-group test (STBVH and refit) plus a
   runtime-index link read. Under words, compiled once per format the kernels
   accept -- ``i64`` and ``i32`` words and ``scalar`` (one value per element,
   the misaligned-storage fallback); under the lanes arm, once for the lane
   vectors. The kernel loads nothing else from global memory but its
   arguments, so the histograms are the node fetch.
2. **A real render** (``--render``): a small shadowed scene (opaque, half
   transparent and glass casters), and the load histograms of
   ``raster_shadow_trace_arena``, ``shadow_queue_trace_arena``,
   ``wavefront_traverse_events_arena`` and ``wavefront_shade_arena``.

Run on the CUDA machine from the worktree root, once per arm::

    python benchmarks/_bvh_node_load_ptx.py --out ptx_words [--render]
    ALGAN_BVH_NODE_WORDS=0 python benchmarks/_bvh_node_load_ptx.py --out ptx_lanes [--render]

Every PTX file stays in ``--out`` for reading. On another arch it still runs
(the dump is that arch's assembly) and says the PTX checks need CUDA.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Load patterns: ld.global[.nc][.cache hints][.vN].<type><bits>
_LOAD = re.compile(
    r"\bld\.global((?:\.\w+::\w+)|\.nc|\.ca|\.cg|\.cs|\.lu|\.cv)*"
    r"(\.v[248])?\.(?:[bsuf])(8|16|32|64)\b"
)
_ENTRY = re.compile(r"\.entry\s+([\w$]+)")


def _histogram(text):
    """``{entry: Counter}`` of global loads, keyed ``"16"``, ``"64"``, ``"v2.64"``..."""
    out = {}
    current = None
    for line in text.splitlines():
        m = _ENTRY.search(line)
        if m:
            current = m.group(1)
            out.setdefault(current, Counter())
            continue
        if current is None:
            continue
        for lm in _LOAD.finditer(line):
            vec = lm.group(2)
            bits = lm.group(3)
            out[current][(vec[1:] + "." if vec else "") + bits] += 1
    return out


def _fmt(counter):
    keys = sorted(counter, key=lambda k: (("v" in k), int(k.split(".")[-1]), k))
    return (
        ", ".join(
            f"{k}-bit:{counter[k]}" if "v" not in k else f"{k}:{counter[k]}"
            for k in keys
        )
        or "(no global loads)"
    )


def _new_files(directory, before):
    return sorted(p for p in Path(directory).iterdir() if p.name not in before)


def _report(files, wanted=None, title=""):
    rows = []
    for path in files:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if ".entry" not in text:
            continue
        for entry, counts in _histogram(text).items():
            if wanted and not any(w in entry for w in wanted):
                continue
            rows.append((entry, counts, path.name))
    if title:
        print(f"\n== {title}")
    for entry, counts, name in rows:
        print(f"  {entry[:70]:70s} {_fmt(counts)}   [{name}]")
    return rows


def _probe_kernels(ti, torch, device):
    from algan.rendering.raytracing import raytrace_kernels_taichi as rk
    from algan.rendering.raytracing.stbvh import (
        bvh_arity,
        bvh_block_f16,
        bvh_node_words,
    )

    # This module's annotations are strings (``from __future__``), resolved
    # against its globals when the kernel is declared -- so the two modules
    # they name are bound there, after the environment above was set.
    globals().update(rk=rk, ti=ti)

    @ti.kernel
    def node_probe(
        blocks: rk.NODE_ARG, out: ti.types.ndarray(), n: ti.i32, refit: ti.template()
    ):
        for i in range(n):
            ro = ti.math.vec3(0.01 * i, 0.5, -3.0)
            inv_rd = ti.math.vec3(7.0, 3.0, 1.0)
            mask, near = rk._group_test(
                refit, 0, i % blocks.shape[0], 0, ro, inv_rd, -1.0, 1e30, blocks
            )
            link = rk._refit_link(i % blocks.shape[0], i % bvh_arity, blocks)
            out[i] = mask + link + ti.cast(near[0], ti.i32)

    rows = 64
    dtype = torch.float16 if bvh_block_f16 else torch.float32
    lanes = torch.randn((rows, 8, bvh_arity)).to(dtype).to(device)
    flat = lanes.reshape(rows, -1)
    out = torch.zeros(256, dtype=torch.int32, device=device)
    if not bvh_node_words:
        formats = {"lanes": lanes}
    else:
        formats = {
            "i64": flat.view(torch.int64),
            "i32": flat.view(torch.int32),
            "scalar": flat,
        }
    return node_probe, formats, out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="ptx_dump")
    parser.add_argument(
        "--render",
        action="store_true",
        help="also render a small shadowed scene in this process's arm",
    )
    args = parser.parse_args(argv)

    # The dump only covers what the backend compiles: no offline cache, no
    # source-key shortcut (a startup variable, so before algan loads).
    os.environ["ALGAN_TAICHI_SOURCE_KEY"] = "0"
    os.environ.setdefault("ALGAN_USE_DAEMON", "0")
    os.environ.setdefault("ALGAN_PRECOMPILE_JOBS", "0")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    os.chdir(out_dir)  # print_kernel_asm writes into the working directory

    import torch

    from algan.rendering import taichi_runtime
    from algan.settings._startup import render_device
    from algan.taichi_compat import ti

    taichi_runtime._INIT_OVERRIDES["print_kernel_asm"] = True
    taichi_runtime._INIT_OVERRIDES["offline_cache"] = False
    taichi_runtime.start_program()
    arch = ti.lang.impl.current_cfg().arch
    device = render_device()
    is_cuda = arch == ti.cuda
    print(f"arch {arch}, render device {device}; dumping into {out_dir}")

    node_probe, formats, out = _probe_kernels(ti, torch, device)
    probe_rows = {}
    for name, blocks in formats.items():
        for refit in (0, 1):
            before = {p.name for p in out_dir.iterdir()}
            node_probe(blocks, out, out.shape[0], refit)
            ti.sync()
            rows = _report(
                _new_files(out_dir, before),
                ("node_probe",),
                f"probe: {name}, refit={refit}",
            )
            probe_rows[(name, refit)] = rows

    if args.render:
        arm = "lanes" if "lanes" in formats else "words"
        before = {p.name for p in out_dir.iterdir()}
        _render_small_scene(out_dir / f"scene_{arm}.png")
        _report(
            _new_files(out_dir, before),
            (
                "raster_shadow_trace",
                "shadow_queue_trace",
                "wavefront_traverse_events",
                "wavefront_shade",
            ),
            f"render arm: {arm} (traversal kernels)",
        )

    if not is_cuda:
        print(
            "\nNot CUDA: the files above are this arch's assembly; the "
            "ld.global counts (PTX) need the CUDA machine."
        )
        return 0
    if not any(probe_rows.values()):
        print(
            "\nNo PTX .entry sections in the dump. Files written:\n  "
            + "\n  ".join(p.name for p in sorted(out_dir.iterdir()))
            + "\nIf they are LLVM IR instead, look for `load <2 x i64>` / "
            "`load i64` against `load half` / `load i16` in the probe kernels."
        )
        return 2
    verdict = 0
    for refit in (0, 1):
        for name in formats:
            got = sum((c for _, c, _ in probe_rows.get((name, refit), [])), Counter())
            print(f"probe refit={refit}: {name} -> {_fmt(got)}")
            if name == "i64" and got.get("16", 0) != 0:
                print("  FAIL: the i64 node fetch still issues 16-bit loads")
                verdict = 1
    print("\nPASS" if verdict == 0 else "\nFAIL")
    return verdict


def _render_small_scene(path):
    from algan import (
        BLACK,
        BLUE,
        OUT,
        RED,
        RIGHT,
        SETTINGS,
        SMOKE_TEST,
        UP,
        WHITE,
        Cube,
        MeshLambertMaterial,
        MeshPhysicalMaterial,
        Off,
        PointLight,
        Prism,
        Scene,
        Sphere,
    )
    from algan.scene_manager import SceneManager

    SceneManager.reset()
    settings = SMOKE_TEST.set(resolution=(160, 90))
    SETTINGS.raytracing.set(shadows=True, max_bounces=1)
    with Scene(video_settings=settings) as scene:
        scene.set_background(BLACK)
        with Off():
            PointLight(
                location=RIGHT + OUT * 6 + UP * 1.5, color=WHITE, intensity=20.0
            ).spawn(animate=False)
            Prism(width=14.0, height=9.0, depth=0.1).set_material(
                MeshLambertMaterial(color=WHITE)
            ).spawn(animate=False)
            Cube(size=1.6).set_material(MeshLambertMaterial(color=RED)).move(
                RIGHT * -2.5 + OUT * 1.0
            ).spawn(animate=False)
            Cube(size=1.8, opacity=0.5).set_material(
                MeshLambertMaterial(color=BLUE)
            ).move(RIGHT * 0.5 + OUT * 1.4).spawn(animate=False)
            Sphere(radius=1.0).set_material(
                MeshPhysicalMaterial(
                    color=WHITE, transmission=1.0, roughness=0.0, ior=1.5
                )
            ).move(RIGHT * 3.0 + OUT * 1.5 + UP).spawn(animate=False)
        scene.save_frame(str(path), video_settings=settings, overwrite=True)


if __name__ == "__main__":
    sys.exit(main())
