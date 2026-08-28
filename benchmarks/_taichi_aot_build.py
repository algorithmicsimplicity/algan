"""Build an AOT module for Algan's eligible prep kernels, in a subprocess.

``DESIGN_taichi_arch_coexistence.md`` §3.3: ``ti.aot.Module(arch)`` silently
switches to the live arch when asked for a different one, so the x64 module for
a CUDA process cannot be built inside it. This is §5.3's build step -- a
separate process with ``ALGAN_RENDER_DEVICE=cpu``, so ``init_taichi()`` selects
x64 and the AOT module is genuinely x64.

    uv run python benchmarks/_taichi_aot_build.py --out /tmp/aot

§5.2's marker: a kernel module opts in by declaring a module-level

    AOT_KERNELS = ("grid_normals_sides_crosses",)

Modules to scan are listed in :data:`CANDIDATE_MODULES`; a module without the
marker contributes nothing. ``--kernels module:name`` overrides the scan, which
is what lets a Phase 0 probe build a kernel before any marker is committed.

Prints one JSON line of timings on stdout (the caller reads it; every other
message goes to stderr) so ``_taichi_aot_build_cost.py`` can time cold against
warm without parsing prose.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

#: Modules the scan looks in for an ``AOT_KERNELS`` marker. Listed rather than
#: globbed so the build never imports a ``*_taichi.py`` that only the renderer
#: uses -- importing a render kernel module here would compile it for x64 and
#: charge this build minutes it should not pay.
CANDIDATE_MODULES = ("algan.mobs.surfaces.surface_kernels_taichi",)

#: What to build when no marker is committed yet and ``--kernels`` is not given.
#: The one kernel §2 measures as paying its way (``cpunormals``).
FALLBACK_KERNELS = (
    ("algan.mobs.surfaces.surface_kernels_taichi", "grid_normals_sides_crosses"),
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def discover_kernels(explicit=None):
    """Resolve ``[(module, kernel_name)]`` from ``--kernels`` or the markers."""
    if explicit:
        pairs = []
        for item in explicit:
            module_name, _, kernel_name = item.partition(":")
            if not kernel_name:
                raise SystemExit(f"--kernels wants module:name, got {item!r}")
            pairs.append((module_name, kernel_name))
        return pairs

    pairs = []
    for module_name in CANDIDATE_MODULES:
        module = importlib.import_module(module_name)
        for kernel_name in getattr(module, "AOT_KERNELS", ()):
            pairs.append((module_name, kernel_name))
    return pairs or list(FALLBACK_KERNELS)


def build(out_dir: Path, explicit=None) -> dict:
    """Build one AOT module holding every requested kernel, and save it."""
    import taichi as ti

    from algan.rendering.taichi_runtime import init_taichi
    from algan.settings._startup import render_device

    if render_device().type != "cpu":
        raise SystemExit(
            f"render device is {render_device()}; this build must run with "
            "ALGAN_RENDER_DEVICE=cpu so the arch is x64 (§3.3)"
        )

    started = time.perf_counter()
    init_taichi()
    init_seconds = time.perf_counter() - started

    arch = ti.lang.impl.get_runtime().prog.config().arch
    if arch != ti.x64:
        raise SystemExit(f"expected an x64 arch, got {arch}")

    pairs = discover_kernels(explicit)
    _log(f"building {len(pairs)} kernel(s) for x64 -> {out_dir}")

    compile_started = time.perf_counter()
    module = ti.aot.Module(ti.x64)
    names = []
    for module_name, kernel_name in pairs:
        source = importlib.import_module(module_name)
        try:
            kernel = getattr(source, kernel_name)
        except AttributeError:
            raise SystemExit(f"{module_name} has no kernel {kernel_name!r}") from None
        if getattr(kernel, "_primal", None) is None:
            raise SystemExit(f"{module_name}.{kernel_name} is not a @ti.kernel")
        # A ti.template() argument is baked per value, so an AOT entry would
        # have to exist per instantiation. §5.1 rules those out of scope; refuse
        # rather than emit one arbitrary specialization under the plain name.
        for argument in kernel._primal.arguments:
            if type(argument.annotation).__name__ == "template":
                raise SystemExit(
                    f"{module_name}.{kernel_name} takes a ti.template() argument "
                    f"({argument.name}); template kernels are out of scope (§5.1)"
                )
        module.add_kernel(kernel, name=kernel_name)
        names.append(kernel_name)
        _log(f"  added {module_name}.{kernel_name}")
    compile_seconds = time.perf_counter() - compile_started

    out_dir.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    module.save(str(out_dir))
    save_seconds = time.perf_counter() - save_started

    bytes_written = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    return {
        "kernels": names,
        "out": str(out_dir),
        "init_seconds": round(init_seconds, 4),
        "compile_seconds": round(compile_seconds, 4),
        "save_seconds": round(save_seconds, 4),
        "total_seconds": round(time.perf_counter() - started, 4),
        "bytes": bytes_written,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="directory to save into")
    parser.add_argument(
        "--arch",
        default="x64",
        choices=("x64",),
        help="AOT target; only x64 is in scope (§5.1)",
    )
    parser.add_argument(
        "--kernels",
        nargs="*",
        default=None,
        metavar="MODULE:NAME",
        help="explicit kernels, bypassing the AOT_KERNELS scan",
    )
    args = parser.parse_args(argv)

    # Belt and braces: the caller is expected to set this, but a build that
    # silently produced a CUDA module would be §3.3's trap all over again.
    os.environ.setdefault("ALGAN_RENDER_DEVICE", "cpu")

    result = build(Path(args.out).resolve(), args.kernels)
    print(json.dumps(result), flush=True)
    _log(
        f"built {len(result['kernels'])} kernel(s) in "
        f"{result['total_seconds']:.2f}s ({result['bytes'] / 1024:.1f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
