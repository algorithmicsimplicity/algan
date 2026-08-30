"""Capture ``sheet_resolve_shade``'s real arguments from a real render.

The kernel is the renderer's largest (49 ndarray parameters, 47 declared);
replaying it needs genuine inputs, so this hooks the call site in
``raster_pipeline``, snapshots one launch's arguments to disk, and aborts the
render before anything else runs.

Consumed by ``_arena_view_real_kernel_ab.py``, which replays the captured
launch through the shipped kernel and through an arena/``View`` variant
generated from the same source.

Usage (one render process at a time on Windows):
  .venv/Scripts/python.exe benchmarks/_arena_view_real_capture.py <out.pt> [W H SSAA]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Serial prep, so the hook fires on the main thread in a deterministic order.
os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")
os.environ.setdefault("ALGAN_AUTO_DAEMON", "0")

import contextlib  # noqa: E402

import torch  # noqa: E402


class _CaptureDone(Exception):
    pass


CAPTURE = {}


def _snap(x):
    if isinstance(x, torch.Tensor):
        return ("tensor", x.detach().clone())
    return ("value", x)


def install_hook(which_call=0):
    # raster_pipeline imports the kernel INSIDE the function that calls it, so
    # the patch has to land on the defining module, not on the call site's.
    import algan.rendering.raytracing.sheet_resolve_taichi as srt

    real_kernel = srt.sheet_resolve_shade

    seen = {"n": 0}
    # Parameter names, in declaration order, straight off the shipped kernel --
    # so the replay can rebuild the call positionally without a hand-written
    # list that could drift from the signature.
    import inspect

    _fn = getattr(real_kernel, "__wrapped__", real_kernel)
    names = list(inspect.signature(_fn).parameters)

    def hook(*args, **kwargs):
        assert not kwargs, "call site is positional"
        if seen["n"] != which_call:
            seen["n"] += 1
            return real_kernel(*args)
        CAPTURE["names"] = names[: len(args)]
        CAPTURE["args"] = [_snap(a) for a in args]
        raise _CaptureDone

    srt.sheet_resolve_shade = hook


def build_scene(width, height, ssaa):
    """A mixed scene: PN spheres, flat 2-D shapes, text, a transparent pane, a
    reflective floor and motion -- so the captured launch exercises the
    kernel's four-way material split rather than one easy branch.
    """
    from algan import (
        BLUE,
        DOWN,
        GREEN,
        LEFT,
        OUT,
        RED,
        RIGHT,
        SETTINGS,
        UP,
        Circle,
        MeshStandardMaterial,
        Off,
        PointLight,
        Scene,
        Sphere,
        Square,
        Sync,
        Text,
    )

    SETTINGS.video.set(
        resolution=(width, height),
        frames_per_second=10,
        super_sampling_anti_aliasing=ssaa,
    )
    with Off():
        floor = Square().scale(8).rotate(90, RIGHT).move(DOWN * 2.2)
        floor.set_material(MeshStandardMaterial(roughness=0.15, metalness=0.6))
        floor.spawn()
        s1 = Sphere().scale(0.8).move(LEFT * 2.5).spawn()
        s2 = Sphere().scale(0.5).move(RIGHT * 2.5 + UP).spawn()
        sq = Square().set_color(RED).move(DOWN * 1.2).spawn()
        c = Circle().set_color(GREEN).move(UP * 1.5).spawn()
        pane = Square().scale(2.5).set_color(BLUE)
        pane.opacity = 0.4
        pane.spawn()
        Text("mixed scene").move(DOWN * 2.5).spawn()
        PointLight().move(UP * 3 + OUT * 4).spawn()
    with Sync():
        s1.move(RIGHT * 1.5)
        s2.move(DOWN * 0.8)
        sq.rotate(90, OUT)
        c.scale(1.6)
        pane.move(LEFT * 0.8)
    return Scene


def main():
    out_path = sys.argv[1]
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 720
    ssaa = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    import algan  # noqa: F401
    from algan import Scene

    install_hook()
    build_scene(width, height, ssaa)
    with contextlib.suppress(_CaptureDone):
        Scene.save_video(
            os.path.join("benchmarks", "_arena_capture_probe"), overwrite=True
        )

    if "args" not in CAPTURE:
        raise SystemExit("kernel was never launched -- scene took another path")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(CAPTURE, out_path)
    total = 0
    print(
        f"captured {len(CAPTURE['args'])} args from sheet_resolve_shade "
        f"at {width}x{height} ssaa={ssaa}"
    )
    for name, (kind, val) in zip(CAPTURE["names"], CAPTURE["args"]):
        if kind == "tensor":
            total += val.numel() * val.element_size()
            print(f"  {name:24s} {str(tuple(val.shape)):26s} {val.dtype}")
        else:
            print(f"  {name:24s} = {val!r}")
    print(f"tensor bytes: {total / 1e6:.1f} MB -> {out_path}")


if __name__ == "__main__":
    main()
