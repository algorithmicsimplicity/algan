"""Which parameters of the wide kernels are ARENA-BACKED, by name, per launch.

``test_arena_binding_live.py`` counts how many of a kernel's arguments sit in the
``ManualMemory`` arena. Converting a kernel to the arena calling convention needs
the next question answered: *which* ones, by parameter name, and whether that is
true on **every** launch of every path. A kernel signature is fixed at authoring
time, so a parameter that is arena-backed in one scene and a standalone
allocation in another cannot be bound through the arena at all.

Prints a per-kernel table of parameter -> (launches seen, launches arena-backed,
dtype, ndim), so the conversion can bind only the parameters that are always
backed and leave the rest as ordinary ndarray arguments.

Usage (one render process at a time on Windows):
  .venv/Scripts/python.exe benchmarks/_arena_param_membership.py <scene> [W H]

  scene: mixed | glass | pathtrace
"""

import importlib
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("ALGAN_AUTO_DAEMON", "0")
os.environ.setdefault("ALGAN_USE_DAEMON", "0")
# Serial prep, so the hooks fire on the main thread in a deterministic order.
os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

import torch  # noqa: E402

WIDE_KERNELS = {
    "algan.rendering.raytracing.sheet_resolve_taichi": ("sheet_resolve_shade",),
    "algan.rendering.raytracing.wavefront_kernels_taichi": (
        "wavefront_shade",
        "wavefront_traverse_events",
    ),
    "algan.rendering.raytracing.raster_taichi": ("raster_shadow_trace",),
    "algan.rendering.raytracing.path_tracer_taichi": ("pt_shade",),
}

SEEN = {}
ARENAS = []


def _arena_ptrs():
    out = set()
    for arena in ARENAS:
        data = getattr(arena, "data", None)
        if data is not None and data.numel():
            out.add(data.untyped_storage().data_ptr())
    return out


def _install(monkeypatch_targets):
    from algan.utils.memory_utils import ManualMemory

    real_init = ManualMemory.__init__

    def init(self, *a, **kw):
        real_init(self, *a, **kw)
        if self.managed and len(self.data):
            ARENAS.append(self)

    ManualMemory.__init__ = init

    for mod_name, names in WIDE_KERNELS.items():
        mod = importlib.import_module(mod_name)
        for n in names:
            original = getattr(mod, n, None)
            if original is None:
                continue
            # A converted kernel's public name is a ``def f(*args)`` that
            # delegates to its `arena_packed` launcher, so the parameter names
            # live on the launcher rather than in the signature.
            launcher = getattr(mod, f"_{n}_launch", None)
            if launcher is not None:
                params = list(launcher.call_params)
            else:
                fn = getattr(original, "__wrapped__", original)
                params = list(inspect.signature(fn).parameters)
            rec = SEEN.setdefault(n, {"launches": 0, "params": {}})

            def wrap(kernel, params=params, rec=rec):
                def recorder(*args, **kwargs):
                    ptrs = _arena_ptrs()
                    rec["launches"] += 1
                    for i, v in enumerate(args):
                        name = params[i] if i < len(params) else f"arg{i}"
                        p = rec["params"].setdefault(
                            name,
                            {
                                "seen": 0,
                                "arena": 0,
                                "dtype": None,
                                "ndim": None,
                                "tensor": 0,
                                "trailing": set(),
                            },
                        )
                        p["seen"] += 1
                        if not isinstance(v, torch.Tensor):
                            continue
                        p["tensor"] += 1
                        p["dtype"] = str(v.dtype).replace("torch.", "")
                        p["ndim"] = v.dim()
                        p["trailing"].add(tuple(v.shape[1:]))
                        if v.untyped_storage().data_ptr() in ptrs:
                            p["arena"] += 1
                    return kernel(*args, **kwargs)

                return recorder

            wrapper = wrap(original)
            setattr(mod, n, wrapper)
            for other in list(sys.modules.values()):
                if other is None or other is mod:
                    continue
                if getattr(other, n, None) is original:
                    setattr(other, n, wrapper)
            monkeypatch_targets.append((mod, n, original))


def scene_mixed(width, height):
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
        resolution=(width, height), frames_per_second=10, super_sampling_anti_aliasing=1
    )
    SETTINGS.raytracing.set(shadows=True)
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


def scene_glass(width, height):
    """Refraction forces the wavefront tracer (see agent_guidance/rendering)."""
    from algan import (
        DOWN,
        LEFT,
        OUT,
        RIGHT,
        SETTINGS,
        UP,
        MeshPhysicalMaterial,
        MeshStandardMaterial,
        Off,
        PointLight,
        Scene,
        Sphere,
        Square,
        Sync,
    )

    SETTINGS.video.set(
        resolution=(width, height), frames_per_second=10, super_sampling_anti_aliasing=1
    )
    SETTINGS.raytracing.set(shadows=True)
    with Off():
        floor = Square().scale(8).rotate(90, RIGHT).move(DOWN * 2.2)
        floor.set_material(MeshStandardMaterial(roughness=0.2, metalness=0.4))
        floor.spawn()
        glass = Sphere().scale(1.2).move(LEFT)
        glass.set_material(
            MeshPhysicalMaterial(transmission=0.95, roughness=0.05, ior=1.5)
        )
        glass.spawn()
        mirror = Sphere().scale(0.9).move(RIGHT * 1.6)
        mirror.set_material(MeshStandardMaterial(roughness=0.05, metalness=1.0))
        mirror.spawn()
        PointLight().move(UP * 3 + OUT * 4).spawn()
    with Sync():
        glass.move(RIGHT * 0.6)
        mirror.move(UP * 0.4)
    return Scene


def scene_pathtrace(width, height):
    from algan import (
        DOWN,
        LEFT,
        OUT,
        RIGHT,
        SETTINGS,
        UP,
        MeshStandardMaterial,
        Off,
        PointLight,
        Scene,
        Sphere,
        Square,
        Sync,
    )

    SETTINGS.video.set(
        resolution=(width, height), frames_per_second=10, super_sampling_anti_aliasing=1
    )
    SETTINGS.raytracing.set(shadows=True, samples_per_pixel=4, max_bounces=3)
    with Off():
        floor = Square().scale(8).rotate(90, RIGHT).move(DOWN * 2.2)
        floor.set_material(MeshStandardMaterial(roughness=0.4))
        floor.spawn()
        a = Sphere().scale(0.9).move(LEFT).spawn()
        b = Sphere().scale(0.7).move(RIGHT * 1.5).spawn()
        PointLight().move(UP * 3 + OUT * 4).spawn()
    with Sync():
        a.move(RIGHT * 0.4)
        b.move(UP * 0.3)
    return Scene


SCENES = {"mixed": scene_mixed, "glass": scene_glass, "pathtrace": scene_pathtrace}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "mixed"
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 320
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 180

    import algan  # noqa: F401
    from algan import Scene

    targets = []
    _install(targets)
    SCENES[which](width, height)
    Scene.save_video(
        os.path.join("benchmarks", f"_arena_member_{which}"), overwrite=True
    )

    out = {}
    print()
    for name, rec in sorted(SEEN.items()):
        if not rec["launches"]:
            continue
        print(f"=== {name}  ({rec['launches']} launches)")
        krec = {}
        for pname, p in rec["params"].items():
            if not p["tensor"]:
                continue
            always = p["arena"] == p["tensor"]
            krec[pname] = {
                "tensor": p["tensor"],
                "arena": p["arena"],
                "dtype": p["dtype"],
                "ndim": p["ndim"],
                "always": always,
                "trailing": sorted(str(t) for t in p["trailing"]),
            }
            flag = "" if always else "   <== NOT ALWAYS ARENA"
            print(
                f"   {pname:24s} {p['dtype']:8s} nd={p['ndim']} "
                f"arena {p['arena']}/{p['tensor']}{flag}"
            )
        out[name] = {"launches": rec["launches"], "params": krec}

    path = os.path.join("benchmarks", f"_arena_member_{which}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
