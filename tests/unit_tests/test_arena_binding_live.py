"""Do the renderer's widest kernels actually pack into 31 Metal buffer slots?

``DESIGN_metal_native_port.md`` §1.2 says yes, and the whole native-Metal port
rests on it: Metal binds at most 31 buffers to a compute stage (measured — the
32nd is a compile error), ``sheet_resolve_shade`` takes 49 ndarray arguments,
and the reason that gap is survivable is that ``ManualMemory`` hands out views
of **one** allocation, so 49 arrays are one buffer at 49 offsets.

The part of that which can rot is the part about arguments that are *not*
arena-backed: those keep their own binding, and enough of them would put a
kernel back over the ceiling however well the arena works. Nothing else in the
suite would notice a new persistent argument being threaded into a megakernel,
because on CUDA and CPU it costs nothing -- Taichi allows 64. It would cost the
Metal port the kernel.

So this renders a real frame, watches what the megakernels are actually handed,
and fails if any of them stops fitting. It is the regression guard for a claim
that is otherwise only in a document.
"""

import importlib
import sys

import pytest
import torch

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import BLUE, RED
from algan.constants.spatial import LEFT, RIGHT
from algan.mobs.shapes_2d import Square
from algan.mobs.shapes_3d import Cube
from algan.rendering.arena_binding import (
    METAL_BUFFER_SLOTS,
    ArenaBindingError,
    arena_storage_ptr,
    plan_bindings,
)
from algan.scene import Scene
from algan.settings.video_settings import LD
from algan.utils.memory_utils import ManualMemory

#: The kernels whose width is the reason the convention exists. Any that a
#: given frame does not launch is skipped rather than failed -- one scene
#: cannot reach every tracer path, and a guard that demanded it would be
#: fragile for no gain.
WIDE_KERNELS = {
    "algan.rendering.raytracing.sheet_resolve_taichi": ("sheet_resolve_shade",),
    "algan.rendering.raytracing.wavefront_kernels_taichi": (
        "wavefront_shade",
        "wavefront_traverse",
        "wavefront_traverse_events",
        "wavefront_shadow",
    ),
    "algan.rendering.raytracing.raster_taichi": (
        "raster_shadow_trace",
        "raster_tri_count",
        "raster_tri_write",
        "raster_bez_count",
        "raster_bez_write",
    ),
    "algan.rendering.raytracing.raytrace_kernels_taichi": (
        "path_trace_physical_stbvh",
        "path_trace_scene_stbvh",
    ),
}

#: At least these must have run, or the render did not exercise the path this
#: file exists to guard and a green result would mean nothing.
REQUIRED = {"sheet_resolve_shade", "raster_tri_count", "raster_tri_write"}


def _install_recorder(monkeypatch, arenas, seen):
    """Patch each wide kernel to plan its bindings, then call through.

    The plan is computed **inside the launch**, for two reasons that both bit
    the first version of this file. ``render_loop`` drops the arena at teardown
    (``render_memory.data = None``, `render_loop.py:2843`), so an argument list
    kept for later inspection can no longer be resolved against the arena it
    came from. And keeping those argument lists at all pins every tensor in
    them, holding the whole multi-hundred-megabyte arena alive across the
    render it is supposed to be observing. Only the small summary escapes.
    """

    def wrap(kernel, name):
        def recorder(*args, **kwargs):
            arena = _best_arena(args, arenas)
            if arena is not None:
                try:
                    plan = plan_bindings(args, arena)
                    record = {
                        "arena": len(plan.slots),
                        "passthrough": len(plan.passthrough),
                        "bindings": plan.bindings,
                        "fits": plan.fits,
                        "describe": plan.describe(),
                        "error": None,
                    }
                except ArenaBindingError as exc:
                    record = {"error": str(exc)}
                seen.setdefault(name, []).append(record)
            return kernel(*args, **kwargs)

        return recorder

    for mod_name, names in WIDE_KERNELS.items():
        mod = importlib.import_module(mod_name)
        for n in names:
            original = getattr(mod, n, None)
            if original is None:
                continue
            wrapper = wrap(original, n)
            monkeypatch.setattr(mod, n, wrapper)
            # Launch sites do `from ... import name` (some at module scope), so
            # rebind every module already holding the original too, or the
            # patch is invisible to exactly the caller under test.
            for other in list(sys.modules.values()):
                if other is None or other is mod:
                    continue
                if getattr(other, n, None) is original:
                    monkeypatch.setattr(other, n, wrapper)

    real_init = ManualMemory.__init__

    def init(self, *a, **kw):
        real_init(self, *a, **kw)
        if self.managed and len(self.data):
            arenas.append(self)

    monkeypatch.setattr(ManualMemory, "__init__", init)


def _best_arena(args, arenas):
    """The arena most of ``args`` came from, or ``None``.

    A render builds more than one ``ManualMemory`` (the unmanaged scratch ones
    among them), so which arena a launch belongs to is decided by counting
    rather than assumed. Released arenas are skipped: the loop frees a chunk's
    buffer as soon as it is finished with it, so by the time a later chunk
    launches, earlier arenas in this list are already dead.
    """
    best, best_hits = None, 0
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    for arena in arenas:
        if getattr(arena, "data", None) is None:
            continue
        ptr = arena_storage_ptr(arena)
        hits = sum(1 for t in tensors if t.untyped_storage().data_ptr() == ptr)
        if hits > best_hits:
            best, best_hits = arena, hits
    return best


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """One frame of flat 2-D and 3-D geometry, kernels recorded. Rendered once.

    Module-scoped deliberately. A function-scoped fixture renders once per test,
    which is both three times the cost and *not equivalent*: a later render in
    the same process can reuse an arena built before this fixture's patch went
    in, and launches against an arena the recorder never saw are silently
    dropped. The first version of this file did that and the three tests
    disagreed with each other about how many kernels had run.

    ``pytest.MonkeyPatch`` is used directly because the ``monkeypatch`` fixture
    is function-scoped and cannot be requested from here.
    """
    arenas: list[ManualMemory] = []
    seen: dict[str, list] = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        _install_recorder(monkeypatch, arenas, seen)
        out = tmp_path_factory.mktemp("arena_binding") / "frame"
        with Scene() as scene:
            with Off():
                square = Square(side_length=1.5, color=RED).move(LEFT)
                cube = Cube(side_length=1.2, color=BLUE).move(RIGHT)
            square.spawn(animate=False)
            cube.spawn(animate=False)
            scene.save_frame(str(out), video_settings=LD)
    assert arenas, "no managed arena was constructed during the render"
    return seen


def test_the_render_exercised_the_kernels_this_guards(rendered):
    seen = rendered
    missing = REQUIRED - set(seen)
    assert not missing, (
        f"the scene never launched {sorted(missing)}, so this file proves "
        "nothing; fix the scene rather than the assertion"
    )


def test_every_wide_kernel_packs_into_metal_s_buffer_limit(rendered):
    seen = rendered
    report = []
    over = []
    for name, launches in sorted(seen.items()):
        planned = [r for r in launches if not r.get("error")]
        if not planned:
            continue
        worst = max(planned, key=lambda r: r["bindings"])
        report.append(
            f"  {name:<28} {worst['arena']:>3} arena "
            f"+ {worst['passthrough']:>2} passthrough "
            f"-> {worst['bindings']:>2} bindings"
        )
        if not worst["fits"]:
            over.append(f"{name}: {worst['describe']}")

    print("\n" + "\n".join(report))
    assert not over, (
        "these kernels no longer fit Metal's "
        f"{METAL_BUFFER_SLOTS}-buffer limit even with the arena packed into "
        "one binding:\n  " + "\n  ".join(over)
    )


def test_arena_arguments_are_addressable_by_offset(rendered):
    """Every packed argument is dense and aligned, so a shader can rebuild it.

    ``plan_bindings`` raises rather than returns for an argument that is in the
    arena but cannot be addressed as (base + offset, dtype, shape) — a
    non-contiguous view has strides a shader has no way to honour. On the
    Metal path that is wrong pixels, not a crash, which is why it is asserted
    on real launch arguments rather than only on hand-built ones.
    """
    seen = rendered
    for name, launches in seen.items():
        for record in launches:
            if record.get("error"):
                pytest.fail(f"{name}: {record['error']}")

    # Tied to the specific claim rather than to a round number: the kernel the
    # whole argument is about must have been seen with essentially all of its
    # 49 arguments resolved into the arena. A drop here means arguments moved
    # out of the arena, which is exactly what would sink the packing.
    widest = max(r["arena"] for r in seen["sheet_resolve_shade"])
    assert widest >= 40, (
        f"sheet_resolve_shade resolved only {widest} of its arguments into the "
        "arena; DESIGN_metal_native_port.md §1.2 assumes essentially all of them"
    )
