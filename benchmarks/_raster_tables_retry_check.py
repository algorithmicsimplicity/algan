"""The out-of-memory split retry must not reclaim the batch-wide raster tables.

``_build_raster_tables`` allocates the projection / screen-bounds tables at the
arena's *persistent* (reverse) end from inside the first render chunk, caches
them on the merged device scene -- which lives for the whole batch -- and
publishes the reverse pointer it reached, so every later chunk reads that cache
instead of rebuilding.

``render_chunk``'s rewind used to restore the reverse pointer to the chunk's
entry value unconditionally, handing that range back to the allocator while the
cache still pointed into it. Nothing notices in the ordinary case: the render
loop re-protects the range the moment ``render_batch_raytraced`` returns. But an
out-of-memory retry rewinds and then **re-enters** ``render_chunk`` on each
half, and those halves allocate forward into the freed range. The tables then
read as garbage, and a negative bbox width reaches ``torch.repeat_interleave``
as ``RuntimeError: repeats can not be negative`` -- which is how this was found,
in post-process bloom's OOM on the reference scene.

Two arms of the same scene:

* **A** -- rendered ordinarily.
* **B** -- with one :class:`InsufficientMemoryException` injected into
  ``post_process_frames`` after the first chunk, which is the chunk that builds
  and caches the tables. That forces the split retry, whose halves are the first
  consumers of the cache.

Three things are checked, because no one of them is sufficient on its own:

1. **The invariant.** While the tables are cached, no rewind inside
   ``render_batch_raytraced`` may raise the reverse pointer above the value the
   build reached. This is the property the fix establishes, and it fails
   deterministically without it.
2. **The table bytes.** Fingerprinted right after the build and again when the
   render returns; an actual overwrite changes them. Whether the retry's forward
   allocations really reach that far depends on how much arena headroom this
   machine has, so this catches the real corruption when it happens but cannot
   be relied on alone -- hence (1).
3. **The frames.** A and B must be identical.

Non-vacuity is asserted rather than assumed: the tables must have been built,
the fault must have fired exactly once, and the retry must have **re-read** the
cache (exactly one build across the whole render) rather than rebuilt it -- a
rebuild would pass every check while testing nothing.

``--mutate`` reproduces the pre-fix rewind by dropping the published reverse
pointer, and the run is then *required* to fail.

    .venv/Scripts/python.exe benchmarks/_raster_tables_retry_check.py
    .venv/Scripts/python.exe benchmarks/_raster_tables_retry_check.py --mutate
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    ORANGE,
    RIGHT,
    SMOKE_TEST,
    UP,
    Circle,
    Group,
    Off,
    Square,
    Tetrahedron,
    Text,
)
from algan.rendering.raytracing import tracer  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402
from algan.settings.kernel_settings import KERNEL_REGISTRY  # noqa: E402
from algan.utils.memory_utils import (  # noqa: E402
    InsufficientMemoryException,
    ManualMemory,
)

MUTATE = "--mutate" in sys.argv

STATE = {
    "tables_built": 0,
    "faults_fired": 0,
    "retained": None,  # reverse pointer the build reached
    "fingerprint": None,  # table bytes right after the build
    "tables": None,
    "armed": False,  # inside render_batch_raytraced
    "violations": [],  # (reverse, retained) rewinds that freed the tables
}


def build_scene():
    """A batch carrying both geometry kinds the tables cover.

    Bezier circuits are what ``_window_pairs`` reads the corrupted bounds for;
    the flat triangles keep ``precompute_triangle_screen_bounds`` in the tables
    too, so either path reaches a corrupted table.
    """
    with Off():
        glyphs = Text("RETRY", font_size=54, color=BLUE).move(UP * 1.2)
        shapes = Group(
            Circle(radius=0.55, color=ORANGE),
            Square(size=1.0, color=GREEN),
            Tetrahedron(color=BLUE),
        ).arrange_in_line(RIGHT, buffer=0.7)
        shapes.move(DOWN * 0.9 - shapes.get_center())
        glyphs.spawn()
        shapes.spawn()
    # Something has to move: the window must span several frames for the split
    # retry to have halves to render.
    shapes.move(LEFT * 0.4)
    glyphs.move(RIGHT * 0.3)


def _fingerprint(tables):
    """Order-sensitive sum over every table tensor.

    Floats and bools both go through ``float64`` sums; the tables are small and
    this only has to notice that the bytes moved, not characterize how.
    """
    total = []
    for group in tables:
        if group is None:
            continue
        entries = group if isinstance(group, tuple) else (group,)
        for t in entries:
            if torch.is_tensor(t):
                total.append(float(t.detach().to(torch.float64).sum()))
    return tuple(total)


def _install_probes():
    """Wrap the build, the arena rewind and the render entry point.

    The published pointer is *not* read from the merged dict: ``--mutate``
    removes that key, and the probe still has to know where the tables are.
    """
    build = tracer._build_raster_tables

    def counted_build(merged, memory, *args, **kwargs):
        out = build(merged, memory, *args, **kwargs)
        STATE["tables_built"] += 1
        STATE["retained"] = memory.get_pointers()[1]
        STATE["tables"] = out
        STATE["fingerprint"] = _fingerprint(out)
        return out

    tracer._build_raster_tables = counted_build

    # The render loop reaches the tracer through the registry, which holds a
    # direct reference captured at import -- patching the module attribute
    # would silently never run.
    render = KERNEL_REGISTRY.render_kernel

    def armed_render(*args, **kwargs):
        STATE["armed"] = True
        try:
            return render(*args, **kwargs)
        finally:
            STATE["armed"] = False

    KERNEL_REGISTRY.render_kernel = armed_render

    set_pointers = ManualMemory.set_pointers

    def watched_set_pointers(self, pointers):
        retained = STATE["retained"]
        if STATE["armed"] and retained is not None:
            reverse = [*pointers][1]
            if reverse > retained:
                STATE["violations"].append((reverse, retained))
        return set_pointers(self, pointers)

    ManualMemory.set_pointers = watched_set_pointers


def _install_mutation():
    """Restore the pre-fix behaviour: no retained-range clamp on the rewind.

    ``rewind_to`` clamps the reverse pointer to the value the tracer publishes
    on the merged scene. Dropping that key after every wavefront call leaves the
    clamp nothing to read, which is exactly what the old code did. (It also
    disables the render loop's cross-chunk protection, which this single-batch
    scene never reaches.)
    """
    wavefront = tracer.raytrace_render_wavefront

    def unpublishing(*args, **kwargs):
        try:
            return wavefront(*args, **kwargs)
        finally:
            merged = args[3]
            if isinstance(merged, dict):
                merged.pop("_raster_tables_reverse_pointer", None)

    tracer.raytrace_render_wavefront = unpublishing


def _render(fault=False):
    """Render the whole recorded window, optionally failing one post-process.

    The fault waits for a chunk of **two or more** frames, for two reasons: the
    memory model always probes with a single frame first, so chunk 0 is the one
    that builds and caches the tables (which has to happen before the retry can
    re-read them), and a one-frame chunk cannot split at all -- it raises
    ``OutOfRenderMemory`` straight past the code under test, and the render loop
    re-preps the whole batch with a fresh merged scene.
    """
    scene = SceneManager.reset()
    scene.set_video_settings(SMOKE_TEST)
    build_scene()
    end_ind = round(scene._recorded_end_time_for_render() * scene.frames_per_second)

    post = tracer.post_process_frames
    fired = {"done": False}

    def faulting_post(*args, **kwargs):
        frames = args[1] if len(args) > 1 else kwargs["frames"]
        if fault and not fired["done"] and frames.shape[0] >= 2:
            fired["done"] = True
            STATE["faults_fired"] += 1
            raise InsufficientMemoryException
        return post(*args, **kwargs)

    tracer.post_process_frames = faulting_post
    try:
        # get_frames yields whatever the tracer produced for a batch: a tensor,
        # or a list of per-chunk tensors when the batch was split.
        frames = []
        for batch in scene.get_frames(0, end_ind):
            for part in batch if isinstance(batch, list) else [batch]:
                frames.append(part.detach().to("cpu", torch.float32).clone())
    finally:
        tracer.post_process_frames = post
    return torch.cat(frames, 0), end_ind


def main():
    _install_probes()
    if MUTATE:
        _install_mutation()

    failures = []

    reference, end_ind = _render()
    violations_a = list(STATE["violations"])
    print(
        f"arm A: {end_ind} frames requested, {tuple(reference.shape)} rendered, "
        f"table builds={STATE['tables_built']}, "
        f"table-freeing rewinds={len(violations_a)}"
    )
    assert end_ind >= 2, (
        f"the scene renders {end_ind} frame(s); a split retry needs at least 2"
    )
    assert STATE["tables_built"] >= 1, (
        "the raster tables were never built -- this scene took a non-raster "
        "path, so the check would be vacuous"
    )
    assert STATE["faults_fired"] == 0
    if violations_a:
        # Benign in this interleaving -- the render loop re-protects the range
        # before the next chunk allocates -- but it is the same defect, and the
        # recursive halves below have no such protection.
        failures.append(
            f"arm A: {len(violations_a)} rewind(s) freed the cached tables: "
            f"{violations_a[:2]}"
        )

    STATE.update(tables_built=0, retained=None, fingerprint=None, violations=[])
    try:
        retried, _ = _render(fault=True)
    except RuntimeError as exc:
        if MUTATE:
            print(f"MUTATED ARM FAILED AS REQUIRED: {type(exc).__name__}: {exc}")
            return 0
        raise

    after = _fingerprint(STATE["tables"])
    print(
        f"arm B: {tuple(retried.shape)} rendered, table builds="
        f"{STATE['tables_built']}, faults={STATE['faults_fired']}, "
        f"table-freeing rewinds={len(STATE['violations'])}"
    )
    assert STATE["faults_fired"] == 1, (
        f"the injected fault fired {STATE['faults_fired']} times, expected 1 -- "
        "the split retry was not exercised"
    )
    assert STATE["tables_built"] == 1, (
        f"the tables were built {STATE['tables_built']} times in the retried "
        "render; the retry must re-read the cache (exactly 1 build) or the "
        "corruption this checks for cannot occur"
    )

    if STATE["violations"]:
        failures.append(
            f"arm B: {len(STATE['violations'])} rewind(s) raised the reverse "
            f"pointer above the cached tables: {STATE['violations'][:2]}"
        )
    if after != STATE["fingerprint"]:
        failures.append("the table bytes changed across the retried render")
    if reference.shape != retried.shape:
        failures.append(
            f"frame shapes differ: {tuple(reference.shape)} vs {tuple(retried.shape)}"
        )
    else:
        diff = (reference - retried).abs()
        peak = float(diff.max())
        print(f"peak |A - B| = {peak}")
        if peak != 0.0:
            failures.append(f"{int((diff > 0).sum())} frame values differ, peak {peak}")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        if MUTATE:
            print("MUTATED ARM FAILED AS REQUIRED")
            return 0
        return 1
    if MUTATE:
        print("MUTATION DID NOT FAIL -- the check is vacuous, fix it")
        return 1
    print("OK: the split retry keeps the tables and reproduces the un-split render")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
