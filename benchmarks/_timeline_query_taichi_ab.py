"""A/B: the Taichi timeline-query kernels vs the shipped torch query, on real shapes.

``ALGAN_OPT_DISABLE=torchquery`` swaps ``generate_array_states``' torch
implementation (``timeline._query_row_states`` over an ``EditQueryIndex``) for
the original Taichi kernels (``timeline._generate_array_states_taichi`` ->
``utils_taichi._query_state_from_edits`` / ``_query_selected_state_from_edits``).
They were disabled for *staging*, not speed -- Taichi's arch is the render
device, so on a CUDA render the kernels stage every argument, including the
whole ``[T, N, D]`` result, through VRAM. This box has no GPU: the arch is x64,
the animation tensors are already host tensors, nothing stages, and the two
arms can be compared on their own merits.

Unlike the other timeline-query benches, the shapes here are not parameters.
The script instruments real ``save_video`` renders (default settings, defaults
not changed) and records the ``(T, N, D)``, dtype, edit count and
``active_rows`` size the query stage actually receives, then asserts the arms
agree on every captured case and times them. Both dispatch branches are
covered with genuinely received inputs:

* **full width** (``active_rows is None``, kernel ``_query_state_from_edits``)
  -- a scene whose animation includes one user-defined function of time. Its
  ``__module__`` is not ``algan.*``, so the conservative actor working-set
  gives up (``timeline._active_mob_ids`` returns ``None``) and every batch's
  base-state query runs through ``generate_array_states`` full width.
* **selected rows** (kernel ``_query_selected_state_from_edits``) -- captured
  off the compact call site (``AttributeTimeline``'s working-set query through
  ``_query_row_states``): those are the exact ``(times, prepared, rows)``
  triples the kernel arm receives at its two selected-row call sites
  (``rematerialize_state_at_times`` under ``ALGAN_OPT_DISABLE=compactstate``,
  ``materialize_additional_rows`` under ``ALGAN_OPT_DISABLE=torchquery``).
  The lazy-discovery route into ``materialize_additional_rows`` never fires in
  these scenes -- the conservative working set already contains every actor
  (``timeline._active_mob_ids`` seeds itself from the whole window's actor
  list), so there is nothing left to discover -- and the wrapper records that
  it did not fire rather than manufacturing a synthetic call.

No ``cap_process_memory``: every tensor here comes from a real scene rather
than from parameters (see CLAUDE.md's memory rules; do not cap a real render).

    ALGAN_USE_DAEMON=0 uv run python benchmarks/_timeline_query_taichi_ab.py
"""

from __future__ import annotations

import os

if os.environ.get("ALGAN_USE_DAEMON", "0") != "0":
    raise SystemExit("A warm daemon serves stale modules; run with ALGAN_USE_DAEMON=0.")
os.environ["ALGAN_USE_DAEMON"] = "0"

import statistics
import threading
import time

import torch

import algan.animation_timeline.timeline as tl
from algan import PREVIEW, Scene
from algan.animation_timeline.animation_contexts import Lag, Off, Seq, Sync
from algan.constants.color import BLUE, YELLOW
from algan.constants.spatial import OUT, RIGHT, UP
from algan.mobs.shapes_2d import Square
from algan.rendering.taichi_runtime import (
    _sync_devices,  # noqa: E402
    init_taichi,
    taichi_arch_is_cpu,
)
from algan.scene_manager import SceneManager

init_taichi()

ROUNDS = int(os.environ.get("AB_ROUNDS", "9"))
N_MOBS = int(os.environ.get("AB_N_MOBS", "300"))

_TORCH_ARM = frozenset()
_TAICHI_ARM = frozenset(("torchquery",))


# ---------------------------------------------------------------------------
# Instrumentation. Every wrapper delegates to the original and only records.


class Capture:
    """One observed call's inputs, held by reference (nothing is copied).

    The edit *list* is deliberately not kept -- with ``prepared`` given, both
    arms read nothing from it, so only its length is recorded. The prepared
    index and the query times are the actual objects the call received.
    """

    __slots__ = ("label", "times", "N", "E", "U", "prepared", "rows")

    def __init__(self, label, times, n, edits_or_none, prepared, rows):
        self.label = label
        self.times = times
        self.N = n
        self.E = -1 if edits_or_none is None else len(edits_or_none)
        self.U = int(prepared.sorted_edit_ids.shape[0])
        self.prepared = prepared
        self.rows = rows

    @property
    def D(self):
        return int(self.prepared.sorted_values.shape[1])

    @property
    def T(self):
        return int(self.times.shape[0])

    @property
    def R(self):
        return -1 if self.rows is None else int(self.rows.numel())

    @property
    def key(self):
        return (self.label, self.T, self.N, self.D, self.U, self.E, self.R)

    @property
    def out_mb(self):
        return self.T * self.N * self.D * 4 / 1e6


class Profiler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.wall_seconds = 0.0
        self.gas_seconds = 0.0
        self.gas_calls = 0
        self.qrs_seconds = 0.0
        self.qrs_calls = 0
        self.prep_seconds = 0.0
        self.prep_calls = 0
        self.captures = []
        self.mar_calls = 0

    def render_report(self, label):
        wall = self.wall_seconds
        parts = (
            ("full-width generate_array_states", self.gas_seconds, self.gas_calls),
            ("compact _query_row_states", self.qrs_seconds, self.qrs_calls),
            (
                "index build _prepare_array_state_queries",
                self.prep_seconds,
                self.prep_calls,
            ),
        )
        total = sum(seconds for _, seconds, _ in parts)
        lines = [
            f"  {label}: save_video wall {wall:.1f}s, query stage {total * 1e3:.0f}ms "
            f"({total / wall:.2%} of the render)"
        ]
        for name, seconds, calls in parts:
            share = "" if wall == 0 else f" = {seconds / wall:.2%}"
            lines.append(
                f"    {name:<42} {seconds * 1e3:8.1f}ms{share}  ({calls} calls)"
            )
        return "\n".join(lines)


_PROF = Profiler()
_TLS = threading.local()

_orig_generate = tl.generate_array_states
_orig_query_rows = tl._query_row_states
_orig_prepare = tl._prepare_array_state_queries
_orig_materialize_more = tl.AttributeTimeline.materialize_additional_rows


def _busy():
    return getattr(_TLS, "in_query_stage", False)


def _wrapped_generate(times, N, edits, *, active_rows=None, prepared=None):
    _TLS.in_query_stage = True
    start = time.perf_counter()
    try:
        return _orig_generate(
            times, N, edits, active_rows=active_rows, prepared=prepared
        )
    finally:
        _PROF.gas_seconds += time.perf_counter() - start
        _PROF.gas_calls += 1
        _TLS.in_query_stage = False
        if prepared is None:
            prepared = _orig_prepare(times, N, edits)
        _PROF.captures.append(
            Capture("full-width", times, N, edits, prepared, active_rows)
        )


def _wrapped_query_rows(times, prepared, rows=None):
    outside = not _busy()
    start = time.perf_counter()
    try:
        return _orig_query_rows(times, prepared, rows)
    finally:
        elapsed = time.perf_counter() - start
        if outside:
            # Calls coming from inside generate_array_states are already
            # counted by its own wrapper.
            _PROF.qrs_seconds += elapsed
            _PROF.qrs_calls += 1
            if rows is not None:
                _PROF.captures.append(
                    Capture(
                        "selected-rows/working-set",
                        times,
                        int(prepared.head.shape[0]) - 1,
                        None,
                        prepared,
                        rows,
                    )
                )


def _wrapped_prepare(times, N, edits):
    if _busy():
        return _orig_prepare(times, N, edits)
    start = time.perf_counter()
    try:
        return _orig_prepare(times, N, edits)
    finally:
        _PROF.prep_seconds += time.perf_counter() - start
        _PROF.prep_calls += 1


def _wrapped_materialize_more(self, times, rows):
    result = _orig_materialize_more(self, times, rows)
    # Captured only after the delegate ran: it calls prepare_for_queries()
    # first, and _prepared_queries before that would cache an index built
    # against the not-yet-sorted edit log.
    _PROF.mar_calls += 1
    _PROF.captures.append(
        Capture(
            "selected-rows/lazy-discovery",
            times,
            self.pointer,
            self._edits_sorted,
            self._prepared_queries(times),
            rows,
        )
    )
    return result


def install_wrappers():
    tl.generate_array_states = _wrapped_generate
    tl._query_row_states = _wrapped_query_rows
    tl._prepare_array_state_queries = _wrapped_prepare
    tl.AttributeTimeline.materialize_additional_rows = _wrapped_materialize_more


# ---------------------------------------------------------------------------
# Scenes. Spawns happen under Off(): each spawn outside a context is a 1s
# animation that advances the authoring clock, which would stretch the scene
# to minutes and hand the query one giant full-span batch.


def _user_wave(mob, elapsed):
    # Defined in this module, so its __module__ does not start with "algan."
    # -- which is exactly what makes _active_mob_ids give up on the working
    # set and every batch query run full width.
    mob.set(location=UP * 0.12 * torch.sin(elapsed * 5.0))


def _layout(mobs):
    with Off():
        for i, m in enumerate(mobs):
            m.move(RIGHT * (((i % 13) - 6) * 0.28))


def _animate(block, duration=6.0):
    with Seq(duration=duration):
        with Lag(0.15, duration=2.5):
            for i, m in enumerate(block):
                m.move(UP * (((i % 7) - 3) * 0.22))
        with Sync(duration=2.0):
            for i, m in enumerate(block[::2]):
                m.rotate(150 * ((i % 5) - 2), OUT)


def build_full_width_scene(n_mobs):
    """Builtin animations over every mob, plus one user-defined function of time."""
    mobs = []
    with Off():
        for i in range(n_mobs):
            mobs.append(Square(color=BLUE if i % 2 else YELLOW).spawn())
    _layout(mobs)
    _animate(mobs)
    with Sync(duration=1.5):
        for i, m in enumerate(mobs[1::3]):
            m.set(opacity=0.25 + 0.1 * (i % 4))
    with Sync(duration=1.0):
        mobs[0].animate_function_of_time(_user_wave)


def build_working_set_scene(n_mobs):
    """Only half the mobs are referenced by recorded events, so the traced
    working set -- and with it the selected-rows query -- covers a real
    subset of the rows.
    """
    mobs = []
    with Off():
        for i in range(n_mobs):
            mobs.append(Square(color=BLUE if i % 2 else YELLOW).spawn())
    _layout(mobs)
    animated = mobs[::2]
    _animate(animated)
    with Sync(duration=1.0):
        for m in animated[1::3]:
            m.set(opacity=0.35)


# ---------------------------------------------------------------------------
# Render drivers.


def _swap_profiler():
    """Start a fresh recording and hand back the one that just filled."""
    global _PROF
    finished = _PROF
    _PROF = Profiler()
    return finished


def run_render(label, build_scene):
    SceneManager.reset()
    _swap_profiler()
    with Scene() as scene:
        scene.set_video_settings(PREVIEW)
        build_scene(N_MOBS)
        start = time.perf_counter()
        scene.save_video(f"_timeline_query_ab_{label}", overwrite=True)
        _PROF.wall_seconds = time.perf_counter() - start
    finished = _swap_profiler()
    SceneManager.reset()
    return finished


def dedupe(captures):
    """One instance per distinct shape, keeping the latest (edit logs grow
    over a render, so the last instance is the fullest).
    """
    keep = {}
    for capture in captures:
        keep[capture.key] = capture
    return [keep[key] for key in sorted(keep)]


def run_case(capture, arm):
    # A one-element sentinel stands in for the edit log: with ``prepared``
    # given, both arms read nothing else from it, and the empty-log early
    # return did not fire on the observed call either.
    tl._OPT_DISABLED = arm
    try:
        return _orig_generate(
            capture.times,
            capture.N,
            [object()],
            active_rows=capture.rows,
            prepared=capture.prepared,
        )
    finally:
        tl._OPT_DISABLED = _TORCH_ARM


def check_correctness(cases):
    """Assert the arms agree on every case before anything is timed."""
    failures = 0
    worst = 0.0
    for capture in cases:
        reference = run_case(capture, _TORCH_ARM)
        candidate = run_case(capture, _TAICHI_ARM)
        if reference.shape != candidate.shape:
            print(
                f"  FAIL {capture.label}: shape {tuple(reference.shape)} vs"
                f" {tuple(candidate.shape)}"
            )
            failures += 1
            continue
        deviation = (reference - candidate).abs().max().item()
        worst = max(worst, deviation)
        if not torch.equal(reference, candidate):
            differing = int((reference != candidate).sum())
            print(
                f"  FAIL {capture.label} T={capture.T} N={capture.N} D={capture.D}"
                f": {differing} elements differ, max {deviation:g}"
            )
            failures += 1
        del reference, candidate
    verdict = "PASS" if failures == 0 else f"{failures} FAILURES"
    print(
        f"correctness over {len(cases)} cases: {verdict}; max abs deviation {worst:g}"
    )
    return failures


def time_case(capture, rounds):
    """Median of ``rounds`` alternating measurements per arm.

    Alternating rather than blocked: a shared 4-vCPU box drifts more across a
    block than the effect being measured. One untimed call per arm precedes
    the rounds, so cold kernel compilation stays out of the timed region.
    """
    for arm in (_TORCH_ARM, _TAICHI_ARM):
        out = run_case(capture, arm)
        del out

    samples = {"torch": [], "taichi": []}
    for _ in range(rounds):
        for name, arm in (("torch", _TORCH_ARM), ("taichi", _TAICHI_ARM)):
            _sync_devices()
            start = time.perf_counter()
            out = run_case(capture, arm)
            _sync_devices()
            samples[name].append((time.perf_counter() - start) * 1e3)
            del out
    return statistics.median(samples["torch"]), statistics.median(samples["taichi"])


def main():
    if not taichi_arch_is_cpu():
        print("Taichi is not on a CPU arch; staging confounds the comparison here.")
        return 1
    install_wrappers()

    print(f"instrumenting two PREVIEW renders of {N_MOBS}-mob scenes...\n")
    full_width_render = run_render("userfn", build_full_width_scene)
    working_set_render = run_render("workingset", build_working_set_scene)

    for label, prof in (
        ("scene A (user function of time -> full-width queries)", full_width_render),
        ("scene B (traced working set -> selected-rows queries)", working_set_render),
    ):
        print(prof.render_report(label))
    print(
        f"\n  materialize_additional_rows calls: "
        f"{full_width_render.mar_calls} (scene A) + "
        f"{working_set_render.mar_calls} (scene B)"
    )

    cases = dedupe(full_width_render.captures + working_set_render.captures)
    print(f"\ncaptured query-stage shapes ({len(cases)} distinct):")
    for c in cases:
        print(
            f"  {c.label:<30} T={c.T:>5} N={c.N:>6} D={c.D} U={c.U:>7} "
            f"E={c.E:>7} R={c.R:>7} out={c.out_mb:7.2f}MB [{c.times.dtype}]"
        )

    print("\ncorrectness (both arms on identical captured inputs):")
    failures = check_correctness(cases)
    if failures:
        return 1

    print(f"\ntiming ({ROUNDS} alternating rounds, median, cold compile excluded):")
    print(f"  {'case':<44}{'torch ms':>10}{'taichi ms':>11}{'speedup':>9}   branch")
    for c in cases:
        torch_ms, taichi_ms = time_case(c, ROUNDS)
        name = (
            f"T={c.T} N={c.N} D={c.D} U={c.U} E={c.E} R={c.R}"
            if c.label.startswith("full-width")
            else f"{c.label} T={c.T} D={c.D} U={c.U} R={c.R}"
        )
        print(
            f"  {name:<44}{torch_ms:>10.2f}{taichi_ms:>11.2f}"
            f"{torch_ms / max(taichi_ms, 1e-9):>8.2f}x   {c.label}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
