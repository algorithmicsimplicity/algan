"""Precompiling must be invisible to what a render draws, and must actually engage.

``algan/rendering/kernel_precompile.py`` writes kernel specializations down as
portable specs, compiles them in worker processes with placeholder arguments,
and lets a render pick the artifacts up (or wait for one in flight). The
claims held here:

* a spec round-trips: the placeholder arguments built from it encode back to
  the same spec, with types preserved (``True`` is not ``1``, a tuple is not a
  list) -- otherwise a worker would compile a different specialization;
* anything that cannot travel (a lambda, an instance, a scene's own function)
  is refused rather than approximated;
* the source key does not see the worker's own switches, so a worker writes
  index entries under the keys the render computes;
* the manifest merges, bounds itself and forgets confirmed work only when the
  environment stamp moves; built-in specs are filtered by the settings their
  variant needs;
* an implicit pool is scoped to the running script, and a process's first
  kernel waits for a pool under its own settings (and stops one under
  others) -- once, because the compiler reads the cache's index once;
* end to end, a kernel a worker compiled is a source-key *hit* in the render,
  with output identical to a locally compiled one, and a process that used
  the cache before the worker wrote does not see it (the compiler behaviour
  the first-touch wait is built around).

The built-in list itself is checked against the live kernels, so a signature
change shows up here rather than as a first render that quietly lost its
parallelism.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import types

import pytest
import torch

from algan.rendering import kernel_precompile as kp
from algan.taichi_compat import BACKEND, ti

quadrants_only = pytest.mark.skipif(
    BACKEND != "quadrants", reason="precompiling rests on the Quadrants-only index"
)


# --- value encoding -----------------------------------------------------------


def test_plain_values_round_trip_with_their_types():
    for value in (None, True, 1, 1.5, "x", (1, (True, 2.0)), [1, 2], ()):
        decoded = kp._decode(kp._encode(value))
        assert decoded == value
        assert type(decoded) is type(value)
    # `True == 1` and they specialize alike, but the source key renders them
    # differently: a worker must see the bool.
    assert kp._decode(kp._encode(True)) is True
    assert type(kp._decode(kp._encode((1,)))) is tuple


def test_algan_objects_round_trip_by_name():
    from algan.rendering.raytracing import shading_taichi
    from algan.rendering.raytracing.arena_args_taichi import ArenaView
    from algan.rendering.shaders.materials import Side

    member = list(Side)[-1]
    assert kp._decode(kp._encode(member)) is member
    assert kp._decode(kp._encode(ArenaView)) is ArenaView
    func = shading_taichi._as_written_bytes
    assert kp._decode(kp._encode(func)) is func
    assert kp._decode(kp._encode(ti.f32)) == ti.f32
    assert kp._decode(kp._encode(torch.int64)) is torch.int64


class _Local:
    pass


@pytest.mark.parametrize(
    "value",
    [lambda: 1, _Local(), _Local, {"a": 1}, object()],
    ids=["lambda", "instance", "test-module-class", "dict", "object"],
)
def test_values_that_cannot_travel_are_refused(value):
    with pytest.raises(kp.Unportable):
        kp._encode(value)


# --- specs ----------------------------------------------------------------------


def _tonemap():
    from algan.rendering.post_processing.tonemap_kernels_taichi import tonemap_to_u8

    return tonemap_to_u8._primal


def _tonemap_args():
    return (
        torch.rand((1, 4, 6, 4)),
        torch.zeros((1, 4, 6, 3), dtype=torch.uint8),
        1,
        1.0,
        0,
        1,
        0,
        0.49,
    )


def test_a_spec_is_a_fixed_point_of_its_placeholders():
    kernel = _tonemap()
    spec = kp.encode_spec(kernel, _tonemap_args())
    assert spec["kernel"] == (
        "algan.rendering.post_processing.tonemap_kernels_taichi:tonemap_to_u8"
    )
    assert spec["args"][0] == ["A", "float32", 4, []]
    assert spec["args"][2] == ["T", ["int", 1]]
    # A scalar is keyed by type: its value is a runtime input, and two batches
    # of one scene pass different values to one specialization.
    assert spec["args"][3] == ["S", "float"]
    other = list(_tonemap_args())
    other[3] = 2.5
    assert kp.encode_spec(kernel, tuple(other)) == spec
    assert kp.resolve_kernel(spec) is kernel
    placeholders = kp.placeholder_args(spec, torch.device("cpu"))
    assert all(t.numel() == 1 for t in placeholders[:2])
    assert kp.encode_spec(kernel, placeholders) == spec


def test_a_spec_is_refused_for_a_non_tensor_array_or_a_foreign_kernel():
    kernel = _tonemap()
    args = list(_tonemap_args())
    args[0] = args[0].numpy()
    assert kp.encode_spec(kernel, tuple(args)) is None
    stub = types.SimpleNamespace(func=lambda: None, arg_metas=[])
    assert kp.encode_spec(stub, ()) is None


def test_a_spec_that_no_longer_matches_its_kernel_is_unportable():
    spec = kp.encode_spec(_tonemap(), _tonemap_args())
    with pytest.raises(kp.Unportable):
        kp.resolve_kernel({**spec, "args": spec["args"][:-1]})
    with pytest.raises(kp.Unportable):
        kp.resolve_kernel(
            {**spec, "kernel": "algan.rendering.kernel_precompile:nothing"}
        )


@pytest.mark.fast
def test_every_builtin_spec_still_matches_a_live_kernel():
    """The built-in list against today's kernels (``scripts/generate_kernel_specs.py``).

    A stale entry is skipped by the workers, never an error, so this is the
    only place a kernel signature change that invalidates it is noticed.
    Regenerate the list when it fails.

    Marked ``fast`` alone in this file (``tests/README.md`` says why): the
    signature change that breaks it is made in a kernel module, and outside
    the fast loop it went unnoticed -- ``det_sum`` left all ten megakernel
    entries stale on master.
    """
    builtin = kp.read_builtin_specs(with_requirements=True)
    if not builtin:
        pytest.skip("no built-in spec list is committed")
    stale = []
    for spec, seconds, requires in builtin:
        try:
            kernel = kp.resolve_kernel(spec)
            placeholders = kp.placeholder_args(spec, torch.device("cpu"))
        except kp.Unportable as exc:
            stale.append(f"{spec['kernel']}: {exc}")
            continue
        if kp.encode_spec(kernel, placeholders) != spec:
            stale.append(f"{spec['kernel']}: no longer encodes to itself")
        assert seconds >= 0
        assert set(requires) <= set(kp.current_context()["raytracing"])
    assert not stale, "regenerate with scripts/generate_kernel_specs.py:\n" + "\n".join(
        stale
    )


# --- the context and the key --------------------------------------------------


def test_the_workers_own_switches_do_not_reach_the_source_key(monkeypatch):
    from algan.utils import taichi_source_key as sk

    before = sk._environment_fingerprint()
    context_before = kp.context_id(kp.current_context())
    monkeypatch.setenv("ALGAN_PRECOMPILE_WORKER", "1")
    monkeypatch.setenv("ALGAN_PRECOMPILE_JOBS", "3")
    assert sk._environment_fingerprint() == before
    assert kp.context_id(kp.current_context()) == context_before


def test_a_keyed_environment_variable_moves_the_context(monkeypatch):
    before = kp.context_id(kp.current_context())
    monkeypatch.setenv("ALGAN_OPT_LEVEL", "2")
    assert kp.context_id(kp.current_context()) != before


def test_the_device_pool_size_is_not_keyed():
    """Workers shrink it; the index must still match (see ``_CONFIG_EXCLUDE_NAMES``)."""
    from algan.utils import taichi_source_key as sk

    assert {"device_memory_GB", "device_memory_fraction"} <= sk._CONFIG_EXCLUDE_NAMES


# --- the manifest ---------------------------------------------------------------


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    path = tmp_path / "specs.json"
    monkeypatch.setattr(kp, "manifest_path", lambda: path)
    monkeypatch.setattr(kp, "_PENDING", {})
    monkeypatch.setattr(kp, "_PENDING_CONTEXTS", {})
    monkeypatch.setattr(kp, "_AWAITING_ARTIFACT", {})
    monkeypatch.setattr(kp, "BUILTIN_MANIFEST", tmp_path / "builtin.json")
    return path


def _spec(n):
    return {"kernel": f"algan.x:k{n}", "args": [["T", ["int", n]]]}


def test_rows_merge_across_writers_and_the_file_is_bounded(manifest, monkeypatch):
    context = kp.current_context()
    kp._note_entry(_spec(1), context, seconds=3.0, confirmed="s1", used=True)
    assert kp.flush_manifest()
    kp._note_entry(_spec(1), context, used=True)
    kp._note_entry(_spec(2), context, seconds=1.0)
    assert kp.flush_manifest()
    data = kp.read_manifest()
    rows = {row["spec"]["kernel"]: row for row in data["entries"].values()}
    assert rows["algan.x:k1"]["uses"] == 2
    assert rows["algan.x:k1"]["seconds"] == 3.0
    assert rows["algan.x:k1"]["confirmed"] == "s1"
    assert list(data["contexts"]) == [kp.context_id(context)]

    monkeypatch.setattr(kp, "_MANIFEST_MAX_ENTRIES", 1)
    kp._note_entry(_spec(3), context, used=True)
    assert kp.flush_manifest()
    assert [
        row["spec"]["kernel"] for row in kp.read_manifest()["entries"].values()
    ] == ["algan.x:k3"]


def test_an_unreadable_manifest_reads_as_empty(manifest):
    manifest.write_text("{not json", encoding="utf-8")
    assert kp.read_manifest()["entries"] == {}


@quadrants_only
def test_a_compiled_spec_is_confirmed_once_its_artifact_exists(
    manifest, tmp_path, monkeypatch
):
    """A real compile is on disk only after a dump, so its row waits for the file.

    Confirming it at once let a process that never flushed -- the kernel
    profiler's, a killed one -- record specs as cached whose ``.qdc`` was never
    written: the next process's pool skipped them and its render compiled them.
    """
    from algan.settings import _startup

    monkeypatch.setattr(_startup, "_TAICHI_CACHE_DIRECTORY", tmp_path)
    monkeypatch.setattr(kp, "environment_stamp", lambda: "now")
    artifacts = tmp_path / "kernel_compilation_manager"
    artifacts.mkdir()
    (artifacts / "Tdumped.qdc").write_bytes(b"")

    def confirmations():
        return {
            row["spec"]["kernel"]: row.get("confirmed")
            for row in kp.read_manifest()["entries"].values()
        }

    # Confirmed by an earlier process, then compiled here anyway: the artifact
    # was not in the cache this process read, so the confirmation goes.
    kp._note_entry(_spec(1), kp.current_context(), confirmed="now")
    assert kp.flush_manifest()
    kp.note_materialized(_spec(1), seconds=2.0, compiled=True, cache_key="Tnew")
    # Served from a cache: on disk already.
    kp.note_materialized(_spec(2), seconds=0.1, compiled=False, cache_key="Tserved")
    # Compiled, but another process had already written the same artifact.
    kp.note_materialized(_spec(3), seconds=2.0, compiled=True, cache_key="Tdumped")
    # No key to look for (the Taichi backend): the old rule.
    kp.note_materialized(_spec(4), seconds=2.0, compiled=True, cache_key=None)
    assert kp.flush_manifest()
    assert confirmations() == {
        "algan.x:k1": False,
        "algan.x:k2": "now",
        "algan.x:k3": "now",
        "algan.x:k4": "now",
    }
    compiled = kp.entry_id(_spec(1), kp.current_context())
    assert kp.read_manifest()["entries"][compiled]["seconds"] == 2.0

    assert not kp.flush_manifest(), "nothing new to write while the file is missing"
    (artifacts / "Tnew.qdc").write_bytes(b"")
    assert kp.flush_manifest()
    assert confirmations()["algan.x:k1"] == "now"
    assert kp._AWAITING_ARTIFACT == {}


def test_pending_work_is_what_the_current_stamp_has_not_confirmed(
    manifest, monkeypatch
):
    monkeypatch.setattr(kp, "environment_stamp", lambda: "now")
    context = kp.current_context()
    other = {**context, "raytracing": {**context["raytracing"], "shadows": True}}
    kp._note_entry(_spec(1), context, seconds=20.0, confirmed="now")
    kp._note_entry(_spec(2), context, seconds=20.0, confirmed="before")
    kp._note_entry(_spec(3), other, seconds=20.0, confirmed="before")
    kp.flush_manifest()
    kp.BUILTIN_MANIFEST.write_text(
        json.dumps(
            {
                "schema": kp._MANIFEST_SCHEMA,
                "entries": {
                    "a": {"spec": _spec(4), "seconds": 9.0, "requires": {}},
                    "b": {
                        "spec": _spec(5),
                        "seconds": 9.0,
                        "requires": {"shadows": True},
                    },
                    # Already confirmed through the manifest under this context.
                    "c": {"spec": _spec(1), "seconds": 9.0, "requires": {}},
                },
            }
        ),
        encoding="utf-8",
    )
    pending = {job.spec["kernel"] for job in kp.pending_jobs(context)}
    assert pending == {"algan.x:k2", "algan.x:k4"}
    shadowed = {job.spec["kernel"] for job in kp.pending_jobs(other)}
    assert shadowed == {"algan.x:k3", "algan.x:k4", "algan.x:k5", "algan.x:k1"}


def test_a_small_batch_is_left_to_the_render():
    context = kp.current_context()
    tiny = [kp._Job(_spec(n), context, 1.0) for n in range(3)]
    assert not kp.worthwhile(tiny)
    assert kp.worthwhile(tiny + [kp._Job(_spec(9), context, 30.0)])


def test_the_job_count_setting_wins_and_zero_turns_the_pool_off(monkeypatch):
    monkeypatch.setenv("ALGAN_PRECOMPILE_JOBS", "2")
    assert kp.worker_budget(10) == 2
    assert kp.worker_budget(1) == 1
    monkeypatch.setenv("ALGAN_PRECOMPILE_JOBS", "0")
    assert kp.worker_budget(10) == 0
    assert "ALGAN_PRECOMPILE_JOBS=0" in kp.skipped_reason()


def test_a_test_runner_never_starts_a_pool_at_import(monkeypatch):
    started = []
    monkeypatch.setattr(
        kp, "start_in_background", lambda reason: started.append(reason)
    )
    monkeypatch.delenv("ALGAN_PRECOMPILE_JOBS", raising=False)
    kp.start_at_import()
    assert started == []


# --- scoping and the first touch -------------------------------------------------


def test_an_implicit_pool_is_scoped_to_the_script(manifest, monkeypatch):
    monkeypatch.setattr(kp, "environment_stamp", lambda: "now")
    context = kp.current_context()
    kp._note_entry(_spec(1), context, seconds=20.0, script="/a.py")
    kp._note_entry(_spec(2), context, seconds=20.0, script="/b.py")
    kp.flush_manifest()
    kp._note_entry(_spec(2), context, script="/a.py")
    kp.flush_manifest()
    rows = {
        row["spec"]["kernel"]: row for row in kp.read_manifest()["entries"].values()
    }
    assert rows["algan.x:k2"]["scripts"] == ["/a.py", "/b.py"]
    mine = kp.pending_jobs(context, include_builtin=False, script="/a.py")
    assert {job.spec["kernel"] for job in mine} == {"algan.x:k1", "algan.x:k2"}
    theirs = kp.pending_jobs(context, include_builtin=False, script="/b.py")
    assert {job.spec["kernel"] for job in theirs} == {"algan.x:k2"}


def test_a_script_that_sets_its_settings_is_judged_by_the_context_it_rendered_under(
    manifest, monkeypatch
):
    """Rows recorded after the script changed ``SETTINGS`` are found from the import."""
    monkeypatch.setattr(kp, "environment_stamp", lambda: "now")
    imported = kp.current_context()
    shadows = {**imported, "raytracing": {**imported["raytracing"], "shadows": True}}
    kp._note_entry(_spec(1), shadows, seconds=20.0, used=True, script="/s.py")
    kp.flush_manifest()
    data = kp.read_manifest()
    guess, rows_from = kp._script_context(data, "/s.py", imported)
    assert kp.context_id(guess) == kp.context_id(shadows)
    assert rows_from == kp.context_id(shadows)
    # A script with rows under the import-time context keeps it.
    kp._note_entry(_spec(2), imported, seconds=20.0, used=True, script="/s.py")
    kp.flush_manifest()
    data = kp.read_manifest()
    assert kp._script_context(data, "/s.py", imported) == (imported, None)
    # A script with no rows at all keeps it too.
    assert kp._script_context(data, "/other.py", imported) == (imported, None)


def _old_schema(context, **raytracing):
    """``context`` as an earlier version recorded it: one field since removed."""
    return {
        **context,
        "raytracing": {
            **context["raytracing"],
            "a_field_an_update_removed": 1,
            **raytracing,
        },
    }


def test_a_guess_never_uses_a_context_from_another_settings_schema(
    manifest, monkeypatch
):
    """After an update that changed SETTINGS' fields, a stored snapshot would make
    every worker's ``apply_context`` raise; it is not used (no overrides recorded).
    """
    monkeypatch.setattr(kp, "environment_stamp", lambda: "after")
    monkeypatch.setattr(kp, "_BASELINE", None)
    now = kp.current_context()
    old = _old_schema(now, shadows=True)
    for k in (1, 2):
        kp._note_entry(
            _spec(k), old, seconds=20.0, confirmed="before", used=True, script="/s.py"
        )
    kp.flush_manifest()
    data = kp.read_manifest()
    assert data["overrides"] == {}
    assert kp._script_context(data, "/s.py", now) == (now, None)
    assert (
        kp.pending_jobs(now, include_builtin=False, manifest=data, script="/s.py") == []
    )


def test_a_guess_is_rebuilt_from_current_defaults_and_what_the_script_changed(
    manifest, monkeypatch
):
    monkeypatch.setattr(kp, "environment_stamp", lambda: "after")
    now = kp.current_context()
    # Recorded by an earlier version: its baseline had the removed field too,
    # and the script turned shadows on.
    old_baseline = _old_schema(now)
    old = _old_schema(now, shadows=not now["raytracing"]["shadows"])
    monkeypatch.setattr(kp, "_BASELINE", old_baseline)
    for k in (1, 2):
        kp._note_entry(
            _spec(k), old, seconds=20.0, confirmed="before", used=True, script="/s.py"
        )
    kp.flush_manifest()
    data = kp.read_manifest()
    assert data["overrides"][kp.context_id(old)] == {
        "raytracing": {"shadows": old["raytracing"]["shadows"]},
        "computing": {},
    }
    guess, rows_from = kp._script_context(data, "/s.py", now)
    assert rows_from == kp.context_id(old)
    assert set(guess["raytracing"]) == set(now["raytracing"])
    assert guess["raytracing"]["shadows"] == old["raytracing"]["shadows"]
    assert {k: v for k, v in guess["raytracing"].items() if k != "shadows"} == {
        k: v for k, v in now["raytracing"].items() if k != "shadows"
    }
    # A worker could apply it, and the script's specs are compiled under it.
    try:
        kp.apply_context(guess)
    finally:
        kp.apply_context(now)
    jobs = kp.pending_jobs(
        guess, include_builtin=False, manifest=data, script="/s.py", rows_from=rows_from
    )
    assert {job.spec["kernel"] for job in jobs} == {"algan.x:k1", "algan.x:k2"}
    assert {job.cid for job in jobs} == {kp.context_id(guess)}


class _RecordingPool:
    def __init__(self, jobs, workers, reason=""):
        self.jobs = jobs
        self.workers = workers
        self.reason = reason
        self.active = True
        self.listeners = []
        self.terminated = False

    def start(self):
        return self

    def terminate(self):
        self.terminated = True
        self.active = False


@pytest.fixture
def recorded_pools(manifest, monkeypatch):
    """`_start_pool` with workers that are only recorded, never spawned."""
    pools = []

    def make(jobs, workers, reason=""):
        pools.append(_RecordingPool(jobs, workers, reason))
        return pools[-1]

    monkeypatch.setattr(kp, "PrecompilePool", make)
    monkeypatch.setattr(kp, "worker_budget", lambda count: min(count, 7))
    monkeypatch.setattr(kp, "environment_stamp", lambda: "now")
    monkeypatch.setattr(kp, "_POOL", None)
    monkeypatch.setattr(kp, "_POOL_CONTEXT_ID", None)
    monkeypatch.setattr(kp, "_SETTLED", set())
    from algan.rendering import kernel_progress

    monkeypatch.setattr(kernel_progress, "pool_started", lambda *a, **k: None)
    kp.BUILTIN_MANIFEST.write_text(
        json.dumps(
            {
                "schema": kp._MANIFEST_SCHEMA,
                "entries": {
                    str(n): {"spec": _spec(n), "seconds": 30.0, "requires": {}}
                    for n in (10, 11, 12)
                },
            }
        ),
        encoding="utf-8",
    )
    return pools


def test_a_first_run_precompiles_the_builtin_list(recorded_pools, monkeypatch):
    monkeypatch.delenv("ALGAN_FIRST_RUN_POOL", raising=False)
    # Independent of the torch warm-up's switch.
    monkeypatch.setenv("ALGAN_STARTUP_WARMUP", "0")
    pool = kp._start_pool(kp.current_context(), "test", "/new.py")
    assert pool is recorded_pools[0]
    assert {job.spec["kernel"] for job in pool.jobs} == {
        "algan.x:k10",
        "algan.x:k11",
        "algan.x:k12",
    }
    assert "built-in" in pool.reason


def test_the_builtin_list_needs_an_empty_manifest_enough_workers_and_the_switch(
    recorded_pools, monkeypatch
):
    context = kp.current_context()
    monkeypatch.setenv("ALGAN_FIRST_RUN_POOL", "0")
    assert kp._start_pool(context, "test", "/new.py") is None
    monkeypatch.delenv("ALGAN_FIRST_RUN_POOL")
    monkeypatch.setattr(kp, "_SETTLED", set())
    monkeypatch.setattr(kp, "worker_budget", lambda count: min(count, 3))
    assert kp._start_pool(context, "test", "/new.py") is None
    # Another script's record means the cache is not new: its own list, or
    # nothing, rather than variants it may never use.
    monkeypatch.setattr(kp, "_SETTLED", set())
    monkeypatch.setattr(kp, "worker_budget", lambda count: min(count, 7))
    kp._note_entry(_spec(1), context, confirmed="now", used=True, script="/old.py")
    kp.flush_manifest()
    assert kp._start_pool(context, "test", "/new.py") is None
    assert recorded_pools == []


def test_settings_changed_after_import_replace_the_pool_at_the_first_kernel(
    recorded_pools, monkeypatch
):
    """The import-time pool was for the defaults; the kernel is traced under others."""
    monkeypatch.delenv("ALGAN_PRECOMPILE_JOBS", raising=False)
    monkeypatch.setattr(kp, "current_script", lambda: "/s.py")
    stale = _RecordingPool([], 1)
    monkeypatch.setattr(kp, "_POOL", stale)
    monkeypatch.setattr(kp, "_POOL_CONTEXT_ID", "the-import-time-context")
    # An empty manifest: this late, the built-in list is not worth a pool.
    assert kp._pool_for_current_settings("test") is None
    assert stale.terminated
    assert recorded_pools == []
    # The script's own record under the current settings is.
    monkeypatch.setattr(kp, "_SETTLED", set())
    context = kp.current_context()
    for k in (1, 2):
        kp._note_entry(_spec(k), context, seconds=20.0, used=True, script="/s.py")
    kp.flush_manifest()
    pool = kp._pool_for_current_settings("test")
    assert pool is recorded_pools[0]
    assert {job.spec["kernel"] for job in pool.jobs} == {"algan.x:k1", "algan.x:k2"}
    current = kp.context_id(context)
    assert current == kp._POOL_CONTEXT_ID
    # Asked again under the same settings, it is the same pool.
    assert kp._pool_for_current_settings("test") is pool


def test_a_worker_never_starts_a_pool_of_its_own(recorded_pools, monkeypatch):
    monkeypatch.setenv("ALGAN_PRECOMPILE_WORKER", "1")
    monkeypatch.setattr(kp, "current_script", lambda: "/worker.py")
    assert kp._pool_for_current_settings("test") is None
    assert recorded_pools == []


def test_stopping_a_worker_kills_the_interpreter_its_launcher_started(tmp_path):
    """A venv's ``python.exe`` on Windows is a launcher; the worker is its child."""
    psutil = pytest.importorskip("psutil")
    marker = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(120)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", code])
    try:
        for _ in range(200):
            if marker.exists() and marker.read_text():
                break
            threading.Event().wait(0.05)
        assert marker.read_text()
        tree = psutil.Process(parent.pid).children(recursive=True)
        assert tree
        kp._kill_tree(parent)
        parent.wait(timeout=10)
        _, alive = psutil.wait_procs(tree, timeout=10)
        assert alive == []
    finally:
        with contextlib.suppress(Exception):
            parent.kill()


def test_a_reaped_process_is_never_used_to_find_children(monkeypatch):
    """Its PID may already belong to an unrelated process."""
    psutil = pytest.importorskip("psutil")
    looked_up = []
    monkeypatch.setattr(psutil, "Process", lambda pid: looked_up.append(pid))

    class Reaped:
        pid = os.getpid()

        def poll(self):
            return 0

        def kill(self):
            raise AssertionError("killed a reaped process")

    kp._kill_tree(Reaped())
    assert looked_up == []


def test_two_deciders_never_start_two_pools(recorded_pools, monkeypatch):
    """A pool assigned but not yet started is not ``active``; deciding is serialized."""
    import time as _time

    class SlowStart(_RecordingPool):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.active = False

        def start(self):
            _time.sleep(0.2)
            self.active = True
            return self

    def make(jobs, workers, reason=""):
        recorded_pools.append(SlowStart(jobs, workers, reason))
        return recorded_pools[-1]

    monkeypatch.setattr(kp, "PrecompilePool", make)
    context = kp.current_context()
    threads = [
        threading.Thread(target=kp._start_pool, args=(context, "test", f"/{n}.py"))
        for n in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(recorded_pools) == 1


_SCRIPTED_WORKER = r"""
import json, os, sys, time
PREFIX = "@@algan-precompile@@ "
d = os.environ["FAKE_DIR"]
init_s = float(os.environ.get("FAKE_INIT_S", "0"))
compile_s = float(os.environ.get("FAKE_COMPILE_S", "0"))
dump_s = float(os.environ.get("FAKE_DUMP_S", "0"))

def emit(**event):
    sys.stdout.write(PREFIX + json.dumps(event) + "\n")
    sys.stdout.flush()

def mark(name):
    open(os.path.join(d, name), "w").close()

def enter(section, eid):
    emit(event="enter", id=eid, section=section)
    line = sys.stdin.readline()
    answer = json.loads(line).get("op") if line else None
    if answer not in ("go", "finish"):
        mark("refused_" + section)
        sys.exit(0)

emit(event="ready", pid=os.getpid())
applied = False
while True:
    line = sys.stdin.readline()
    if not line:
        break
    op = json.loads(line)
    if op.get("op") in ("quit", "finish"):
        break
    if op.get("op") != "compile":
        continue
    eid = op["id"]
    if not applied:
        enter("init", eid)
        time.sleep(init_s)
        applied = True
    emit(event="started", id=eid, fast_key="k")
    mark("compiling")
    time.sleep(compile_s)
    enter("dump", eid)
    lock = os.path.join(d, "qdcache.lock")
    open(lock, "w").close()
    time.sleep(dump_s)
    os.remove(lock)
    mark("dumped")
    emit(event="done", id=eid, status="compiled", seconds=compile_s)
"""


@pytest.fixture
def scripted_pool(tmp_path, monkeypatch):
    """A one-job pool whose worker is a script that speaks the protocol."""
    script = tmp_path / "worker.py"
    script.write_text(_SCRIPTED_WORKER, encoding="utf-8")
    monkeypatch.setattr(
        kp.PrecompilePool, "worker_command", (sys.executable, str(script))
    )
    monkeypatch.setattr(kp, "_note_entry", lambda *a, **k: None)
    monkeypatch.setattr(kp, "flush_manifest", lambda *a, **k: None)
    monkeypatch.setattr(kp, "_open_log", lambda: subprocess.DEVNULL)

    def make(**timings):
        env = {**os.environ, "FAKE_DIR": str(tmp_path)}
        env.update({f"FAKE_{k.upper()}": str(v) for k, v in timings.items()})
        job = kp._Job(_spec(1), kp.current_context(), 10.0)
        pool = kp.PrecompilePool([job], 1, env=env, reason="test")
        return pool.start()

    return make


def _wait_for(predicate, timeout=20.0):
    deadline = threading.Event()
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return True
        deadline.wait(0.02)
    return predicate()


def test_stopping_a_pool_waits_for_a_worker_inside_a_dump(scripted_pool, tmp_path):
    """Killed mid-dump, a worker would leave the cache's lock file behind."""
    import time as _time

    pool = scripted_pool(compile_s=0.05, dump_s=1.0)
    assert _wait_for(lambda: pool._workers and pool._workers[0].section == "dump")
    began = _time.monotonic()
    pool.terminate(grace=10)
    assert _time.monotonic() - began >= 0.3
    assert (tmp_path / "dumped").exists()
    assert not (tmp_path / "qdcache.lock").exists()
    assert pool.wait(10)


def test_stopping_a_pool_kills_a_compiling_worker_at_once(scripted_pool, tmp_path):
    import time as _time

    pool = scripted_pool(compile_s=60)
    assert _wait_for(lambda: (tmp_path / "compiling").exists())
    assert _wait_for(lambda: pool._workers[0].section is None)
    began = _time.monotonic()
    pool.terminate(grace=10)
    assert _time.monotonic() - began < 5
    assert pool.wait(10)
    assert not (tmp_path / "dumped").exists()


def test_a_worker_still_inside_a_section_at_the_deadline_is_refused_the_next(
    scripted_pool, tmp_path
):
    """Left alive past the grace period, it may not take the lock again."""
    pool = scripted_pool(init_s=1.0, compile_s=0.05, dump_s=0.05)
    assert _wait_for(lambda: pool._workers and pool._workers[0].section == "init")
    pool.terminate(grace=0.1)
    assert pool.wait(20)
    assert (tmp_path / "refused_dump").exists()
    assert not (tmp_path / "dumped").exists()


def test_releasing_a_pool_lets_the_compile_in_hand_finish_and_dump(
    scripted_pool, tmp_path
):
    pool = scripted_pool(compile_s=0.5, dump_s=0.05)
    assert _wait_for(lambda: (tmp_path / "compiling").exists())
    pool.release()
    assert pool.wait(20)
    assert (tmp_path / "dumped").exists()


def test_a_reader_that_fails_never_leaves_its_worker_waiting(
    scripted_pool, tmp_path, monkeypatch
):
    """An exception in the reader must not hang ``wait`` (and a render with it)."""
    original = kp.PrecompilePool._emit
    exploded = []

    def exploding(self, events):
        if not exploded and any(event[0] == "started" for event in events):
            exploded.append(True)
            raise RuntimeError("the reader broke")
        return original(self, events)

    monkeypatch.setattr(kp.PrecompilePool, "_emit", exploding)
    pool = scripted_pool(compile_s=0.2, dump_s=0.05)
    assert pool.wait(20)
    assert exploded
    # Its stdin closed with the reader, the worker's request to dump met
    # "stop" rather than a parent that would never answer.
    assert (tmp_path / "refused_dump").exists()
    assert not (tmp_path / "dumped").exists()


_TOUCH_CHILD = r"""
import os, sys, time
import algan  # noqa: F401
from algan.rendering import kernel_precompile as kp
from algan.rendering import taichi_runtime
from algan.settings._startup import _TAICHI_CACHE_DIRECTORY

spec = [
    s for s, _sec, req in kp.read_builtin_specs(with_requirements=True)
    if not req and s["kernel"].endswith("tonemap_to_u8")
][0]
taichi_runtime.ensure_taichi_for_render()
if sys.argv[1] == "populate":
    kp.compile_spec(spec)
    sys.exit(0)
lock = os.path.join(
    str(_TAICHI_CACHE_DIRECTORY), "kernel_compilation_manager", "qdcache.lock"
)
open(lock, "w").close()
sys.stderr.write("@@TOUCH-BEGIN\n"); sys.stderr.flush()
began = time.perf_counter()
kp._touch_cache_index()
took = time.perf_counter() - began
sys.stderr.write("@@TOUCH-END\n"); sys.stderr.flush()
os.remove(lock)
print("TOOK", took)
"""


@quadrants_only
def test_the_cache_index_is_read_inside_the_touch(tmp_path):
    """Pins what the worker's lock protocol rests on.

    A worker counts as killable while it compiles, so the program's one read
    of the offline cache's index -- taken under the metadata lock -- must
    happen inside :func:`kp._touch_cache_index`, which runs in a granted
    section. With the lock held by someone else, the compiler's retry (five
    50 ms attempts, then a warning) must therefore happen inside the touch: if
    a Quadrants update moves the read elsewhere, this fails rather than
    silently reopening a window to kill a worker holding the lock.
    """
    env = {
        **os.environ,
        "ALGAN_USE_DAEMON": "0",
        "ALGAN_AUTO_DAEMON": "0",
        "ALGAN_PRECOMPILE_JOBS": "0",
        "ALGAN_CACHE_DIR": str(tmp_path / "cache"),
        "ALGAN_HOME": str(tmp_path / "home"),
        "ALGAN_RENDER_DEVICE": "cpu",
        "CUDA_VISIBLE_DEVICES": "-1",
    }
    for mode in ("populate", "locked"):
        result = subprocess.run(
            [sys.executable, "-c", _TOUCH_CHILD, mode],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr[-3000:]
    stderr = result.stderr
    inside = stderr[stderr.index("@@TOUCH-BEGIN") : stderr.index("@@TOUCH-END")]
    assert "qdcache.lock" in inside, stderr[-3000:]
    took = float(
        next(x for x in result.stdout.splitlines() if x.startswith("TOOK"))[5:]
    )
    assert took >= 0.15


def test_scripts_are_remembered_most_recent_first_and_bounded():
    merged = kp._merge_scripts(["/new.py"], [f"/{n}.py" for n in range(10)])
    assert merged[0] == "/new.py"
    assert len(merged) == kp._SCRIPTS_PER_ROW


class _FakePool:
    def __init__(self):
        self.active = True
        self.waited = False
        self.terminated = False
        self.listeners = []
        self.jobs = []

    def wait(self, timeout=None):
        self.waited = True
        self.active = False
        return True

    def counts(self):
        return 0, 2

    def terminate(self):
        self.terminated = True
        self.active = False


@pytest.fixture
def untouched(monkeypatch):
    monkeypatch.setattr(kp, "_TOUCHED_PROGRAM", None)
    monkeypatch.setattr(kp, "_STARTER", None)
    fake_program = object()
    import algan.taichi_compat as compat

    monkeypatch.setattr(compat, "program", lambda: fake_program)
    return fake_program


def test_the_first_materialization_waits_for_a_pool_under_the_same_settings(
    untouched, monkeypatch
):
    pool = _FakePool()
    monkeypatch.setattr(kp, "_POOL", pool)
    monkeypatch.setattr(kp, "_POOL_CONTEXT_ID", kp.context_id(kp.current_context()))
    kp.before_first_materialization()
    assert pool.waited
    # Only the first: the index is read once, so later waits would buy nothing.
    again = _FakePool()
    monkeypatch.setattr(kp, "_POOL", again)
    kp.before_first_materialization()
    assert not again.waited


def test_a_pool_for_other_settings_is_stopped_not_waited_for(untouched, monkeypatch):
    pool = _FakePool()
    monkeypatch.setattr(kp, "_POOL", pool)
    monkeypatch.setattr(kp, "_POOL_CONTEXT_ID", "some-other-context")
    kp.before_first_materialization()
    assert pool.terminated
    assert not pool.waited


def test_a_touched_program_starts_no_pool(untouched, monkeypatch):
    monkeypatch.setattr(kp, "_TOUCHED_PROGRAM", id(untouched))
    monkeypatch.delenv("ALGAN_PRECOMPILE_JOBS", raising=False)
    started = []
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: started.append(a))
    kp.start_in_background("test", "/some/script.py")
    assert started == []


# --- end to end -------------------------------------------------------------------

_CHILD = r"""
import json, os, sys, time
import torch
import algan  # noqa: F401
from algan.rendering import kernel_precompile as kp
from algan.rendering.taichi_runtime import ensure_taichi_for_render
from algan.rendering.post_processing.tonemap_kernels_taichi import tonemap_to_u8
from algan.utils import taichi_source_key as sk

ensure_taichi_for_render()
torch.manual_seed(0)
frame = torch.rand((1, 8, 12, 4))
out = torch.zeros((1, 8, 12, 3), dtype=torch.uint8)
args = (frame, out, 1, 1.0, 0, 1, 0, 0.49)
mode = sys.argv[1]
report = {}
if mode == "touched":
    # This process uses the cache before the worker writes: the compiler has
    # read the index by then, and the worker's artifact must stay invisible.
    from algan.rendering.raytracing.sheet_compact_taichi import opaque_prefix_keep
    u8 = torch.zeros((4,), dtype=torch.uint8)
    i64 = torch.zeros((4,), dtype=torch.int64)
    opaque_prefix_keep(u8, i64, i64, 4, u8)
if mode in ("pool", "touched"):
    spec = kp.encode_spec(tonemap_to_u8._primal, args)
    context = kp.current_context()
    pool = kp.PrecompilePool([kp._Job(spec, context, 30.0)], 1, reason="test")
    kp._POOL, kp._POOL_CONTEXT_ID = pool, kp.context_id(context)
    pool.start()
    if mode == "touched":
        pool.wait(120)
# In "pool" mode this is the process's first kernel: it waits for the pool.
tonemap_to_u8(*args)
if mode in ("pool", "touched"):
    report["job"] = [pool.jobs[0].state, pool.jobs[0].status]
    pool.wait(60)
report.update(hits=sk.STATS["hits"], misses=sk.STATS["misses"], out=out.flatten().tolist())
print("REPORT " + json.dumps(report))
"""


def _run(mode, tmp_path, name):
    env = {
        **os.environ,
        "ALGAN_USE_DAEMON": "0",
        "ALGAN_AUTO_DAEMON": "0",
        "ALGAN_CACHE_DIR": str(tmp_path / name / "cache"),
        "ALGAN_HOME": str(tmp_path / name / "home"),
    }
    env.pop("ALGAN_PRECOMPILE_JOBS", None)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, mode], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-3000:]
    line = next(x for x in result.stdout.splitlines() if x.startswith("REPORT "))
    return {**json.loads(line[len("REPORT ") :]), "stderr": result.stderr}


@quadrants_only
def test_a_worker_compiled_kernel_is_an_index_hit_with_identical_output(tmp_path):
    """One process runs a worker for the kernel it is about to launch; another compiles it itself.

    The pooled process must end with a source-key *hit* for the launch (it
    waited for the worker, or found its artifact already on disk) and the
    same bytes out as the process that compiled the kernel locally.
    """
    local = _run("local", tmp_path, "local")
    assert (local["hits"], local["misses"]) == (0, 1)
    # The progress account for a real cold compile: the device is named for
    # what it is (the old notice said "GPU kernels" on the CPU too), and the
    # kernel gets a numbered line.
    from algan.settings._startup import render_device

    device = "the CPU" if render_device().type == "cpu" else "the GPU"
    assert f"Preparing render kernels for {device}" in local["stderr"]
    assert "kernel 1: tonemap_to_u8 compiled in" in local["stderr"]

    pooled = _run("pool", tmp_path, "pooled")
    assert pooled["job"] == ["done", "compiled"]
    assert (pooled["hits"], pooled["misses"]) == (1, 0)
    assert pooled["out"] == local["out"]
    assert (
        "waiting for 1 of 1 kernels compiling in background workers"
        in (pooled["stderr"])
    )


@quadrants_only
def test_a_process_that_used_the_cache_does_not_see_later_artifacts(tmp_path):
    """The compiler behaviour the first-touch wait exists for, pinned.

    Once a program has used the kernel cache, an artifact another process
    writes afterwards is invisible to it (Quadrants 1.3 reads the index once).
    If this ever starts to fail, a newer compiler refreshes its index, and the
    precompile pool could hand kernels to a render mid-flight instead of
    making its first kernel wait.
    """
    touched = _run("touched", tmp_path, "touched")
    assert touched["job"] == ["done", "compiled"]
    # tonemap_to_u8: the worker compiled and indexed it, and still a miss.
    assert touched["hits"] == 0
    assert touched["misses"] == 2


# --- what an MPS render binds ---------------------------------------------------


def test_an_imported_ndarray_is_recorded_as_the_tensor_it_imports():
    """An MPS render's arguments reach the recorder as compiler ndarrays.

    The zero-copy wrapper imports every arena tensor before the call reaches
    ``timed_materialize``, and a spec that refused them left every arena
    kernel out of the manifest on a Mac. Described as the tensor would have
    been, the spec is the one a CPU or CUDA render records, so a worker's
    placeholders build the same specialization everywhere.
    """
    from algan.rendering import taichi_runtime

    taichi_runtime.ensure_taichi_for_render()
    kernel = _tonemap()
    tensors = _tonemap_args()
    imported = (
        ti.ndarray(ti.f32, shape=tuple(tensors[0].shape)),
        ti.ndarray(ti.u8, shape=tuple(tensors[1].shape)),
        *tensors[2:],
    )
    assert kp.encode_spec(kernel, imported) == kp.encode_spec(kernel, tensors)
    vector = ti.Vector.ndarray(4, ti.f16, shape=(3, 5))
    assert kp._encode_compiler_ndarray(vector) == ["A", "float16", 3, [4]]
    assert kp._encode_compiler_ndarray(ti.ndarray(ti.f64, shape=(2,))) is None
    assert kp._encode_compiler_ndarray(object()) is None


def test_a_worker_binds_its_placeholders_the_way_a_launch_would(monkeypatch):
    """``compile_spec`` hands its placeholders through ``launch_arguments``.

    On MPS that is what imports them, so the worker's source key describes the
    imported ndarrays the render's does; off MPS it is the identity.
    """
    from algan.rendering import mps_zero_copy

    seen = []

    def recording(kernel, args):
        seen.append((kernel, args))
        return tuple(args)

    monkeypatch.setattr(mps_zero_copy, "launch_arguments", recording)
    kernel = _tonemap()
    spec = kp.encode_spec(kernel, _tonemap_args())
    kp.compile_spec(spec)
    assert len(seen) == 1
    assert seen[0][0] is kernel
    assert kp.encode_spec(kernel, seen[0][1]) == spec


@pytest.mark.parametrize(
    ("arch", "flag", "outlives"),
    [
        ("cuda", None, True),
        ("cpu", True, True),
        ("x64", True, True),
        ("cpu", False, False),
        ("cpu", None, False),
        ("metal", True, False),
    ],
)
def test_a_worker_prepares_only_where_the_result_reaches_the_disk(arch, flag, outlives):
    """CUDA's PTX, and the CPU's machine code under Quadrants patch 0012.

    Without 0012's object cache a CPU worker's JIT would die with the worker.
    """
    cfg = types.SimpleNamespace(arch=getattr(ti, arch))
    if flag is not None:
        cfg.offline_cache_cpu_object_by_content_hash = flag
    assert kp._prepared_artifact_outlives_worker(cfg, ti) is outlives
