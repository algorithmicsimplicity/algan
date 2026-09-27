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


def test_every_builtin_spec_still_matches_a_live_kernel():
    """The built-in list against today's kernels (``scripts/generate_kernel_specs.py``).

    A stale entry is skipped by the workers, never an error, so this is the
    only place a kernel signature change that invalidates it is noticed.
    Regenerate the list when it fails.
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
args = (frame, out, 1, 1.0, 0, 1, 0)
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
