"""Kernels a daemon run compiled outlive the daemon.

A script writes its newly compiled kernels to the offline cache at exit
(``taichi_runtime.flush_kernel_cache``). A daemon executes scripts through
``runpy`` and never exits between them, so until it wrote them after each run
everything it compiled died with it: on 2026-10-23 a daemon killed by a CUDA
error took ~21 kernels along, and the next process spent ~95 s recompiling
them.

Each arm runs a real daemon loop (``algan.daemon.main``) in its own process on
the CPU arch, against a cache directory of its own. Its first run compiles one
kernel; its second dies outright (``os._exit``: no cleanup, no ``atexit``), the
way a crashed or killed daemon does. A fresh process then launches the same
kernel, and whether it had to compile it is the verdict.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

KERNELS = """
from algan.taichi_compat import ti


@ti.kernel
def persisted(out: ti.types.ndarray(), salt: ti.template()):
    out[0] += salt
"""

DAEMON = """
import os
import sys

import torch

from algan import daemon as d
from algan.rendering import taichi_runtime as rt

mode, scene = sys.argv[1], sys.argv[2]
events = None
runs = 0


def stdin(queue):
    global events
    events = queue


d._start_stdin = stdin
if mode == "control":
    # What a daemon did before: nothing between runs.
    d._persist_new_kernels = lambda: None


def run(path, run_name):
    global runs
    runs += 1
    if runs == 1:
        import kernels

        kernels.persisted(torch.zeros(1), 7)
        assert rt._UNSAVED_KERNELS
        if mode == "dead":
            # What a CUDA context an error left unusable does to the
            # teardown: Program::finalize raises at its first synchronize,
            # before it has written anything.
            def dead_reset():
                raise RuntimeError("CUDA error: out of memory")

            rt.ti.reset = dead_reset
        events.put(("render", "the run that dies"))
        return
    print("second run started", flush=True)
    os._exit(17)


d.runpy.run_path = run
d.main([scene, "--no-serve"])
print("daemon loop returned", flush=True)
os._exit(18)
"""

FRESH = """
import json

import torch

from algan.rendering import taichi_runtime as rt
import kernels

out = torch.zeros(1)
kernels.persisted(out, 7)
print(json.dumps({
    "built": rt._BUILT_A_SPECIALIZATION,
    "compiled": rt._UNSAVED_KERNELS,
    "value": out.item(),
}))
"""


def _environment(tmp_path):
    env = {
        **os.environ,
        "ALGAN_HOME": str(tmp_path / "h"),
        "ALGAN_CACHE_DIR": str(tmp_path / "c"),
        "ALGAN_RENDER_DEVICE": "cpu",
        "ALGAN_USE_DAEMON": "0",
        "ALGAN_DAEMON_CHILD": "1",
        "ALGAN_PROGRESS": "none",
        "ALGAN_DAEMON_RELEASE_MEMORY": "0",
        "ALGAN_PRECOMPILE_JOBS": "0",
        "ALGAN_STARTUP_WARMUP": "0",
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]
        ),
    }
    env.pop("ALGAN_FLUSH_KERNEL_CACHE", None)
    return env


@pytest.mark.parametrize("mode", ["persist", "dead", "control"])
def test_a_daemon_runs_kernels_survive_its_death(tmp_path, mode):
    (tmp_path / "kernels.py").write_text(textwrap.dedent(KERNELS), encoding="utf-8")
    (tmp_path / "daemon_run.py").write_text(DAEMON, encoding="utf-8")
    (tmp_path / "fresh.py").write_text(FRESH, encoding="utf-8")
    scene = tmp_path / "scene.py"
    scene.write_text("# served by the test controller\n", encoding="utf-8")
    env = _environment(tmp_path)

    daemon = subprocess.run(
        [sys.executable, str(tmp_path / "daemon_run.py"), mode, str(scene)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    daemon_output = daemon.stdout + daemon.stderr
    if mode == "dead":
        # The write left the runtime unusable, so the daemon stood down
        # instead of serving another run on it.
        assert daemon.returncode == 18, daemon_output
        assert "failed to shut down" in daemon_output, daemon_output
        assert "runtime cleanup failed" in daemon_output, daemon_output
    else:
        assert daemon.returncode == 17, daemon_output
        assert "second run started" in daemon_output, daemon_output
    if mode == "persist":
        assert "saved this run's newly compiled kernels" in daemon_output

    fresh = subprocess.run(
        [sys.executable, str(tmp_path / "fresh.py")],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    verdict = json.loads(fresh.stdout.strip().splitlines()[-1])
    assert verdict["value"] == 7.0
    if mode == "control":
        # The source-key index entry survived, the kernel it names did not.
        assert verdict["compiled"] is True, verdict
    else:
        # Served from disk by the source-key index: no front end, no compile.
        assert verdict == {"built": False, "compiled": False, "value": 7.0}
