"""Prove patches 0010 and 0011 do what they claim, on a CUDA box.

`0010-cuda-ptx-cache-by-content-hash.patch` keys CUDA's PTX cache by the
SHA-256 each `.qdc` already stores (its *content hash*) under
`qd.init(offline_cache_ptx_by_content_hash=True)`, and defers parsing a cached
kernel's LLVM IR until a PTX miss needs the module. `0011-prepare-kernel-
without-launch.patch` adds `Program.prepare_kernel` (a first launch's work
without the launch), writes the PTX cache in `dump_cache_data_to_disk`, and
lets a PTX lookup find a file another process wrote after this one started.

Every claim is visible in the compiler's own debug log, so each step is a
child process run at `log_level="debug"` (or `"trace"`) whose output is
counted:

    served     "PTX for content hash ... served from the PTX cache"   (0010)
    stored     "PTX for content hash ... stored in the PTX cache"     (0010)
    parsed     "Parsing the deferred LLVM IR of a cached kernel"      (0010)
    ir_hit     "Loaded PTX from cache for module"   (trace: the IR-keyed path)
    late       "... on disk, outside the index read at start-up"      (0011)

The sequence, one fresh process per step, against one temporary cache:

    1 populate  on   compile + launch            stored >= K, *.qdc and C*.ptx on disk
    2 warm      on   fresh process, launch       served >= K, parsed == stored == 0
    3 off       off  fresh process, launch       served == parsed == 0 (the gate)
    4 migrate   on   C*.ptx deleted, T*.ptx kept parsed >= K, stored >= K, and
                                                 ir_hit beyond warm2's >= K
    5 warm2     on   fresh process, launch       served >= K, parsed == 0

Steps 4 and 5 run at `"trace"`, because the IR-key hit is a `QD_TRACE` line; the
runtime module takes the IR key in every process, so warm2's count is the
baseline migrate's is read against. A build whose trace lines are compiled out
leaves that one claim unobserved and says so (a `NOTE`, not a failure).

and, each against a cache of its own, for 0011:

    6 prepare   on   compile, prepare_kernel, dump, os._exit -- never launched,
                     never finalized: C*.ptx and *.qdc on disk, and the driver's
                     compute cache (CUDA_CACHE_PATH) non-empty
      warm      on   served >= K, parsed == 0
    7 late      on   init, wait; a `prepare` process fills the cache; then
                     launch: late >= K, served >= K, parsed == 0

K is the number of probe kernels (two: one plain, one `fastcache=True`, so both
`KernelCompilationManager::load_or_compile` and `load_fast_cache` -- the path
Algan's source-key index takes -- are covered). Every launch checks its output
against numpy. The first-launch time of each step is printed too; it is
informational, and the speed claims are Algan's A/B to make, not this
script's.

    python verify_ptx_cache.py run result.json      # the CUDA sequence
    python verify_ptx_cache.py binding result.json  # the CPU-only gate (CI)

`binding` needs no GPU: the config field exists, defaults off and reads back
on, `Program.prepare_kernel` exists, and on the CPU arch preparing a kernel
and then launching it gives the right answer. **No CUDA device is a skip, not
a failure**: `run` writes `{"skipped": true}` and exits 0, as
`verify_cuda_patches.py` does. The result goes to a file, never stdout, which
`qd.init()` prints its banner to.
"""

# ruff: noqa: I002 -- `from __future__ import annotations` would turn the
# kernels' runtime-evaluated annotations into strings; see
# verify_cuda_patches.py, which carries the same exemption for the same reason.
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

# Module-level, as the annotations resolve against this module's globals.
import quadrants as qd

N = 4096
SELF = os.path.abspath(__file__)

PATTERNS = {
    "served": r"PTX for content hash \S+ served from the PTX cache",
    "stored": r"PTX for content hash \S+ stored in the PTX cache",
    "parsed": r"Parsing the deferred LLVM IR of a cached kernel",
    "ir_hit": r"Loaded PTX from cache for module",
    "late": r"on disk, outside the index read at start-up",
}


@qd.kernel
def verify_ptx_probe_plain(
    n: int,
    a: float,
    x: qd.types.ndarray(),
    y: qd.types.ndarray(),
    out: qd.types.ndarray(),
):
    for i in range(n):
        out[i] = a * x[i] + y[i]


@qd.kernel(fastcache=True)
def verify_ptx_probe_fast(n: int, x: qd.types.ndarray(), out: qd.types.ndarray()):
    for i in range(n):
        out[i] = qd.sqrt(x[i]) + 1.0


def _inputs():
    x = np.linspace(0.0, 4.0, N, dtype=np.float32)
    y = np.full(N, 0.25, dtype=np.float32)
    return x, y


def _probes():
    """`(kernel, args, check)` for each probe: `check(out)` is the numpy answer."""
    x, y = _inputs()
    return [
        (
            verify_ptx_probe_plain,
            lambda out: (N, 1.5, x, y, out),
            1.5 * x + y,
        ),
        (
            verify_ptx_probe_fast,
            lambda out: (N, x, out),
            np.sqrt(x) + 1.0,
        ),
    ]


K = 2


def _init(arch, on, cache_dir, log_level):
    kwargs = {
        "arch": arch,
        "offline_cache": True,
        "offline_cache_file_path": cache_dir,
        "advanced_optimization": False,
        "fast_math": True,
        "log_level": log_level,
    }
    if on is not None:
        kwargs["offline_cache_ptx_by_content_hash"] = on
    qd.init(**kwargs)


def _launch_all():
    """Launch every probe once, checking its output; returns the seconds the first launches took."""
    started = time.perf_counter()
    bad = []
    for kernel, args, expected in _probes():
        out = np.zeros(N, dtype=np.float32)
        kernel(*args(out))
        qd.sync()
        if not np.allclose(out, expected, rtol=1e-4, atol=1e-5):
            bad.append(kernel.__name__ if hasattr(kernel, "__name__") else str(kernel))
    return time.perf_counter() - started, bad


def _compile_without_launch():
    """Each probe's CompiledKernelData, the way Algan's precompile worker gets it."""
    from quadrants.lang import impl

    prog = impl.get_runtime().prog
    data = []
    for kernel, args, _ in _probes():
        primal = kernel._primal
        # What Kernel.__call__ sets before ensure_compiled.
        primal.raise_on_templated_floats = impl.current_cfg().raise_on_templated_floats
        out = np.zeros(N, dtype=np.float32)
        key = primal.ensure_compiled(*args(out))
        ckd = primal.compiled_kernel_data_by_key.get(key)
        if ckd is None:
            result = prog.compile_kernel(
                prog.config(), prog.get_device_caps(), primal.materialized_kernels[key]
            )
            ckd = result.compiled_kernel_data
            primal.compiled_kernel_data_by_key[key] = ckd
        data.append(ckd)
    return prog, data


# --- the child: one step, in a process of its own ----------------------------


def child(mode, cache_dir, flag_dir):
    on = mode != "off"
    # The IR-key hit is a QD_TRACE line; warm2 is migrate's baseline for it.
    log_level = "trace" if mode in ("migrate", "warm2") else "debug"
    try:
        _init(qd.cuda, on, cache_dir, log_level)
    except KeyError as exc:
        reason = f"qd.init refused the 0010 keyword ({exc}): this build does not carry the patch"
        print("@@RESULT " + json.dumps({"error": reason}))
        return
    from quadrants.lang import impl

    if str(impl.current_cfg().arch) != str(qd.cuda):
        print("@@RESULT " + json.dumps({"skipped": True, "reason": "no CUDA device"}))
        return
    result = {"mode": mode}
    if mode == "prepare":
        prog, data = _compile_without_launch()
        started = time.perf_counter()
        for ckd in data:
            prog.prepare_kernel(ckd)
        result["prepare_s"] = time.perf_counter() - started
        prog.dump_cache_data_to_disk()
        print("@@RESULT " + json.dumps(result), flush=True)
        sys.stdout.flush()
        os._exit(0)  # no finalize: the dump above has to have written everything
    if mode == "late":
        pathlib.Path(flag_dir, "ready").touch()
        deadline = time.time() + 600
        while not pathlib.Path(flag_dir, "go").exists():
            if time.time() > deadline:
                result["error"] = "never told to go"
                print("@@RESULT " + json.dumps(result), flush=True)
                return
            time.sleep(0.05)
    seconds, bad = _launch_all()
    result["first_launch_s"] = seconds
    result["wrong_output"] = bad
    print("@@RESULT " + json.dumps(result), flush=True)


# --- the parent ---------------------------------------------------------------


def _counts(text):
    return {name: len(re.findall(pattern, text)) for name, pattern in PATTERNS.items()}


def _step(mode, cache_dir, env, flag_dir="", background=False):
    command = [sys.executable, SELF, "--child", mode, cache_dir, flag_dir]
    if background:
        return subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    done = subprocess.run(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return _parse(mode, done.returncode, done.stdout)


def _parse(mode, returncode, text):
    lines = [x for x in text.splitlines() if x.startswith("@@RESULT ")]
    result = (
        json.loads(lines[-1][len("@@RESULT ") :])
        if lines
        else {"error": "no result line"}
    )
    result.update(_counts(text))
    result["returncode"] = returncode
    if returncode != 0 or "error" in result:
        result["log_tail"] = text[-4000:]
    print(
        f"[verify_ptx_cache] {mode:8s} {json.dumps({k: v for k, v in result.items() if k != 'log_tail'})}"
    )
    return result


def _ok(step):
    """The step ran, raised nothing, and every launch computed the right output."""
    return (
        step["returncode"] == 0 and "error" not in step and not step.get("wrong_output")
    )


def _files(cache_dir, pattern):
    return glob.glob(os.path.join(cache_dir, "**", pattern), recursive=True)


def run():
    try:
        from quadrants._lib import core as qd_core

        if not qd_core.with_cuda():
            return {"skipped": True, "reason": "no CUDA driver"}
    except Exception as exc:  # noqa: BLE001 -- the probe's absence is itself the answer
        return {"skipped": True, "reason": f"cannot probe for CUDA ({exc!r})"}
    root = tempfile.mkdtemp(prefix="qd_ptx_cache_")
    env = dict(os.environ)
    failures = []
    notes = []
    steps = {}

    def expect(condition, message):
        if not condition:
            failures.append(message)

    try:
        cache = os.path.join(root, "a")
        env["CUDA_CACHE_PATH"] = os.path.join(root, "a_cuda")
        steps["populate"] = s = _step("populate", cache, env)
        if s.get("skipped"):
            return {"skipped": True, "reason": s.get("reason")}
        if "error" in s:
            return {"skipped": False, "steps": steps, "failures": [s["error"]]}
        expect(_ok(s), "populate failed or computed wrong output")
        expect(
            s["stored"] >= K,
            f"populate stored {s['stored']} PTX by content hash, expected >= {K}",
        )
        expect(len(_files(cache, "C*.ptx")) >= K, "populate left no C*.ptx files")
        expect(len(_files(cache, "*.qdc")) >= K, "populate left no .qdc files")

        steps["warm"] = s = _step("warm", cache, env)
        expect(_ok(s), "warm failed or computed wrong output")
        expect(
            s["served"] >= K,
            f"warm served {s['served']} kernels by content hash, expected >= {K}",
        )
        expect(
            s["parsed"] == 0,
            f"warm parsed {s['parsed']} cached modules; with the PTX cached it must parse none",
        )
        expect(s["stored"] == 0, "warm stored PTX; everything should have been served")

        steps["off"] = s = _step("off", cache, env)
        expect(_ok(s), "off failed or computed wrong output")
        expect(
            s["served"] == 0 and s["parsed"] == 0,
            "the off arm used the content-hash path; the flag does not gate",
        )

        removed = _files(cache, "C*.ptx")
        for path in removed:
            os.remove(path)
        steps["migrate"] = s = _step("migrate", cache, env)
        expect(_ok(s), "migrate failed or computed wrong output")
        expect(
            s["parsed"] >= K,
            f"migrate parsed {s['parsed']} deferred modules, expected >= {K} (content keys removed)",
        )
        expect(
            s["stored"] >= K,
            "migrate did not re-key the IR-keyed PTX under content hashes",
        )

        steps["warm2"] = s = _step("warm2", cache, env)
        expect(
            s["served"] >= K and s["parsed"] == 0,
            "after migrating, a fresh process was not served by content hash",
        )
        # The runtime module always takes the IR key, so it hits in both steps;
        # what migrate has beyond warm2 is the kernels'.
        kernel_ir_hits = steps["migrate"]["ir_hit"] - s["ir_hit"]
        if steps["migrate"]["ir_hit"] == 0 and s["ir_hit"] == 0:
            notes.append(
                "no QD_TRACE output at log_level=trace: whether migrate re-keyed the off "
                "arm's PTX or recompiled it was not observed"
            )
        else:
            expect(
                kernel_ir_hits >= K,
                f"migrate found {kernel_ir_hits} kernels' PTX by IR key, expected >= {K} "
                "(the off arm wrote them)",
            )

        # 0011: prepare without launching, in a process that never finalizes.
        cache = os.path.join(root, "b")
        driver_cache = os.path.join(root, "b_cuda")
        env["CUDA_CACHE_PATH"] = driver_cache
        steps["prepare"] = s = _step("prepare", cache, env)
        expect(_ok(s), "prepare failed")
        expect(
            len(_files(cache, "C*.ptx")) >= K,
            "prepare_kernel + dump_cache_data_to_disk left no C*.ptx files",
        )
        expect(len(_files(cache, "*.qdc")) >= K, "prepare left no .qdc files")
        expect(
            os.path.isdir(driver_cache) and any(os.scandir(driver_cache)),
            "the driver's compute cache is empty after prepare_kernel; the module was not JIT-loaded",
        )
        steps["after_prepare"] = s = _step("warm", cache, env)
        expect(_ok(s), "the launch after prepare failed")
        expect(
            s["served"] >= K and s["parsed"] == 0,
            "the PTX prepare_kernel wrote did not serve a fresh process",
        )

        # 0011: PTX written after this process read its index.
        cache = os.path.join(root, "c")
        env["CUDA_CACHE_PATH"] = os.path.join(root, "c_cuda")
        flags = os.path.join(root, "c_flags")
        os.makedirs(flags)
        waiting = _step("late", cache, env, flags, background=True)
        deadline = time.time() + 300
        while (
            not os.path.exists(os.path.join(flags, "ready"))
            and waiting.poll() is None
            and time.time() < deadline
        ):
            time.sleep(0.05)
        steps["late_prepare"] = _step("prepare", cache, env)
        pathlib.Path(flags, "go").touch()
        text, _ = waiting.communicate(timeout=600)
        steps["late"] = s = _parse("late", waiting.returncode, text)
        expect(_ok(s), "late failed or computed wrong output")
        expect(
            s["late"] >= K,
            f"late found {s['late']} PTX files outside its start-up index, expected >= {K}",
        )
        expect(
            s["served"] >= K and s["parsed"] == 0,
            "late was not served by the PTX another process wrote",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return {"skipped": False, "steps": steps, "failures": failures, "notes": notes}


def binding():
    """The CPU gate: the field and the entry point exist, and a prepared kernel launches right."""
    cache = tempfile.mkdtemp(prefix="qd_ptx_binding_")
    failures = []
    try:
        qd.init(arch=qd.cpu, offline_cache=True, offline_cache_file_path=cache)
        from quadrants.lang import impl

        cfg = impl.current_cfg()
        if not hasattr(cfg, "offline_cache_ptx_by_content_hash"):
            return {
                "skipped": False,
                "failures": [
                    "no offline_cache_ptx_by_content_hash field: 0010 is not in this build"
                ],
            }
        if cfg.offline_cache_ptx_by_content_hash:
            failures.append("offline_cache_ptx_by_content_hash defaults on")
        _init(qd.cpu, True, cache, "warn")
        if not impl.current_cfg().offline_cache_ptx_by_content_hash:
            failures.append(
                "qd.init(offline_cache_ptx_by_content_hash=True) did not read back"
            )
        prog = impl.get_runtime().prog
        if not hasattr(prog, "prepare_kernel"):
            failures.append("Program has no prepare_kernel: 0011 is not in this build")
        else:
            prog, data = _compile_without_launch()
            for ckd in data:
                prog.prepare_kernel(ckd)
            _, bad = _launch_all()
            if bad:
                failures.append(
                    f"a prepared kernel launched with the wrong output: {bad}"
                )
        qd.reset()
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    return {"skipped": False, "failures": failures}


def main(argv):
    if argv[1] == "--child":
        child(argv[2], argv[3], argv[4] if len(argv) > 4 else "")
        return 0
    mode, out_path = argv[1], argv[2]
    result = run() if mode == "run" else binding()
    pathlib.Path(out_path).write_text(json.dumps(result, indent=1))
    if result.get("skipped"):
        print(f"SKIP: {result.get('reason')} -- patches 0010/0011 were not exercised")
        return 0
    if result["failures"]:
        print("FAIL")
        for failure in result["failures"]:
            print(f"  - {failure}")
        return 1
    for note in result.get("notes", ()):
        print(f"NOTE: {note}")
    if mode == "run":
        timing = {
            k: round(v.get("first_launch_s", 0.0), 3)
            for k, v in result["steps"].items()
            if "first_launch_s" in v
        }
        print(f"PASS: 0010 and 0011 hold on CUDA. First launches (s): {timing}")
    else:
        print(
            "PASS: the 0010 field and the 0011 entry point are present, and a prepared kernel launches correctly"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
