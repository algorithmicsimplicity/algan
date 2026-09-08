"""Capture native stacks and actual launch timings around the existing Mac probe.

Run on an ephemeral macOS runner. The supervisor never imports torch; it can
sample a child whose GIL or Metal driver has blocked. No renderer settings are
changed except the explicitly selected arena cap and existing compile disable.
"""
from __future__ import annotations

import faulthandler
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "algan_outputs" / "mac_diagnose"


def child():
    import runpy

    # Only this ephemeral diagnostic child imports the rewritten kernel.
    limit = int(os.environ.get("DIAG_DISPATCH_LIMIT", "0"))
    if limit:
        path = ROOT / "algan/rendering/raytracing/wavefront_kernels_taichi.py"
        source = path.read_text()
        begin = source.index("def wavefront_traverse_events_arena(")
        end = source.index("\n#: What ``wavefront_traverse_events``", begin)
        kernel = source[begin:end]
        assert kernel.count("ashp: ti.types.ndarray()):") == 1
        assert kernel.count("for i in range(num_active):") == 1
        kernel = kernel.replace("ashp: ti.types.ndarray()):", "ashp: ti.types.ndarray(), active_begin: int):")
        kernel = kernel.replace("for i in range(num_active):", "for i in range(active_begin, num_active):")
        path.write_text(source[:begin] + kernel + source[end:])

    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(ROOT / "benchmarks/_mps_warm_regression.py"), "1", "UHD", "1000"]
    ns = runpy.run_path(sys.argv[0], run_name="diagnostic_target")
    import torch
    from algan.taichi_compat import ti
    import algan.rendering.mps_zero_copy as zc

    import algan.rendering.raytracing.wavefront_kernels_taichi as wf
    kernel = wf.wavefront_traverse_events_arena
    traversal = {"calls": 0, "dispatches": 0, "rays": 0, "max_active": 0,
                 "seconds": 0.0, "max_seconds": 0.0}
    def traverse(*args):
        n = int(args[1])
        traversal["calls"] += 1
        traversal["max_active"] = max(traversal["max_active"], n)
        print("DIAG_TRAVERSE_BEGIN", traversal["calls"], n, "limit", limit, flush=True)
        for start in range(0, n, limit or max(n, 1)):
            call = list(args)
            if limit:
                call[1] = min(start + limit, n)
                call.append(start)
            before = time.perf_counter()
            kernel(*call)
            # The MPS zero-copy wrapper already synchronizes each real kernel.
            elapsed = time.perf_counter() - before
            traversal["dispatches"] += 1
            traversal["rays"] += call[1] - start
            traversal["seconds"] += elapsed
            traversal["max_seconds"] = max(traversal["max_seconds"], elapsed)
            if elapsed > 1:
                print("DIAG_SLOW_DISPATCH", start, call[1], elapsed, flush=True)
        print("DIAG_TRAVERSE_END", traversal["calls"], json.dumps(traversal), flush=True)
    wf.wavefront_traverse_events_arena = traverse

    import algan.rendering.raytracing.tracer as tracer
    real_select = tracer._ArenaRayCompactor.select
    select_calls = 0
    def checked_select(self, rs_int, desired_status, *, source=None, scan_pool=False,
                       rs_key=None, desired_key=0):
        nonlocal select_calls
        source = self.current[:self.size] if source is None else source
        state_cpu = rs_int.cpu()
        source_cpu = source.cpu()
        keys_cpu = rs_key.cpu() if rs_key is not None else None
        candidates = torch.arange(self.capacity, dtype=torch.int32) if scan_pool else source_cpu
        keep = state_cpu[candidates.long(), 2] == desired_status
        if keys_cpu is not None:
            keep &= keys_cpu[candidates.long()] == desired_key
        expected = candidates[keep]
        result = real_select(self, rs_int, desired_status, source=source, scan_pool=scan_pool,
                             rs_key=rs_key, desired_key=desired_key)
        actual = result.cpu()
        select_calls += 1
        good = torch.equal(actual.sort().values, expected.sort().values)
        print("DIAG_COMPACT", select_calls, "pool", self.capacity, "source", source.numel(),
              "expected", expected.numel(), "actual", actual.numel(), "equal", good,
              "status", torch.unique(state_cpu[:, 2], return_counts=True),
              "hits", torch.unique(state_cpu[:, 3], return_counts=True), flush=True)
        if not good:
            torch.save(dict(state=state_cpu, source=source_cpu, keys=keys_cpu,
                            actual=actual, expected=expected, capacity=self.capacity,
                            scan_pool=scan_pool, desired_status=desired_status,
                            desired_key=desired_key),
                       OUT / "compaction-mismatch.pt")
            raise AssertionError("GPU compaction disagrees with CPU status filter")
        if select_calls >= 5:
            print("DIAG_COMPACT_FIRST_FIVE_PASSED", flush=True)
            os._exit(0)
        return result
    tracer._ArenaRayCompactor.select = checked_select

    times = {"torch_sync": 0.0, "ti_sync": 0.0, "import": 0.0,
             "torch_sync_calls": 0, "ti_sync_calls": 0, "import_calls": 0,
             "import_misses": 0}
    def wrap(name, original):
        def measured(*args, **kwargs):
            start = time.perf_counter()
            times[name + "_calls"] += 1
            try:
                return original(*args, **kwargs)
            finally:
                times[name] += time.perf_counter() - start
        return measured

    torch.mps.synchronize = wrap("torch_sync", torch.mps.synchronize)
    ti.sync = wrap("ti_sync", ti.sync)
    original_import = zc.import_tensor
    def imported(*args, **kwargs):
        before = len(zc._IMPORTS)
        start = time.perf_counter()
        times["import_calls"] += 1
        try:
            return original_import(*args, **kwargs)
        finally:
            times["import"] += time.perf_counter() - start
            times["import_misses"] += int(len(zc._IMPORTS) > before)
    zc.import_tensor = imported
    original_wavefront = ns["_wavefront"].__globals__["_real_wavefront"]
    chunk_count = 0
    def traced_chunk(*args, **kwargs):
        nonlocal chunk_count
        start = time.perf_counter()
        before_t = dict(times)
        before_z = dict(zc.STATS)
        result = original_wavefront(*args, **kwargs)
        print("DIAG_CHUNK " + json.dumps({
            "seconds": time.perf_counter() - start,
            "timings": {k: times[k] - before_t[k] for k in times},
            "zero_copy": {k: zc.STATS[k] - before_z[k] for k in zc.STATS},
            "cache": zc.cache_stats(), "traversal": dict(traversal),
        }), flush=True)
        chunk_count += 1
        if chunk_count >= int(os.environ.get("DIAG_CHUNK_LIMIT", "999")):
            print("DIAG_CHUNK_LIMIT reached", flush=True)
            os._exit(0)  # Supervisor cleans up this diagnostic child group.
        return result
    ns["_wavefront"].__globals__["_real_wavefront"] = traced_chunk
    raise SystemExit(ns["main"]())


def command(argv, path, timeout=15):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        path.write_text(result.stdout + result.stderr)
        print("DIAG_TOOL", argv[0], result.returncode, path.name, flush=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        path.write_text(repr(exc))
        print("DIAG_TOOL_ERROR", argv[0], repr(exc), flush=True)


def snapshot(proc, arm, number):
    tag = OUT / f"{arm}-{number}"
    try:
        os.kill(proc.pid, signal.SIGUSR1)
    except ProcessLookupError:
        return
    command(["/usr/bin/sample", str(proc.pid), "3", "-file", str(tag) + "-sample.txt"],
            Path(str(tag) + "-sample-command.txt"), timeout=15)
    command(["/usr/bin/vm_stat"], Path(str(tag) + "-vm-stat.txt"))
    command(["/usr/sbin/sysctl", "vm.swapusage"], Path(str(tag) + "-swap.txt"))
    command(["/bin/ps", "-axo", "pid,ppid,state,%cpu,rss,vsz,comm"],
            Path(str(tag) + "-processes.txt"))
    for suffix in ["-vm-stat.txt", "-swap.txt"]:
        print("DIAG_MEMORY", arm, number, suffix,
              Path(str(tag) + suffix).read_text(), flush=True)
    sample = Path(str(tag) + "-sample.txt")
    if sample.exists():
        # Stream the relevant stack as well as preserving the full artifact.
        lines = sample.read_text(errors="replace").splitlines()
        print("DIAG_SAMPLE_BEGIN", arm, number, flush=True)
        print("\n".join(lines[:260]), flush=True)
        print("DIAG_SAMPLE_END", flush=True)


def supervise(arm, cap):
    env = dict(os.environ, ALGAN_MPS_HOST_SHARE=cap, ALGAN_TORCH_COMPILE="0",
               ALGAN_VIDEO_ENCODER="software", PYTHONUNBUFFERED="1",
               DIAG_CHUNK_LIMIT="999")
    path = OUT / f"{arm}.txt"
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child"],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    started = last_progress = time.monotonic()
    next_status = started + 30
    next_sample = started + 180
    sample_number = 0
    pending = b""
    reason = "exited"
    with path.open("wb") as log:
        try:
            while proc.poll() is None:
                for key, _ in sel.select(timeout=1):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        continue
                    log.write(data)
                    log.flush()
                    pending += data
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        text = line.decode(errors="replace")
                        print(f"[{arm}] {text}", flush=True)
                        if "DIAG_CHUNK " in text or "DIAG_TRAVERSE_END" in text or "chunk " in text and "begins at" in text:
                            last_progress = time.monotonic()
                now = time.monotonic()
                if now >= next_status:
                    print("DIAG_WATCHDOG", arm, "elapsed", round(now-started),
                          "no_chunk_progress", round(now-last_progress), flush=True)
                    next_status = now + 30
                if now >= next_sample:
                    sample_number += 1
                    snapshot(proc, arm, sample_number)
                    next_sample = time.monotonic() + 180
                if now - last_progress > 240 or now - started > 1200:
                    reason = "no_progress" if now-last_progress > 240 else "time_budget"
                    snapshot(proc, arm, sample_number + 1)
                    break
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            # Dispose only descendants of this diagnostic child on this runner.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            remainder = proc.stdout.read()
            log.write(pending + remainder)
            if remainder:
                print(remainder.decode(errors="replace"), flush=True)
            sel.close()
    command(["/usr/bin/log", "show", "--last", "15m", "--style", "compact",
             "--predicate", 'eventMessage CONTAINS[c] "GPU" OR eventMessage CONTAINS[c] "AGX" OR eventMessage CONTAINS[c] "Paravirtual"'],
            OUT / f"{arm}-system-log.txt", timeout=30)
    logtext = (OUT / f"{arm}-system-log.txt").read_text(errors="replace")
    print("DIAG_SYSTEM_LOG", arm, logtext[:20000] + "\n[...tail...]\n" + logtext[-20000:], flush=True)
    print("DIAG_ARM " + json.dumps({"arm": arm, "reason": reason,
           "returncode": proc.returncode, "seconds": time.monotonic()-started}), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    command(["/usr/sbin/system_profiler", "SPHardwareDataType", "SPDisplaysDataType"],
            OUT / "hardware.txt", timeout=30)
    print((OUT / "hardware.txt").read_text(), flush=True)
    command(["/usr/bin/xcrun", "swift", "-e", 'import Metal; for d in MTLCopyAllDevices() { print("METAL_DEVICE", d.name, "unified", d.hasUnifiedMemory, "maxBufferLength", d.maxBufferLength, "recommended", d.recommendedMaxWorkingSetSize, "allocated", d.currentAllocatedSize) }'], OUT / "metal-devices.txt", timeout=60)
    print((OUT / "metal-devices.txt").read_text(), flush=True)
    command(["/usr/bin/vm_stat"], OUT / "baseline-vm-stat.txt")
    print((OUT / "baseline-vm-stat.txt").read_text(), flush=True)
    arm = "limited" if int(os.environ.get("DIAG_DISPATCH_LIMIT", "0")) else "validation"
    kernel_path = ROOT / "algan/rendering/raytracing/wavefront_kernels_taichi.py"
    original = kernel_path.read_bytes()
    try:
        supervise(arm, "0")
    finally:
        kernel_path.write_bytes(original)
    return 0


if __name__ == "__main__":
    raise SystemExit(child() if "--child" in sys.argv else main())
