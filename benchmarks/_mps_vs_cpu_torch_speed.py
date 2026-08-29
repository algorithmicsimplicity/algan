"""Does GitHub's free macOS runner have a real GPU behind MPS, or only a CPU?

    uv run python benchmarks/_mps_vs_cpu_torch_speed.py

``_mps_capability_probe.py`` established what the runner's Metal backend *can
do*; its header then says the runner is a virtualized-GPU instance and that its
Q5 milliseconds are directional only. That is a claim about the machine, and
nobody has measured it: what is not established is whether MPS work on this
runner is executed by a GPU at all, or by the same three cores that serve the
CPU arm. The two are indistinguishable from a capability result -- a software
Metal implementation compiles the same shaders and returns the same bits.

Throughput separates them, and it does so without any appeal to what the
hypervisor reports. Run identical torch tensor work on ``cpu`` and on ``mps``
and compare:

* **Compute-bound work is the discriminator.** A dense f32 matmul is ~2N^3
  FLOPs over 3N^2 bytes, so it is limited by arithmetic units and nothing else.
  An Apple GPU is worth roughly a teraflop and up here (an M1's 7-core GPU is
  ~2.3 TFLOP/s f32, an M2's 10-core ~3.6); these runners are small VMs, so the
  share of one on offer may be a fraction of that, but it is a fraction of a
  number the CPU cannot reach. A CPU-executed Metal path cannot beat the CPU
  arm by much, because it *is* the CPU arm with a driver in front of it.
* **Bandwidth-bound work is not.** On unified memory both devices read the same
  DRAM, so an elementwise chain or a reduction can land at parity on a machine
  whose GPU is entirely real. A near-1.0 ratio there is not evidence of
  anything, which is why the verdict below is drawn from the matmul ladder and
  the elementwise rows are reported beside it rather than averaged into it.
* **Dispatch latency is the third axis**, and the one Algan's many-small-kernel
  stages care about: a tiny op timed with a synchronize per iteration measures
  the cost of getting to the GPU at all, not of the arithmetic.

Every case runs on both devices from the *same* seeded inputs, and each arm
reports a checksum of its result. The orchestrator compares them, so a row whose
timings are being compared is also a row that has been shown to compute the same
answer -- otherwise "MPS is faster" could just mean MPS did less.

``PYTORCH_ENABLE_MPS_FALLBACK`` is cleared before anything runs. With it set, an
unimplemented op runs on the CPU and is timed as if it were the GPU, which is
precisely the confusion this script exists to resolve.

Each device runs in its **own subprocess** (``--device cpu`` / ``--device mps``),
following ``_mps_capability_probe.py``: a Metal failure can abort rather than
raise, and losing the mps arm should not cost the cpu arm's numbers. Each arm
prints one ``ALGAN_SPEED_JSON <json>`` line; the orchestrator collects them,
writes ``mps_vs_cpu_speed.json`` beside itself, and prints the report.

No ``cap_process_memory``: the sizes here are fixed constants rather than
parameters (largest working set ~400 MB, in the f32 4096 matmul), and the POSIX
ceiling is an ``RLIMIT_AS`` on *address space*, which Metal's own reservations
sit inside -- capping this would break the arm it exists to measure, the same
reason ``_mps_capability_probe.py`` does not cap either.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: One line per arm, parsed by the orchestrator. A marker rather than bare
#: stdout because torch and the MPS driver both write warnings to the same
#: stream.
_MARKER = "ALGAN_SPEED_JSON "

#: A timed batch aims for this long, so that a case which runs in microseconds
#: is still measured over an interval the clock can resolve.
_TARGET_BATCH_SECONDS = 0.05

#: How many timed batches per case. The report takes the median (typical) and
#: the minimum (least disturbed by the runner's neighbours).
_REPS = 5

#: Seconds a single case may spend on its timed batches, down to one batch for a
#: case whose single iteration already exceeds it. A case that slow gets fewer
#: batches rather than a long job:
#: the f32 4096 matmul on a small runner is most of a second an iteration, and
#: five batches of that is a minute for one row.
_CASE_TIME_BUDGET_SECONDS = 12.0


# ---------------------------------------------------------------------------
# What machine is this
# ---------------------------------------------------------------------------


def _run_text(argv, timeout=20):
    """Best-effort stdout of a command, or None if it is not there / fails."""
    try:
        out = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _sysctl(name):
    return _run_text(["sysctl", "-n", name])


def _machine():
    """Everything about the host that bears on reading the numbers below."""
    import torch

    facts = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_count": os.cpu_count(),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        # Which BLAS serves the cpu arm decides how to read the whole
        # comparison. On Apple silicon, Accelerate reaches the AMX matrix
        # coprocessor and a NEON-only GEMM does not -- a factor of several,
        # in the denominator of every speedup on this page.
        "blas": _blas_identity(),
        "parallel_info": _parallel_info(),
    }

    if sys.platform == "darwin":
        # kern.hv_vmm_present is the machine's own answer to "am I a guest",
        # which is worth having beside the throughput rather than instead of it:
        # a virtualized host can still be passed a real GPU, and that is exactly
        # the ambiguity the timings are here to settle.
        facts["mac"] = {
            "hw.model": _sysctl("hw.model"),
            "machdep.cpu.brand_string": _sysctl("machdep.cpu.brand_string"),
            "hw.ncpu": _sysctl("hw.ncpu"),
            "hw.memsize": _sysctl("hw.memsize"),
            "kern.hv_vmm_present": _sysctl("kern.hv_vmm_present"),
            "gpu": _mac_gpu(),
        }

    if facts["mps_available"]:
        try:
            facts["mps_recommended_max_memory"] = int(
                torch.mps.recommended_max_memory()
            )
        except Exception as exc:  # informational only
            facts["mps_recommended_max_memory_error"] = str(exc)[:200]

    return facts


def _blas_identity():
    """The BLAS/LAPACK lines out of ``torch.__config__.show()``, if any."""
    import torch

    try:
        text = torch.__config__.show()
    except Exception as exc:  # informational only
        return f"unavailable: {type(exc).__name__}: {exc}"[:200]
    # Pulled out of the one enormous "Build settings:" line as key=value pairs,
    # rather than printed whole: the line is ~2 kB of compiler flags and the two
    # tokens that matter here are BLAS_INFO and LAPACK_INFO.
    wanted = ("BLAS_INFO", "LAPACK_INFO", "USE_MKL", "USE_MKLDNN")
    found = {}
    for token in text.replace("\n", ",").split(","):
        key, _, value = token.strip().partition("=")
        # The first pair on the line carries the "- Build settings: " prefix.
        key = key.rsplit(" ", 1)[-1]
        if key in wanted:
            found[key] = value
    if not found:
        # Accelerate builds name it in prose rather than in BLAS_INFO.
        if "Accelerate" in text:
            return "Accelerate (named in torch.__config__.show(), no BLAS_INFO)"
        return "(no BLAS token in torch.__config__.show())"
    return " | ".join(f"{k}={v}" for k, v in found.items())[:400]


def _parallel_info():
    import torch

    try:
        text = torch.__config__.parallel_info()
    except Exception as exc:  # informational only
        return f"unavailable: {type(exc).__name__}: {exc}"[:200]
    keep = ("ATen parallel backend", "OpenMP", "Intra-op", "Inter-op", "thread")
    lines = [ln.strip() for ln in text.splitlines() if any(k in ln for k in keep)]
    return " | ".join(lines)[:400]


def _mac_gpu():
    """What macOS says the display/compute device is, if it will say anything.

    ``system_profiler`` is the naming authority, but a headless VM can report an
    empty display tree, so the ioreg core count is asked for separately -- the
    two fail independently.
    """
    gpu = {}
    raw = _run_text(["system_profiler", "-json", "SPDisplaysDataType"], timeout=60)
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            gpu["system_profiler_raw"] = raw[:800]
        else:
            gpu["system_profiler"] = parsed.get("SPDisplaysDataType", parsed)
    cores = _run_text(["bash", "-c", "ioreg -l | grep -i 'gpu-core-count' | head -5"])
    if cores:
        gpu["ioreg_gpu_core_count"] = cores
    return gpu


# ---------------------------------------------------------------------------
# The workloads
# ---------------------------------------------------------------------------


def _seeded(shape, dtype, device, seed):
    """Identical values on every device, so the two arms compute the same thing.

    Generated on the CPU under an explicit generator and then moved: an MPS
    generator seeded with the same number does not produce the CPU's stream, and
    a checksum comparison between arms is only meaningful on shared inputs.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    host = torch.randn(shape, generator=gen, dtype=torch.float32)
    return host.to(device=device, dtype=dtype)


def _matmul_case(n, dtype_name):
    def setup(device):
        import torch

        dtype = getattr(torch, dtype_name)
        a = _seeded((n, n), dtype, device, seed=1)
        b = _seeded((n, n), dtype, device, seed=2)
        out = {}

        def run():
            out["r"] = a @ b

        return {
            # 2 FLOPs (one multiply, one add) per element of the N^3 products.
            "work": 2.0 * n * n * n,
            "unit": "GFLOP/s",
            "run": run,
            "result": lambda: out["r"],
            "hold": (a, b, out),
        }

    return setup


def _elementwise_case(n):
    def setup(device):
        import torch

        a = _seeded((n,), torch.float32, device, seed=3)
        b = _seeded((n,), torch.float32, device, seed=4)
        c = _seeded((n,), torch.float32, device, seed=5)
        out = {}

        def run():
            out["r"] = torch.addcmul(c, a, b)

        return {
            # Three reads and one write of f32, which is what a chain like this
            # is limited by -- the arithmetic is two ops per element.
            "work": 4.0 * 4 * n,
            "unit": "GB/s",
            "run": run,
            "result": lambda: out["r"],
            "hold": (a, b, c, out),
        }

    return setup


def _reduction_case(n):
    def setup(device):
        import torch

        a = _seeded((n,), torch.float32, device, seed=6)
        out = {}

        def run():
            out["r"] = a.sum()

        return {
            "work": 4.0 * n,
            "unit": "GB/s",
            "run": run,
            "result": lambda: out["r"],
            "hold": (a, out),
        }

    return setup


def _softmax_case(rows, cols):
    def setup(device):
        import torch

        a = _seeded((rows, cols), torch.float32, device, seed=7)
        out = {}

        def run():
            out["r"] = torch.softmax(a, dim=-1)

        return {
            # Two passes over the input plus one write, at f32.
            "work": 3.0 * 4 * rows * cols,
            "unit": "GB/s",
            "run": run,
            "result": lambda: out["r"],
            "hold": (a, out),
        }

    return setup


def _conv2d_case(batch, channels, size, kernel):
    def setup(device):
        import torch

        x = _seeded((batch, channels, size, size), torch.float32, device, seed=8)
        w = _seeded((channels, channels, kernel, kernel), torch.float32, device, seed=9)
        out = {}

        def run():
            out["r"] = torch.nn.functional.conv2d(x, w, padding=kernel // 2)

        return {
            # Same-padding keeps the spatial size, so every output element costs
            # channels * kernel^2 multiply-adds.
            "work": 2.0 * batch * channels * size * size * channels * kernel * kernel,
            "unit": "GFLOP/s",
            "run": run,
            "result": lambda: out["r"],
            "hold": (x, w, out),
        }

    return setup


#: Ordered cheapest-first, so an arm that dies partway through still reports the
#: small end of the ladder.
_CASES = [
    ("matmul_f32_512", _matmul_case(512, "float32")),
    ("matmul_f32_1024", _matmul_case(1024, "float32")),
    ("matmul_f32_2048", _matmul_case(2048, "float32")),
    ("matmul_f32_4096", _matmul_case(4096, "float32")),
    # Smaller than the f32 ladder on purpose. Apple's GPU is at its best in f16
    # and this row is where that would show, but torch has no native f16 GEMM on
    # x86: the Linux control runs it at 0.7 GFLOP/s against 360 for f32, so at
    # 2048 one control row costs longer than every other row on both arms.
    ("matmul_f16_1024", _matmul_case(1024, "float16")),
    ("conv2d_f32_4x64x128", _conv2d_case(4, 64, 128, 3)),
    ("softmax_f32_8192x1024", _softmax_case(8192, 1024)),
    ("elementwise_f32_16M", _elementwise_case(16 << 20)),
    ("reduction_f32_16M", _reduction_case(16 << 20)),
]

#: The matmul rows are the ones the verdict is drawn from: they are the only
#: cases whose rate is set by arithmetic throughput rather than by the DRAM both
#: devices share.
_COMPUTE_BOUND = {name for name, _ in _CASES if name.startswith(("matmul", "conv2d"))}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _sync(device):
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _time_case(case, device):
    """Median and minimum seconds per iteration, plus the checksum of the result.

    Iterations per batch are chosen from a single measured iteration rather than
    fixed, because the same case can differ by two orders of magnitude between
    the two devices and a count that suits one wastes minutes or resolves
    nothing on the other.
    """
    run = case["run"]

    # Warm-up. On MPS the first launch of a shape compiles a Metal pipeline, and
    # on the CPU the first touch of a fresh buffer takes the page faults; both
    # belong outside the measurement.
    _sync(device)
    t0 = time.perf_counter()
    run()
    _sync(device)
    single = time.perf_counter() - t0

    # Capped as well as floored: on MPS every iteration in a batch enqueues a
    # command buffer and a fresh result allocation before the batch's one
    # synchronize, so an unbounded count on a cheap case would measure the
    # allocator rather than the kernel.
    iters = 1
    if single > 0:
        iters = max(1, min(200, int(_TARGET_BATCH_SECONDS / single)))

    # Further warm-up only where it is affordable. A case measured in seconds
    # has already had its pipeline compiled and its pages faulted by the
    # iteration above.
    batch_seconds = single * iters
    if batch_seconds < 1.0:
        for _ in range(2):
            run()
        _sync(device)

    reps = _REPS
    if batch_seconds > 0:
        reps = max(1, min(_REPS, int(_CASE_TIME_BUDGET_SECONDS / batch_seconds)))

    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            run()
        _sync(device)
        samples.append((time.perf_counter() - t0) / iters)

    result = case["result"]()
    checksum = _checksum(result)
    return {
        "iters_per_batch": iters,
        "reps": reps,
        "seconds_median": statistics.median(samples),
        "seconds_min": min(samples),
        "seconds_all": samples,
        "checksum": checksum,
    }


def _checksum(tensor):
    """A small, device-independent summary of a result tensor.

    Summed in f64 on the host: an f32 reduction's ordering differs between
    backends, and the point of the comparison is to catch an arm that computed
    something else entirely, not to police the last ulp.
    """
    flat = tensor.detach().cpu().reshape(-1).double()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sum": float(flat.sum()),
        "abs_max": float(flat.abs().max()),
        "n": int(flat.numel()),
    }


def _transfer_bandwidth(device, megabytes=64):
    """Host to device and back, which on unified memory is a question worth asking.

    A discrete GPU pays a PCIe crossing here. An Apple GPU shares the pointer, so
    a number near host memcpy speed is the expected answer and a number far below
    it says the runtime is copying when it did not have to.
    """
    import torch

    if device == "cpu":
        return None

    n = (megabytes << 20) // 4
    host = torch.full((n,), 1.5, dtype=torch.float32)
    dev = host.to(device)
    _sync(device)

    def timed(fn, reps=5):
        fn()
        _sync(device)
        out = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            _sync(device)
            out.append(time.perf_counter() - t0)
        return statistics.median(out)

    h2d = timed(lambda: host.to(device))
    d2h = timed(lambda: dev.to("cpu"))
    nbytes = float(n * 4)
    return {
        "megabytes": megabytes,
        "h2d_seconds": h2d,
        "d2h_seconds": d2h,
        "h2d_gb_s": nbytes / h2d / 1e9,
        "d2h_gb_s": nbytes / d2h / 1e9,
    }


#: Sizes for the ceiling sweep. The comparison table stops at 4096 because a
#: bigger matmul costs the cpu arm seconds per iteration for a row whose ratio
#: is already established; the ceiling sweep is asking a different question and
#: needs the sizes where each device stops being launch- or cache-limited.
_CEILING_SIZES = (2048, 4096, 6144, 8192)


def _ceiling(device):
    """The best sustained matmul rate this device reaches, at its best size.

    The comparison table answers "which is faster here". This answers "is either
    number near what the hardware can do", which is what says whether a modest
    ratio means a weak GPU or a strong CPU baseline. f16 is swept on the GPU
    only: torch has no native f16 GEMM on either machine's CPU, so a CPU f16 row
    measures a fallback loop rather than a ceiling.
    """
    import torch

    dtypes = ["float32"] if device == "cpu" else ["float32", "float16"]
    rows = []
    for dtype_name in dtypes:
        for n in _CEILING_SIZES:
            try:
                case = _matmul_case(n, dtype_name)(device)
                measured = _time_case(case, device)
            except Exception as exc:
                rows.append(
                    {
                        "n": n,
                        "dtype": dtype_name,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    }
                )
                continue
            rows.append(
                {
                    "n": n,
                    "dtype": dtype_name,
                    "status": "ok",
                    "seconds_median": measured["seconds_median"],
                    "gflop_s": case["work"] / measured["seconds_median"] / 1e9,
                }
            )
            del case
            if device == "mps":
                torch.mps.empty_cache()
            print(
                f"  [{device}] ceiling {dtype_name} {n}: {rows[-1].get('gflop_s', 0):.1f}"
                f" GFLOP/s",
                flush=True,
            )
    return rows


def _dispatch_latency(device, reps=200):
    """Cost of getting one trivial op to the device, synchronized every time.

    This is the axis Algan's many-small-kernel stages live on: the raster path
    launches far more kernels per frame than it does teraflops of arithmetic, so
    a backend can be fast per FLOP and still lose on the dispatch.
    """
    import torch

    x = torch.zeros(1024, device=device)
    for _ in range(10):
        x.add_(1.0)
    _sync(device)

    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        x.add_(1.0)
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return {
        "reps": reps,
        "seconds_median": statistics.median(samples),
        "seconds_min": min(samples),
    }


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------


def _run_arm(device):
    # A silent CPU fallback would time the CPU and report it as the GPU, which
    # is the exact confusion this script exists to remove.
    os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

    import torch

    payload = {"device": device, "machine": _machine()}

    if device == "mps" and not torch.backends.mps.is_available():
        payload["status"] = "skipped"
        payload["reason"] = (
            "torch.backends.mps.is_available() is False "
            f"(built={torch.backends.mps.is_built()})"
        )
        _emit(payload)
        return 0

    payload["status"] = "ok"
    payload["cases"] = {}

    for name, setup in _CASES:
        try:
            case = setup(device)
        except Exception as exc:  # a refused allocation is a result
            payload["cases"][name] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:400],
            }
            continue
        try:
            measured = _time_case(case, device)
        except Exception as exc:  # an unimplemented op is a result
            payload["cases"][name] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:400],
            }
        else:
            measured["status"] = "ok"
            measured["work"] = case["work"]
            measured["unit"] = case["unit"]
            measured["rate"] = case["work"] / measured["seconds_median"] / 1e9
            payload["cases"][name] = measured
        finally:
            del case
            if device == "mps":
                torch.mps.empty_cache()
        # Printed as it goes, and relayed line by line by the orchestrator, so
        # that a job which times out still says which case it was on.
        row = payload["cases"][name]
        if row.get("status") == "ok":
            print(
                f"  [{device}] {name}: {row['seconds_median'] * 1e3:.3f} ms  "
                f"{row['rate']:.1f} {row['unit']}",
                flush=True,
            )
        else:
            print(f"  [{device}] {name}: {row['error_type']}", flush=True)

    try:
        payload["transfer"] = _transfer_bandwidth(device)
    except Exception as exc:
        payload["transfer_error"] = f"{type(exc).__name__}: {exc}"[:400]

    try:
        payload["dispatch"] = _dispatch_latency(device)
    except Exception as exc:
        payload["dispatch_error"] = f"{type(exc).__name__}: {exc}"[:400]

    try:
        payload["ceiling"] = _ceiling(device)
    except Exception as exc:
        payload["ceiling_error"] = f"{type(exc).__name__}: {exc}"[:400]

    _emit(payload)
    return 0


def _emit(payload):
    print(_MARKER + json.dumps(payload), flush=True)


# ---------------------------------------------------------------------------
# Orchestration and report
# ---------------------------------------------------------------------------


def _spawn(device, timeout=1800):
    """Run one arm as a child process, relaying its output as it arrives.

    Relayed rather than captured and printed at the end, because the arm prints
    a line per case and a job that hits its timeout should still say which case
    it was on -- with ``capture_output`` that log arrives only if the process
    finishes, which is exactly the case where it is not needed.
    """
    env = dict(os.environ)
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    argv = [sys.executable, str(Path(__file__).resolve()), "--device", device]
    print(f"== running the {device} arm ==", flush=True)

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    # A watchdog rather than a deadline checked between lines: an arm that hangs
    # stops producing lines, which is the case a between-lines check cannot see.
    watchdog = threading.Timer(timeout, proc.kill)
    watchdog.daemon = True
    watchdog.start()

    payload = None
    tail = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(_MARKER):
                payload = json.loads(line[len(_MARKER) :])
                continue
            print(line, flush=True)
            tail.append(line)
            del tail[:-12]
        returncode = proc.wait()
    finally:
        watchdog.cancel()

    if payload is not None:
        return payload
    return {
        "device": device,
        "status": "crashed",
        "returncode": returncode,
        "output_tail": tail,
    }


def _fmt_ms(seconds):
    if seconds is None:
        return "-"
    return f"{seconds * 1e3:.3f}"


def _report(results):
    cpu = results.get("cpu", {})
    mps = results.get("mps", {})

    print()
    print("=" * 78)
    print("torch on CPU vs MPS")
    print("=" * 78)

    machine = cpu.get("machine") or mps.get("machine") or {}
    print("\n== machine ==")
    for key in (
        "platform",
        "machine",
        "processor",
        "python",
        "torch",
        "cpu_count",
        "torch_num_threads",
        "mps_built",
        "mps_available",
        "mps_fallback_env",
        "mps_recommended_max_memory",
        "blas",
        "parallel_info",
    ):
        if key in machine:
            print(f"  {key:28s} {machine[key]}")
    mac = machine.get("mac")
    if mac:
        for key, value in mac.items():
            if key == "gpu":
                continue
            print(f"  {key:28s} {value}")
        gpu = mac.get("gpu") or {}
        print(f"  {'gpu':28s} {json.dumps(gpu)[:600]}")

    for name, arm in (("cpu", cpu), ("mps", mps)):
        if arm.get("status") not in (None, "ok"):
            detail = arm.get("reason") or arm.get("output_tail") or ""
            print(f"\n  !! the {name} arm did not run: {arm.get('status')} {detail}")

    print("\n== throughput (seconds per iteration, median of up to 5 batches) ==")
    header = (
        f"  {'case':24s} {'cpu ms':>10s} {'mps ms':>10s} {'speedup':>9s} "
        f"{'cpu rate':>11s} {'mps rate':>11s}  unit"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    ratios = {}
    for name, _ in _CASES:
        c = (cpu.get("cases") or {}).get(name, {})
        m = (mps.get("cases") or {}).get(name, {})
        c_ok = c.get("status") == "ok"
        m_ok = m.get("status") == "ok"
        c_s = c.get("seconds_median") if c_ok else None
        m_s = m.get("seconds_median") if m_ok else None
        speedup = (c_s / m_s) if (c_s and m_s) else None
        if speedup is not None:
            ratios[name] = speedup
        unit = c.get("unit") or m.get("unit") or ""
        speedup_text = f"{speedup:.2f}x" if speedup else "-"
        c_rate = f"{c['rate']:.1f}" if c_ok else "-"
        m_rate = f"{m['rate']:.1f}" if m_ok else "-"
        print(
            f"  {name:24s} {_fmt_ms(c_s):>10s} {_fmt_ms(m_s):>10s} "
            f"{speedup_text:>9s} {c_rate:>11s} {m_rate:>11s}  {unit}"
        )
        for arm_name, arm in (("cpu", c), ("mps", m)):
            if arm.get("status") == "error":
                print(
                    f"      {arm_name} error: {arm['error_type']}: {arm['error'][:160]}"
                )

    _report_checksums(cpu, mps)
    _report_transfer(mps)
    _report_dispatch(cpu, mps)
    _report_ceiling(cpu, mps)
    _verdict(cpu, mps, ratios)


def _report_checksums(cpu, mps):
    cpu_cases = cpu.get("cases") or {}
    mps_cases = mps.get("cases") or {}
    rows = []
    for name, _ in _CASES:
        c = cpu_cases.get(name, {}).get("checksum")
        m = mps_cases.get(name, {}).get("checksum")
        if not c or not m:
            continue
        # Scaled by the magnitude a random-sign sum of this many terms lands at,
        # so a near-zero total does not read as a mismatch over its own noise.
        scale = max(
            abs(c["sum"]), abs(m["sum"]), c["abs_max"] * max(c["n"], 1) ** 0.5, 1e-6
        )
        rel = abs(c["sum"] - m["sum"]) / scale
        rows.append((name, c["sum"], m["sum"], rel))
    if not rows:
        return
    print("\n== same answer? (f64 sum of each arm's result) ==")
    print(f"  {'case':24s} {'cpu sum':>16s} {'mps sum':>16s} {'rel. diff':>12s}")
    for name, c_sum, m_sum, rel in rows:
        flag = "" if rel < 1e-3 else "   <-- DIFFERS"
        print(f"  {name:24s} {c_sum:16.6g} {m_sum:16.6g} {rel:12.3e}{flag}")


def _report_transfer(mps):
    t = mps.get("transfer")
    if not t:
        return
    print("\n== host <-> device transfer (64 MB) ==")
    print(
        f"  host->mps  {t['h2d_seconds'] * 1e3:8.3f} ms  {t['h2d_gb_s']:7.2f} GB/s\n"
        f"  mps->host  {t['d2h_seconds'] * 1e3:8.3f} ms  {t['d2h_gb_s']:7.2f} GB/s"
    )


def _report_dispatch(cpu, mps):
    c = cpu.get("dispatch")
    m = mps.get("dispatch")
    if not c and not m:
        return
    print("\n== per-dispatch latency (one tiny op, synchronized each time) ==")
    for label, arm in (("cpu", c), ("mps", m)):
        if arm:
            print(
                f"  {label}  median {arm['seconds_median'] * 1e6:9.2f} us"
                f"   min {arm['seconds_min'] * 1e6:9.2f} us"
            )


def _report_ceiling(cpu, mps):
    """Each device's best sustained matmul rate, at the size that reaches it."""
    rows = {"cpu": cpu.get("ceiling"), "mps": mps.get("ceiling")}
    if not any(rows.values()):
        return
    print("\n== sustained matmul ceiling (GFLOP/s by size) ==")
    header = f"  {'device':6s} {'dtype':8s}" + "".join(
        f"{n:>10d}" for n in _CEILING_SIZES
    )
    print(header)
    for device, sweep in rows.items():
        if not sweep:
            continue
        by_dtype = {}
        for row in sweep:
            by_dtype.setdefault(row["dtype"], {})[row["n"]] = row
        for dtype_name, sizes in by_dtype.items():
            cells = ""
            for n in _CEILING_SIZES:
                row = sizes.get(n)
                if row and row.get("status") == "ok":
                    cells += f"{row['gflop_s']:10.1f}"
                else:
                    cells += f"{'-':>10s}"
            print(f"  {device:6s} {dtype_name:8s}{cells}")
    print(
        "  (the comparison table's ratio is only as impressive as its denominator:\n"
        "   read the cpu row before reading the speedup)"
    )


def _verdict(cpu, mps, ratios):
    print("\n== verdict ==")
    if mps.get("status") == "skipped":
        print(f"  no MPS device on this host: {mps.get('reason')}")
        print("  (expected on the Linux control arm, which is here to show the")
        print("   harness itself works and to give the CPU numbers a second reading)")
        return
    if mps.get("status") not in ("ok",):
        print(f"  the mps arm did not report: {mps.get('status')}")
        return

    compute = {k: v for k, v in ratios.items() if k in _COMPUTE_BOUND}
    memory = {k: v for k, v in ratios.items() if k not in _COMPUTE_BOUND}

    def peak(arm, names):
        cases = arm.get("cases") or {}
        rates = [
            cases[n]["rate"]
            for n in names
            if cases.get(n, {}).get("status") == "ok" and cases[n]["unit"] == "GFLOP/s"
        ]
        return max(rates) if rates else None

    cpu_peak = peak(cpu, _COMPUTE_BOUND)
    mps_peak = peak(mps, _COMPUTE_BOUND)

    if cpu_peak:
        print(f"  peak f32 compute, cpu arm: {cpu_peak:8.1f} GFLOP/s")
    if mps_peak:
        print(f"  peak f32 compute, mps arm: {mps_peak:8.1f} GFLOP/s")

    if compute:
        best = max(compute.values())
        print(
            f"  compute-bound speedup: {min(compute.values()):.2f}x to {best:.2f}x "
            f"over {len(compute)} cases"
        )
    if memory:
        print(
            f"  bandwidth-bound speedup: {min(memory.values()):.2f}x to "
            f"{max(memory.values()):.2f}x over {len(memory)} cases "
            f"(unified memory: parity here is expected either way)"
        )

    if not compute:
        print("  no compute-bound case completed on both arms; nothing to conclude")
        return

    best = max(compute.values())
    if best >= 3.0:
        print(
            "  READS AS REAL GPU HARDWARE. Arithmetic-bound work is several times\n"
            "  faster than the CPU arm, which a CPU-executed Metal path cannot do."
        )
    elif best >= 1.5:
        print(
            "  READS AS A GPU, BUT A SMALL SHARE OF ONE. Compute-bound work is\n"
            "  ahead of the CPU arm but not by the margin a full Apple GPU gives;\n"
            "  consistent with a virtualized slice, or with a backend whose\n"
            "  overhead eats the win at these sizes."
        )
    else:
        print(
            "  NO GPU ADVANTAGE MEASURED. Compute-bound work is at or below the\n"
            "  CPU arm's throughput, which is what a CPU-executed Metal path looks\n"
            "  like. It is not proof of one -- an overloaded shared runner reads the\n"
            "  same -- so re-run before treating it as settled."
        )
    print(
        "  Read the bandwidth-bound rows as context only: both devices address the\n"
        "  same DRAM on Apple silicon, so parity there says nothing either way."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        help="run one arm in this process and print its JSON (used by the orchestrator)",
    )
    parser.add_argument(
        "--json",
        default=str(_HERE / "mps_vs_cpu_speed.json"),
        help="where the orchestrator writes the collected results",
    )
    args = parser.parse_args()

    if args.device:
        return _run_arm(args.device)

    results = {}
    for device in ("cpu", "mps"):
        results[device] = _spawn(device)

    Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    _report(results)
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
