"""Smoke-test a freshly built Quadrants wheel, the same way on every platform.

`quadrants_build.yaml` runs this after `pip install dist/*.whl` on each of its
legs. It is a file rather than three `python -c` one-liners because the legs
run under bash, inside a manylinux container, and under PowerShell, and a
one-liner that survives all three sets of quoting rules is not one anyone can
read.

What it checks is what Quadrants' own `scripts_new/*/3_install.sh` checks
(`import quadrants as qd; qd.init(arch=...)`) plus the three facts the Algan
patches add, each behind a flag so a stock build can be smoked with the same
command:

  --arch cpu|metal        which backend to `qd.init` (their install scripts
                          run cpu everywhere and metal on macOS, one process
                          each -- this script does one, call it twice)
  --expect-cuda-runtime   `_lib/runtime/runtime_cuda.bc` must be in the
                          installed package. It exists if and only if the
                          build had CUDA on -- the honest check, and not
                          `qd._lib.core.with_cuda()`, which also probes for
                          libcuda.so and is False on every GPU-less runner
                          however the binary was built
                          (`scripts/gate/quadrants_linux_build.sh`).
  --expect-patched        `quadrants.lang._ndarray.ExternalMetalNdarray` must
                          import: it is what `quadrants_patches/0001` adds and
                          what Algan's `mps_zero_copy.zero_copy_available()`
                          tests for, so a wheel without it is not the wheel
                          the Apple-GPU path needs.
  --expect-version X      the *distribution* version (`importlib.metadata`)
                          must be X. The release path pins it, because a
                          setuptools_scm version like `1.3.1.dev0+gab9a58ab5`
                          has a `+`, GitHub rewrites `+` to `.` in release
                          asset names, and pip then refuses the renamed wheel.
                          `qd.__version__` is not the thing to compare: it is
                          a `(major, minor, patch)` tuple CMake cuts from the
                          leading digits, `(1, 3, 0)` for `1.3.0.post1` too.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import pathlib
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--arch", default="cpu")
    parser.add_argument("--expect-cuda-runtime", action="store_true")
    parser.add_argument("--expect-patched", action="store_true")
    parser.add_argument("--expect-version", default="")
    args = parser.parse_args(argv)

    import quadrants as qd

    root = pathlib.Path(qd.__path__[0])
    dist_version = importlib.metadata.version("quadrants")
    print(f"quadrants {dist_version} (qd.__version__={qd.__version__}) from {root}")

    if args.expect_version and dist_version != args.expect_version:
        print(
            f"FAIL: distribution version is {dist_version!r}, expected "
            f"{args.expect_version!r} -- SETUPTOOLS_SCM_PRETEND_VERSION did not "
            "reach the build"
        )
        return 1

    if args.expect_cuda_runtime:
        bc = root / "_lib" / "runtime" / "runtime_cuda.bc"
        if not bc.exists():
            print(f"FAIL: {bc} is ABSENT -- CUDA was off, this build did not")
            print("compile the code quadrants_patches/0003 changes. Check that")
            print("CMAKE_ARGS reached build.py. _lib/runtime holds:")
            for entry in sorted((root / "_lib" / "runtime").glob("*")):
                print(f"  {entry.name}")
            return 1
        print(f"{bc.relative_to(root)} present -- CUDA backend compiled")

    if args.expect_patched:
        try:
            from quadrants.lang._ndarray import ExternalMetalNdarray  # noqa: F401
        except ImportError as exc:
            print(f"FAIL: ExternalMetalNdarray is not importable ({exc}) -- the")
            print("wheel does not carry quadrants_patches/0001, so Algan's")
            print("Apple-GPU path (mps_zero_copy.zero_copy_available) would say False")
            return 1
        print("ExternalMetalNdarray present -- quadrants_patches/0001 is in the wheel")

    arch = getattr(qd, args.arch)
    qd.init(arch=arch)
    print(f"qd.init(arch=qd.{args.arch}) OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
