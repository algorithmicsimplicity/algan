"""Build the small Metal owner for macOS wheels and editable installs only.

No Torch import, header, or C++ library is needed. Wheels use the CPython 3.10
stable ABI and contain both Mac architectures. Other platforms remain pure
Python. The source distribution carries this hook and the Objective-C++ source.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class MetalArenaBuildHook(BuildHookInterface):
    """Compile native storage ownership at install/build time, never at render time."""

    def initialize(self, version, build_data):
        if self.target_name != "wheel" or sys.platform != "darwin":
            return
        root = Path(self.root)
        relative = "algan/rendering/_mps_arena_native.abi3.so"
        # A unique staging path also avoids readers observing a partial binary
        # when two editable installs happen to target the same source tree.
        self._temporary = tempfile.TemporaryDirectory(prefix="algan-metal-build-")
        output = Path(self._temporary.name) / "_mps_arena_native.abi3.so"
        command = [
            "xcrun",
            "clang++",
            "-std=c++17",
            "-O2",
            "-bundle",
            "-undefined",
            "dynamic_lookup",
            "-fno-objc-arc",
            "-arch",
            "arm64",
            "-arch",
            "x86_64",
            "-mmacosx-version-min=11.0",
            "-I" + sysconfig.get_path("include"),
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            str(root / "algan/rendering/_mps_arena_native.mm"),
            "-o",
            str(output),
        ]
        try:
            subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=180
            )
            build_data["pure_python"] = False
            build_data["tag"] = "cp310-abi3-macosx_11_0_universal2"
            if version == "editable":
                destination = root / relative
                # Staging in the destination directory makes replace atomic
                # even when the OS temp directory is on a different volume.
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(output.read_bytes())
                try:
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                build_data.setdefault("force_include", {})[str(output)] = relative
        except (OSError, subprocess.SubprocessError) as error:
            self._temporary.cleanup()
            detail = getattr(error, "stderr", None) or str(error)
            raise RuntimeError(
                "Building Algan's MPS arena requires Apple's Command Line Tools "
                "(xcode-select --install), a macOS SDK, and Python development headers. "
                "No Torch C++ toolchain is required.\n" + detail
            ) from error

    def finalize(self, version, build_data, artifact_path):
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()
