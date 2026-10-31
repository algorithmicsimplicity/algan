"""How much Metal source does ``pt_shade`` cross-compile to, with and without caustics?

On the Mac runner the caustic variant of ``pt_shade_arena`` (``pt_caustics``, and
``pt_subsurface_delta_lights`` in a scene with a refracting scattering interior)
cross-compiles to ~17 MB of MSL, and Apple's compiler service drops its XPC
connection on it: every such render fails to create its pipeline. This measures
the variant that does compile beside the one that does not, so the gap is a
number rather than a guess.

Each arm renders the glass-slab furnace of ``test_pt_caustics_render`` in a
child process with ``QD_DUMP_MSL=1`` (every translated shader is printed) and
``QD_SHADER_DUMP_DIR`` (a shader that fails is written out), and the parent
reports each ``pt_shade`` shader's size the moment its arm finishes.

    uv run python benchmarks/_mps_pt_shade_msl_size.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_CHILD = r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "tests" / "unit_tests"))
from test_path_tracer import _floor_under_glass, _furnace_opts, _render_scene_result
caustics = sys.argv[2] == "1"
opts = _furnace_opts(True, 4)
opts["experimental"] = {**opts["experimental"], "pt_caustics": caustics}
try:
    _render_scene_result(Path(sys.argv[3]), "slab.png", _floor_under_glass(1.5), 4, **opts)
    print("ARM-RENDERED", flush=True)
except Exception as exc:  # noqa: BLE001 -- the failure is the measurement
    print(f"ARM-FAILED {type(exc).__name__}: {str(exc)[:300]}", flush=True)
"""


def _shaders(text):
    """``(kernel names, bytes)`` for each translated shader in a dump."""
    out = []
    for chunk in text.split("#include <metal_stdlib>")[1:]:
        names = re.findall(r"kernel void (\w+)\(", chunk)
        out.append((names, len(chunk)))
    return out


def _arrays(text):
    """Total elements of the function-scope arrays SPIRV-Cross declared."""
    total = 0
    biggest = []
    for elem, n in re.findall(r"spvUnsafeArray<([\w:<> ]+?), (\d+)>", text):
        total += int(n)
        biggest.append((int(n), elem))
    biggest.sort(reverse=True)
    return total, biggest[:8]


def main():
    for caustics in ("0", "1"):
        work = Path(tempfile.mkdtemp(prefix=f"msl{caustics}-"))
        dump = work / "dump"
        dump.mkdir()
        env = dict(
            os.environ,
            QD_DUMP_MSL="1",
            QD_SHADER_DUMP_DIR=str(dump),
            ALGAN_USE_DAEMON="0",
            ALGAN_AUTO_DAEMON="0",
            ALGAN_PRECOMPILE_JOBS="0",
        )
        log = work / "child.log"
        with open(log, "w") as handle:
            subprocess.run(
                [sys.executable, "-c", _CHILD, str(_REPO), caustics, str(work)],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=str(_REPO),
                timeout=2400,
            )
        text = log.read_text(errors="replace")
        status = [ln for ln in text.splitlines() if ln.startswith("ARM-")]
        print(f"RESULT caustics={caustics}: {status[-1] if status else 'no status'}")
        print(f"RESULT caustics={caustics}: log {len(text)} bytes")
        for names, size in _shaders(text):
            if any("pt_shade" in n for n in names):
                print(f"RESULT caustics={caustics}: translated {names} {size} bytes")
        for path in sorted(dump.rglob("*")):
            if path.is_file():
                print(
                    f"RESULT caustics={caustics}: dumped {path.name} {path.stat().st_size} bytes"
                )
                if path.suffix == ".metal" and "pt_shade" in path.name:
                    source = path.read_text(errors="replace")
                    total, biggest = _arrays(source)
                    print(
                        f"RESULT caustics={caustics}: {path.name} lines "
                        f"{source.count(chr(10))}, array elements {total}, "
                        f"largest {biggest}"
                    )
        # Function-scope arrays of each translated pt_shade shader.
        for chunk in text.split("#include <metal_stdlib>")[1:]:
            names = re.findall(r"kernel void (\w+)\(", chunk)
            if any("pt_shade" in n for n in names):
                total, biggest = _arrays(chunk)
                print(
                    f"RESULT caustics={caustics}: {names} lines {chunk.count(chr(10))}, "
                    f"array elements {total}, largest {biggest}",
                    flush=True,
                )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
