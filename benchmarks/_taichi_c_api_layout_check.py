"""Check the ctypes struct layouts against a C oracle built from taichi's headers.

``DESIGN_taichi_arch_coexistence.md`` calls the ctypes union "the single most
dangerous part of the design" (§5.5) and makes a layout guard a Phase 1 exit
criterion; §10 makes "the struct-layout guard cannot be made to hold" a kill
criterion. This is that guard, built the strong way: rather than asserting
hand-copied constants, it compiles a C program against the *installed*
``taichi_core.h`` that prints every ``sizeof``/``offsetof``, and compares it to
what ``_taichi_c_api_shim`` declares. A taichi upgrade that moves a field fails
here instead of corrupting memory at a launch.

    uv run python benchmarks/_taichi_c_api_layout_check.py

Exits non-zero on any mismatch. Needs a C compiler; skips (exit 0, loudly) if
there is none, since that is a property of the box rather than of the code.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _taichi_c_api_shim import LAYOUT, taichi_lib_root  # noqa: E402

# One entry per struct the shim declares, naming the fields to probe. Kept
# explicit so a field added to the C header without a shim change shows up as a
# size mismatch rather than being quietly skipped.
_PROBES = {
    "TiNdShape": ("dim_count", "dims"),
    "TiNdArray": ("memory", "shape", "elem_shape", "elem_type"),
    "TiImageExtent": ("width", "height", "depth", "array_layer_count"),
    "TiTexture": ("image", "sampler", "dimension", "extent", "format"),
    "TiScalarValue": ("x8", "x16", "x32", "x64"),
    "TiScalar": ("type", "value"),
    "TiTensorValue": ("x8", "x16", "x32", "x64"),
    "TiTensorValueWithLength": ("length", "data"),
    "TiTensor": ("type", "contents"),
    "TiArgumentValue": ("i32", "f32", "ndarray", "texture", "scalar", "tensor"),
    "TiArgument": ("type", "value"),
}

# Enum values the shim hard-codes. A renumbering here is silent and total:
# every argument would be typed wrong.
_ENUM_PROBES = {
    "TI_ARCH_CUDA": 3,
    "TI_ARCH_X64": 4,
    "TI_ARGUMENT_TYPE_I32": 0,
    "TI_ARGUMENT_TYPE_F32": 1,
    "TI_ARGUMENT_TYPE_NDARRAY": 2,
    "TI_ARGUMENT_TYPE_SCALAR": 4,
    "TI_DATA_TYPE_F16": 0,
    "TI_DATA_TYPE_F32": 1,
    "TI_DATA_TYPE_F64": 2,
    "TI_DATA_TYPE_I32": 5,
    "TI_DATA_TYPE_I64": 6,
    "TI_DATA_TYPE_U8": 8,
    "TI_ERROR_SUCCESS": 0,
}


def _c_source() -> str:
    lines = [
        "#include <stdio.h>",
        "#include <stddef.h>",
        "#include <taichi/taichi.h>",
        "int main(void) {",
        '  printf("{\\n");',
    ]
    entries = []
    for struct, fields in _PROBES.items():
        rows = ['\\"sizeof\\": %zu, \\"alignof\\": %zu']
        args = [f"sizeof({struct})", f"_Alignof({struct})"]
        for field in fields:
            rows.append(f'\\"offsetof.{field}\\": %zu')
            args.append(f"offsetof({struct}, {field})")
        entries.append(
            f'  printf("  \\"{struct}\\": {{{", ".join(rows)}}},\\n", '
            + ", ".join(args)
            + ");"
        )
    for name in _ENUM_PROBES:
        entries.append(f'  printf("  \\"{name}\\": %d,\\n", (int)({name}));')
    lines.extend(entries)
    # Trailing sentinel keeps the JSON valid without tracking the last comma.
    lines.append('  printf("  \\"_end\\": 0\\n}\\n");')
    lines.append("  return 0;")
    lines.append("}")
    return "\n".join(lines)


def _oracle() -> dict:
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        return {}
    include = taichi_lib_root() / "c_api" / "include"
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "probe.c"
        binary = Path(tmp) / "probe"
        source.write_text(_c_source())
        subprocess.run(
            [compiler, "-std=c11", f"-I{include}", str(source), "-o", str(binary)],
            check=True,
            capture_output=True,
        )
        out = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def main() -> int:
    try:
        oracle = _oracle()
    except subprocess.CalledProcessError as error:
        print("C oracle failed to build:")
        print(error.stderr.decode("utf-8", "replace"))
        return 1
    if not oracle:
        print("SKIP: no C compiler, cannot build the layout oracle")
        return 0

    include = taichi_lib_root() / "c_api" / "include"
    print(f"oracle built against {include}")

    failures = []
    for struct, fields in _PROBES.items():
        want = oracle[struct]
        got = LAYOUT[struct]
        mismatched = [
            f"{struct}.{key}: C says {want[key]}, ctypes says {got.get(key)}"
            for key in ("sizeof", "alignof", *(f"offsetof.{f}" for f in fields))
            if want[key] != got.get(key)
        ]
        failures.extend(mismatched)
        print(
            f"  {struct:26s} sizeof={want['sizeof']:4d} align={want['alignof']:2d}"
            f"  {'MISMATCH' if mismatched else 'ok'}"
        )

    import _taichi_c_api_shim as shim

    for name, expected in _ENUM_PROBES.items():
        c_value = oracle[name]
        py_value = getattr(shim, name)
        if c_value != py_value:
            failures.append(f"{name}: C says {c_value}, shim says {py_value}")
        if c_value != expected:
            failures.append(f"{name}: C says {c_value}, this check expected {expected}")
    print(f"  {len(_ENUM_PROBES)} enum values checked")

    if failures:
        print("\nFAIL — the ctypes declarations do not match the installed headers:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nPASS — every struct size, field offset and enum value agrees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
