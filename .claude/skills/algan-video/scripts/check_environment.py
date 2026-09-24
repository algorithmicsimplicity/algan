#!/usr/bin/env python3
"""Non-invasive Algan preflight; does not import Algan or initialize a GPU.

Uses the Python standard library only. Run with the same interpreter that will
render. Missing optional capabilities are reported, not treated as core failures,
unless explicitly requested. Algan's own `algan check` and a render remain needed.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any


def package_info(distribution: str, module: str | None = None) -> dict[str, Any]:
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        version = None
    origin = None
    available = False
    error = None
    if module:
        try:
            spec = importlib.util.find_spec(module)
            available = spec is not None
            if spec is not None:
                origin = spec.origin
        except (ImportError, ValueError, AttributeError) as exc:
            error = str(exc)
    return {"distribution": distribution, "version": version,
            "module": module, "discoverable": available,
            "module_origin": origin, "discovery_error": error}


def bundled_ffmpeg_candidates() -> list[str]:
    """Discover packaged executables without importing imageio or running them."""
    try:
        dist = metadata.distribution("imageio-ffmpeg")
    except metadata.PackageNotFoundError:
        return []
    paths = []
    for relative in dist.files or ():
        value = str(relative).replace("\\", "/")
        if "/binaries/" in value and "ffmpeg" in Path(value).name.lower():
            path = Path(dist.locate_file(relative))
            if path.is_file():
                paths.append(str(path.resolve()))
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-pango", action="store_true",
                        help="Fail if the ManimPango binding is not discoverable.")
    parser.add_argument("--require-latex", action="store_true",
                        help="Fail unless latex and dvisvgm are on PATH.")
    parser.add_argument("--require-ffprobe", action="store_true",
                        help="Fail unless FFprobe is on PATH.")
    parser.add_argument("--json", action="store_true", help="Print a JSON report.")
    args = parser.parse_args(argv)

    packages = {
        "algan": package_info("algan", "algan"),
        "torch": package_info("torch", "torch"),
        "algan_quadrants": package_info("algan-quadrants", "quadrants"),
        "quadrants_distribution": package_info("quadrants"),
        "taichi": package_info("taichi", "taichi"),
        "manimpango": package_info("manimpango", "manimpango"),
        "moviepy": package_info("moviepy", "moviepy"),
        "imageio_ffmpeg": package_info("imageio-ffmpeg", "imageio_ffmpeg"),
    }
    executables = {name: shutil.which(name) for name in
                   ("algan", "ffmpeg", "ffprobe", "latex", "dvisvgm",
                    "espeak-ng", "espeak")}
    bundled = bundled_ffmpeg_candidates()
    problems: list[str] = []
    warnings: list[str] = []

    for name in ("algan", "torch", "moviepy"):
        if not packages[name]["discoverable"]:
            problems.append(f"{name} is not discoverable with this interpreter.")
    if not (packages["algan_quadrants"]["discoverable"]
            or packages["taichi"]["discoverable"]):
        problems.append("Neither quadrants nor taichi is discoverable; install matching Algan dependencies.")
    if (packages["algan_quadrants"]["version"] is not None
            and packages["quadrants_distribution"]["version"] is not None):
        problems.append("Both algan-quadrants and quadrants distributions are installed; they share an import package.")
    if not executables["ffmpeg"] and not bundled:
        warnings.append("No PATH or imageio-bundled FFmpeg found; check Algan/MoviePy's actual configured binary.")
    if args.require_pango and not packages["manimpango"]["discoverable"]:
        problems.append("Requested Pango-backed text but manimpango is not discoverable.")
    if args.require_latex:
        for name in ("latex", "dvisvgm"):
            if not executables[name]:
                problems.append(f"Requested LaTeX support but {name} is not on PATH.")
    if args.require_ffprobe and not executables["ffprobe"]:
        problems.append("Requested FFprobe verification but ffprobe is not on PATH.")
    if sys.version_info[:2] < (3, 10) or sys.version_info[:2] > (3, 13):
        warnings.append("This interpreter is outside the source-checked Python 3.10–3.13 range; verify current package support.")
    if packages["algan"]["discoverable"] and not packages["algan"]["version"]:
        warnings.append("Algan source is discoverable without distribution metadata; record its checkout revision.")
    if executables["algan"]:
        warnings.append("Confirm the PATH algan CLI belongs to the Python interpreter reported here.")

    report = {"python_executable": sys.executable, "python_version": sys.version,
              "packages": packages, "executables_on_path": executables,
              "unexecuted_bundled_ffmpeg_candidates": bundled,
              "problems": problems, "warnings": warnings,
              "preflight_passed": not problems,
              "scope": "Discovery only: no imports, GPU checks, shader compilation, synthesis, or rendering performed."}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Python: {sys.executable}\nVersion: {sys.version.split()[0]}")
        for key, info in packages.items():
            print(f"{key}: version={info['version']!r}, discoverable={info['discoverable']}, origin={info['module_origin']}")
        for name, path in executables.items():
            print(f"{name}: {path or 'not on PATH'}")
        for path in bundled:
            print(f"Bundled FFmpeg candidate (not executed): {path}")
        for message in warnings:
            print(f"WARNING: {message}")
        for message in problems:
            print(f"PROBLEM: {message}")
        print(report["scope"])
        print("Preflight: " + ("passed discovery checks" if not problems else "missing/conflicting requirements"))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
