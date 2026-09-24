"""Check an installed Algan wheel, without consulting the source checkout.

Run with the wheel environment's Python in isolated mode (``-I``), from an
empty directory. This script uses only the standard library until the import
whose provenance it checks.
"""

from __future__ import annotations

import importlib.resources as resources
import os
import sys
import sysconfig
from pathlib import Path


def main():
    if not sys.flags.isolated:
        raise SystemExit("Run the wheel check with isolated Python (-I)")

    # A warm daemon uses its own interpreter and import path. This check must
    # inspect the wheel in this process, even if a daemon is already running.
    os.environ["ALGAN_USE_DAEMON"] = "0"
    import algan

    installed_roots = {
        Path(sysconfig.get_path(kind)).resolve() / "algan"
        for kind in ("purelib", "platlib")
    }
    origin = Path(algan.__file__).resolve()
    if origin.parent not in installed_roots:
        raise SystemExit(f"algan imported outside the wheel environment: {origin}")
    print(f"algan imported from {origin}")

    root = resources.files("algan")
    required = (
        "rendering/post_processing/anti_aliasing/AreaTex.png",
        "rendering/post_processing/anti_aliasing/SearchTex.png",
        "rendering/raytracing/_glass_energy_lut.npz",
        "external_libraries/manim/_config/default.cfg",
        "viewer/static/index.html",
        "viewer/static/viewer.css",
        "viewer/static/viewer.js",
        "viewer/static/transcript.js",
        "viewer/static/audio.js",
    )
    missing = [name for name in required if not root.joinpath(name).is_file()]
    if missing:
        raise SystemExit("missing from the installed wheel: " + ", ".join(missing))
    print("all required data files present")

    # The fitted table can change size; the installed module owns its shape.
    from algan.rendering.raytracing.glass_energy import (
        GLASS_ENERGY_SHAPE,
        glass_energy_table,
    )

    assert glass_energy_table().shape == GLASS_ENERGY_SHAPE
    print("glass energy lookup decodes from installed wheel")


if __name__ == "__main__":
    main()
