r"""Regenerate ``algan/rendering/kernel_specs_builtin.json``, the built-in kernel list.

The built-in list is what lets the *first* render on a machine compile in
parallel: with no manifest of its own yet, a process can still hand the common
set -- the kernel specializations the ``algan warmup`` variant scenes use -- to
background workers (``algan/rendering/kernel_precompile.py``). This script
records that set by rendering each variant scene in a fresh process against an
empty kernel cache, and collects what each one materialized, with its cold
compile time (the pool schedules longest first by it)::

    <venv-python> scripts/generate_kernel_specs.py

Run it when a kernel's signature or a pipeline's choice of kernels changes; a
spec that no longer resolves is skipped by the workers rather than failing
anything, so a stale list costs parallelism, not correctness.
``tests/unit_tests/test_kernel_precompile.py`` checks that every entry still
resolves to a kernel with the recorded arity. The times are the generating
machine's and serve only as relative weights.

Scenes run one at a time so each cold compile time is measured without
competition. ``--variants`` limits the set; ``--check`` reports whether the
committed list would change instead of writing it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "algan" / "rendering" / "kernel_specs_builtin.json"

# This process never imports algan: everything it needs comes back from its
# children, so it cannot be handed to a render daemon, start a precompile
# pool, or be told at exit that it rendered nothing.
_INFO = """
import json
from algan.rendering.kernel_precompile import _MANIFEST_SCHEMA
from algan.rendering.kernel_warmup import VARIANTS
from algan.settings._startup import render_device
from algan.taichi_compat import describe_backend
print("REPORT " + json.dumps({
    "schema": _MANIFEST_SCHEMA,
    "variants": {name: v.raytracing for name, v in VARIANTS.items()},
    "compiler": describe_backend(),
    "device": str(render_device()),
}))
"""

_RECORD = """
import json
from algan.rendering.kernel_warmup import run_variant
from algan.rendering import kernel_precompile as kp
run_variant({name!r}, {scratch!r})
kp.flush_manifest()
rows = kp.read_manifest()["entries"].values()
print("REPORT " + json.dumps({{
    kp.spec_id(row["spec"]): [row["spec"], float(row.get("seconds") or 0.0)]
    for row in rows
}}))
"""


def _child(code, scratch):
    env = dict(os.environ)
    env.update(
        {
            "ALGAN_CACHE_DIR": str(Path(scratch) / "cache"),
            "ALGAN_HOME": str(Path(scratch) / "home"),
            "ALGAN_USE_DAEMON": "0",
            "ALGAN_AUTO_DAEMON": "0",
            "ALGAN_PRECOMPILE_JOBS": "0",
            "ALGAN_LOG_LEVEL": "WARNING",
        }
    )
    env.pop("ALGAN_DAEMON_CHILD", None)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(f"child failed:\n{result.stderr[-4000:]}")
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("REPORT ")
    )
    return json.loads(line[len("REPORT ") :])


def _record_variant(name, echo):
    """Render one variant cold in a child; return ``{spec_id: [spec, seconds]}``."""
    with tempfile.TemporaryDirectory(prefix="algan_specs_") as scratch:
        recorded = _child(_RECORD.format(name=name, scratch=scratch), scratch)
    total = sum(seconds for _, seconds in recorded.values())
    echo(f"  {name}: {len(recorded)} specializations, {total:.1f} s of cold compiling")
    return recorded


def generate(variants, info, echo=print):
    entries = {}
    for name in variants:
        requires = dict(info["variants"][name])
        for sid, (spec, seconds) in _record_variant(name, echo).items():
            entry = entries.get(sid)
            if entry is None:
                entries[sid] = {
                    "spec": spec,
                    "seconds": round(seconds, 2),
                    "variants": [name],
                    "requires": requires,
                }
                continue
            entry["seconds"] = round(max(entry["seconds"], seconds), 2)
            entry["variants"].append(name)
            # Needed by a scene with fewer requirements: keep only the ones
            # every variant that uses it shares.
            entry["requires"] = {
                key: value
                for key, value in entry["requires"].items()
                if requires.get(key) == value
            }
    return {
        "schema": info["schema"],
        "generated": {
            "compiler": info["compiler"],
            "device": info["device"],
            "python": ".".join(map(str, sys.version_info[:2])),
            "by": "scripts/generate_kernel_specs.py",
        },
        "entries": dict(sorted(entries.items())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variants", default=None, help="comma-separated subset")
    parser.add_argument(
        "--check", action="store_true", help="report whether the list would change"
    )
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="algan_specs_") as scratch:
        info = _child(_INFO, scratch)
    variants = args.variants.split(",") if args.variants else list(info["variants"])
    print(f"Recording {len(variants)} variant(s) against empty kernel caches:")
    data = generate(variants, info)
    text = json.dumps(data, indent=1, sort_keys=True) + "\n"
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        old = json.loads(current)["entries"] if current else {}
        changed = set(old) ^ set(data["entries"])
        print(
            f"{len(changed)} specialization(s) differ from the committed list."
            if changed
            else "The committed list is current."
        )
        return 1 if changed else 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(
        f"Wrote {len(data['entries'])} specializations to {OUTPUT.relative_to(REPO)}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
