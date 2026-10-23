"""Which cached kernels have PTX under patch 0010's content-hash key: read off the disk.

    python check_ptx_cache.py <cache dir> [--fast-math 1] [--llvm 22.1.0]
                                          [--version 1.3.3 --commit <sha>]

`<cache dir>` is any directory above Quadrants' `kernel_compilation_manager/`
and `ptx_cache_sm_<cc>/` (Algan's `ALGAN_CACHE_DIR` will do). For every `.qdc`
it reads the SHA-256 the file carries, rebuilds the key
`PtxCache::make_content_cache_key` would look its PTX up under, and checks
for `<key>.ptx` in each `ptx_cache_sm_<cc>`. It needs no GPU and no program:
run it after a render, a precompile pool or a verify step to see what reached
the disk.

The key also names the compiler, so `--version` and `--commit` must be those
of the build that wrote the cache; they default to the installed Quadrants'
(`get_version_string`, `get_commit_hash`), which is right when that build is
still installed. `--llvm` is the LLVM the build links (the start-up banner
prints it). Keep `kContentKeyRevision` below in step with the patch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

#: `PtxCache::kContentKeyRevision` in 0010-cuda-ptx-cache-by-content-hash.patch.
CONTENT_KEY_REVISION = 1


def content_key(content_hash, sm, fast_math, llvm, version, commit):
    """`PtxCache::make_content_cache_key`, in Python."""
    material = (
        f"ptx-by-content-hash/{CONTENT_KEY_REVISION};sm_{sm};"
        f"fast_math={1 if fast_math else 0};llvm={llvm};"
        f"quadrants={version}+{commit};content={content_hash}"
    )
    return "C" + hashlib.sha256(material.encode()).hexdigest()


def read_qdc(path):
    """`(content hash, first task name)` of a `.qdc`, or `None` if it is not one."""
    blob = Path(path).read_bytes()
    if len(blob) < 24 + 64 or blob[:4] != b"QDC\x00":
        return None
    _arch, meta_size, src_size = struct.unpack("<IQQ", blob[4:24])
    if 24 + meta_size + src_size + 64 != len(blob):
        return None
    metadata = blob[24 : 24 + meta_size].decode("utf-8", "replace")
    found = re.search(r'"name"\s*:\s*"([^"]+)"', metadata)
    return blob[-64:].decode("ascii", "replace"), found.group(1) if found else "?"


def summarize(root, *, fast_math, llvm, version, commit):
    root = Path(root)
    qdcs = sorted(root.rglob("kernel_compilation_manager/*.qdc"))
    ptx_dirs = sorted(p for p in root.rglob("ptx_cache_sm_*") if p.is_dir())
    report = {"qdc": len(qdcs), "ptx_dirs": {}}
    kernels = []
    for path in qdcs:
        read = read_qdc(path)
        if read is not None:
            kernels.append((path.stem, *read))
    for ptx_dir in ptx_dirs:
        sm = ptx_dir.name.removeprefix("ptx_cache_sm_")
        files = {p.name for p in ptx_dir.iterdir()}
        wanted = {
            content_key(digest, sm, fast_math, llvm, version, commit) + ".ptx": (
                key,
                name,
            )
            for key, digest, name in kernels
        }
        c_files = {f for f in files if f.startswith("C") and f.endswith(".ptx")}
        report["ptx_dirs"][ptx_dir.name] = {
            "with_content_ptx": sum(f in files for f in wanted),
            "without": sorted(
                f"{name} ({key[:12]})"
                for f, (key, name) in wanted.items()
                if f not in files
            ),
            "content_ptx_files": len(c_files),
            "content_ptx_without_qdc": len(c_files - set(wanted)),
            "ir_keyed_ptx_files": sum(
                f.startswith("T") and f.endswith(".ptx") for f in files
            ),
            "leftover_tmp_files": sum(f.endswith(".tmp") for f in files),
        }
    return report


def _installed_identity():
    try:
        from quadrants._lib import core

        return core.get_version_string(), core.get_commit_hash()
    except Exception:  # noqa: BLE001 -- the caller says what to pass instead
        return None, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cache_dir")
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--llvm", default="22.1.0")
    parser.add_argument("--version", default=None)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args(argv)
    version, commit = args.version, args.commit
    if version is None or commit is None:
        installed = _installed_identity()
        version = version or installed[0]
        commit = commit or installed[1]
    if not version or not commit:
        sys.exit("no Quadrants installed here: pass --version and --commit")
    if not os.path.isdir(args.cache_dir):
        sys.exit(f"{args.cache_dir} is not a directory")
    report = summarize(
        args.cache_dir,
        fast_math=args.fast_math,
        llvm=args.llvm,
        version=version,
        commit=commit,
    )
    print(
        f"keys for quadrants={version}+{commit}, llvm={args.llvm}, "
        f"fast_math={args.fast_math}"
    )
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
