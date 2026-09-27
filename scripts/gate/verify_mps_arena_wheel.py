"""Refuse a macOS distribution without the native owner or with a false ABI tag."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def verify(path):
    path = Path(path)
    assert path.name.endswith("-cp310-abi3-macosx_11_0_universal2.whl"), path.name
    with zipfile.ZipFile(path) as wheel:
        name = "algan/rendering/_mps_arena_native.abi3.so"
        binary = wheel.read(name)
        assert len(binary) > 4096, "native owner is missing or truncated"
        # Universal Mach-O (FAT_MAGIC); a thin host-architecture binary must
        # never be tagged universal2, even if it imports on the build host.
        assert binary[:4] == b"\xca\xfe\xba\xbe", "owner is not universal Mach-O"
        count = int.from_bytes(binary[4:8], "big")
        assert count == 2
        cpu_types = {
            int.from_bytes(binary[8 + 20 * i : 12 + 20 * i], "big")
            for i in range(count)
        }
        assert cpu_types == {0x01000007, 0x0100000C}, cpu_types
        metadata = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL")
        )
        contents = wheel.read(metadata).decode()
        assert "Root-Is-Purelib: false" in contents
        assert "Tag: cp310-abi3-macosx_11_0_universal2" in contents
    print("macOS arena wheel verified:", path.name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_mps_arena_wheel.py <macOS-wheel>")
    verify(sys.argv[1])
