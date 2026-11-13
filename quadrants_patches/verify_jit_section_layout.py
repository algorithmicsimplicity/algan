"""Prove `0012-windows-cpu-jit-section-layout.patch` did what it claims.

On Windows the CPU backend JIT-links each kernel with LLVM's RuntimeDyld. A
kernel's `.pdata` unwind entries address its code with
`IMAGE_REL_AMD64_ADDR32NB`, a 32-bit offset from the lowest of that object's
sections, so all of one object's sections must sit within 4 GB of each other.
Unpatched, Quadrants gives each object a default `SectionMemoryManager`, which
maps code, read-only data and read-write data with separate
`VirtualAlloc(NULL, ...)` calls. Windows puts each at the lowest free range
large enough. If the read-only data fits a low hole while a code section larger
than one 64 KB allocation granule only fits more than 4 GB higher, the link
aborts with

    LLVM Fatal Error: IMAGE_REL_AMD64_ADDR32NB relocation requires an ordered
    section layout

How often that happens depends on how fragmented the process's address space
is when a kernel is linked (also when it is loaded from the offline cache), so
on a real machine it is intermittent. 0012 maps each object's sections as one
block, which this gate checks on the CPU backend::

    python verify_jit_section_layout.py

1. Compiles a kernel whose code is well over 64 KB, and checks that it really
   is that large and computes what a numpy model of it computes -- otherwise
   the gate would prove nothing.
2. **The criterion.** Checks that the `VirtualAlloc` block holding the new
   code also holds a read-only region: the kernel's `.pdata`, `.xdata` and
   constants. With 0012 code and read-only data are carved from one block;
   without it they never share one. This does not depend on where Windows
   happens to put anything, so it gives the same answer on every run.
3. **The failure itself.** Reserves (never commits) the address space above
   the process's allocation floor so that only isolated 64 KB holes remain in
   the first 1 GB and nothing is free for about 9 GB above them, then links a
   second kernel there -- one constant different, so Quadrants cannot reuse
   the first's code -- and checks it links, runs and matches the model, and
   repeats step 2 on it. Without 0012 this usually aborts with the error
   above, but not always: memory the process allocated before the layout and
   freed during the compile opens gaps inside the wall, and the code can fit
   one. That is why step 2, not this step, decides the gate.

Exit 0 when both kernels pass, 1 when either fails (the bug), and 2 when the
gate could not test anything -- the kernel was too small, the model disagreed,
the layout did not take or nothing was linked -- which is reported rather
than passed. Off Windows it prints a skip line and exits 0: macOS arm64 links
with JITLink and Linux with ELF, neither of which emits these relocations, and
0012 leaves both unchanged.
"""

# ruff: noqa: I002 -- I002 would insert `from __future__ import annotations`,
# which turns a kernel's runtime-evaluated annotations into strings and breaks
# decoration; see `verify_invariant_load.py` beside this file for the long form.
import ctypes
import sys
import time

import numpy as np

# Module level on purpose: kernel annotations resolve against this module's
# globals, so a function-local import would fail to decorate them.
import quadrants as qd

#: Unrolled steps in the test kernels: a 64-bit multiply and add with
#: ``movabs`` immediates, a shift and an xor each. On the RTX 5080 box 1200
#: steps came to 116 KB of code in 4 s of compiling (3500 gave 336 KB in 35 s).
STEPS = 1200
#: The code must exceed this for the gate to exercise the bug: a section that
#: fits one Windows allocation granule fits the holes the data sections use.
MIN_CODE = 80 * 1024
MULTIPLIER = 6364136223846793005
INCREMENT = 1442695040888963407
STRIDE = 2654435761

GRANULE = 64 * 1024
GB = 1 << 30
MEM_COMMIT, MEM_RESERVE, MEM_RELEASE, MEM_FREE = 0x1000, 0x2000, 0x8000, 0x10000
MEM_PRIVATE = 0x20000
PAGE_NOACCESS, PAGE_READONLY = 0x01, 0x02
#: PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE.
EXECUTABLE = (0x10, 0x20, 0x40)


def _make_kernel(salt):
    """A kernel with a large unrolled body.

    ``salt`` changes its constants: identical kernels share compiled code, so
    two salts are two links.
    """

    @qd.kernel
    def mix(x: qd.types.ndarray(qd.i64, ndim=1)):
        for i in range(x.shape[0]):
            v = x[i]
            for k in qd.static(range(STEPS)):
                v = (v * MULTIPLIER + (INCREMENT + k * STRIDE + salt)) ^ (v >> 29)
            x[i] = v

    return mix


def _model(seed, salt):
    """``_make_kernel(salt)`` in numpy: wrapping int64, arithmetic shift."""
    v = seed.copy()
    with np.errstate(over="ignore"):
        for k in range(STEPS):
            step = np.int64(INCREMENT + k * STRIDE + salt)
            v = (v * np.int64(MULTIPLIER) + step) ^ (v >> 29)
    return v


class _MBI(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", ctypes.c_ulong),
        ("PartitionId", ctypes.c_ushort),
        ("RegionSize", ctypes.c_size_t),
        ("State", ctypes.c_ulong),
        ("Protect", ctypes.c_ulong),
        ("Type", ctypes.c_ulong),
    ]


class _AddressSpace:
    """The few kernel32 calls the gate needs, through ctypes."""

    def __init__(self):
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualAlloc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
        k32.VirtualQuery.restype = ctypes.c_size_t
        k32.VirtualQuery.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_MBI),
            ctypes.c_size_t,
        ]
        self._k32 = k32
        self._reserved = []

    def regions(self, stop=1 << 47):
        """``(base, size, state, protect, type)`` for every region below ``stop``."""
        addr, mbi = GRANULE, _MBI()
        size_of = ctypes.sizeof(mbi)
        while addr < stop and self._k32.VirtualQuery(addr, ctypes.byref(mbi), size_of):
            base = mbi.BaseAddress or 0
            yield base, mbi.RegionSize, mbi.State, mbi.Protect, mbi.Type
            addr = base + mbi.RegionSize

    def committed(self, protections):
        """Committed private regions with one of ``protections``, as ``(base, size)``.

        JIT code is private and executable; its constants and unwind tables
        private and read-only.
        """
        return {
            (base, size)
            for base, size, state, protect, kind in self.regions()
            if state == MEM_COMMIT and kind == MEM_PRIVATE and protect in protections
        }

    def snapshot(self):
        return self.committed(EXECUTABLE), self.committed((PAGE_READONLY,))

    def protections_in_block(self, address):
        """Protections of the committed regions of the block holding ``address``.

        The block is one ``VirtualAlloc``: every region sharing its allocation base.
        """
        mbi, size_of = _MBI(), ctypes.sizeof(_MBI)
        if not self._k32.VirtualQuery(address, ctypes.byref(mbi), size_of):
            return []
        block, addr, out = mbi.AllocationBase, mbi.AllocationBase, []
        while self._k32.VirtualQuery(addr, ctypes.byref(mbi), size_of):
            if mbi.AllocationBase != block:
                break
            if mbi.State == MEM_COMMIT:
                out.append(mbi.Protect)
            addr = (mbi.BaseAddress or 0) + mbi.RegionSize
        return out

    def where_null_lands(self, size):
        """Where ``VirtualAlloc(NULL, size)`` would put a block right now."""
        p = self._k32.VirtualAlloc(None, size, MEM_RESERVE, PAGE_NOACCESS)
        if p:
            self._k32.VirtualFree(p, 0, MEM_RELEASE)
        return p or 0

    def reserve(self, base, size):
        p = self._k32.VirtualAlloc(base, size, MEM_RESERVE, PAGE_NOACCESS)
        if p:
            self._reserved.append(p)
        return size if p else 0

    def release_all(self):
        for p in self._reserved:
            self._k32.VirtualFree(p, 0, MEM_RELEASE)
        self._reserved.clear()


def _build_layout(space):
    """Leave isolated 64 KB holes in the first GB above the floor, a wall above.

    Returns ``(hole_ceiling, wall_top)``, or ``None`` if the allocator did not
    end up placing small and large blocks as the gate needs.
    """
    floor = min(space.where_null_lands(GRANULE), space.where_null_lands(16 << 20))
    hole_ceiling, wall_top = floor + 1 * GB, floor + 10 * GB
    holes = reserved = 0
    for base, size, state, _protect, _kind in list(space.regions(stop=wall_top)):
        if state != MEM_FREE:
            continue
        # A free region ends where an allocation begins, on a granule boundary,
        # but often starts mid-granule, after the last page of the one below.
        # An allocation can only start on a granule boundary, so that tail is
        # unusable anyway, and a reservation based there would be rounded down
        # into the neighbour and fail.
        end = min(base + size, wall_top)
        start = -(-base // GRANULE) * GRANULE
        if start < hole_ceiling and end - start > GRANULE:
            start += GRANULE  # an isolated granule at the start of the region
            holes += 1
        if end > start:
            reserved += space.reserve(start, end - start)
    small = space.where_null_lands(GRANULE)
    large = space.where_null_lands(16 << 20)
    print(
        f"layout: {holes} isolated 64 KB holes below {hole_ceiling:#x}, "
        f"{reserved / GB:.0f} GB reserved up to {wall_top:#x}; a 64 KB block now "
        f"lands at {small:#x}, a 16 MB one at {large:#x}"
    )
    if not (small and small < hole_ceiling and large >= wall_top):
        return None
    return hole_ceiling, wall_top


def _cannot_test(why):
    print(f"0012 gate: could not test -- {why}")
    return 2


def _one_block(space, name, before):
    """Step 2 for the kernel linked since ``before``: is its read-only data in
    its code's block? Returns an exit code, or ``None`` when it passes.
    """
    exec_before, readonly_before = before
    exec_now, readonly_now = space.snapshot()
    linked, readonly = exec_now - exec_before, readonly_now - readonly_before
    code = sum(size for _base, size in linked)
    if code < MIN_CODE:
        return _cannot_test(
            f"{name}: {code // 1024} KB of new code, not over {MIN_CODE // 1024} KB"
        )
    if not readonly:
        return _cannot_test(f"{name}: no new read-only data, so nothing to look for")
    base = max(linked, key=lambda region: region[1])[0]
    protections = space.protections_in_block(base)
    names = ", ".join(_PROTECTION_NAMES.get(p, hex(p)) for p in protections)
    print(f"{name}: {code // 1024} KB of code at {base:#x}; its block holds {names}")
    if PAGE_READONLY not in protections:
        nearest = min(abs(start - base) for start, _size in readonly)
        print(
            f"0012 gate: failed -- {name}'s read-only data is in another block "
            f"({nearest / (1 << 20):.1f} MB from its code, where nothing keeps it "
            "within 4 GB): 0012 is missing or ineffective"
        )
        return 1
    return None


_PROTECTION_NAMES = {
    0x02: "read-only",
    0x04: "read-write",
    0x10: "execute",
    0x20: "execute-read",
    0x40: "execute-read-write",
}


def main() -> int:
    if sys.platform != "win32":
        print(
            f"0012 gate: skipped on {sys.platform} "
            "(the relocation is COFF-only, and 0012 changes nothing here)"
        )
        return 0
    qd.init(
        arch=qd.cpu,
        offline_cache=False,
        src_ll_cache=False,
        unrolling_limit=0,
        default_ip=qd.i64,
    )
    space = _AddressSpace()
    seed = np.arange(1, 1025, dtype=np.int64)

    # 1-2. An ordinary layout: big enough, the model is right, one block.
    before = space.snapshot()
    t0 = time.perf_counter()
    first = seed.copy()
    _make_kernel(salt=0)(first)
    qd.sync()
    print(f"first kernel: compiled and ran in {time.perf_counter() - t0:.1f} s")
    if not np.array_equal(first, _model(seed, salt=0)):
        return _cannot_test("the first kernel and its numpy model disagree")
    failed = _one_block(space, "first kernel", before)
    if failed is not None:
        return failed

    # 3. The adversarial layout, and a kernel that has to be linked into it.
    second = _make_kernel(salt=1)
    want = _model(seed, salt=1)
    got = seed.copy()
    try:
        if _build_layout(space) is None:
            return _cannot_test("the address-space layout did not take")
        before = space.snapshot()
        try:
            second(got)
            qd.sync()
        except RuntimeError as exc:
            if "ADDR32NB" not in str(exc):
                raise
            print(f"FAIL  {str(exc).strip().splitlines()[-1][-160:]}")
            print(
                "0012 gate: failed -- the second kernel's sections were mapped "
                "more than 4 GB apart (0012 is missing or ineffective)"
            )
            return 1
        failed = _one_block(space, "second kernel", before)
    finally:
        space.release_all()
    if failed is not None:
        return failed
    if not np.array_equal(got, want):
        print("0012 gate: failed -- the second kernel computed the wrong result")
        return 1
    print(
        "0012 gate: passed -- each kernel's code and read-only data share one "
        "block, and a kernel linked while only 64 KB holes were free low down"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
