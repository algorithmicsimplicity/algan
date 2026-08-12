"""Hard host-RAM ceiling for benchmark / probe processes.

A benchmark that mis-sizes its synthetic data can exhaust system RAM and take
the whole machine down with it (this has happened: a generator produced a
250x-too-large edit log and blue-screened Windows). Ceilings belong on any
script that sizes tensors from parameters rather than from a real scene.

``cap_process_memory()`` puts the process in a Windows job object with
``JOB_OBJECT_LIMIT_PROCESS_MEMORY``, so an over-large allocation fails with
``MemoryError`` in *this* process instead of paging the machine to death. On
non-Windows it falls back to ``resource.RLIMIT_AS``. Call it before importing
torch-heavy modules::

    from _memory_cap import cap_process_memory

    cap_process_memory(2)

**Do not cap a real render.** On Windows, WDDM charges GPU allocations against
the process's system-memory commit, so the render arena (~0.4 of free VRAM)
comes out of this budget; and native CUDA/Taichi code does not handle a failed
commit gracefully -- capping a full scene render at 4 GB segfaults it (exit
139) rather than raising. Cap the scripts whose tensor sizes come from
*parameters* (where a typo produces a 250x-too-large allocation); a render is
sized by the scene and by the engine's own arena logic, which already has an
OOM-retry path.
"""

from __future__ import annotations

import sys

DEFAULT_LIMIT_GB = 2


def cap_process_memory(gigabytes=DEFAULT_LIMIT_GB, *, strict=True):
    """Fail allocations past ``gigabytes`` of committed memory.

    Raises by default when the ceiling could not be installed -- a silently
    uncapped run is exactly the failure mode this guards against. Pass
    ``strict=False`` to get a warning and False instead.
    """
    limit = int(gigabytes * (1 << 30))
    installed = _cap_windows(limit) if sys.platform == "win32" else _cap_posix(limit)
    if not installed and strict:
        raise RuntimeError(
            f"could not cap this process at {gigabytes} GB of host memory; "
            f"refusing to run uncapped"
        )
    return installed


def _cap_posix(limit):
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return True


def _cap_windows(limit):
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JobObjectExtendedLimitInformation = 9

    # Declare the handle-returning calls explicitly: ctypes defaults restype to
    # a 32-bit int, which truncates GetCurrentProcess's (HANDLE)-1 pseudo-handle
    # to 0x00000000FFFFFFFF and fails the assignment with ERROR_INVALID_HANDLE.
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        print("memory cap: CreateJobObject failed", file=sys.stderr)
        return False
    # The handle must outlive this call: closing it would end the job (and
    # kill the process). Park it on the module.
    globals()["_JOB_HANDLE"] = job

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = limit
    ok = kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        print(
            f"memory cap: SetInformationJobObject failed "
            f"({ctypes.get_last_error()})",
            file=sys.stderr,
        )
        return False
    ok = kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
    if not ok:
        print(
            f"memory cap: AssignProcessToJobObject failed "
            f"({ctypes.get_last_error()})",
            file=sys.stderr,
        )
        return False
    return True
