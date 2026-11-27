"""This process's Linux memory cgroup: its working set and its hard limit.

Kept apart from :mod:`algan.utils.memory_utils`, which imports ``SETTINGS``,
so that ``SETTINGS.computing`` can size its default CPU budget against the
cgroup limit while ``SETTINGS`` is being built, without an import cycle. It
imports nothing from Algan.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _read_positive_int(path):
    try:
        raw = Path(path).read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        return None
    if not raw or raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _reclaimable_page_cache(stat_path, key):
    """The cgroup's inactive file cache from its ``memory.stat``, or 0."""
    try:
        lines = Path(stat_path).read_text(encoding="ascii").splitlines()
    except (OSError, ValueError):
        return 0
    for line in lines:
        name, _, value = line.partition(" ")
        if name == key:
            try:
                return max(int(value), 0)
            except ValueError:
                return 0
    return 0


def _linux_cgroup_memory_usage():
    """Return ``(current, limit)`` for this process's finite memory cgroup.

    Supports cgroup v2 and the common v1 memory-controller mount. The direct
    mount root is tried first because container mount namespaces normally make
    it the process's own cgroup already; ``/proc/self/cgroup`` supplies the
    nested path on hosts that expose the hierarchy root instead.

    ``current`` is the working set: usage less the inactive file cache, the
    figure the kernel reclaims before it would OOM-kill anything (and the one
    ``kubectl top`` and ``docker stats`` report). Raw usage counts every page
    of file cache. On a 14.3 GB cloud container that had read a few GB of
    kernel cache, wheels and test data it stood at 9.4 GB, 5.5 GB of it
    inactive file cache, so :func:`_host_memory_pressure` reported pressure
    after every render and reset the Quadrants program each time.
    """
    if not sys.platform.startswith("linux"):
        return None

    candidates = []
    v2_root = Path("/sys/fs/cgroup")
    candidates.append(
        (v2_root / "memory.current", v2_root / "memory.max", "inactive_file")
    )
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError:
        lines = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, rel = parts
        rel = rel.lstrip("/")
        if hierarchy == "0" and controllers == "":
            root = v2_root / rel
            candidates.append(
                (root / "memory.current", root / "memory.max", "inactive_file")
            )
        elif "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory") / rel
            candidates.append(
                (
                    root / "memory.usage_in_bytes",
                    root / "memory.limit_in_bytes",
                    # v1's own ``inactive_file`` leaves out child cgroups,
                    # which its usage counts.
                    "total_inactive_file",
                )
            )

    seen = set()
    for current_path, limit_path, cache_key in candidates:
        key = (str(current_path), str(limit_path))
        if key in seen:
            continue
        seen.add(key)
        current = _read_positive_int(current_path)
        limit = _read_positive_int(limit_path)
        if current is None or limit is None:
            continue
        # v1 often exposes an enormous sentinel when the controller is
        # effectively unlimited. Do not mistake that for a useful hard cap.
        if limit >= (1 << 60):
            continue
        cache = _reclaimable_page_cache(
            current_path.with_name("memory.stat"), cache_key
        )
        return max(current - cache, 0), limit
    return None


def linux_cgroup_memory_limit():
    """The finite memory limit of this process's cgroup in bytes, or None.

    Inside a container ``/proc/meminfo`` -- and so ``psutil`` -- reports the
    host's RAM, while the kernel OOM-kills the process at this limit.
    """
    usage = _linux_cgroup_memory_usage()
    return None if usage is None else usage[1]
