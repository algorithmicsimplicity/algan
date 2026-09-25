"""Per-file process/MPS telemetry for the MPS suite diagnostic."""
import os
import subprocess

import torch

def _rss_mb():
    try:
        raw = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
        ).strip()
        return int(raw) / 1024
    except Exception:
        return -1.0

def _zero_copy():
    try:
        from algan.rendering.mps_zero_copy import cache_stats
        return cache_stats()
    except Exception as exc:
        return {"error": repr(exc)}

def pytest_runtest_teardown(item, nextitem):
    if nextitem is not None and nextitem.path == item.path:
        return
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        cur = torch.mps.current_allocated_memory() / (1024 * 1024)
        drv = torch.mps.driver_allocated_memory() / (1024 * 1024)
    else:
        cur = drv = -1.0
    print(
        "FILEMEM",
        str(item.path),
        "rss_mb", round(_rss_mb(), 1),
        "mps_cur_mb", round(cur, 1),
        "mps_drv_mb", round(drv, 1),
        "zero_copy", _zero_copy(),
        flush=True,
    )
