"""The traversal's ray and node-visit counters (``trace_counters_taichi``).

Off by default and compiled out: no field exists and :func:`read` is empty.
On (``ALGAN_TRACE_COUNTERS``, read at import, so a subprocess), the counts
must add up -- every shadow ray answered exactly one way, every opaque-first
answer re-derived by the march in ``verify`` with no mismatch -- and counting
must not move a pixel.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from algan.rendering.raytracing import trace_counters_taichi as tc

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

_CHILD = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import test_opaque_first_shadows as scene
from algan.animation_timeline import timeline
from algan.rendering.raytracing import trace_counters_taichi as tc
out = {}
for arm, disabled in (("on", ()), ("off", ("opaquefirst",))):
    timeline._OPT_DISABLED = frozenset(disabled)
    tc.reset()
    frame = scene._render(Path(sys.argv[2]) / f"{arm}.png")
    out[arm] = {"counts": tc.read(), "md5": hashlib.md5(frame.tobytes()).hexdigest()}
print("COUNTS " + json.dumps(out))
"""


def test_counting_is_off_and_compiled_out_by_default():
    assert not tc.counting()
    assert tc._COUNTS is None
    assert tc.read() == {}
    assert "off" in tc.format_counts(tc.read())


def test_counts_add_up_verify_finds_no_mismatch_and_pixels_do_not_move(tmp_path):
    env = dict(
        os.environ,
        ALGAN_TRACE_COUNTERS="verify",
        ALGAN_USE_DAEMON="0",
        ALGAN_AUTO_DAEMON="0",
    )
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, str(_HERE), str(tmp_path)],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("COUNTS "))
    result = json.loads(line[len("COUNTS ") :])
    on, off = result["on"]["counts"], result["off"]["counts"]

    rays = on["shadow_rays"]
    assert rays > 1000
    answered = (
        on["shadow_opaque_first_blocked"]
        + on["shadow_opaque_first_clear"]
        + on["shadow_opaque_first_march"]
    )
    assert answered == rays
    assert (
        min(
            on["shadow_opaque_first_blocked"],
            on["shadow_opaque_first_clear"],
            on["shadow_opaque_first_march"],
        )
        > 0
    )
    assert on["shadow_opaque_first_mismatch"] == 0
    assert on["shadow_march_capped"] == 0
    # Every finite ray walks once for the first pass; verify marches them all.
    assert on["shadow_anyhit_walks"] == rays
    assert on["shadow_nearest_walks"] >= rays
    assert on["shadow_anyhit_nodes"] >= on["shadow_anyhit_walks"]
    assert on["shadow_anyhit_tri_tests"] > 0
    assert on["shadow_anyhit_bez_tests"] > 0

    # The kill-switch arm: the same rays, all marched.
    assert off["shadow_rays"] == rays
    assert off["shadow_opaque_first_blocked"] == off["shadow_opaque_first_march"] == 0
    assert off["shadow_anyhit_walks"] == 0
    assert off["shadow_nearest_walks"] >= rays

    # Counting is a diagnostic: both arms' frames are the uncounted frame.
    import test_opaque_first_shadows as scene

    plain = scene._render(tmp_path / "plain.png")
    md5 = hashlib.md5(plain.tobytes()).hexdigest()
    assert result["on"]["md5"] == md5
    assert result["off"]["md5"] == md5
