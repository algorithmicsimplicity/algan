"""A pixel the count pass accepts and the write pass then misses is inert.

The emission's COUNT kernel decides which candidate pixels each triangle
covers and reserves one fragment slot apiece; the WRITE kernel replays those
decisions and fills the slots from its own evaluation of the same geometry.
The two run the same code, so they should never disagree. If they did, the
write used to store its miss as it came back -- pixel 0, depth 0, full
coverage: an opaque fragment nobody emitted, which the opt-in key check could
not tell from a real one. It now stores an inert record (the candidate's own
pixel, no samples, zero coverage, the smallest positive depth) and the key
check counts those.

The disagreement is forced here by widening the count pass's acceptance bits to
every candidate pixel of every pair, so the write pass misses at each box pixel
the triangle does not cover. The frame must come out as it does unforced.

One way to cause it for real was a setting: the acceptance bits are an i32, one
per candidate pixel of a pair, so ``ALGAN_RASTER_CHUNK`` above 32 left pixels
with no bit. It is now clamped to 32, with a warning.
"""

import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import RED
from algan.mobs.shapes_3d import Cube
from algan.rendering.raytracing import raster_pipeline as rp
from algan.scene import Scene
from algan.settings.video_settings import LD

VIDEO = LD.set(resolution=(96, 54))


def _accept_every_candidate(real_count):
    """``raster_tri_count`` with every in-box candidate pixel accepted."""

    def count(pairs, num_pairs, *args):
        real_count(pairs, num_pairs, *args)
        counts, accepts = args[-2], args[-1]
        rows = pairs[:num_pairs].long()
        n = (rows[:, 4] * rows[:, 5] - rows[:, 6]).clamp(0, rp.raster_chunk)
        bits = torch.where(n >= 32, torch.full_like(n, -1), (1 << n.clamp_max(31)) - 1)
        accepts[:num_pairs] = bits.to(torch.int32)
        counts[:num_pairs] = n.to(torch.int32)

    return count


def _render(path):
    with Scene() as scene:
        with Off():
            cube = Cube(color=RED).scale(2).rotate(30, torch.tensor([1.0, 1.0, 0.0]))
        cube.spawn(animate=False)
        scene.save_frame(str(path), video_settings=VIDEO)
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int32)


def test_a_write_pass_miss_is_stored_inertly_and_counted(tmp_path, monkeypatch, capsys):
    reference = _render(tmp_path / "reference.png")
    monkeypatch.setenv("ALGAN_RASTER_KEY_CHECK", "1")
    monkeypatch.setattr(
        rp, "raster_tri_count", _accept_every_candidate(rp.raster_tri_count)
    )
    capsys.readouterr()
    forced = _render(tmp_path / "forced.png")
    log = capsys.readouterr().out
    misses = [
        int(line.split("] ")[1].split(" of ")[0])
        for line in log.splitlines()
        if "write pass's miss record" in line
    ]
    assert misses, log
    assert sum(misses) > 100, log
    # No key outside the tile either: every miss record is a real pixel.
    assert "outside [0," not in log, log
    assert (reference.sum(-1) > 30).sum() > 500, "the cube is meant to be on screen"
    diff = np.abs(forced - reference).max(-1)
    assert int(diff.max()) <= 1, f"{int((diff > 1).sum())} pixels moved"


def test_a_raster_chunk_past_the_acceptance_mask_is_clamped_with_a_warning():
    env = dict(os.environ, ALGAN_RASTER_CHUNK="64", ALGAN_AUTO_DAEMON="0")
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "always",
            "-c",
            "import algan.rendering.raytracing.raster_taichi as rt\n"
            "print(rt.raster_chunk)\n",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "32"
    assert "ALGAN_RASTER_CHUNK=64 exceeds the 32 pixels" in result.stderr
