"""A/B the opt-in tiled primary frontend and simple-interior compaction.

usage: tiled_primary_ab.py <arm> [scene] [quality]

Arms select ``SETTINGS.raytracing.experimental`` switches
(``DESIGN_tiled_primary.md``); one arm per process, like ``nn_ablation.py``,
because each arm compiles its own kernel set and a warm reading must not
include the other arm's JIT.

Arms
----
base     both switches off -- this branch's reference configuration
tile     ``raster_tile_binning=True``   (audit Rank 3)
simple   ``raster_simple_interiors=True`` (audit Rank 4)
both     both on

Scenes
------
nn        the ``nn_scene_UHD.py`` workload (shadows off), the standing anchor
overdraw  stacked frame-filling flat opaque quads -- the audit's "heavy opaque
          overlap behind a front surface" workload, which is what Rank 3's
          conservative early rejection exists for. It is also the only shape
          measured so far on which Rank 4 certifies anything: on this scene
          both switches engage (LD, CPU: 8060 of 20088 tile candidates
          occlusion-rejected, 418446 of 419904 pixels certified simple),
          while on ``nn`` both counters are exactly zero. If the switches
          cannot win here they cannot win anywhere.

Every arm prints a DIAG line with the coverage counters the design doc
exposes, so a neutral timing can be read as "the feature did nothing" or "the
feature did a lot and paid for it" rather than being ambiguous.
"""

import os
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"

from algan import *  # noqa: F403
from algan.utils.profiling_utils import profile_scene

ARM = (sys.argv[1] if len(sys.argv) > 1 else "base").lower()
SCENE = (sys.argv[2] if len(sys.argv) > 2 else "nn").lower()
QUALITY = (sys.argv[3] if len(sys.argv) > 3 else "UHD").upper()

_ARMS = {
    "base": (False, False),
    "tile": (True, False),
    "simple": (False, True),
    "both": (True, True),
}
if ARM not in _ARMS:
    raise SystemExit(f"unknown arm {ARM!r}; pick one of {sorted(_ARMS)}")
TILE, SIMPLE = _ARMS[ARM]


# -- coverage counters ---------------------------------------------------
# Summed over every coverage window of every run. The absolute values are
# scene/arena dependent; what matters is whether the switch changed anything
# at all, and by how much, beside the wall time.
import algan.rendering.raytracing.raster_pipeline as _rp  # noqa: E402

_KEYS = (
    "num_fragments",
    "num_covered",
    "num_sheets",
    "num_simple_pixels",
    "num_general_pixels",
    "tile_candidates",
    "tile_bbox_rejected",
    "tile_occluded",
)
_STATS = {}
_inner = _rp.prepare_sparse_raster_coverage


def _counted(*args, **kwargs):
    out = _inner(*args, **kwargs)
    if isinstance(out, dict):
        _STATS["windows"] = _STATS.get("windows", 0) + 1
        for key in _KEYS:
            if key in out:
                _STATS[key] = _STATS.get(key, 0) + int(out[key])
    return out


_rp.prepare_sparse_raster_coverage = _counted


def _nn_scene():
    from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3

    duration = 5.0 if QUALITY == "PREVIEW" else 0.5
    SETTINGS.raytracing.set(shadows=False)
    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob("world_map.png").move_next_to(nn, LEFT).spawn()
        label = (
            Text("Neural Net MLP v3 processing an image of the globe")
            .move_next_to(nn, DOWN)
            .spawn()
        )
    with Sync(runtime=duration):
        nn.move(UP)
        x.color_texture = x.color_texture * 0.5
        label.move(RIGHT * 2)


#: Layers of the overdraw stack. Every layer fills the frame, so only the
#: nearest one is visible: the other 15 are exactly the work Rank 3's
#: pre-emission opaque rejection is supposed to never generate.
OVERDRAW_LAYERS = 16
_LAYER_COLORS = (BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE)


def _overdraw_scene():
    duration = 5.0 if QUALITY == "PREVIEW" else 0.5
    SETTINGS.raytracing.set(shadows=False)
    quads = []
    with Off():
        for i in range(OVERDRAW_LAYERS):
            # Flat OPEN TriangleMesh quads, and every word of that is load
            # bearing (all four gates were measured on this branch, CPU, LD):
            #   * a ``Square`` is a *circuit*, and the tiled frontend never
            #     culls a circuit nor certifies one as an occluder;
            #   * a ``Cube`` defaults to opacity 0.75, so ``tri_frame_opaque``
            #     is false for every one of its faces and
            #     ``opaque_material_proofs`` certifies none of them -- and it
            #     is a closed shell besides, which
            #     ``interior_fragment_proofs`` rejects outright;
            #   * a textured or shaded surface is alpha-uncertain, which
            #     ``opaque_material_proofs`` refuses to certify.
            # Each layer is larger than the one in front of it so perspective
            # cannot shrink it inside the frame: every layer covers the whole
            # screen, so 15 of the 16 are invisible.
            half = 20.0 + 8.0 * i
            z = -(2.0 + 4.0 * i)
            q = TriangleMesh(
                [[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]],
                [[0, 1, 2], [0, 2, 3]],
            )
            q.set_color(_LAYER_COLORS[i % len(_LAYER_COLORS)])
            q.spawn()
            quads.append(q)
    with Sync(runtime=duration):
        # Animate the front layer only: the batch still re-prepares every
        # frame, so the frontend runs per frame like a real render.
        quads[0].move(RIGHT * 0.5)


_SCENES = {"nn": _nn_scene, "overdraw": _overdraw_scene}
if SCENE not in _SCENES:
    raise SystemExit(f"unknown scene {SCENE!r}; pick one of {sorted(_SCENES)}")


def scene():
    # Set inside the scene builder, like nn_ablation's arms: profile_scene
    # resets the SceneManager between runs, and these are per-render settings.
    SETTINGS.raytracing.experimental.set(
        raster_tile_binning=TILE,
        raster_simple_interiors=SIMPLE,
    )
    _SCENES[SCENE]()


_PRESETS = {"UHD": UHD, "HD": HD, "PREVIEW": PREVIEW, "MD": MD, "LD": LD}
print(
    f"ARM={ARM}  SCENE={SCENE}  QUALITY={QUALITY}  "
    f"raster_tile_binning={TILE}  raster_simple_interiors={SIMPLE}",
    flush=True,
)

# The tag carries the arm AND the scene: steps of one harness run share an
# `algan_outputs/`, so two arms on one tag would overwrite each other's video
# and the digest parity check would silently compare a file with itself
# (gpu_harnesses.md, "Reading the numbers").
profile_scene(
    scene,
    _PRESETS[QUALITY],
    f"tiled_{SCENE}_{ARM}_{QUALITY}",
    runs=2,
    kernel_profiler=False,
    save_video_kwargs={"ffmpeg_params": ["-crf", "17", "-preset", "ultrafast"]},
)

print(
    "DIAG "
    + f"arm={ARM} scene={SCENE} quality={QUALITY} "
    + " ".join(f"{k}={_STATS.get(k, 0)}" for k in ("windows",) + _KEYS),
    flush=True,
)
