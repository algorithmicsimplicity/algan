"""Unit tests for the path-tracer denoiser (``algan/rendering/denoise/``).

Three layers, by what they need:

* **Pure pieces** (tza parser, U-Net topology, PU transfer, autoexposure)
  run everywhere with synthetic data -- no weights, no network, no render.
* **AOV guides** run one small real render with ``get_denoiser``
  monkeypatched to a spy, so the kernel's albedo/normal accumulation is
  tested end-to-end without the weights.
* **The real filter** needs the official weights; those tests resolve them
  through the normal cache-or-download path and SKIP when they cannot be
  had (an offline CI still runs everything above).
"""

from __future__ import annotations

import struct

import cv2
import numpy as np
import pytest
import torch

from algan import (
    BLACK,
    DOWN,
    LEFT,
    OUT,
    RED,
    SETTINGS,
    SMOKE_TEST,
    UP,
    WHITE,
    MeshLambertMaterial,
    MeshStandardMaterial,
    Off,
    PointLight,
    Prism,
    Scene,
    SceneManager,
)
from algan.rendering.denoise import denoise as denoise_mod
from algan.rendering.denoise import weights as weights_mod
from algan.rendering.denoise.oidn_unet import (
    RT_HDR_ALB_NRM_LAYERS,
    OidnUNet,
    WeightShapeError,
)
from algan.rendering.denoise.tza import TzaError, parse_tza

# ---------------------------------------------------------------------------
# tza parser
# ---------------------------------------------------------------------------


def _build_tza(tensors):
    """Serialize ``{name: float16/float32 tensor}`` in the tza v2 layout."""
    blobs = []
    offset = 16  # data begins after a small aligned header region
    table = b""
    payload = b""
    for name, tensor in tensors.items():
        raw = tensor.numpy().tobytes()
        layout = "oihw" if tensor.dim() == 4 else "x"
        type_char = b"h" if tensor.dtype == torch.float16 else b"f"
        entry = struct.pack("<H", len(name)) + name.encode("ascii")
        entry += struct.pack("<B", tensor.dim())
        entry += struct.pack(f"<{tensor.dim()}I", *tensor.shape)
        entry += layout.encode("ascii") + type_char
        entry += struct.pack("<Q", offset)
        table += entry
        blobs.append((offset, raw))
        payload += raw
        offset += len(raw)
    table_offset = offset
    header = struct.pack("<HBB", 0x41D7, 2, 0) + struct.pack("<Q", table_offset)
    body = header + b"\x00" * (16 - len(header)) + payload
    return body + struct.pack("<I", len(tensors)) + table


def test_tza_roundtrip_preserves_names_shapes_and_values():
    original = {
        "enc_conv0.weight": torch.randn(4, 9, 3, 3, dtype=torch.float32).half(),
        "enc_conv0.bias": torch.randn(4, dtype=torch.float32),
    }
    parsed = parse_tza(_build_tza(original))
    assert set(parsed) == set(original)
    for name, tensor in original.items():
        assert parsed[name].dtype == torch.float32
        assert torch.equal(parsed[name], tensor.float())


def test_tza_rejects_garbage_and_truncation():
    with pytest.raises(TzaError):
        parse_tza(b"not a tza file at all")
    good = _build_tza({"enc_conv0.bias": torch.randn(4)})
    with pytest.raises(TzaError):
        parse_tza(good[:10])
    # A version this parser does not speak.
    bad_version = bytearray(good)
    bad_version[2] = 9
    with pytest.raises(TzaError):
        parse_tza(bytes(bad_version))


# ---------------------------------------------------------------------------
# U-Net topology
# ---------------------------------------------------------------------------


def _random_weights():
    gen = torch.Generator().manual_seed(0)
    tensors = {}
    for name, (oc, ic) in RT_HDR_ALB_NRM_LAYERS.items():
        tensors[f"{name}.weight"] = torch.randn((oc, ic, 3, 3), generator=gen) * 0.05
        tensors[f"{name}.bias"] = torch.randn((oc,), generator=gen) * 0.05
    return tensors


def test_unet_runs_on_aligned_input_and_rejects_misaligned():
    net = OidnUNet(_random_weights(), torch.device("cpu"))
    out = net(torch.rand((1, 9, 32, 48)))
    assert tuple(out.shape) == (1, 3, 32, 48)
    with pytest.raises(ValueError, match="aligned"):
        net(torch.rand((1, 9, 30, 48)))


def test_unet_rejects_wrong_shapes():
    tensors = _random_weights()
    tensors["dec_conv0.weight"] = torch.randn(3, 31, 3, 3)
    with pytest.raises(WeightShapeError, match="dec_conv0"):
        OidnUNet(tensors, torch.device("cpu"))
    del tensors["enc_conv1.bias"]
    tensors["dec_conv0.weight"] = torch.randn(3, 32, 3, 3)
    with pytest.raises(WeightShapeError, match="enc_conv1"):
        OidnUNet(tensors, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Transfer function + autoexposure
# ---------------------------------------------------------------------------


def test_pu_transfer_roundtrips_across_the_hdr_range():
    y = torch.tensor([1e-7, 1e-4, 0.01, 0.18, 1.0, 10.0, 1000.0, 60000.0])
    x = denoise_mod._pu_forward(y) * denoise_mod._PU_NORM_SCALE
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0 + 1e-5
    back = denoise_mod._pu_inverse(x / denoise_mod._PU_NORM_SCALE)
    assert torch.allclose(back, y, rtol=1e-3, atol=1e-8)


def test_autoexposure_normalizes_middle_grey():
    grey = torch.full((64, 64, 3), 0.18)
    assert abs(denoise_mod.autoexposure(grey) - 1.0) < 1e-3
    # Scaling the image inversely scales the exposure.
    assert abs(denoise_mod.autoexposure(grey * 4.0) - 0.25) < 1e-3
    # A black frame is left alone rather than dividing by zero.
    assert denoise_mod.autoexposure(torch.zeros((16, 16, 3))) == 1.0


# ---------------------------------------------------------------------------
# Weights resolution fallbacks (no network involved)
# ---------------------------------------------------------------------------


@pytest.fixture
def _fresh_weight_state():
    weights_mod._reset_for_tests()
    denoise_mod._reset_for_tests()
    try:
        yield
    finally:
        weights_mod._reset_for_tests()
        denoise_mod._reset_for_tests()


@pytest.mark.usefixtures("_fresh_weight_state")
def test_missing_override_path_degrades_to_denoise_off():
    SETTINGS.raytracing.experimental.set(
        denoise_weights="/nonexistent/rt_hdr_alb_nrm.tza"
    )
    assert weights_mod.weights_path() is None
    # The answer is memoized: a second ask does not re-warn or re-probe.
    assert weights_mod.weights_path() is None
    assert denoise_mod.get_denoiser(torch.device("cpu")) is None


@pytest.mark.usefixtures("_fresh_weight_state")
def test_unparseable_weights_degrade_to_denoise_off(tmp_path):
    bad = tmp_path / "bad.tza"
    bad.write_bytes(b"junk that is not a tensor archive")
    SETTINGS.raytracing.experimental.set(denoise_weights=str(bad))
    assert weights_mod.weights_path() == str(bad)
    assert denoise_mod.get_denoiser(torch.device("cpu")) is None


# ---------------------------------------------------------------------------
# The AOV guides, through a real render (no weights needed: spy denoiser)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_VS = SMOKE_TEST.set(resolution=(64, 64))


def _render_guided(tmp_path, name, fake_denoiser, build):
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(samples_per_pixel=4, denoise=True)
        # The tracer imports get_denoiser from the package lazily (inside
        # render_chunk), so patching the package attribute reaches it.
        import algan.rendering.denoise as denoise_pkg

        original = denoise_pkg.get_denoiser
        denoise_pkg.get_denoiser = lambda device: fake_denoiser
        try:
            with Scene(video_settings=_VS) as scene:
                with Off():
                    build(scene)
                result = scene.save_frame(
                    tmp_path / name, video_settings=_VS, overwrite=True
                )
        finally:
            denoise_pkg.get_denoiser = original
        frame = cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED)
        assert frame is not None
        return torch.from_numpy(frame.astype(np.int32))
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)


class _SpyDenoiser:
    """Captures the hook's inputs and passes the color through unchanged."""

    def __init__(self):
        self.calls = []

    def __call__(self, color, albedo, normal):
        self.calls.append((color.clone(), albedo.clone(), normal.clone()))
        return color.clone()


def test_aov_guides_carry_surface_albedo_and_normal(tmp_path):
    """The denoiser hook receives real guides: at the center pixel of a lit
    white floor seen face-on, the albedo guide is the surface's base color
    (not its lit radiance) and the normal guide faces the camera; a
    pass-through spy leaves the image exactly as the denoise-off arm
    renders it.
    """
    spy = _SpyDenoiser()

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        PointLight(location=UP * 2.0 + OUT * 4.0, color=WHITE, intensity=0.5).spawn(
            animate=False
        )
        floor = Prism(dimensions=(7.0, 7.0, 0.2))
        floor.set_material(MeshLambertMaterial(color=RED))
        floor.spawn(animate=False)

    img = _render_guided(tmp_path, "aov_spy.png", spy, build)
    assert len(spy.calls) >= 1
    color, albedo, normal = spy.calls[0]
    h, w = albedo.shape[1], albedo.shape[2]
    ctr_alb = albedo[0, h // 2, w // 2]
    ctr_nrm = normal[0, h // 2, w // 2]
    # RED's base color, not its (dimly lit) radiance: the guide's red channel
    # dominates and is far above the lit pixel's brightness.
    assert float(ctr_alb[0]) > 0.5, f"albedo guide {ctr_alb} lost the base color"
    assert float(ctr_alb[0]) > float(ctr_alb[2]) + 0.2
    assert float(color[0, h // 2, w // 2].max()) < float(ctr_alb.max()), (
        "the albedo guide should be unlit (brighter than the dimly lit color)"
    )
    # The floor faces the camera: |z| dominates the normal guide.
    assert abs(float(ctr_nrm[2])) > 0.5
    assert abs(float(ctr_nrm[2])) > abs(float(ctr_nrm[0]))
    assert abs(float(ctr_nrm[2])) > abs(float(ctr_nrm[1]))

    # A pass-through denoiser changes nothing.
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(samples_per_pixel=4, denoise=False)
        with Scene(video_settings=_VS) as scene:
            with Off():
                build(scene)
            result = scene.save_frame(
                tmp_path / "aov_off.png", video_settings=_VS, overwrite=True
            )
        off = torch.from_numpy(
            cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
        )
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
    # The hook's byte-scale round trip (out/255 -> spy -> *255) can move a
    # value by one ulp, which the encode may round differently; anything
    # beyond one channel count would be a real change.
    assert int((img - off).abs().max()) <= 1, (
        "a pass-through denoiser must leave the frame as the denoise-off arm renders it"
    )


def test_byte_buffer_skips_the_denoiser(tmp_path):
    """With post_process_tonemap off the frame buffer is uint8; the hook
    must skip (never call the denoiser) rather than filter encoded bytes.
    """
    spy = _SpyDenoiser()

    def build(scene):
        scene.set_background(BLACK)
        floor = Prism(dimensions=(4.0, 4.0, 0.2))
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)

    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(
            samples_per_pixel=4, denoise=True, linear_color_space=False
        )
        SETTINGS.raytracing.experimental.set(post_process_tonemap=False)
        import algan.rendering.denoise as denoise_pkg

        original = denoise_pkg.get_denoiser
        denoise_pkg.get_denoiser = lambda device: spy
        try:
            with Scene(video_settings=_VS) as scene:
                with Off():
                    build(scene)
                scene.save_frame(
                    tmp_path / "byte_skip.png", video_settings=_VS, overwrite=True
                )
        finally:
            denoise_pkg.get_denoiser = original
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
    assert spy.calls == [], "the uint8 buffer must never reach the denoiser"


# ---------------------------------------------------------------------------
# The real filter (needs the official weights; skips offline)
# ---------------------------------------------------------------------------


def _real_denoiser():
    if weights_mod.weights_path() is None:
        pytest.skip("the OIDN weights are not available on this machine")
    denoiser = denoise_mod.get_denoiser(torch.device("cpu"))
    if denoiser is None:
        pytest.skip("the OIDN weights failed to load on this machine")
    return denoiser


def test_real_weights_reduce_synthetic_noise():
    denoiser = _real_denoiser()
    gen = torch.Generator().manual_seed(3)
    clean = torch.zeros((1, 48, 64, 3))
    clean[0, :, :32] = 0.6
    clean[0, :, 32:] = 0.15
    noisy = (clean + torch.randn(clean.shape, generator=gen) * 0.25).clamp_min(0.0)
    normal = torch.zeros_like(clean)
    normal[..., 2] = -1.0
    out = denoiser(noisy, clean.clone(), normal)

    def rmse(a, b):
        return float(((a - b) ** 2).mean().sqrt())

    assert rmse(out, clean) < 0.4 * rmse(noisy, clean), (
        f"denoising barely helped: {rmse(noisy, clean):.4f} -> {rmse(out, clean):.4f}"
    )


def test_real_weights_tile_seamlessly():
    """Tiled inference (tiny tile, forced) must agree with the single-tile
    result away from nothing -- the overlap blending is copy-the-core, so
    the two runs differ only by floating error at worst.
    """
    denoiser = _real_denoiser()
    gen = torch.Generator().manual_seed(4)
    img = torch.rand((1, 96, 160, 3), generator=gen) * 0.5
    albedo = torch.rand((1, 96, 160, 3), generator=gen)
    normal = torch.zeros_like(img)
    normal[..., 2] = -1.0
    snapshot = SETTINGS.snapshot()
    try:
        SETTINGS.raytracing.experimental.set(denoise_tile_size=4096)
        whole = denoiser(img, albedo, normal)
        SETTINGS.raytracing.experimental.set(denoise_tile_size=128)
        tiled = denoiser(img, albedo, normal)
    finally:
        SETTINGS.restore(snapshot)
    err = float((whole - tiled).abs().max())
    assert err < 0.02, f"tile seams: max deviation {err:.4f}"


def test_denoised_render_is_reproducible_and_less_noisy(tmp_path):
    """End to end on this machine: same frame twice with the denoiser on is
    byte-identical, and the denoised frame sits closer to a high-sample
    reference than the noisy input does.
    """
    _real_denoiser()

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        PointLight(
            location=UP * 2.5 + OUT * 3.5 + LEFT * 1.0,
            color=WHITE,
            intensity=2.0,
        ).spawn(animate=False)
        floor = Prism(dimensions=(8.0, 0.2, 4.0))
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.move(DOWN * 1.4)
        floor.spawn(animate=False)
        metal = Prism(dimensions=(1.2, 1.2, 1.2))
        metal.set_material(
            MeshStandardMaterial(color=WHITE, metalness=1.0, roughness=0.4)
        )
        metal.move(DOWN * 0.6)
        metal.spawn(animate=False)

    def render(name, spp, denoise):
        snapshot = SETTINGS.snapshot()
        SceneManager.reset()
        try:
            SETTINGS.raytracing.set(
                samples_per_pixel=spp, denoise=denoise, shadows=True
            )
            with Scene(video_settings=_VS) as scene:
                with Off():
                    build(scene)
                result = scene.save_frame(
                    tmp_path / name, video_settings=_VS, overwrite=True
                )
            return torch.from_numpy(
                cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED)[
                    ..., :3
                ].astype(np.float32)
            )
        finally:
            SceneManager.reset()
            SETTINGS.restore(snapshot)

    ref = render("dn_ref.png", 128, False)
    noisy = render("dn_noisy.png", 4, False)
    a = render("dn_a.png", 4, True)
    b = render("dn_b.png", 4, True)
    assert torch.equal(a, b), "denoised output changed between identical runs"

    def rmse(x, y):
        return float(((x - y) ** 2).mean().sqrt())

    assert rmse(a, ref) < rmse(noisy, ref), (
        f"denoising did not help: {rmse(noisy, ref):.2f} -> {rmse(a, ref):.2f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
