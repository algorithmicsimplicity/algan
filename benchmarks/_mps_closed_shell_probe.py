"""Where the closed-shell frame's two routes disagree, pixel by pixel.

``test_closed_shell_attenuates_once_at_authored_opacity`` renders one
translucent emissive cube twice -- once path-traced (8 spp), once through the
deterministic sheet route (1 spp) -- and asserts the two agree to within 2
channel values over the centre 12x12 of a 64x64 frame. On an Apple GPU they
agree everywhere in that window **except its last column**, by 86
(``DESIGN_mps_support.md`` §1.2c); on the CPU they agree, in MPS-friendly mode
and out of it.

The assertion says how far apart they are and nothing about where or which one
moved, and those are the two facts that separate the readings:

* a **whole column of the image** that differs, running past the window, is a
  geometry edge the window happens to clip -- the two routes are allowed to
  disagree at an edge, and the finding would be that this window is only just
  clear of one rather than that Metal is wrong;
* a **12-pixel run that stops at the window's edge** is inside the flat
  interior, and then something really is wrong on this backend.

So this prints the frames over a window WIDER than the assertion's, plus every
column of the frame that disagrees within the window's own rows, and says which
route sits at the authored ``0.6 * 255`` and which one moved.

It renders the deterministic route through **each arm** of the
solid-shell opacity ceiling: the fused Taichi kernel
(``sheet_compact_taichi.solid_shell_ceiling``) and the torch block it replaced.
The two are meant to be bit-identical and they share every input, so they split
the remaining question in half -- an arm that agrees with the path tracer while
the other does not puts the defect in that kernel; two arms that agree with each
other put it in what the ceiling was handed (the segment key, the facing bit,
the exclusive prefix).

And it renders one of those arms **twice**, which is the question that comes
first: MPS-friendly mode is documented non-deterministic, and §1.2's amendment
predicts this symptom in as many words -- "a ceiling that wobbles in its low
bits flipping borderline fragments in and out of being clipped". Two renders of
one configuration that disagree are that prediction; two that agree bit for bit
mean there is a fixed wrong answer to find.

    uv run python benchmarks/_mps_closed_shell_probe.py

Renders on whatever ``ALGAN_RENDER_DEVICE`` selects, so the reading is taken by
running it once per device and comparing the two reports. It exits 0 whatever
it finds: it is a measurement, not a check -- the test is the check.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from algan import (  # noqa: E402
    BLACK,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    UP,
    WHITE,
    MeshLambertMaterial,
    Off,
    Prism,
    Scene,
    SceneManager,
)

#: The test's own settings and scene, kept identical on purpose: a probe that
#: renders something slightly different is measuring something slightly
#: different.
SHELL_SETTINGS = SMOKE_TEST.set(resolution=(64, 64))
OPACITY = 0.6
_RAW_KW = {"linear_color_space": False, "tonemapping": False}
_RAW_EXP = {"post_process_tonemap": False}


def _emissive_shell_cube():
    """The test's cube: black albedo, white emission, one closed shell."""
    cube = Prism(width=2.0, height=2.0, depth=2.0)
    cube.set_material(
        MeshLambertMaterial(color=BLACK, emissive=WHITE, emissive_intensity=1.0)
    )
    cube.set_opacity(OPACITY)
    cube.rotate(17, UP).rotate(9, RIGHT)
    return cube


def _render(out_dir, name, samples_per_pixel, shell_ceiling_kernel=None, ceiling=True):
    """One frame, as an int32 HxWxC tensor, plus its truncation counters.

    ``shell_ceiling_kernel`` selects which arm applies the solid-shell opacity
    ceiling: the fused Taichi kernel (the default) or the torch block it
    replaced. They are meant to be bit-identical, and A/B-ing them is what
    separates "the ceiling's INPUTS are wrong on this device" from "the kernel
    is" -- the two arms share every input and nothing else.
    """
    from algan.rendering.raytracing import settings as rt_settings

    snapshot = SETTINGS.snapshot()
    previous_kernel = rt_settings.sheet_shell_ceiling_kernel
    SceneManager.reset()
    try:
        if shell_ceiling_kernel is not None:
            rt_settings.set_sheet_shell_ceiling_kernel(shell_ceiling_kernel)
        SETTINGS.raytracing.set(samples_per_pixel=samples_per_pixel, denoise=False)
        for key, value in _RAW_KW.items():
            SETTINGS.raytracing.set(**{key: value})
        for key, value in _RAW_EXP.items():
            SETTINGS.raytracing.experimental.set(**{key: value})
        if not ceiling:
            SETTINGS.raytracing.experimental.set(solid_shell_alpha=False)
        with Scene(video_settings=SHELL_SETTINGS) as scene:
            with Off():
                scene.set_background(BLACK)
                Scene.clear_lights()
                _emissive_shell_cube().spawn(animate=False)
            result = scene.save_frame(
                out_dir / name, video_settings=SHELL_SETTINGS, overwrite=True
            )
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
        rt_settings.set_sheet_shell_ceiling_kernel(previous_kernel)
    frame = cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise SystemExit(f"unreadable frame at {result.output_path}")
    truncations = getattr(getattr(result, "render_plan", None), "truncations", None)
    return torch.from_numpy(frame.astype(np.int32)), truncations


#: Set while the arm whose sheets are worth dumping is rendering.
_DUMP_SHEETS = {"on": False, "lines": []}
#: Image columns to report, and the rows of the assertion window. 37 is the
#: column that disagrees, 36/38 bracket it, 32 is a control from the middle of
#: the same interior.
_DUMP_COLUMNS = (32, 36, 37, 38)
_DUMP_ROWS = range(26, 38)


def _install_sheet_dump(width, height):
    """Report the sheet list of a few named pixels, once per render.

    The composite a pixel shows IS its sheet list -- each sheet attenuates
    once, by its own coverage against the shell cap -- so an interior pixel
    reading `1 - 0.4**3` instead of `0.6` has three sheets where its
    neighbours have two, or two whose coverages are wrong. Which of those it is
    cannot be inferred from the frame, and it is the whole remaining question
    (``DESIGN_mps_support.md`` §1.2c).

    Read off the coverage dict the resolve is about to be handed, on the host,
    for the same reason the smoke render's diagnostics are: a reading taken on
    the device under investigation lets the defect shape the evidence.
    """
    from algan.rendering.raytracing import raster_pipeline

    original = raster_pipeline.prepare_sparse_raster_coverage

    def reporting(*args, **kwargs):
        coverage = original(*args, **kwargs)
        if not _DUMP_SHEETS["on"] or not coverage or not coverage.get("sheets"):
            return coverage
        _DUMP_SHEETS["on"] = False  # first chunk only; the frame is one chunk
        try:
            covered = coverage["covered_idx"][: coverage["num_covered"]].cpu()
            offsets = coverage["sheet_offsets"].cpu()
            cov = coverage["sheet_cov"].cpu()
            cap = coverage["sheet_cap"].cpu()
            msk = coverage["sheet_msk"].cpu()
            wanted = {
                int(py) * width + int(px): (int(py), int(px))
                for py in _DUMP_ROWS
                for px in _DUMP_COLUMNS
            }
            for t in range(covered.numel()):
                pixel = int(covered[t])
                if pixel not in wanted:
                    continue
                py, px = wanted[pixel]
                lo, hi = int(offsets[t]), int(offsets[t + 1])
                covs = " ".join(f"{float(cov[s]):+.5f}" for s in range(lo, hi))
                caps = " ".join(f"{float(cap[s]):.5f}" for s in range(lo, hi))
                masks = " ".join(f"{int(msk[s]):#06x}" for s in range(lo, hi))
                _DUMP_SHEETS["lines"].append(
                    f"  py={py:3d} px={px:3d} sheets={hi - lo}  cov[{covs}]"
                    f"  cap[{caps}]  msk[{masks}]"
                )
        except Exception as exc:  # noqa: BLE001
            _DUMP_SHEETS["lines"].append(f"  (sheet dump failed: {exc!r})")
        return coverage

    raster_pipeline.prepare_sparse_raster_coverage = reporting


#: Lines from :func:`_install_ceiling_check`.
_CEILING_CHECK: list = []


def _reference_ceiling(key, o2, back, excl, cov):
    """The solid-shell ceiling, in float64 on the host, from the same inputs.

    A transcription of ``sheet_compact_taichi.solid_shell_ceiling``: segments
    are runs of equal ``key`` along ``o2``, each segment's cap is
    ``float32(max(front sum, back sum))``, and each fragment keeps
    ``min(max(cap - spent, 0) / denom, 1)`` of its own area, where ``spent`` is
    the global exclusive prefix differenced at the segment's own base.

    Its purpose is to be the ORACLE the device's answer is checked against, so
    it is deliberately the slow, obvious form: float64 throughout, one Python
    loop per segment, no reassociation to argue about.
    """
    import numpy as np

    order = o2.astype(np.int64)
    ki = key[order]
    out = cov.astype(np.float64).copy()
    n = order.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and ki[j] == ki[i]:
            j += 1
        rows = order[i:j]
        c = cov[rows].astype(np.float64)
        is_back = back[rows] != 0
        cap = np.float64(np.float32(max(c[~is_back].sum(), c[is_back].sum())))
        base = np.float64(excl[i])
        spent = excl[i:j].astype(np.float64) - base
        denom = np.maximum(c, 1e-12)
        scale = np.minimum(np.maximum(cap - spent, 0.0) / denom, 1.0)
        out[rows] = denom * scale
        i = j
    return out


def _install_ceiling_check():
    """Check ``solid_shell_ceiling``'s output against the host oracle, in situ.

    The named next measurement of ``DESIGN_mps_support.md`` §1.2c. It runs the
    kernel on its real inputs and then recomputes the answer from **those same
    inputs** on the host, which is the only comparison that separates the two
    remaining possibilities without a second machine: a kernel that computes
    the wrong thing from correct inputs disagrees with the oracle, and inputs
    that were already wrong agree with it perfectly while still producing the
    wrong picture.
    """
    import numpy as np

    from algan.rendering.raytracing import sheet_compact_taichi as sc

    original = sc.solid_shell_ceiling

    def checked(key, o2, back, excl, scratch, n, cov, acc_t):
        try:
            before = (
                key[:n].detach().cpu().numpy(),
                o2[:n].detach().cpu().numpy(),
                back[:n].detach().cpu().numpy(),
                excl[:n].detach().cpu().to(torch.float64).numpy(),
                cov[:n].detach().cpu().to(torch.float64).numpy(),
            )
        except Exception as exc:  # noqa: BLE001
            _CEILING_CHECK.append(f"  (could not read the inputs: {exc!r})")
            return original(key, o2, back, excl, scratch, n, cov, acc_t)
        result = original(key, o2, back, excl, scratch, n, cov, acc_t)
        try:
            after = cov[:n].detach().cpu().to(torch.float64).numpy()
            oracle = _reference_ceiling(*before)
            delta = np.abs(after - oracle)
            bad = int((delta > 1e-6).sum())
            _CEILING_CHECK.append(
                f"  n={n} fragments; kernel vs host oracle: {bad} differ by "
                f"> 1e-6, worst {float(delta.max()) if delta.size else 0.0:.6g}"
            )
            k_before, o2_before, back_before, excl_before, cov_before = before
            worst = np.argsort(-delta)[:6]
            for row in worst:
                if delta[row] <= 1e-6:
                    break
                _CEILING_CHECK.append(
                    f"    row {int(row)}: key {int(k_before[row])} "
                    f"back {int(back_before[row])} cov in {cov_before[row]:.6f} "
                    f"kernel {after[row]:.6f} oracle {oracle[row]:.6f}"
                )
            # And what the oracle itself says about the shape of the data,
            # which is what an input defect shows up in: how many segments,
            # and how many hold more than the two crossings a closed shell has.
            ki = k_before[o2_before.astype(np.int64)]
            starts = np.flatnonzero(np.r_[True, ki[1:] != ki[:-1]])
            sizes = np.diff(np.r_[starts, n])
            _CEILING_CHECK.append(
                f"  segments={starts.size} (negative-key pass-throughs="
                f"{int((k_before < 0).sum())}); sizes 1:{int((sizes == 1).sum())} "
                f"2:{int((sizes == 2).sum())} 3:{int((sizes == 3).sum())} "
                f">3:{int((sizes > 3).sum())}"
            )
            # A fingerprint of what went in and what the clamp did with it,
            # comparable across devices by eye. If the kernel matches its
            # oracle on both and these differ, the inputs are what moved; if
            # neither differs, nothing here is the defect.
            kept = int((np.abs(after - cov_before) <= 1e-6).sum())
            zeroed = int((np.abs(after) <= 1e-6).sum())
            _CEILING_CHECK.append(
                f"  cov in sum {cov_before.sum():.6f} out sum {after.sum():.6f}; "
                f"back-facing {int((back_before != 0).sum())}; "
                f"excl last {excl_before[-1]:.6f}; "
                f"unchanged {kept} clamped-to-zero {zeroed} "
                f"partly {n - kept - zeroed}"
            )
        except Exception as exc:  # noqa: BLE001
            _CEILING_CHECK.append(f"  (check failed: {exc!r})")
        return result

    sc.solid_shell_ceiling = checked


#: Lines from :func:`_install_nan_trace`.
_NAN_TRACE: list = []


def _install_nan_trace():
    """Say which step of the band aggregation first produces a non-finite value.

    The Apple GPU's sheet coverages at the offending column are **nan**, and a
    NaN never trips the resolve's ``eff <= min_alpha`` branch -- comparisons
    against it are all false -- so the sheet composites instead of dropping
    out, which is precisely the extra crossing. The ceiling is not where it is
    made: its output was compared against a float64 oracle over the same
    inputs and no element differed, and that comparison would have gone
    non-finite itself had a NaN been in either side.

    So this wraps the two steps between the ceiling and the sheet record and
    counts non-finite values in every tensor going in and coming out. The
    first function whose inputs are clean and whose output is not is where the
    NaN is born.
    """
    from algan.rendering.raytracing import sheets as sh

    def counts(label, values):
        parts = []
        for name, value in values:
            if not torch.is_tensor(value) or not value.is_floating_point():
                continue
            bad = int((~torch.isfinite(value)).sum())
            if bad:
                parts.append(f"{name}={bad}/{value.numel()}")
        return f"  {label}: " + (", ".join(parts) if parts else "all finite")

    def wrap(name, function, arg_names, out_names):
        def wrapped(*args, **kwargs):
            result = function(*args, **kwargs)
            if len(_NAN_TRACE) < 12:
                outs = result if isinstance(result, tuple) else (result,)
                _NAN_TRACE.append(counts(f"{name} in ", list(zip(arg_names, args))))
                _NAN_TRACE.append(counts(f"{name} out", list(zip(out_names, outs))))
            return result

        return wrapped

    sh._band_composite = wrap(
        "_band_composite",
        sh._band_composite,
        ("band_of_frag", "nbands", "cov_o", "msk_o"),
        ("area", "union", "corr", "split"),
    )
    sh._sibling_weights = wrap(
        "_sibling_weights",
        sh._sibling_weights,
        ("sheet_band", "cov", "msk", "band_area", "band_union", "band_corr"),
        ("wgt", "wmsk"),
    )


def _grid(values, label, first_col):
    """A numbered grid of one channel, so a column can be pointed at."""
    lines = [f"{label} (columns {first_col}..{first_col + values.shape[1] - 1}):"]
    lines.append(
        "      " + "".join(f"{first_col + i:5d}" for i in range(values.shape[1]))
    )
    for r in range(values.shape[0]):
        lines.append("      " + "".join(f"{int(v):5d}" for v in values[r]))
    return "\n".join(lines)


def main():
    from algan.rendering.mps_compat import mps_friendly

    print(f"render device    : {SETTINGS.computing.render_device}")
    print(f"mps friendly     : {mps_friendly()}")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        pt, pt_trunc = _render(out_dir, "shell_pt.png", 8)
        _install_sheet_dump(*SHELL_SETTINGS.resolution)
        _install_ceiling_check()
        _install_nan_trace()
        _DUMP_SHEETS["on"] = True
        det, det_trunc = _render(out_dir, "shell_det.png", 1, shell_ceiling_kernel=True)
        _DUMP_SHEETS["on"] = False
        # The same arm again, same process, same settings. MPS-friendly mode is
        # documented non-deterministic -- f32 atomics replace the f64
        # accumulator, and DESIGN_mps_support.md 1.2's amendment predicts the
        # symptom as "a ceiling that wobbles in its low bits flipping
        # borderline fragments in and out of being clipped". If two renders of
        # ONE configuration disagree at the offending column, that prediction
        # is what this is and no amount of looking at the kernel will find a
        # bug; if they agree bit for bit, it is a real defect with a fixed
        # answer.
        det_again, _ = _render(
            out_dir, "shell_det_again.png", 1, shell_ceiling_kernel=True
        )
        # The ceiling off entirely, which says WHEN the extra crossing appears.
        # With `solid_shell_alpha` off both crossings composite by design, so
        # the interior reads `a * (2 - a) * 255 = 214`. If the offending column
        # then reads 214 like the rest, the extra layer is the ceiling's doing;
        # if it still reads 239 (three crossings) the compaction handed the
        # ceiling three sheets where the CPU's handed it two, and the ceiling is
        # innocent. This arm uses no `index_copy_`, so unlike the torch arm
        # below it runs on MPS.
        no_ceiling, _ = _render(
            out_dir, "shell_no_ceiling.png", 1, shell_ceiling_kernel=True, ceiling=False
        )
        # LAST, and allowed to fail. The torch arm of the ceiling calls
        # ``index_copy_``, which torch has not implemented for MPS
        # (`aten::index_copy.out`), so on an Apple GPU this arm raises and
        # takes every reading above with it if it runs first. It is the least
        # important of the three and it is the only one that can die.
        torch_det = torch_trunc = None
        try:
            torch_det, torch_trunc = _render(
                out_dir, "shell_det_torch.png", 1, shell_ceiling_kernel=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"det, ceiling torch    : unavailable -- {type(exc).__name__}: {exc}")

    h, w = pt.shape[0], pt.shape[1]
    expected = OPACITY * 255.0
    lo, hi = h // 2 - 6, h // 2 + 6
    clo, chi = w // 2 - 6, w // 2 + 6
    print(f"\nframe            : {tuple(pt.shape)}, authored interior {expected:.0f}")

    def summarize(label, frame):
        err = (pt[..., :3] - frame[..., :3]).abs().amax(-1).float()
        # Whether the run of disagreement stops at the window's edge or runs
        # past it, which is the reading this probe exists to take. Across the
        # window's ROWS but the frame's whole WIDTH: the silhouette crosses
        # other rows in every render, so a full-frame column count says
        # nothing, while these twelve rows are the interior the test asserts
        # about.
        hot = [int(c) for c in torch.nonzero(err[lo:hi].amax(0) > 2).flatten().tolist()]
        core = frame[lo:hi, clo:chi, :3].float()
        print(
            f"{label:22s}: max |pt - this| in window "
            f"{float(err[lo:hi, clo:chi].max()):3.0f}; columns > 2 across the "
            f"window's rows {hot}; interior mean {float(core.mean()):7.2f} "
            f"min {float(core.min()):5.0f} max {float(core.max()):5.0f}"
        )
        return err

    print(
        f"assertion window : rows {lo}..{hi - 1} cols {clo}..{chi - 1}, "
        f"authored {expected:.0f}"
    )
    core_pt = pt[lo:hi, clo:chi, :3].float()
    print(
        f"{'path traced (8 spp)':22s}: interior mean {float(core_pt.mean()):7.2f} "
        f"min {float(core_pt.min()):5.0f} max {float(core_pt.max()):5.0f} "
        "-- the oracle"
    )
    err = summarize("det, ceiling kernel", det)
    # Its own oracle is 214 (`a * (2 - a) * 255`), not the path tracer's 153,
    # so this one is reported against itself: a uniform interior means every
    # pixel crossed the shell twice, and an outlier at 239 means one crossed it
    # three times.
    nc_core = no_ceiling[lo:hi, clo:chi, :3].float()
    # Restricted to the assertion window's own columns: outside the cube the
    # frame is background and deviates from the interior everywhere, which
    # tells you nothing.
    nc_dev = (nc_core - float(nc_core.median())).abs().amax(-1)
    nc_hot = [
        clo + int(c) for c in torch.nonzero(nc_dev.amax(0) > 2).flatten().tolist()
    ]
    print(
        f"{'det, ceiling OFF':22s}: interior mean {float(nc_core.mean()):7.2f} "
        f"min {float(nc_core.min()):5.0f} max {float(nc_core.max()):5.0f} "
        f"(both crossings composite: expect {OPACITY * (2 - OPACITY) * 255:.0f}); "
        f"columns off its own median inside the window {nc_hot}"
    )
    if torch_det is not None:
        summarize("det, ceiling torch", torch_det)

    # Reproducibility, which decides whether there is a defect to find at all.
    repeat = (det[..., :3] - det_again[..., :3]).abs().amax(-1).float()
    repeat_hot = [
        int(c) for c in torch.nonzero(repeat[lo:hi].amax(0) > 0).flatten().tolist()
    ]
    print(
        f"{'det vs det again':22s}: max |diff| whole frame "
        f"{float(repeat.max()):3.0f}; columns differing at all across the "
        f"window's rows {repeat_hot}"
    )

    for label, trunc in (
        ("path traced", pt_trunc),
        ("det, kernel", det_trunc),
        ("det, torch", torch_trunc),
    ):
        if trunc is None:
            continue
        hits = {
            name: getattr(trunc, name)
            for name in dir(trunc)
            if not name.startswith("_") and isinstance(getattr(trunc, name), int)
        }
        hits = {k: v for k, v in hits.items() if v}
        print(f"truncations {label:12s}: {hits or 'none'}")

    if _NAN_TRACE:
        print("\nnon-finite values through the band aggregation:")
        for line in _NAN_TRACE:
            print(line)

    if _CEILING_CHECK:
        print("\nsolid_shell_ceiling against a float64 host oracle on its own inputs:")
        for line in _CEILING_CHECK[:18]:
            print(line)

    if _DUMP_SHEETS["lines"]:
        print(
            "\nsheets of a few pixels, deterministic arm with the ceiling on "
            f"(columns {_DUMP_COLUMNS}, rows {_DUMP_ROWS.start}..{_DUMP_ROWS.stop - 1}):"
        )
        for line in _DUMP_SHEETS["lines"]:
            print(line)

    # Four columns either side of the window, so an edge shows as a run.
    wlo, whi = max(0, clo - 4), min(w, chi + 4)
    green = 1  # BGR from cv2; the channel is arbitrary, the frame is grey.
    print()
    print(_grid(pt[lo:hi, wlo:whi, green], "path traced (8 spp)", wlo))
    print()
    print(_grid(det[lo:hi, wlo:whi, green], "deterministic, ceiling kernel", wlo))
    if torch_det is not None:
        print()
        print(
            _grid(torch_det[lo:hi, wlo:whi, green], "deterministic, ceiling torch", wlo)
        )
    print()
    print(_grid(err[lo:hi, wlo:whi], "|pt - det(kernel)|", wlo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
