"""The tests that do not pass on an Apple GPU yet, and why each does not.

``.github/workflows/test.yaml`` runs ``tests/unit_tests tests/fast`` on
``macos-latest`` with ``ALGAN_RENDER_DEVICE=mps``, and that arm is a **required**
check. When it was first turned on it reported 32 failures and 3 errors against
3311 passes (run 34102515789). Eleven of those failures and all three errors are
fixed; the entries below are what is left, and this file is the record of them.
``algan/rendering/DESIGN_mps_support.md`` §4 is the scoreboard behind it — six
causes, not twenty tests, which is what makes the remainder tractable.

Why a list here rather than a skip in each test file
----------------------------------------------------
* A skip in a test file is a claim about the test. Every entry below is a claim
  about the **render device**: the same test passes on Linux, on CUDA and on the
  macOS CPU arm, and each is on this list because of a defect in torch's MPS
  backend, in the Metal codegen, or in Algan's own handling of one. Putting that
  in ``test_glossy_prefilter.py`` spreads Apple-GPU knowledge across eight files
  that have nothing else to do with it.
* One list is reviewable. Twenty scattered skips are not, and neither is their
  total: the number in this file is how much of the port is left, and it should
  be readable in one place without grep.

Why ``xfail(strict=True)`` rather than a deselect
-------------------------------------------------
The entries **run**. A deselect would make them invisible; an xfail reports them
in the summary, so the arm's output says how much is outstanding on every run.
And ``strict`` makes the list self-cleaning: a fix that makes one of these pass
turns the arm RED with ``XPASS(strict)`` naming the test, so an entry cannot
outlive the defect it describes. Removing the line is how a fix is finished.

Nothing here applies off MPS. On every other device these are ordinary tests
and a failure in one is an ordinary failure.

Each reason names the section of ``algan/rendering/DESIGN_mps_support.md`` that
has the measurement, because "it fails on MPS" is not a reason -- it is the
observation the reason has to explain.
"""

from __future__ import annotations

#: ``nodeid -> reason``. The nodeid is what pytest prints in its ``FAILED``
#: lines, parametrisation included, so an entry is copied from a run rather
#: than reconstructed.
KNOWN_FAILURES: dict[str, str] = {}


def _add(reason: str, *nodeids: str) -> None:
    for nodeid in nodeids:
        KNOWN_FAILURES[nodeid] = reason


# -- B: the glossy tile loop walks off its frame table (DESIGN §4.2) --------
#
# `tracer.py`'s `frame_end = gl_bounds[gl_frame + 1]` raises IndexError. The
# bounds are a correct `searchsorted` (measured: `probe_frame_bounds`) of a
# `covered_idx` whose last ordinal is past the end of the window, so the tile
# loop still has pixels to place after it has finished the last frame. Every
# test here is an ordinary render that reaches the glossy route, which is the
# default; it is one defect, not thirteen.
_add(
    "MPS: the glossy tile loop's frame table runs short -- DESIGN_mps_support.md §4.2",
    "tests/unit_tests/test_area_light_soft_shadow.py::test_soft_shadow_fans_compile_and_render_one_frame[sheet]",
    "tests/unit_tests/test_bezier_group_runs.py::test_run_splitting_leaves_the_rendered_frame_unchanged",
    "tests/unit_tests/test_deterministic_shadow_opacity.py::test_deterministic_shadows_accumulate_every_blocker_opacity[raster]",
    "tests/unit_tests/test_display_referred_coverage.py::test_partial_coverage_matches_a_supersampled_render",
    "tests/unit_tests/test_manim_shader_render.py::test_use_manim_defaults_reaches_bare_solids",
    "tests/unit_tests/test_path_tracer.py::test_author_order_and_depth_compose_like_the_deterministic_route",
    "tests/unit_tests/test_path_tracer.py::test_the_deterministic_renderer_reports_no_path_samples",
    "tests/unit_tests/test_path_tracer.py::test_authored_sampling_is_inert_for_the_deterministic_renderer",
    "tests/unit_tests/test_render_output_pipeline.py::test_save_frame_runs_a_user_post_process_and_writes_its_output",
    "tests/unit_tests/test_render_output_pipeline.py::test_save_frame_defaults_to_bloom_and_honours_an_empty_chain",
    "tests/unit_tests/test_render_output_pipeline.py::test_save_frame_applies_the_post_process_to_every_still_in_a_sequence",
    "tests/unit_tests/test_render_output_pipeline.py::test_save_frame_writes_a_png_at_the_requested_resolution",
    "tests/unit_tests/test_viewer_fragments.py::test_two_mobs_are_told_apart",
)

# -- C: a slab that neither reflects nor is lit comes back black -----------
#
# The centre pixel of a 48x48 frame filled by the slab reads (0, 0, 0) where the
# authored colour should survive the round trip. Not a decode error -- nothing
# is drawn at all -- and a candidate for the same corrupted covered ordinal as
# B, which is why it is not being chased separately yet.
_add(
    "MPS: the slab renders black -- DESIGN_mps_support.md §4.1 (cause C)",
    "tests/unit_tests/test_color_decode_boundary.py::test_an_unlit_authored_colour_renders_as_itself",
    "tests/unit_tests/test_color_decode_boundary.py::test_an_emissive_colour_renders_as_itself",
)

# -- D: the glossy prefilter loses its reflection ---------------------------
#
# One arm divides by a zero signal (`nan > 3 * 74.9`), the other finds four
# interior local maxima where the prefilter should have left at most two. Both
# are the reflection buffer coming back empty or unfiltered.
_add(
    "MPS: the glossy prefilter's reflection is empty -- DESIGN_mps_support.md §4.1 (cause D)",
    "tests/unit_tests/test_glossy_prefilter.py::test_prefiltered_reflection_is_substantially_wider",
    "tests/unit_tests/test_glossy_prefilter.py::test_a_creases_siblings_share_the_pixels_prefiltered_claim",
)

# -- E: the two composites disagree by 107 ----------------------------------
#
# The path-traced and deterministic routes agree everywhere on CPU and CUDA and
# differ by 107 over 34 of 2123 flat pixels here. Distinct from §2.3c's closed
# shell, which was the same shape of failure and is fixed.
_add(
    "MPS: path-traced and deterministic transparency differ -- DESIGN_mps_support.md §4.1 (cause E)",
    "tests/unit_tests/test_path_tracer.py::test_path_traced_transparency_matches_deterministic_compositing",
)

# -- F: a real function's early return will not compile to SPIR-V -----------
_add(
    "MPS: ti.real_func early return breaks the SPIR-V builder -- "
    "DESIGN_mps_support.md §4.3",
    "tests/unit_tests/test_taichi_early_return.py::test_a_real_function_is_not_rewritten",
)

# -- H: one stray fragment in the bottom row --------------------------------
#
# The cube's lit rows are 5..15, correct, plus a single lit pixel in row 35 of
# 36. One fragment composited at a pixel nothing should have written, which is
# the same shape as B and is the reading that suggests B's ordinal is the cause
# of more than B.
_add(
    "MPS: one fragment lands in the wrong pixel -- DESIGN_mps_support.md §4.2",
    "tests/unit_tests/test_viewer_fragments.py::test_pixel_rows_are_not_flipped",
)
