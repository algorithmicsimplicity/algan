"""The tests that do not pass on an Apple GPU yet, and why each does not.

``.github/workflows/test.yaml`` runs ``tests/unit_tests tests/fast`` on
``macos-latest`` with ``ALGAN_RENDER_DEVICE=mps``, and that arm is a **required**
check. When it was first turned on it reported 32 failures and 3 errors against
3311 passes (run 34102515789). **Nine failures are left** (run 34203241437:
12 failed, 3544 passed, 217 skipped, 9 xfailed — of those 12, eleven were this
list's own strict XPASSes and are gone from it, and the twelfth was an unrelated
curation guard). The entries below are what remains, and this file is the record
of them. ``algan/rendering/DESIGN_mps_support.md`` §4 is the scoreboard behind
it — five causes, not nine tests, which is what makes the remainder tractable.

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


# -- B: two renders that still disagree, cause no longer established -------
#
# This was thirteen entries, all raising `IndexError` at `tracer.py`'s
# `frame_end = gl_bounds[gl_frame + 1]` -- the glossy tile loop running past
# the end of its frame table. **Eleven of them now pass** and were removed
# after run 34203241437, and it is worth being exact about why, because it was
# not a fix aimed at them: emptying torch's MPS cache before sizing the arena
# (§4.4) gave the render its full budget back, and the tile sizing that follows
# from the budget no longer produces the window that tripped the loop.
#
# So the loop's guard is still missing -- a `gl_frame` that can walk off the
# end is a latent IndexError at any window the arithmetic happens to pick --
# and the two below are what is left of the group. Their cause is NOT
# re-measured since the arena fix: they may be the same defect at a different
# window, or something else entirely that the group was hiding. Measuring that
# is the first thing to do here, and §4.2 has the covered-pixel reading to do
# it against.
_add(
    "MPS: renders differently, cause not re-measured since the arena fix -- "
    "DESIGN_mps_support.md §4.2",
    "tests/unit_tests/test_deterministic_shadow_opacity.py::test_deterministic_shadows_accumulate_every_blocker_opacity[raster]",
    "tests/unit_tests/test_display_referred_coverage.py::test_partial_coverage_matches_a_supersampled_render",
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
