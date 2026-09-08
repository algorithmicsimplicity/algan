"""The tests that do not pass on an Apple GPU yet, and why each does not.

``.github/workflows/test.yaml`` runs ``tests/unit_tests tests/fast`` on
``macos-latest`` with ``ALGAN_RENDER_DEVICE=mps``, and that arm is a **required**
check. When it was first turned on it reported 32 failures and 3 errors against
3311 passes (run 34102515789). **Two are left** — run 34213125405 is
``1 failed, 3563 passed, 217 skipped, 2 xfailed``, and that one "failure" is
this list's own last strict XPASS, now removed. This file is the record of the
two; ``algan/rendering/DESIGN_mps_support.md`` §4 is the scoreboard behind it.

Counting causes rather than tests is what got it there. Thirty of the
thirty-two came off with two fixes, neither aimed at most of what it cleared:

* the arena measuring torch's own cache as occupied (§4.4) took eleven;
* the raster acceptance mask losing its low bits to the 2**24 gather ceiling
  (§2.3f) took nineteen — every remaining render failure except one, because
  they were one corrupted fragment stream wearing several faces.

Both were found by measuring the runs this list produced, which is the argument
for keeping the list at all rather than deselecting what fails.

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


# -- D: the prefiltered reflection is absent, and it is Metal's -------------
#
# `nan > 3 * 74.9`, and the nan is the finding: `_reflection_spread` divides by
# the signal's sum, so a nan means the sum is zero -- no reflected glow above
# the wall's own level anywhere in the window the mirror image lands in. Not a
# reflection that came out narrow or dim. Absent.
#
# Localized to the device rather than to this port's own substitutions, which
# is as far as it has been taken: the same test passes on the macOS CPU arm of
# the same runner, on both Linux legs, and -- the discriminator that matters --
# on Linux with `ALGAN_MPS_FRIENDLY=1`, which exercises every substitution
# `mps_compat` makes with no Apple GPU in the picture. §4.5 has the table and
# where to look next.
_add(
    "MPS: the prefiltered reflection is absent -- DESIGN_mps_support.md §4.5",
    "tests/unit_tests/test_glossy_prefilter.py::test_prefiltered_reflection_is_substantially_wider",
)

# -- F: a real function's early return will not compile to SPIR-V -----------
#
# The only entry here that is not a render. `@ti.real_func` with an early
# `return`, launched, dies in Quadrants' SPIR-V builder with `Value "tmp6" does
# not yet exist` -- one layer below Algan, and the shape of §1.2c's defect. No
# renderer kernel uses `ti.real_func`, so it blocks a test rather than a
# picture, and the Vulkan loop that would localize it for free no longer
# exists on the published wheel (§1.2c).
_add(
    "MPS: ti.real_func early return breaks the SPIR-V builder -- "
    "DESIGN_mps_support.md §4.3",
    "tests/unit_tests/test_taichi_early_return.py::test_a_real_function_is_not_rewritten",
)
