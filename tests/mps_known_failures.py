"""The tests that do not pass on an Apple GPU yet, and why each does not.

``.github/workflows/test.yaml`` runs ``tests/unit_tests tests/fast`` on
``macos-latest`` with ``ALGAN_RENDER_DEVICE=mps``, and that arm is a **required**
check. When it was first turned on it reported 32 failures and 3 errors against
3311 passes (run 34102515789). **One is left**, and how the other came off is
the caution this file needs most. Run 34302218686 reported the
prefiltered-reflection entry as a strict XPASS -- the self-cleaning rule below
firing -- and reading back through the arm's runs, that render had **never**
failed on master: it was measured on the branch the entry was written on, and
master's own arm was already green by the time that branch merged
(``DESIGN_mps_support.md`` §4.5 has the run table). So an entry copied from a
branch run is a claim about that branch until a master run repeats it. This
file is the record of the one that remains;
``algan/rendering/DESIGN_mps_support.md`` §4 is the scoreboard behind it.

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
  in ``test_taichi_early_return.py`` spreads Apple-GPU knowledge across the
  eight files that have carried an entry and have nothing else to do with it.
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
