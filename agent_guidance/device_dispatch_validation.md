# Q1/Q2 validation

Both original features were implemented before testing.

The original publication reported two render failures (absolute arena offsets incorrectly limited to int32) and one fast-suite failure (continue under a static gate). Both causes are fixed.

## Corrected implementation

targeted: 81 passed, 2 warnings in 186.98s (0:03:06)

fast: 607 passed, 3723 deselected, 2 warnings in 126.99s (0:02:06)

Ruff lint, Ruff format and git diff --check: passed. Kernel modules were not formatted.

Environment: Ubuntu 24.04, Python 3.11, CPU, locked algan-quadrants 1.3.0.post2.

Large-offset tests use sparse file mappings, not multi-gigabyte resident tensors. The shadowed diffuse and reflective scenes test off/on/off output parity and verify that the new dispatch path ran.

Run: https://github.com/algorithmicsimplicity/algan/actions/runs/34540031455

## Local full-suite attempt

The OpenAI Linux container (Python 3.13, Torch 2.10 CPU, diagnostic quadrants 1.3.0.post1) passed all 81 targeted tests. Its fast suite had 606 passes and the known TeX-wrapper baseline failure. A full-suite run with --maxfail=1 reached 1099 passed and 9 skipped, then stopped in test_doc_examples.py at text_and_math.rst:115 because the compatibility dvisvgm wrapper loses SVG groups. The same failing documentation example was reproduced against the unmodified published implementation, 5be7c82, so it is not introduced by these fixes. Native dvisvgm is used for the passing remote fast suite above.

No performance measurements or GPU hardware validation. The complete heavy render-baseline suites are not covered by the remote run.
