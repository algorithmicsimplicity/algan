from __future__ import annotations

import re
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected exactly one occurrence of {old!r}, found {text.count(old)}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


def regex_once(path: str, pattern: str, replacement: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one regex match for {pattern!r}, found {count}")
    file.write_text(updated, encoding="utf-8")


# TODO #3: reconcile the stale pre-Volta ledger with the already-recorded
# 2026-09-05 GTX 1050 run in MIGRATION.md section 11.
replace_once(
    "taichi_patches/MIGRATION.md",
    "| pre-Volta CUDA works on sm_61 | **NOT verified** — compile-only; needs the maintainer's GTX 1050 |",
    "| pre-Volta CUDA works on sm_61 | **verified** on the maintainer's Windows 10 GTX 1050 (sm_61, 4 GB, driver 576.52): `qd.init(qd.cuda)` and kernels run; runtime PTX has 0 `atom.sys`, the expected `atom.gpu.cas.b64`, and `.version 5.0 / .target sm_61` — §11.1 |",
)
replace_once(
    "taichi_patches/MIGRATION.md",
    "4. **Run 0003 on the GTX 1050.** Look for `atom.gpu.cas.b64` and no remaining\n   `atom.sys` (`quadrants_patches/PORTING-NOTES.md` §7).",
    "4. **DONE — run 0003 on the GTX 1050.** §11.1 records the successful sm_61\n   run: `qd.init(qd.cuda)` and kernels execute, runtime PTX contains the expected\n   `atom.gpu.cas.b64`, and no `atom.sys` remains.",
)

regex_once(
    "quadrants_patches/README.md",
    r"\*\*Still unverified: that the CUDA half works\.\*\*.*?(?=\n\n)",
    "**Verified on real pre-Volta CUDA hardware (2026-09-05).** On the maintainer's "
    "Windows 10 GTX 1050 (sm_61, 4 GB, driver 576.52), the patched wheel brings "
    "`qd.init(arch=qd.cuda)` up and runs kernels correctly. Runtime PTX contains "
    "no `atom.sys`, contains the expected `atom.gpu.cas.b64`, and targets "
    "`.version 5.0 / .target sm_61`. The 0005–0007 on/off verifier also passes "
    "on that card: `.maxnreg` appears only on the enabled arm, `ld.global.nc` "
    "is gated by `readonly_ndarray_ldg`, and `fast_math` selects "
    "`__nv_fast_expf`. `../taichi_patches/MIGRATION.md` §11 is the full run "
    "record.",
)
replace_once(
    "quadrants_patches/README.md",
    "**0001-0004 apply, and they compile.** \"Applying them\" above is the apply half.\nThe gate runs below are the compile half, and they predate 0004: they measured\n0001-0003 at 15 files, +492/−8. 0004 adds 5 files and 135 lines and was built\nand checked separately, on Linux — see its own section. **0005-0007 are outside\nevery claim in this section**: they have not been applied, built or run, and\ntheir own section (below, before \"Upstreaming\") is the record of that.",
    "**All seven patches apply and compile, and the CUDA behaviour that requires a\nreal device has now been verified on sm_61.** The historical gate runs below\npredate 0004 and measured 0001-0003 at 15 files, +492/−8; 0004 was built and\nchecked separately on Linux. 0005-0007 were subsequently built in the\nthree-platform wheel workflow and their PTX/IR behaviour was verified on the\nmaintainer's GTX 1050. See the dedicated sections below and\n`../taichi_patches/MIGRATION.md` §11 for the hardware record.",
)

replace_once(
    "taichi_patches/PLAN.md",
    "the patch is **unbuilt and unrun**",
    "the patch was **unbuilt and unrun when this plan was written**, and was later hardware-verified on the maintainer's GTX 1050 (sm_61) on 2026-09-05; see `MIGRATION.md` §11.1",
)
replace_once(
    "taichi_patches/PLAN.md",
    "and only the maintainer's sm_61 box can answer whether it actually works",
    "and at the time only the maintainer's sm_61 box could answer whether it actually worked; that check is now PASS in `MIGRATION.md` §11.1",
)

regex_once(
    "scripts/gate/quadrants_linux_build.sh",
    r"# So this is a \*\*compile check, not a behaviour check\*\*\. It answers \"do the\n# patched files still build\", which is the half that can be automated\. The half\n# that cannot is whether sm_61 now loads the runtime module and runs a kernel;\n# that needs the maintainer's GTX 1050.*?(?=\n[^#])",
    "# This is a **compile check, not a recurring hardware behaviour check**. It\n"
    "# answers \"do the patched files still build\", which is the half that can be\n"
    "# automated. The one-time sm_61 behaviour check was completed on the\n"
    "# maintainer's GTX 1050 on 2026-09-05: the runtime module loads, kernels run,\n"
    "# and the PTX evidence matches the patch design. See taichi_patches/MIGRATION.md\n"
    "# §11.1. A future compiler/base change still needs equivalent hardware\n"
    "# revalidation; this runner cannot provide it.\n",
)

# TODO #6: make the full-render oracle deterministic with respect to
# torch.compile availability. The canonical contract is eager execution.
replace_once(
    "tests/full_renders/test_full_renders.py",
    "This module renders it at ``PREVIEW`` and compares every frame against the\nchecked-in baseline in ``expected_outputs_<device>/``.",
    "This module renders it at ``PREVIEW`` and compares every frame against the\nchecked-in baseline in ``expected_outputs_<device>/``. The baseline contract\npins ``SETTINGS.computing.torch_compile=False``: compiled and eager triangle\nprojection have produced different rounding on otherwise identical renders, so\nthe oracle must not depend on whether ``torch.compile`` happens to work on the\nhost.",
)
replace_once(
    "tests/full_renders/test_full_renders.py",
    "    ``available_memory_override`` pins the frame-window split; see\n    ``AVAILABLE_MEMORY_OVERRIDE``.",
    "    ``available_memory_override`` pins the frame-window split; see\n    ``AVAILABLE_MEMORY_OVERRIDE``. ``torch_compile=False`` is equally part of\n    the baseline contract: every comparison and every rebaseline runs the\n    PyTorch stages eagerly.",
)
replace_once(
    "tests/full_renders/test_full_renders.py",
    "    SETTINGS.computing.set(available_memory_override=AVAILABLE_MEMORY_OVERRIDE)",
    "    # Baseline contract: eager PyTorch. Historical CPU baselines were made on\n    # a Linux box where torch.compile worked; they need one eager rebaseline on\n    # their canonical machine before CPU pixel verification is authoritative.\n    SETTINGS.computing.set(\n        available_memory_override=AVAILABLE_MEMORY_OVERRIDE, torch_compile=False\n    )",
)

print("updated TODO #3 GTX ledger and TODO #6 eager baseline contract")
