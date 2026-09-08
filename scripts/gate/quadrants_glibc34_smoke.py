"""Exercise a built Quadrants wheel in the oldest supported aarch64 userspace.

Quadrants kernels must live in a real source file because its frontend uses
``inspect`` to recover the decorated function body before compilation.  Keep
this out of ``python -`` / stdin smoke snippets for that reason.
"""

from __future__ import annotations

import numpy as np
import quadrants as qd
from quadrants.lang import impl


qd.init(arch=qd.cpu, invariant_arg_loads=False, readonly_ndarray_ldg=True)
cfg = impl.current_cfg()
assert cfg.invariant_arg_loads is False
assert cfg.readonly_ndarray_ldg is True


@qd.kernel
def add_one(a: qd.types.ndarray()):
    for i in range(a.shape[0]):
        a[i] += 1


x = np.arange(8, dtype=np.float32)
add_one(x)
qd.sync()
np.testing.assert_array_equal(x, np.arange(8, dtype=np.float32) + 1)
print("fresh glibc-2.34 import + qd.init + kernel OK")
