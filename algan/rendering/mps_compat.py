"""MPS-friendly mode: the renderer restricted to what an Apple GPU can run.

``DESIGN_mps_support.md`` §1.2 is the measurement this module answers. Metal
has **no float64 at all** -- Taichi's SPIR-V codegen refuses ``f64`` outright,
and Torch refuses to put a float64 tensor on an MPS device before that -- and a
handful of Torch reductions are missing on the backend as well. Algan's
renderer leans on both: several accumulators are deliberately float64
(``DESIGN_sheet_resolve.md`` §6.6.4 -- a float32 atomic sum is not
order-reproducible and these values feed thresholds), the sheet compaction
reduces int64 positions with ``scatter_reduce_(reduce='amin')``, and three
places take a running extremum with ``cummax``/``cummin``.

MPS-friendly mode substitutes, in one place each:

============================ ==================================================
float64 accumulator          float32 (:func:`accumulate_dtype`, and
                             :func:`taichi_accumulate_dtype` for the kernels)
int64 amin/amax reduction    int32 (:func:`reduction_index_dtype`, and
                             :func:`taichi_reduction_index_dtype`; the same
                             narrowing takes Taichi's int64 atomics out of the
                             kernels, which Metal aborts on rather than
                             refusing to compile)
``cummax`` / ``cummin``      a log-step scan of ``maximum`` / ``minimum``
                             (:func:`cummax_values`, :func:`cummin_values`)
an i64 index array a kernel  the array, already int32 (:func:`kernel_index`).
narrows per element          Not a Metal limit but a Taichi **codegen** bug
                             (§1.2b): the narrowing cast comes out of the
                             SPIR-V-to-MSL step as ``int(long(x))``, which C++
                             reads as a function type, and the shader does not
                             compile. Narrowing at the boundary removes the
                             cast rather than working around the parse.
============================ ==================================================

The narrowings are safe as *values* and lossy as *reproducibility*, which are
different claims and only the second one costs anything:

* Every int64 reduction the mode narrows carries a **position, a count or a
  surface id** -- all bounded by the fragment count, which cannot approach
  2**31 because the arrays would not fit in any device's memory. The int32
  answer is the int64 answer, bit for bit. The sentinels move with the width
  (``1 << 40`` becomes ``2**31 - 1``), and they are only ever compared against
  real values, never read.
* The float accumulators genuinely lose the property they were widened for.
  **MPS-friendly mode is not deterministic**, and that is deliberate: §1.2
  measured the one exactly-order-independent alternative -- a Q32 fixed-point
  accumulator -- aborting inside a Metal kernel, so a non-deterministic mode
  has a floor and a deterministic one does not. Two renders of one scene on
  MPS may differ in their low bits.

Nothing here is reached unless the mode is on, and the mode is off unless the
render device is MPS or a script asks for it, so CPU and CUDA renders keep the
float64 path and their byte-identity. That the mode can be asked for on any
device is what makes it testable: ``tests/unit_tests/test_mps_friendly.py``
runs it on the CPU, where the float64 path is right there to compare against.

The substitutions dispatch on the **mode**, never on a tensor's device, for
exactly that reason -- a helper that quietly kept the wide path on a CPU tensor
would make the CPU test prove nothing about MPS.

**Scope: the render device, not the animation device.** The mode covers what
runs on ``SETTINGS.computing.render_device`` -- the renderer, and the render
primitives the mobs build for it. Authoring state lives on
``ALGAN_ANIMATION_DEVICE``, which is initialization-only and **cpu**, and
several authoring paths use float64 there quite happily: a polyhedron's signed
shell volume, a circuit's plane sagitta, the timeline's event index. Torch's
CPU backend has float64 on a Mac like anywhere else, so on a normal Apple
machine -- MPS render device, CPU animation device -- none of that is reached
by Metal. Setting ``ALGAN_ANIMATION_DEVICE=mps`` is a different, unsupported
thing, and this mode does not make it work.
"""

from __future__ import annotations

import torch

from algan.environment import env_flag
from algan.settings._startup import render_device


def mps_friendly() -> bool:
    """Whether the renderer is restricted to MPS-runnable operations.

    ``SETTINGS.computing.mps_friendly`` decides, and its default ``'auto'``
    resolves here rather than at import: the render device is settable between
    renders, so the mode has to be able to follow it. ``ALGAN_MPS_FRIENDLY``
    overrides both, which is how a CPU machine runs the mode's own tests and
    how CI renders one suite each way.

    Call it; never bind the result at import time, for the same reason
    :func:`algan.settings._startup.render_device` says not to.
    """
    from algan.settings import SETTINGS

    configured = SETTINGS.computing.mps_friendly
    if configured == "auto":
        configured = render_device().type == "mps"
    return env_flag("ALGAN_MPS_FRIENDLY", bool(configured))


def accumulate_dtype() -> torch.dtype:
    """The dtype for the renderer's wide float accumulators.

    ``torch.float64`` normally; ``torch.float32`` in MPS-friendly mode, where
    float64 does not exist. Every call site is a §6.6.4 accumulator -- an
    exact-area sum, a coverage sum, a next-event CDF -- that accumulates wide
    and rounds to float32 for use, so narrowing it makes the round a no-op
    rather than changing what the value means.
    """
    return torch.float32 if mps_friendly() else torch.float64


def reduction_index_dtype() -> torch.dtype:
    """The dtype for the renderer's integer min/max reductions.

    ``torch.int64`` normally; ``torch.int32`` in MPS-friendly mode, where
    ``scatter_reduce_(reduce='amin')`` and its ``amax`` twin are unimplemented
    at int64 and Taichi's int64 atomics abort the process. Only reduction
    *accumulators* narrow -- the fragment keys, which really do need 64 bits,
    are never reduced this way and stay int64 (Metal's int64 arithmetic,
    shifts and packing all work; it is the atomics that do not).
    """
    return torch.int32 if mps_friendly() else torch.int64


def reduction_index_sentinel() -> int:
    """The "nothing reduced into this slot" fill for the dtype above.

    ``1 << 40`` at int64 and ``2**31 - 1`` at int32. Both are larger than any
    position, count or surface id a render can produce, and both are only ever
    compared against, so which one a slot holds is unobservable.
    """
    return (1 << 40) if reduction_index_dtype() is torch.int64 else 2147483647


#: Float16's smallest positive subnormal, and the floor every ``clamp_min``
#: bound below it collapses to on MPS. ``probe_epsilon_clamp`` in
#: ``benchmarks/_mps_torch_op_probe.py`` sweeps both sides of it; §2.3c has the
#: table. Bounds at or above this are honoured exactly, which is why
#: :func:`clamp_floor` is only needed underneath it.
_MPS_CLAMP_FLOOR = 5.9604645e-8


def clamp_floor(tensor: torch.Tensor, floor: float) -> torch.Tensor:
    """``tensor.clamp_min(floor)`` for a floor too small for MPS to carry.

    **MPS rounds a clamp's scalar bound through float16.** Every bound at or
    above float16's smallest subnormal (:data:`_MPS_CLAMP_FLOOR`, 5.96e-8) is
    honoured exactly; every bound below it comes back as *that* subnormal
    instead, and so does every input below it. With a 1e-12 floor the Apple GPU
    returns 5.96e-8 for an input of 0, of 1e-13, and of 1e-11 -- the last of
    which is above the floor and should have passed through untouched. So the
    effective floor there is a hard 5.96e-8 whatever was asked for, and the
    three spellings ``clamp_min``, ``clamp(min=)`` and ``torch.clamp_min`` are
    equally affected. §2.3c of ``DESIGN_mps_support.md`` has the sweep.

    The renderer guards twenty-odd divisions this way, at floors of 1e-12,
    1e-20 and 1e-30. On MPS every one of those denominators is silently raised
    to 5.96e-8 -- up to twenty-two orders of magnitude off -- and unlike the two
    defects above it this one produces neither a crash nor a NaN, just a
    plausible wrong number. ``raster_pipeline``'s ``n2.clamp_min(1e-30)`` is a
    squared normal length, so a thin triangle lands squarely in the gap.

    ``where(x < floor, floor, x)`` is ``clamp_min`` **bit for bit** -- over 0,
    -0, ±inf, NaN, the subnormals and both sides of the floor, the two agree in
    every bit of every result. The comparison is ``<`` rather than ``>`` for
    the one input where the order matters: a comparison against a NaN is false
    either way, and only this order leaves the NaN in the ``x`` arm, where
    ``clamp_min`` leaves it. (``where(x > floor, x, floor)`` returns the floor
    there instead -- identical everywhere else, and a silent NaN swallower.)

    Because the two are bit-identical, gating on the mode costs nothing and
    keeps this module's rule intact: off the mode this is exactly
    ``clamp_min``, one op, and CPU and CUDA do not pay a compare and a select
    for a defect they do not have.

    A ``floor`` at or above the cliff takes ``clamp_min`` on **every** backend,
    because there the clamp is measured correct on MPS too and is the cheaper
    of the two. So this is a correct clamp at any magnitude a caller passes,
    and raising a floor past the cliff -- an ordinary numerical tuning change --
    quietly stops paying for the workaround rather than becoming an error. What
    keeps ``clamp_min`` from creeping back into the renderer is a static check,
    ``test_mps_friendly.py::test_the_renderer_floors_a_divide_through_clamp_floor``,
    which is where that belongs: a call-time exception on a valid floor would
    fire mid-render, on whichever scene first reached the branch.

    **One magnitude is beyond either spelling.** At a float32 *subnormal* floor
    (1.4e-45) the sweep has ``where`` failing on MPS as well -- the constant
    itself does not survive -- so neither form is a guard there. No call site
    is anywhere near it, and this says so rather than implying a universal fix.
    """
    if floor >= _MPS_CLAMP_FLOOR or not mps_friendly():
        return tensor.clamp_min(floor)
    return torch.where(tensor < floor, floor, tensor)


def kernel_index(tensor: torch.Tensor) -> torch.Tensor:
    """An index array on its way into a kernel, narrowed in MPS-friendly mode.

    Not a dtype choice about the *value* -- these are stream positions, CSR
    starts and counts, and permutations, all bounded by the fragment count --
    but a way around a **Metal codegen bug**, which is why it is its own
    function rather than a use of :func:`reduction_index_dtype`.

    Taichi's SPIR-V-to-MSL step renders a narrowing cast of a 64-bit ndarray
    load as a nested functional cast, and when the result is bound to a
    temporary the generated line reads::

        int tmp16_i32 = (int(long(_76))) * 8;

    which is C++'s most vexing parse: ``int(long(_76))`` is the function type
    ``int(long)`` with a parameter named ``_76``, so the ``* 8`` after it
    parses as a dereference and the shader does not compile. Metal reports
    that by handing Taichi a nil function, and Taichi builds a pipeline from
    it without checking -- ``failed assertion 'computeFunction must not be
    nil'``, a SIGABRT rather than an exception (``DESIGN_mps_support.md``
    §1.1 saw the same abort from the argument limit).

    The kernels all narrow such a load immediately (``b = ti.cast(band[i],
    ti.i32)``), so passing the array already narrowed removes the cast
    entirely: Taichi emits nothing for a cast to the type a value already
    has, and the same kernel source serves both widths. Off the mode this is
    the identity, down to the object -- ``.to`` returns ``self`` when the
    dtype already matches.
    """
    return tensor.to(torch.int32) if mps_friendly() else tensor


#: Where MPS's integer gather stops being exact, and **why**. Measured on an
#: Apple GPU by ``benchmarks/_mps_torch_op_probe.py``, which builds the values
#: on the host, proves the move to the device is bit-exact, and only then
#: gathers -- so what it catches is the gather and nothing around it:
#:
#:     int32 2**16 ok    int32 2**24 ok    int32 2**25 FAIL   int32 2**30 FAIL
#:     int64 2**24 ok    int64 2**25 FAIL  int64 2**40 FAIL   int64 2**62 FAIL
#:
#: The boundary is 2**24 and it is the same for both widths, which already says
#: this is not about int64. The returned values say what it *is*: every one of
#: them is exactly ``float32(correct_value)``, round-to-nearest --
#: 18271053 -> 18271052, 756440460 -> 756440448, 976314890686 -> 976314892288,
#: 3314435950399956755 -> 3314436020488896512. So the gather round-trips
#: integers through a float32, and 24 bits is its significand.
#:
#: **It is a torch dispatch defect, not a Metal limit.** In the same run, over
#: the same values at 2**40, ``index_select`` and ``torch.gather`` are wrong
#: while **advanced indexing ``v[i]`` is exact**, and so is a
#: ``repeat_interleave`` slice (``torch.take`` is not implemented on MPS at
#: all). The hardware can clearly move those bits; two particular aten paths
#: do not. Storing, moving, comparing, shifting, masking and multiply-add are
#: exact at every rung.
#:
#: So :func:`gather_packed_key` uses ``v[i]``: one gather, exact, and the
#: narrowest possible change. What it costs is a **dependency on a dispatch**
#: -- ``v[i]`` is right because of which aten kernel torch routes it to, and
#: nothing in torch's API promises that stays true. That dependency is made
#: loud rather than left implicit:
#: ``tests/unit_tests/test_mps_friendly.py::test_advanced_indexing_is_exact_above_the_mps_ceiling``
#: gathers values past this ceiling on MPS whenever the machine has one and
#: fails the suite if the answer ever stops being exact. It is the guard that
#: makes the fast path safe to take.
_MPS_EXACT_INT_BITS = 24


def gather_packed_key(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``tensor.index_select(0, index)`` for a packed 64-bit key.

    **MPS gathers an integer through 24 bits of mantissa.** Below 2**24 it is
    exact; above, it silently rounds -- see ``_MPS_EXACT_INT_BITS`` for the
    measurement. That is not a rounding difference but a lost image, and the
    arithmetic says so exactly. The renderer's fragment key is ``pixel << 32 |
    bit_cast(depth)`` (``raster_taichi.py:2039``), about 2**50 for a 1080p
    frame, so a 24-bit gather destroys bits 25..0 and leaves the low word
    masked with ``0xFC000000``. Every depth in ``[4, 8)`` -- which is the whole
    smoke scene -- then reads back as **exactly 2.0**, and that is what the
    Apple GPU produced: ``depth min=2.000000 max=2.000000 distinct=1`` against
    the CPU's ``5.317023 .. 7.723102, distinct=37899``. With every fragment at
    one depth the compaction cannot order or band them, and 40956 sheets
    collapsed to 128.

    **The defect is the dispatch, not the hardware**, which is what makes the
    fix a one-liner: over the same values at 2**40, ``index_select`` and
    ``torch.gather`` are wrong while ``v[i]`` -- advanced indexing, which lands
    on ``aten::index`` -- is exact. So the mode gathers with ``v[i]``. For a
    1-D index that selects along dim 0 and copies, which is ``index_select``'s
    contract exactly.

    Two earlier attempts are worth knowing about, because they say why the
    ceiling is 2**24 and not something more convenient. Splitting the key into
    two 32-bit halves does **not** work: a half still reaches 2**32, and the
    render that shipped it came back with the low word rounded from ``40e68475``
    to ``40e68480`` and its distinct depths down from 37899 to 22292 -- better
    than the 2.0 it replaced and still wrong. Splitting into four 16-bit lanes
    does work, and was what shipped before this: correct on any dispatch path,
    at four gathers instead of one. ``v[i]`` replaced it because the guard below
    makes the cheaper form safe.

    **What this costs is a dependency on a dispatch.** Nothing in torch's API
    promises ``v[i]`` keeps routing to a kernel that is exact here, and if it
    ever stops the failure is silent -- the same shape of failure this module
    exists to contain. That is why
    ``test_advanced_indexing_is_exact_above_the_mps_ceiling`` exists: it gathers
    past the ceiling on MPS whenever the machine has one, and fails the suite
    loudly if the answer changes. The fast path is only safe with that guard in
    place; do not drop one without the other.

    Off the mode, and for any dtype that is not int64, this is exactly
    ``index_select``.

    **Not the only gather at risk, only the confirmed one.** ``sheets._lexsort``
    gathers ``pix``, which is ``frame_rel * width * height + pixel``: 282179 for
    the 1080p smoke frame, comfortably exact, but a multi-frame 4K chunk would
    put it past 2**24. Nothing has rendered at that size on MPS yet, so this
    stays scoped to the key the hardware actually caught -- it is a real
    exposure and it is written down rather than guessed at.
    """
    if not mps_friendly() or tensor.dtype is not torch.int64:
        return tensor.index_select(0, index)
    return tensor[index]


def band_class_groups(band_of_frag, cls_eff, base):
    """Group the fragments by ``(band, shading class)``, without a wide key.

    ``compact_sheets`` subdivides each band by shading class with a composite
    key, ``band * _SHADE_CLASS_BASE + cls``, and a ``unique`` over it
    (``sheets.py`` §4.4). The base is ``1 << 25``, so for a 1080p frame with
    40956 bands the key reaches **2**40** -- past where MPS int64 stops being
    exact, and rows that differ only in their low bits merge.

    Measured, and measured as the *only* thing left: with the split off the
    same Apple GPU compaction produced 40956 sheets, exactly the CPU's, and
    with it on, 128.

    So this groups the pairs directly. It sorts by class and then stably by
    band -- a two-pass LSD sort, the same trick ``sheets._lexsort`` uses -- and
    walks the result for boundaries, which never multiplies the two together
    and never handles a value wider than the larger of them. Returns what the
    ``unique`` returned: the group count, the per-fragment group id, and each
    group's band.

    The group ORDER is the same. ``unique(..., sorted=True)`` orders by the
    composite, and because ``base`` exceeds every class the composite orders by
    ``(band, class)`` -- which is what the sort here produces, so the ids match
    the wide-key ones exactly and every consumer downstream is unaffected.

    Off the mode this is the wide key, unchanged: the composite form is one
    sort where this is two, and CPU and CUDA have no reason to pay for that.
    """
    if not mps_friendly():
        skey = band_of_frag * base + cls_eff
        uniq_skey, inverse = torch.unique(skey, sorted=True, return_inverse=True)
        return int(uniq_skey.numel()), inverse, uniq_skey // base

    if band_of_frag.numel() == 0:
        empty = band_of_frag.new_zeros(0)
        return 0, empty, empty
    # Least-significant key first, so the stable sort on the band leaves
    # equal-band runs ordered by class.
    order = torch.argsort(cls_eff, stable=True)
    order = order.index_select(
        0, torch.argsort(band_of_frag.index_select(0, order), stable=True)
    )
    bands = band_of_frag.index_select(0, order)
    classes = cls_eff.index_select(0, order)
    starts = torch.ones_like(bands, dtype=torch.bool)
    if bands.numel() > 1:
        starts[1:] = (bands[1:] != bands[:-1]) | (classes[1:] != classes[:-1])
    del classes
    group_sorted = torch.cumsum(starts.to(torch.int64), 0) - 1
    inverse = torch.empty_like(group_sorted)
    inverse.scatter_(0, order, group_sorted)
    del group_sorted, order
    band_of_group = bands[starts]
    return int(band_of_group.numel()), inverse, band_of_group


def taichi_accumulate_dtype():
    """:func:`accumulate_dtype`'s Taichi twin, for a kernel's ``acc_t``.

    Passed as a ``ti.template()`` argument rather than gated with
    ``ti.static``: Taichi specialises on template arguments, so the two widths
    compile to two kernel variants and a process that renders both ways gets
    both. A ``ti.static`` gate is resolved once, at the first compile, and the
    second arm would silently reuse the first arm's code (``CLAUDE.md``).
    """
    import taichi as ti

    return ti.f32 if mps_friendly() else ti.f64


def taichi_reduction_index_dtype():
    """:func:`reduction_index_dtype`'s Taichi twin, for a kernel's ``idx_t``.

    The kernels cast every atomic operand to this before the atomic, so the
    narrowing is what keeps ``ti.atomic_min`` off int64 -- which Metal answers
    with ``Assertion failed: (p != nullptr), function bind_pipeline`` rather
    than with an error (§1.2).
    """
    import taichi as ti

    return ti.i32 if mps_friendly() else ti.i64


def cummax_values(x: torch.Tensor, dim: int) -> torch.Tensor:
    """``torch.cummax(x, dim).values``, without ``cummax`` in MPS-friendly mode."""
    if not mps_friendly():
        return torch.cummax(x, dim)[0]
    return _scan(x, dim, torch.maximum)


def cummin_values(x: torch.Tensor, dim: int) -> torch.Tensor:
    """``torch.cummin(x, dim).values``, without ``cummin`` in MPS-friendly mode."""
    if not mps_friendly():
        return torch.cummin(x, dim)[0]
    return _scan(x, dim, torch.minimum)


def _scan(x: torch.Tensor, dim: int, combine) -> torch.Tensor:
    """Inclusive scan of an idempotent, associative ``combine`` along ``dim``.

    Hillis-Steele: after the step of stride ``s`` every element holds the
    combination of itself and its ``s`` predecessors, so ``ceil(log2(n))``
    steps reach the whole prefix. ``maximum`` and ``minimum`` are both
    idempotent, so the overlapping ranges the doubling produces are harmless
    and the result is exactly ``cummax``/``cummin``'s -- no floating-point
    reassociation is involved, because neither op reassociates.

    ``n log n`` element visits against the sequential scan's ``n``, in
    ``log n`` launches against one. That is a real cost and it is why this is
    reached only in MPS-friendly mode; the arrays it runs on are either short
    (a BVH's split bins) or reached only through a torch A/B arm.
    """
    n = int(x.shape[dim])
    if n <= 1:
        return x.clone()
    out = x
    step = 1
    while step < n:
        head = out.narrow(dim, 0, step)
        tail = combine(out.narrow(dim, step, n - step), out.narrow(dim, 0, n - step))
        out = torch.cat((head, tail), dim=dim)
        step *= 2
    return out
