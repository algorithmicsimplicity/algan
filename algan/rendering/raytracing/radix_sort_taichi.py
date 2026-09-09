"""A stable argsort over a device key array, composed from Quadrants' radix sort.

``qd.algorithms.sort`` is an LSB radix sort published as a ``@qd.func`` that
must be called at the **top level** of a kernel: each of its phases is one
top-level ``for``, which the compiler offloads as its own serialized launch,
and that serialization is the grid-wide barrier the algorithm needs between
histogram, scan and scatter. Nesting the call inside ordinary control flow
demotes those loops and corrupts the sort, so the whole wrapper below is one
kernel whose body is a seed loop followed by the call, and nothing else.

The sort orders ``keys`` in place and carries ``values`` along, so an
**argsort** is the sort of ``(key, position)`` pairs with the positions read
back. Two things then follow from writing the seed loop ourselves rather than
handing torch the job:

* the caller's key array is never mutated -- the seed loop copies it into the
  working buffer, which the sort is free to destroy;
* the gather that composes an LSD multi-key sort happens **in the kernel**.
  ``_lexsort``'s torch form is ``argsort`` / ``index_select`` / ``index_select``
  per key; here the running permutation arrives as ``perm_in`` and the seed
  loop reads ``src_key[perm_in[i]]`` directly, so a three-key order costs three
  launches and no torch gather at all. That matters most on MPS, where an
  integer ``index_select`` is both slow and inexact above 2**24
  (``mps_compat._MPS_EXACT_INT_BITS``); a kernel subscript is neither.

``mode`` selects between the three seedings actually used, as a compile-time
template so each is its own specialization with no branch in the loop:

===== ================================ =========================================
mode  ``keys[i]``                      ``values[i]``
===== ================================ =========================================
0     ``src_key[i]``                   ``i``            (a fresh argsort)
1     ``src_key[i]``                   ``perm_in[i]``   (keys already permuted)
2     ``src_key[perm_in[i]]``          ``perm_in[i]``   (one LSD pass)
===== ================================ =========================================

Mode 1 is what a two-stage sort whose second key was built in the first
stage's order needs (``raster_pipeline._exact_fragment_order``): the keys are
already in ``perm_in``'s space, and only the permutation has to be composed.

**Stability.** Every ingredient is stable: ``block.radix_rank_match_atomic_or``
returns a thread's stable rank within its block, tile histograms are scanned
digit-major over blocks in block order, and the pass loop is
least-significant-digit first. So the emitted permutation is the one
``torch.argsort(..., stable=True)`` emits, and an LSD composition over several
keys is a lexicographic order, exactly as :func:`~algan.rendering.raytracing.sheets._lexsort`'s
torch form is.

**Float keys.** ``f32`` is a supported key dtype -- the sort maps IEEE bits to
a monotone unsigned order and back -- so a depth key needs no packing. A bit
order and a comparison sort disagree about two things, and the seed loop
canonicalizes both (see the comment there) so the permutation is the one
``torch.argsort`` produces on the CPU: signed zeros, which compare equal but
have different bit patterns, and NaN, which torch orders last whatever its
sign where the twiddle puts a negative one below ``-inf``.

**Scratch dtype.** ``sort`` documents its scratch as ``u32``; it is passed here
as **int32**. Torch's unsigned integer dtypes are barely implemented (and
:func:`algan.rendering.mps_zero_copy._taichi_dtype` maps none of them, so a
``uint32`` argument would be *staged through the host* on Metal, which is the
one thing this path exists to avoid). Every value the buffer holds is a tile
histogram count (at most ``BLOCK_DIM`` = 256) or a scanned offset into the
input (at most ``n``), so nothing in it can reach the sign bit and the two
readings of the bits agree.
"""

from algan.taichi_compat import ti


@ti.kernel
def argsort_pairs(
    src_key: ti.types.ndarray(),
    perm_in: ti.types.ndarray(),
    keys: ti.types.ndarray(),
    tmp_keys: ti.types.ndarray(),
    values: ti.types.ndarray(),
    tmp_values: ti.types.ndarray(),
    scratch: ti.types.ndarray(),
    count_buf: ti.types.ndarray(),
    count: ti.i32,
    key_dtype: ti.template(),
    end_bit: ti.template(),
    log256_max_n: ti.template(),
    mode: ti.template(),
):
    """Write the stable order of ``src_key`` (seeded per ``mode``) into ``values``.

    ``keys`` / ``tmp_keys`` / ``tmp_values`` / ``scratch`` are caller-owned
    working buffers; ``count_buf`` is the 0-d int32 the sort reads its count
    from, filled here so the host never has to write a device scalar of its
    own. Only ``values`` is an output.
    """
    # One top-level loop, so one offload: seed the pairs and publish the count
    # the sort's own phases will read. Every later phase is a separate offload,
    # which is what orders this write before that read.
    for i in range(count):
        j = i
        if ti.static(mode != 0):
            j = ti.cast(perm_in[i], ti.i32)
        k = i
        if ti.static(mode == 2):
            k = j
        value = src_key[k]
        if ti.static(key_dtype == ti.f32):
            # Two float values a comparison sort calls EQUAL and a bit order
            # does not, canonicalized here so the permutation is torch's:
            #
            # * -0.0 and +0.0 compare equal, and the twiddle puts every -0.0
            #   before every +0.0. (Torch on MPS separates them too, so this
            #   agrees with the CPU rather than with the local backend -- and
            #   the CPU is what the renderer's baselines were taken on.)
            # * NaN compares greater than everything under torch whatever its
            #   sign, while the twiddle sends a NEGATIVE NaN below -inf. One
            #   canonical quiet NaN also makes payloads tie rather than order.
            #
            # A depth is a distance along a ray and can be neither, so this
            # buys agreement on inputs the renderer does not produce; it costs
            # two integer compares in a loop already reading every element.
            #
            # Both tests read the BITS, and the NaN one has to: ``value !=
            # value`` is the obvious spelling and a fast-math backend folds it
            # to false, which is how the first version of this shipped a
            # canonicalization that did nothing. ``sheet_sort_taichi._after``
            # inspects the bits for the same reason. Measured on Metal by
            # ``_device_sort_probe.py``'s "signed zeros and NaN" arm, which
            # disagreed with torch on every one of 2.9M positions until this
            # was bits rather than comparisons -- a whole NaN group moving from
            # the end of the order to the front shifts everything after it.
            bits = ti.bit_cast(value, ti.u32)
            if bits == ti.u32(0x80000000):
                value = ti.bit_cast(ti.u32(0), ti.f32)
            elif (bits & ti.u32(0x7FFFFFFF)) > ti.u32(0x7F800000):
                value = ti.bit_cast(ti.u32(0x7FC00000), ti.f32)
        keys[i] = value
        values[i] = j
        if i == 0:
            count_buf[()] = count
    ti.algorithms.sort(
        keys,
        tmp_keys,
        values,
        tmp_values,
        scratch,
        count_buf,
        key_dtype,
        True,
        end_bit,
        log256_max_n,
    )
