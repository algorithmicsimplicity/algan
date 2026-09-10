from pathlib import Path
import hashlib


def edit(path, old, new):
    p = Path(path)
    s = p.read_text()
    assert s.count(old) == 1, (path, old[:80], s.count(old))
    p.write_text(s.replace(old, new))


regions = 'algan/rendering/arena_regions.py'
edit(regions, '''        if any(d < 0 or d > _INT32_MAX for d in shape):
            raise ArenaRegionError("Region shape exceeds the signed int32 kernel ABI")
        if any(s < 0 or s > _INT32_MAX for s in strides):
            raise ArenaRegionError("Region stride exceeds the signed int32 kernel ABI")
        if offset // itemsize + tensor.numel() > _INT32_MAX:
            raise ArenaRegionError("Region end exceeds the signed int32 kernel ABI")
''', '''        # Allocation tracking uses host byte addresses, not kernel indices.
        # A small persistent region may sit beyond 2 GiB in a large arena.
''')
edit(regions, '''    @property
    def byte_end(self):''', '''    def validate_kernel_indexing(self, *, byte_origin=None):
        """Check the int32 ABI relative to the buffer passed to the kernel."""
        if any(d < 0 or d > _INT32_MAX for d in self.shape):
            raise ArenaRegionError("Region shape exceeds the signed int32 kernel ABI")
        if any(s < 0 or s > _INT32_MAX for s in self.strides):
            raise ArenaRegionError("Region stride exceeds the signed int32 kernel ABI")
        # Ordinary ndarray arguments start at the view's data pointer. Packed
        # arguments use the start of the narrowed dtype hull instead.
        origin = self.byte_offset if byte_origin is None else byte_origin
        itemsize = self.dtype.itemsize
        relative = self.byte_offset - origin
        if origin < 0 or relative < 0 or relative % itemsize:
            raise ArenaRegionError("Invalid or misaligned kernel region origin")
        if (relative + self.byte_length) // itemsize > _INT32_MAX:
            raise ArenaRegionError("Rebased region end exceeds the signed int32 kernel ABI")

    @property
    def byte_end(self):''')
edit(regions, '''        layout = RegionLayout.from_tensor(tensor)
        if dtype is not None''', '''        layout = RegionLayout.from_tensor(tensor)
        layout.validate_kernel_indexing()
        if dtype is not None''')
edit(regions, '''            raise ArenaRegionError(f"{self.name}: tensor metadata changed after binding")
        if self.allocation''', '''            raise ArenaRegionError(f"{self.name}: tensor metadata changed after binding")
        self.layout.validate_kernel_indexing()
        if self.allocation''')
edit('algan/rendering/arena_region_args.py', '''        end = max(v.layout.byte_end for v in group)
        sample = group[0].tensor''', '''        end = max(v.layout.byte_end for v in group)
        # Validate after rebasing, before building native views or int32
        # tables. Individually small views can still span an oversized hull.
        for view in group:
            view.layout.validate_kernel_indexing(byte_origin=begin)
        sample = group[0].tensor''')
edit('algan/rendering/raytracing/raster_taichi.py', '''        if ti.static(counted_dispatch):
            # Guard before any payload access or visibility write.
            if dispatch_header[2] != 0 or dispatch_header[3] != 0 \\
                    or e >= dispatch_header[0]:
                continue
''', '''        skip_event = 0
        if ti.static(counted_dispatch):
            skip_event = (dispatch_header[2] != 0 or dispatch_header[3] != 0
                          or e >= dispatch_header[0])
        # Keep the continue outside the static gate for SPIR-V backends,
        # and before any payload access or visibility write.
        if skip_event:
            continue
''')

# Require byte-for-byte equality with the locally tested production fixes.
expected = {
    regions: 'dac57114bb7f8b738aca97b673ea2c6ee93c0151',
    'algan/rendering/arena_region_args.py': '9153b13121c7df8b16c06cffc26ff23d96545e07',
    'algan/rendering/raytracing/raster_taichi.py': 'd5866e715df45e2d8dd4ae9f5bba4c9bcb5a4ab1',
}
for path, want in expected.items():
    data = Path(path).read_bytes()
    got = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
    assert got == want, (path, got, want)

edit('agent_guidance/device_dispatch.md', '''actual alignment and allocation identity. End offsets are checked, not just
starts. Empty tensors use storage offsets rather than their zero data pointer.
All host metadata arithmetic uses integers; out-of-range layouts fail explicitly.''', '''actual alignment and allocation identity. Allocation tracking retains full-width
host byte offsets: a small persistent view above 2 GiB is legal. The int32 kernel
ABI is checked separately, relative to the submitted view or narrowed dtype
buffer. Both rebased starts and ends must fit, as must shapes and strides;
individually small views cannot be packed into an oversized shared span. Empty
tensors use storage offsets rather than their zero data pointer. All host
metadata arithmetic uses integers; out-of-range bindings fail explicitly.''')

# Bind callback arguments immediately rather than capturing loop variables.
p = Path('algan/rendering/raytracing/raster_pipeline.py')
s = p.read_text()
a = s.index('            def trace_events():\n')
b = s.index('            if dispatch is not None:\n', a)
block = s[a:b].splitlines(keepends=True)[1:]
call = ''.join(line[4:] if line.strip() else line for line in block)
call = call.replace('            raster_shadow_trace(\n', '            trace_events = partial(\n                raster_shadow_trace,\n', 1)
s = s[:a] + call + s[b:]
s = s.replace('from __future__ import annotations\n', 'from __future__ import annotations\n\nfrom functools import partial\n', 1)
p.write_text(s)
