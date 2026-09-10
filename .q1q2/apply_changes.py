"""Apply both complete features to the pinned source tree before any tests."""
from pathlib import Path
import re

ROOT = Path.cwd()


def edit(path, old, new, count=1):
    p = ROOT / path
    s = p.read_text()
    assert s.count(old) == count, (path, old[:100], s.count(old))
    p.write_text(s.replace(old, new))


edit('algan/environment.py', '    "ALGAN_DIRECT_SPECULAR_LOBE",',
     '    "ALGAN_DEVICE_DISPATCH",\n    "ALGAN_DIRECT_SPECULAR_LOBE",')
edit('algan/rendering/raytracing/settings.py', '# Maximum number of ray bounces',
'''# Opt-in audit Q1/Q2: device-counted shadow queues and typed arena regions.
# Capacity-guarded portable submission; no change to coverage or defaults.
device_dispatch = env_flag("ALGAN_DEVICE_DISPATCH", False)

# Maximum number of ray bounces''')

mem = 'algan/utils/memory_utils.py'
edit(mem, '        self.current_pointer = 0\n        self.max_pointer = 0',
     '        self._region_tracker = None\n        self.current_pointer = 0\n        self.max_pointer = 0')
edit(mem, '    def get_pointers(self):\n', '''    @property
    def current_pointer(self):
        return self._current_pointer

    @current_pointer.setter
    def current_pointer(self, value):
        tracker = getattr(self, "_region_tracker", None)
        if tracker is not None and value < self._current_pointer:
            tracker.rewind(forward=value)
        self._current_pointer = value

    @property
    def current_reverse_pointer(self):
        return self._current_reverse_pointer

    @current_reverse_pointer.setter
    def current_reverse_pointer(self, value):
        tracker = getattr(self, "_region_tracker", None)
        if tracker is not None and value > self._current_reverse_pointer:
            tracker.rewind(reverse=value)
        self._current_reverse_pointer = value

    def get_pointers(self):
''')
edit(mem, '''        pointers = [*pointers]
        self.current_pointer = pointers[0]
        self.current_reverse_pointer = pointers[1]''', '''        pointers = [*pointers]
        if self._region_tracker is not None:
            self._region_tracker.rewind(
                forward=pointers[0] if pointers[0] < self.current_pointer else None,
                reverse=pointers[1] if pointers[1] > self.current_reverse_pointer else None,
            )
        self._current_pointer, self._current_reverse_pointer = pointers''')
edit(mem, '''            if scalar:
                x = x.view(())
            return x''', '''            if scalar:
                x = x.view(())
            if self._region_tracker is not None or SETTINGS.raytracing.device_dispatch:
                if self._region_tracker is None:
                    from algan.rendering.arena_regions import ArenaRegionTracker

                    self._region_tracker = ArenaRegionTracker(self)
                self._region_tracker.register(x, bool(reverse))
            return x''')

arena = 'algan/rendering/raytracing/arena_args_taichi.py'
edit(arena, '''    def __new__(cls, buf, base, shape):
        return super().__new__(cls, (buf, base, tuple(shape)))''', '''    def __new__(cls, buf, base, shape, *, hoist=False):
        shape = tuple(shape)
        strides = None
        if hoist:
            # Materialize layout values inside the parallel thread prologue,
            # not at each access and not as invariant mutable payload loads.
            base = _ti_impl.expr_init(base)
            shape = tuple(_ti_impl.expr_init(d) for d in shape)
            strides = [1] * len(shape)
            for d in range(len(shape) - 2, -1, -1):
                strides[d] = _ti_impl.expr_init(strides[d + 1] * shape[d + 1])
            strides = tuple(strides)
        return super().__new__(cls, (buf, base, shape, strides))''')
edit(arena, '''        shape = self.shape
        flat = idx[0]
        for d in range(1, len(idx)):
            flat = flat * shape[d] + idx[d]''', '''        shape = self.shape
        strides = tuple.__getitem__(self, 3)
        flat = idx[0]
        if strides is not None:
            flat = idx[0] * strides[0]
            for d in range(1, len(idx)):
                flat = flat + idx[d] * strides[d]
        else:
            for d in range(1, len(idx)):
                flat = flat * shape[d] + idx[d]''')
edit(arena, '    _table_cache.clear()\n', '''    _table_cache.clear()
    from algan.rendering.arena_region_args import clear_region_pack_cache

    clear_region_pack_cache()
''')
edit(arena, 'def arena_packed(module_name, kernel_attr, call_params, spec):',
     'def arena_packed(module_name, kernel_attr, call_params, spec, *, region_access=None):')
edit(arena, '''        packed = pack(spec, [args[i] for i in bound_idx])
        kernel = getattr(sys.modules[module_name], kernel_attr)''', '''        if region_access is not None and args[position["counted_dispatch"]]:
            from algan.rendering.arena_region_args import (
                checked_launch_bindings,
                pack_regions,
            )
            from algan.rendering.device_dispatch import retain_dispatch_regions

            views = checked_launch_bindings(call_params, args, region_access)
            packed = pack_regions(spec, [args[i] for i in bound_idx])
            # Keep leases/native views through the enclosing plan's fence,
            # not merely through this asynchronous wrapper's return.
            retain_dispatch_regions(views, packed)
        else:
            packed = pack(spec, [args[i] for i in bound_idx])
        kernel = getattr(sys.modules[module_name], kernel_attr)''')

p = ROOT / 'algan/rendering/raytracing/raster_taichi.py'
s = p.read_text()
start = s.index('def raster_shadow_trace_arena(')
head, body = s[:start], s[start:]
old = '''        adaptive_taps: ti.template(),
        arena_f32: ti.types.ndarray(),'''
assert body.count(old) == 1
body = body.replace(old, '''        adaptive_taps: ti.template(),
        dispatch_header: ti.types.ndarray(),
        counted_dispatch: ti.template(),
        arena_f32: ti.types.ndarray(),''')
b0 = body.index('    t_node_miss = ti.static(ArenaView(')
b1 = body.index('    # One thread per (event, light)', b0)
prologue = body[b0:b1]
assert prologue.count('ArenaView(') == 21
prologue = re.sub(r'\)\)\)(?=\n)', '), hoist=counted_dispatch))', prologue)
assert prologue.count('hoist=counted_dispatch') == 21
body = body[:b0] + body[b1:]
anchor = '''        li = idx - e * num_lights
        f = event_frame[e]'''
assert body.count(anchor) == 1
body = body.replace(anchor, '''        li = idx - e * num_lights
        if ti.static(counted_dispatch):
            # Guard before any payload access or visibility write.
            if dispatch_header[2] != 0 or dispatch_header[3] != 0 \\
                    or e >= dispatch_header[0]:
                continue
''' + ''.join('    ' + line if line.strip() else line for line in prologue.splitlines(True)) + '''        f = event_frame[e]''')
old = '''    "shadow_term", "adaptive_taps",
)'''
assert body.count(old) == 1
body = body.replace(old, '''    "shadow_term", "adaptive_taps", "dispatch_header", "counted_dispatch",
)''')
old = '    _RASTER_SHADOW_TRACE_PARAMS, _RASTER_SHADOW_TRACE_ARENA)'
assert body.count(old) == 1
body = body.replace(old, '''    _RASTER_SHADOW_TRACE_PARAMS, _RASTER_SHADOW_TRACE_ARENA,
    region_access={name: "write" if name == "shadow_vis" else "read"
                   for name in _RASTER_SHADOW_TRACE_PARAMS})
_raster_shadow_trace_launch.public_call_params = _RASTER_SHADOW_TRACE_PARAMS[:-2]''')
old = '    return _raster_shadow_trace_launch(*args)'
assert body.count(old) == 1
body = body.replace(old, '''    from algan.rendering.device_dispatch import DeviceCount

    if len(args) != len(_RASTER_SHADOW_TRACE_PARAMS) - 2:
        raise ValueError("Invalid public raster_shadow_trace argument count")
    if isinstance(args[0], DeviceCount):
        extent = args[0]
        extent.validate()
        extent.launch_capacity(int(args[30]))  # num_lights, checked host integer
        return _raster_shadow_trace_launch(
            extent.capacity, *args[1:], extent.header.tensor, 1)
    # The template-off variant never reads this event_frame dummy header.
    return _raster_shadow_trace_launch(*args, args[4], 0)''')
p.write_text(head + body)

p = ROOT / 'algan/rendering/raytracing/raster_pipeline.py'
s = p.read_text()
a = s.index('        acc_idx = sheet_accept[:num_slice_sheets].nonzero(as_tuple=True)[0]')
b = s.index('        sheet_resolve_shade(\n', a)
old = s[a:b]
trace_call = old[old.index('            raster_shadow_trace(\n'):]
assert trace_call.rstrip().endswith(')')
new = '''        from algan.rendering.taichi_runtime import taichi_launch_is_local

        dispatch = None
        if rt_settings.device_dispatch and taichi_launch_is_local(covered_idx.device):
            from algan.rendering.raytracing.shadow_dispatch import (
                prepare_primary_shadow_dispatch,
            )

            dispatch = prepare_primary_shadow_dispatch(
                memory, n=num_slice_sheets, accepted=sheet_accept,
                source=coverage["sheet_ref"][s_start:s_end],
                pos=event_pos, snrm=event_snrm, fnrm=event_fnrm,
                frame=event_frame, mask=event_msk, dp=event_dp, toff=event_toff,
                reverse=sheet_event_id, footprint=sec_aa > 1, terminator=term_on,
            )
            num_events = dispatch.extent
            event_capacity = dispatch.extent.capacity
            ev_pos = dispatch.outputs["pos"]
            ev_snrm = dispatch.outputs["snrm"]
            ev_fnrm = dispatch.outputs["fnrm"]
            ev_frame = dispatch.outputs["frame"]
            ev_msk = dispatch.outputs["mask"]
            ev_dp = dispatch.outputs["dp"]
            ev_toff = dispatch.outputs["toff"]
            ordered_sources = dispatch.outputs["source"]
        else:
            acc_idx = sheet_accept[:num_slice_sheets].nonzero(as_tuple=True)[0]
            num_events = int(acc_idx.numel())
            event_capacity = num_events
            ordered_sources = None
            if num_events:
                acc_idx, ordered_sources = _order_primary_shadow_events(
                    acc_idx, coverage["sheet_ref"][s_start:s_end]
                )
                sheet_event_id[:num_slice_sheets].scatter_(
                    0, acc_idx,
                    torch.arange(num_events, dtype=torch.int32, device=acc_idx.device),
                )
                ev_pos = event_pos.index_select(0, acc_idx)
                ev_snrm = event_snrm.index_select(0, acc_idx)
                ev_fnrm = event_fnrm.index_select(0, acc_idx)
                ev_frame = event_frame.index_select(0, acc_idx)
                ev_msk = event_msk.index_select(0, acc_idx)
                ev_dp = event_dp.index_select(0, acc_idx) if sec_aa > 1 else event_dp
                ev_toff = event_toff.index_select(0, acc_idx) if term_on else event_toff
        shadow_vis = _arena_tensor(
            memory, (max(1, event_capacity), max(1, int(num_lights)), 3),
            torch.float32, 1.0,
        )
        if dispatch is not None or num_events:
            identity_on = bool(rt_settings.shadow_identity_reject)
            if identity_on:
                ev_src_prim = (
                    ordered_sources if ordered_sources is not None
                    else coverage["sheet_ref"][s_start:s_end]
                    .index_select(0, acc_idx).to(torch.int32)
                )
                eps_self, eps_near = _shadow_identity_epsilons(merged)
            else:
                ev_src_prim = dummy_i
                eps_self, eps_near = float(min_hit_distance), 0.0
            from algan.rendering.raytracing.refit_bvh import RefitBVH

            def trace_events():
'''
new += ''.join('    ' + line if line.strip() else line for line in trace_call.splitlines(True))
new += '''
            if dispatch is not None:
                # Consolidated status, before mode 2 commits final shading.
                dispatch.run(trace_events, shadow_vis)
            else:
                trace_events()
'''
p.write_text(s[:a] + new + s[b:])

tests = 'tests/unit_tests/test_arena_args.py'
edit(tests, '    for stmt in fn.body:', '    for stmt in ast.walk(fn):')
edit(tests, '        expected[name] = len(_launcher(module, name).call_params)',
'''        launcher = _launcher(module, name)
        expected[name] = len(getattr(launcher, "public_call_params", launcher.call_params))''')
print('Both Q1 and Q2 applied. No testing has run before this point.')
