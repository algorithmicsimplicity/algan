# OX_AREA_SHADOW_FIX2 — the wavefront fan did not compile; fixed, swept, compile-tested

Fixes the one defect in the previous round (`OX_AREA_SHADOW_FIX.md`), sweeps the
whole diff for its siblings, and adds the regression test that class of defect
needs. Branch `claude/area-shadow-banding-fix-i411ly`. **No render and no
pytest suite beyond the file named in §5 was run; nothing is committed or
pushed.**

In one sentence: `wavefront_shade`'s soft-shadow fan assigned `off` inside both
arms of a new if/else and read it after — which is a `TaichiNameError` at
kernel-compile time, because Taichi scopes a local to the block it is FIRST
assigned in, not to the function — and the fix hoists one initialiser above
the gate, exactly where the raster fan already had its own.

## 1. The defect, measured

Reproduced standalone on this environment (Taichi 1.7.4, arch=x64) before
touching anything:

```python
@ti.kernel
def k(flag: ti.i32) -> ti.f32:
    acc = 0.0
    for i in range(4):
        if i >= 0:
            if flag > 0:
                off = 1.0
            else:
                off = 2.0
            acc += off          # line 15
    return acc
```

```
TaichiNameError:
File ".../taichi_scope_repro.py", line 15, in k:
            acc += off
                   ^^^
Name "off" is not defined
```

Assigning in every arm is not enough; the arms are separate scopes. This is
the invariant of §2 of the brief, and it is what separates the two fans: the
raster fan already carried `off = ti.math.vec3(0.0, 0.0, 0.0)` before its
radius gate (`raster_taichi.py:2954`), so its branch assignments landed in an
existing local; the wavefront fan had no such line, because there `off` had
only ever been assigned once, unconditionally.

## 2. The fix

One hunk, `algan/rendering/raytracing/wavefront_kernels_taichi.py`, sample
loop of the inline fan (now line 2853):

```python
for s in range(ns):
    wis = wi
    ldn = ldist
    ok = 1
    # Declared here, not in the arms below: ... (comment)
    off = ti.math.vec3(0.0, 0.0, 0.0)
    if radius > 0.0:
        if (hu > 0.0) or (hv > 0.0):
            ...
            off = b1 * (hu * ru) + b2 * (hv * rv)
        else:
            ...
            off = (ti.cos(ang) * b1 + ti.sin(ang) * b2) * rr
        if ltype == _LT_DIRECTIONAL:
            wis = (wi + off).normalized()      # now in scope
        else:
            tls = lp + off - spos              # now in scope
```

Why this keeps every path byte-identical rather than merely correct:

- When `radius > 0.0`, exactly one arm runs and each assigns `off` fully
  before any read of it; the initialiser is dead there. No float expression
  gained, lost, or reordered an operation.
- When `radius <= 0.0`, every read of `off` sits inside `if radius > 0.0:`,
  so the initialiser's value is never observed.
- It mirrors the raster fan's pre-existing pattern line for line, so the two
  fans stay shaped alike.
- The reads were NOT duplicated into the arms (explicitly ruled out by the
  brief); there is still exactly one read site per consumer.

## 3. The sibling sweep

Every local the diff touches or that the touched code reaches, both fans,
declaration site first. Line numbers are the current working tree;
"unconditional" means not inside any `if` at that scope.

### `wavefront_shade` inline fan (`wavefront_kernels_taichi.py`)

| Name | First declared | Conditionally reassigned | Read | Verdict |
| --- | --- | --- | --- | --- |
| `ltype` | :2748 unconditional | :2753 under `shape[2] > 3` | gates :2771/:2781/:2880, `_light_zero_radiance` :2796 | OK |
| `radius` | :2749 unconditional | :2755 under `shape[2] > 3` | gate :2808, disk arm :2873 | OK |
| `hu` | :2750 unconditional (this diff) | :2765 under `ltype == _LT_AREA_SAMPLE` | gate :2810/:2855, rect arm :2869 | OK |
| `hv` | :2751 unconditional (this diff) | :2766 same guard | gate :2810/:2855, rect arm :2870 | OK |
| `b1` | :2806 unconditional vec3(0), **pre-existing** | rect :2820 / disk :2835 | rect arm :2869, disk arm :2877 | OK |
| `b2` | :2807 unconditional vec3(0), **pre-existing** | rect :2824 / disk :2836 | rect arm :2870, disk arm :2878 | OK |
| `aref` | only inside disk `else:` (:2830, :2833) | inner `ti.abs(wi[0])` check :2833 | :2835, inside the same block | OK |
| `u` | rect arm only :2861 | — | :2865 same block | OK |
| `v` | rect arm only :2863 | — | :2867 same block | OK |
| `ru` | rect arm only :2865 | — | :2869 same block | OK |
| `rv` | rect arm only :2867 | — | :2870 same block | OK |
| `ang` | disk arm only :2872 | — | :2877 same block | OK |
| `rr` | disk arm only :2873 | — | :2878 same block | OK |
| `ns` | :2805 unconditional | :2809 under radius gate | `range(ns)` :2839, :2876 | OK |
| `wis` | loop top :2840 unconditional | :2881/:2887 gated | horizon culls :2903ff, `_shadow_occluded` | OK |
| `ldn` | loop top :2841 unconditional | finite-distance arm :2885 | `_shadow_occluded` call | OK |
| `ok` | loop top :2842 unconditional | degenerate-ray else :2889 | validity gate | OK |
| `off` | **was** born in the arms (:2869/:2877, this diff's branch split) | — | :2881 and :2884, OUTSIDE the arms | **THE DEFECT** — now declared unconditionally at :2853 |
| `tls` | finite-distance arm only :2884 | — | :2885–:2886 same block | OK |
| `occ_sum`, `n_valid` | :2837–:2838 unconditional | mutated in loop | after loop | OK |
| `horizon_ok` | :2903 unconditional | :2908 under `shadow_term` static gate | :2910 | OK |

### `raster_shadow_trace` fan (`raster_taichi.py`)

| Name | First declared | Conditionally reassigned | Read | Verdict |
| --- | --- | --- | --- | --- |
| `ltype` | :2872 unconditional | :2877 under `shape[2] > 3` | later gates, `_light_zero_radiance` | OK |
| `radius` | :2873 unconditional | :2879 under `shape[2] > 11` | gates :2917/:2955, disk arm :2967 | OK |
| `hu` | :2874 unconditional (this diff) | :2886 under area guard | gates :2919/:2956, rect arm :2964 | OK |
| `hv` | :2875 unconditional (this diff) | :2887 same guard | gates :2919/:2956, rect arm :2964 | OK |
| `b1` | :2915 unconditional vec3(0), **pre-existing** | rect :2925 / disk :2935 | rect :2964, disk :2970 | OK |
| `b2` | :2916 unconditional vec3(0), **pre-existing** | rect :2928 / disk :2936 | rect :2964, disk :2970 | OK |
| `aref` | disk arm only (:2932, :2934) | — | :2935 same block | OK |
| `u`/`v` | rect arm only :2960–:2961 | — | :2962–:2963 same block | OK |
| `ru`/`rv` | rect arm only :2962–:2963 | — | :2964 same block | OK |
| `ang`/`rr` | disk arm only :2966–:2967 | — | :2970 same block | OK |
| `ns` | :2914 unconditional | :2918 radius gate, :2941 static `sec_aa` | `range(ns)` :2945, :2968 | OK |
| `wis` | loop top :2946 unconditional | :2972/:2981 gated | horizon culls, `_shadow_occluded` | OK |
| `ldn` | loop top :2947 unconditional | point-arm :2979 | `_shadow_occluded` | OK |
| `ok` | loop top :2948 unconditional | :2953 static sec_aa mask, :2983 | validity gate | OK |
| `sorg` | loop top :2949 unconditional, **pre-existing** | :2951 static `sec_aa` | :2978 outside the static block | OK — declared before the block, so the static reassignment lands in scope |
| `off` | :2954 unconditional, **pre-existing init** | rect :2964 / disk :2970 (this diff) | :2972/:2978 | OK — why this fan never broke |
| `tls`, `horizon_ok`, `occ_sum`, `n_valid` | declared before their conditional reads/mutations | as noted | after | OK |

### Non-kernel hunks of the diff

`lights.py` `build_aux`'s new locals (`k`, `right`, `hu`, `hv`) live and die
inside one `if rt_settings.AREA_LIGHT_SOFT_SHADOWS:` block — ordinary Python
function scoping, nothing reads them after. `environment.py`,
`raytracing_settings.py`, `scene_builder.py`, `settings.py` hunks are module
constants, dict entries and docstrings; no locals at all.

### Did my change strand any local that was safe only because the branch did not exist?

Yes — exactly one, and it was `off`: safe as a single unconditional assignment
inside `if radius > 0.0:`, broken by splitting that assignment across two new
arms. Fixed. Beyond it:

- `b1`/`b2` in the wavefront fan survive only because pre-existing code
  happened to declare them unconditionally (:2806–:2807). Had the original
  author not done that, the new rect arm would have broken them identically.
  They are fine as they stand; nothing to change.
- `aref` moved INTO the new `else:` arm; its reads moved with it. Fine.
- No other local in either fan has its first assignment inside a branch this
  diff added.
- Pre-existing, out of scope but adjacent (already recorded in the audit): the
  wavefront fan guards the column-11 `radius` load with `shape[2] > 3` while
  raster uses `> 11`; harmless while extended packing is always 16 wide, and
  untouched here.

## 4. The regression test

Added to `tests/unit_tests/test_area_light_soft_shadow.py`
(`test_soft_shadow_fans_compile_and_render_one_frame`, parametrized over two
routes), plus a shared `_render_one_area_shadow_frame` helper:

- Scene: one `RectAreaLight` (samples=4 → K rows, cell extents packed),
  shadows on via `SETTINGS.raytracing.set(shadows=True, ...)`, over a lambert
  ground plane and one blocker square — rendered by `save_frame` at 32×32
  (`SMOKE_TEST`).
- Route forcing, confirmed from source not assumed:
  `analytic_raster_route_active` (`tracer.py:567-581`) returns False when
  `not rt_settings.ANALYTIC_AA`, falling the batch back to the classic
  wavefront tracer — so `analytic_aa=False` forces the wavefront fan, the
  default leaves the sheet fan.
- A monkeypatch spy wraps `tracer.analytic_raster_route_active` (called as a
  module global at render time, `tracer.py:1200`) and asserts every recorded
  decision matches the arm — so neither arm can silently stop exercising its
  kernel while staying green.
- Assertion: the frame renders and the output file exists. Pixel-free, cheap.
- Unmarked (not `fast`), per CLAUDE.md; its docstring says explicitly that it
  exists to compile kernels so nobody deletes it as a redundant render test.

Honest note on process: the test's first draft forgot to apply its own
parametrized settings, and its failure (route reported `[True]`) is what the
route-spy assertion exists to catch — including against its own author.

### Before / after evidence

BEFORE the fix, on the broken tree (both arms run):

```
tests/unit_tests/test_area_light_soft_shadow.py::test_soft_shadow_fans_compile_and_render_one_frame[wavefront] FAILED
...
algan/rendering/raytracing/tracer.py:3193: in run_tile
    wavefront_shade(
...
E           taichi.lang.exception.TaichiNameError:
E           File "/home/user/algan/algan/rendering/raytracing/wavefront_kernels_taichi.py", line 2870, in wavefront_shade:
E                                                               wis = (wi + off) \
E                                                                           ^^^
E           Name "off" is not defined
...
1 failed, 1 passed in 25.43s
```

The `[wavefront]` arm failed with exactly the predicted error, through exactly
the predicted route (`raytrace_render_wavefront → _run_wavefront_tiles →
run_tile → wavefront_shade`); the `[sheet]` arm passed even then — the
differential the brief asked for, and the proof the sheet route really was
exercised separately.

AFTER the fix:

```
$ .venv/bin/python -m pytest -q tests/unit_tests/test_area_light_soft_shadow.py
..........                                                                [100%]
10 passed in 16.46s
```

(8 pre-existing host-side tests + the 2 new arms.) A repeat verbose run shows
both arms passing and writing their PNGs. Renders ran in-process: no
`$ALGAN_HOME/daemon.json` existed and no daemon process was running.

## 5. Verification commands and outcomes

- `ruff check --no-fix algan/ tests/` → **25 errors, the established
  pre-existing set**, none in files this round changed except
  `wavefront_kernels_taichi.py`, whose five findings are the documented ones
  with F841 shifted :3569→:3580 by the fix's +11 lines. Zero new findings;
  `test_area_light_soft_shadow.py` clean.
- `ruff format --check tests/unit_tests/test_area_light_soft_shadow.py` →
  "1 file already formatted". (The kernel files remain formatter-excluded by
  config; no other non-`_taichi` file was touched this round.)
- `.venv/bin/python -m pytest -q tests/unit_tests/test_area_light_soft_shadow.py`
  → **10 passed** (see §4). Kernel compiles were served partly warm from the
  offline cache, hence the modest wall time; the brief's several-minutes
  allowance was not needed on this machine.

## What I did not verify

- **Disk-path byte-identity is argued structurally, not A/B-measured**: both
  arms assign `off` fully before any read, so the added initialiser is dead on
  the disk path. There is no pre-defect binary to compare against — the
  defective kernel never compiled anywhere.
- **CUDA**: no GPU here. The scoping rule is AST-level so the defect and the
  fix should behave identically on CUDA, but that is reasoning, not
  measurement; a CUDA machine still owes this branch its usual scrutiny.
- **Template specialisations beyond the ones the test compiles**: SMOKE_TEST
  renders at anti_alias_level 1, so the `sec_aa == 1` specialisation of each
  fan is what is compiled. The `sec_aa > 1` sub-pixel variants contain the
  same AST shape and would have failed the same way, but this suite does not
  literally compile them.
- **The golden-angle disk arm executes nowhere in these tests** (the scene's
  only soft emitter is a rect). It compiles in both fans; its runtime
  behaviour remains covered (or not) by the suites this brief told me not to
  run, e.g. full-renders act 3's directional `shadow_angle`.
- Nothing else re-run: no fast/full-render suite, no baseline comparison, no
  benchmark. FIX.md's statement of what those will show stands unchanged.
- Wall-clock numbers above are this container's, cold-cache state included;
  they are not transferable.
