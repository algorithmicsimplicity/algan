# Validating the runtime render device on a CUDA machine

Everything here was written on a **CPU-only 4-vCPU cloud box**. That machine can
prove the mechanism works and that nothing regressed on the CPU path; it cannot
see the case the change exists for, because there `cpu` and `cuda` are the same
device and Taichi's arch never moves. This document is the list of what is
therefore still unproven, and exactly how to prove it.

It is self-contained: every command and probe below can be run without reading
the diff first. §1 says what changed and why, §2 is a 10-minute smoke pass, §3
is the full checklist, §4 lists what a failure would look like, §5 is the
numbers to record.

---

## 1. What changed

Before: `ALGAN_RENDER_DEVICE` was read at `import algan`, and four kernel
modules called `init_taichi()` at module scope, so Taichi's arch was fixed
before any user code ran. The device could only be chosen through the
environment, and `SETTINGS.computing.set(render_device=...)` raised.

After:

| | |
| --- | --- |
| `SETTINGS.computing.render_device` | the runtime source of truth, seeded from `ALGAN_RENDER_DEVICE`, settable between renders |
| `algan.settings._startup.render_device()` | how engine code reads it -- **never** bind it at import |
| `taichi_runtime.ensure_taichi_for_render()` | called once per render job; re-runs `ti.init` **only** when the live arch no longer matches the device |
| `taichi_runtime.install_render_arch_guard()` | wraps `Kernel.__call__` so a kernel launched outside a render brings Taichi up first |
| `timeline.wide_attribute_device_pin()` | refuses a device change once a texture has been placed on the render device |
| `taichi_runtime.render_is_active()` | refuses a device change while a render job is running |
| `daemon._adopt_render_device()` | a warm daemon now adopts a client's differing `ALGAN_RENDER_DEVICE` instead of refusing the run |

Two properties this leans on, both worth re-checking on CUDA:

* **Nothing needs a live Taichi program while algan imports.** `@ti.kernel`
  only registers a kernel; materialization at first launch is what needs a
  program.
* **A re-init is safe.** `ti.init` calls `impl.reset()`, which clears
  `compiled_kernels` on every registered kernel. Reading a `ti.field` created
  before a re-init segfaults with no Python exception -- Algan holds no
  `ti.field` or `ti.Ndarray` anywhere, only torch tensors passed as
  `ti.types.ndarray()`, which is why this is safe and why it stops being safe
  the day that changes.

### Measured on CPU (what CUDA should be compared against)

| | |
| --- | --- |
| `import algan` with `ti.init` stubbed out | completes, 124 kernels registered, `prog is None` |
| `ti.init` itself | 0.17-0.19 s |
| a re-init's cost to the *next* render | +4.0 s on a single-square scene, warm offline cache (22 compiled specializations dropped to 0) |
| arch guard, per kernel launch | 0.31 us, against ~72 us for the launch |
| output across a re-init | 4 frames, sha256-identical |

---

## 2. Ten-minute smoke pass

Run these five in order on the CUDA machine. If all five pass, the mechanism
works there; §3 is what makes it trustworthy.

```bash
# 0. From the repo root, on the branch under test.
cd /path/to/algan

# 1. Importing algan must not create a Taichi program -- or a CUDA context.
ALGAN_USE_DAEMON=0 uv run python - <<'PY'
import taichi as ti, torch, algan
print("taichi program:", ti.lang.impl.get_runtime().prog)     # expect: None
print("kernels registered:", len(ti.lang.impl.get_runtime().kernels))  # expect: >0
print("render device:", algan.SETTINGS.computing.render_device)        # expect: cuda
PY

# 2. The arch follows the setting, and a render works on each device.
ALGAN_USE_DAEMON=0 uv run python - <<'PY'
import taichi as ti
from algan import *
from algan.settings import SETTINGS
from algan.rendering.taichi_runtime import taichi_arch_is_cpu

Square(color=RED).spawn()
Scene.save_frame("cuda_arm")
print("after CUDA render, arch_is_cpu:", taichi_arch_is_cpu())   # expect: False

SETTINGS.computing.set(render_device="cpu")
Scene.save_frame("cpu_arm")
print("after CPU render, arch_is_cpu:", taichi_arch_is_cpu())    # expect: True

SETTINGS.computing.set(render_device="cuda")
Scene.save_frame("cuda_arm_again")
print("back on CUDA, arch_is_cpu:", taichi_arch_is_cpu())        # expect: False
PY

# 3. A texture freezes the device (this branch is unreachable on a CPU box).
ALGAN_USE_DAEMON=0 uv run python - <<'PY'
from algan import *
from algan.settings import SETTINGS
from algan.animation_timeline.timeline import (
    WIDE_ATTR_MIN_CHANNELS, AttributeTimeline, wide_attribute_device_pin,
)

AttributeTimeline(WIDE_ATTR_MIN_CHANNELS, buffer_size=2)
print("pin:", wide_attribute_device_pin())          # expect: cuda, NOT None
try:
    SETTINGS.computing.set(render_device="cpu")
    print("FAIL: the change was allowed")
except AlganConfigurationError as exc:
    print("refused as intended:", exc)
PY

# 4. The suites.
uv run -m pytest -q --fast          # includes one pixel-compared CUDA render
uv run -m pytest -q                 # everything

# 5. The full render suites, which only mean something on their own machine.
uv run -m pytest -q tests/full_renders
```

---

## 3. The checklist

### 3.1 No CUDA context is created for a CPU render *(new capability)*

The point of deferring `ti.init`: asking for the CPU on a CUDA box should no
longer spin up a CUDA Taichi program. Watch the driver, not torch's allocator --
Taichi allocates through its own.

```bash
ALGAN_USE_DAEMON=0 ALGAN_RENDER_DEVICE=cpu uv run python - <<'PY'
import torch
free_before, total = torch.cuda.mem_get_info()
from algan import *
free_after_import, _ = torch.cuda.mem_get_info()
Square(color=RED).spawn()
Scene.save_frame("cpu_on_a_cuda_box")
free_after_render, _ = torch.cuda.mem_get_info()
mb = lambda a, b: (a - b) / 1e6
print(f"VRAM taken by the import : {mb(free_before, free_after_import):.1f} MB")
print(f"VRAM taken by the render : {mb(free_before, free_after_render):.1f} MB")
PY
```

**Expect** both close to zero. A large second number means something still put
the render on the GPU; a large first means something initialized Taichi (or a
CUDA context) during the import.

> Caveat: `_cuda_is_usable()` in `algan/settings/_startup.py` allocates a
> one-element CUDA tensor at import to probe the device, so a small torch-side
> context cost is expected and is not what this measures. Compare the two
> numbers against each other rather than against zero.

### 3.2 The arch actually changes, and kernels are dropped with it

```bash
ALGAN_USE_DAEMON=0 uv run python - <<'PY'
import taichi as ti
from algan import *
from algan.settings import SETTINGS
from algan.rendering import taichi_runtime as tr

Square(color=RED).spawn()
Scene.save_frame("arm_a")
program = ti.lang.impl.get_runtime().prog
compiled = sum(len(k.compiled_kernels) for k in ti.lang.impl.get_runtime().kernels)
print("arch:", program.config().arch, "| compiled specializations:", compiled)

SETTINGS.computing.set(render_device="cpu")
print("re-initialized:", tr.ensure_taichi_for_render())         # expect: True
after = ti.lang.impl.get_runtime().prog
print("new program:", after is not program)                      # expect: True
print("arch now:", after.config().arch)                          # expect: x64/arm64
print("compiled now:", sum(len(k.compiled_kernels)
                           for k in ti.lang.impl.get_runtime().kernels))  # expect: 0

# And it is a no-op when nothing moved.
print("second call re-initialized:", tr.ensure_taichi_for_render())  # expect: False
PY
```

### 3.3 A device switch does not change what a render produces

This is the one that decides whether the change is shippable, and it is the one
a CPU box cannot ask. Render the same scene two ways and compare bytes:

* **arm A** -- one process, `ALGAN_RENDER_DEVICE=cuda`, render.
* **arm B** -- one process, started on the CPU, switched to CUDA with
  `SETTINGS.computing.set(render_device="cuda")` *before any Mob is created*,
  render.

```bash
cat > /tmp/switch_arm.py <<'PY'
import hashlib, os, sys
from algan import *
from algan.settings import SETTINGS

if os.environ.get("SWITCH") == "1":
    SETTINGS.computing.set(render_device="cuda")   # arrive by the new route
print("rendering on", SETTINGS.computing.render_device, flush=True)

Sphere(radius=1.0, color=BLUE).spawn()
Square(color=RED).spawn().move(RIGHT * 2)
result = Scene.save_video(sys.argv[1], SMOKE_TEST)
print(sys.argv[1], hashlib.sha256(open(result.output_path, "rb").read()).hexdigest())
PY

ALGAN_USE_DAEMON=0 ALGAN_RENDER_DEVICE=cuda uv run python /tmp/switch_arm.py direct
ALGAN_USE_DAEMON=0 ALGAN_RENDER_DEVICE=cpu SWITCH=1 uv run python /tmp/switch_arm.py switched
```

**Expect identical digests.** They are the same device rendering the same scene;
only the route to selecting it differs. A difference means the switch left some
state (a cache, a projection budget, a wide-attribute placement) disagreeing
with the arch, and §4 is where to look.

If the digests differ, re-run with `SETTINGS.computing.available_memory_override`
pinned in both arms before concluding: free VRAM is not reproducible, and the
render loop sizes its frame windows from it, which on its own moves silhouette
pixels (see `ComputingSettings`).

### 3.4 The prep-kernel gates still answer correctly after a switch

`taichi_arch_is_cpu()` and `taichi_launch_is_local()` decide whether a kernel
would stage its arguments through VRAM. They read the **live** arch, so they are
correct only if `ensure_taichi_for_render()` ran first.

```bash
ALGAN_USE_DAEMON=0 uv run python - <<'PY'
import torch
from algan import *
from algan.settings import SETTINGS
from algan.rendering.taichi_runtime import (
    cpu_prep_kernel_enabled, taichi_arch_is_cpu, taichi_launch_is_local,
)

Square(color=RED).spawn()
for device in ("cuda", "cpu"):
    SETTINGS.computing.set(render_device=device)
    Scene.save_frame(f"gate_{device}")
    print(device,
          "| arch_is_cpu:", taichi_arch_is_cpu(),
          "| cpunormals:", cpu_prep_kernel_enabled("cpunormals"),
          "| host tensor local:", taichi_launch_is_local(torch.device("cpu")),
          "| cuda tensor local:", taichi_launch_is_local(torch.device("cuda")))
PY
```

**Expect** on `cuda`: `arch_is_cpu False`, `cpunormals False`, host tensor
**not** local, cuda tensor local. On `cpu`: the exact mirror. A row where
`cpunormals` is True while `arch_is_cpu` is False is the staging bug the whole
gate exists to prevent.

### 3.5 The wide-attribute pin, on the device where it arms

Only a `cuda`/`mps` render device places a wide attribute on the render device,
so `wide_attribute_device_pin()` is always `None` on a CPU box and the guard is
untested there. Smoke item 3 covers the synthetic case; this covers the real
one, with an actual textured Mob:

```bash
ALGAN_USE_DAEMON=0 uv run python - <<'PY'
from algan import *
from algan.settings import SETTINGS
from algan.animation_timeline.timeline import wide_attribute_device_pin

print("pin before any Mob:", wide_attribute_device_pin())   # expect: None
SETTINGS.computing.set(render_device="cpu")                 # expect: allowed
SETTINGS.computing.set(render_device="cuda")

surface = Sphere(radius=1.0).spawn()
surface.color_texture = ...   # substitute a real texture assignment
print("pin after a texture:", wide_attribute_device_pin())  # expect: cuda
try:
    SETTINGS.computing.set(render_device="cpu")
    print("FAIL: allowed after a texture existed")
except AlganConfigurationError as exc:
    print("refused:", exc)

SceneManager.reset()
print("pin after reset:", wide_attribute_device_pin())      # expect: None
SETTINGS.computing.set(render_device="cpu")                 # expect: allowed again
PY
```

Also confirm the pin does **not** arm for ordinary attributes: a scene of plain
`Square`s must leave `wide_attribute_device_pin()` at `None`, or the guard will
freeze the device for every script rather than for textured ones.

### 3.6 A change is refused mid-render, on the thread that would corrupt

The one path here that could corrupt rather than merely be slow: batch prep
launches kernels from a worker thread, so a change *during* a job could have
that thread run `ti.init` -- dropping every compiled kernel -- while the render
thread is inside one. `tests/unit_tests/test_settings_api.py` covers the
refusal synthetically; this is the real arrangement, and it only has a second
thread to race with when there is real prep work to overlap.

```bash
ALGAN_USE_DAEMON=0 uv run python - <<'PY'
from algan import *
from algan.settings import SETTINGS

# A scene with enough frames that batch prep actually overlaps the render.
sphere = Sphere(radius=1.0, color=BLUE).spawn()
sphere.rotate(360, OUT)

def switch(mob, t):
    try:
        SETTINGS.computing.set(render_device="cpu")
        print("FAIL: a mid-render change was allowed")
    except AlganConfigurationError as exc:
        print("refused mid-render:", str(exc)[:60])
    return mob

sphere.add_updater(switch)
Scene.save_video("mid_render_switch", SMOKE_TEST)
print("device after the render:", SETTINGS.computing.render_device)  # expect: cuda
PY
```

**Expect** every attempt refused and the render to finish on CUDA. A crash, a
second `Starting on arch=` line, or a device that came out as `cpu` all mean the
job counter is not covering the whole render.

### 3.7 The daemon adopts a differing render device

The bonus. On a CPU box both values resolve to `cpu`, so the adoption is a
no-op there and only the refusal-was-lifted half is observable. On CUDA it is
a real switch.

```bash
# terminal 1 -- a daemon on the default device (CUDA)
env -u ALGAN_RENDER_DEVICE uv run python -m algan.daemon

# terminal 2 -- a client that wants the other one
cat > /tmp/dev_scene.py <<'PY'
from algan import *
from algan.settings import SETTINGS
print("SCRIPT render_device:", SETTINGS.computing.render_device)
Square(color=RED).spawn()
Scene.save_frame("daemon_device")
PY
ALGAN_RENDER_DEVICE=cpu uv run python /tmp/dev_scene.py
```

**Expect** the daemon log to show `run #1 (client)` and
`adopting this script's render device: cpu`, the script to print `cpu`, and
Taichi to report `Starting on arch=x64`. **Not** expected: a refusal naming
`ALGAN_RENDER_DEVICE`, or the script rendering on CUDA.

Then run a second client **without** `ALGAN_RENDER_DEVICE` and confirm it goes
back to CUDA -- the daemon's `SETTINGS.restore()` between runs is what makes one
script's device not leak into the next.

### 3.8 The suites

```bash
uv run -m pytest -q --fast
uv run -m pytest -q
uv run -m pytest -q tests/full_renders
```

`tests/fast` and `tests/full_renders` compare pixels against
`expected_outputs_cuda/`. **Nothing in this change should move a pixel** -- it
changes when the arch is selected, not what any kernel computes. A deviation is
a finding, not a re-baseline: report it rather than running
`ALGAN_UPDATE_*_BASELINE=1`.

Note that `tests/full_renders` baselines are per **machine**, not merely per
device (`pn_criterion_kernel` runs under Taichi's `fast_math`, and which
tessellation levels sit on the borderline depends on the CPU evaluating the
criterion). So establish the baseline behaviour **on the base branch first**,
on the same machine, and compare the two runs -- not the run against the
committed files.

---

## 4. What a failure looks like, and where to look

| symptom | most likely cause |
| --- | --- |
| `TaichiRuntimeError: Please call init() first` | a kernel launched through a path the arch guard does not wrap. `install_render_arch_guard()` must be installed **after** `taichi_fast_launch.apply()` (`algan/__init__.py`) -- the fast dispatcher goes straight to C++ on a plan hit and would bypass a guard installed under it. |
| Segfault (exit 139) with no traceback right after a device switch | something now holds a `ti.field` or `ti.Ndarray` across the re-init. Grep for both under `algan/`; the design assumes there are none. |
| A CPU render on a CUDA box is slow and holds VRAM | `ensure_taichi_for_render()` did not run, or `_arch_matches_render_device()` returned a false positive. `ti.gpu` is a *list* of backends, which is why that helper compares the live `prog.config().arch` rather than the value `_taichi_arch()` returned. |
| Pixels move between §3.3's two arms | a cache keyed on something other than the device, or a wide attribute placed before the switch. The cross-render caches (`_EDGE_CACHE`, `_SAMPLE_TENSOR_CACHE`, the five in `rendering/logical_pn.py`) are all device-keyed; a new one that is not would show up exactly here. |
| A second render after a switch is slow | expected once -- the re-init dropped the compiled kernels. A *third* render that is still slow means the arch is flipping every job. |
| A crash or a stray `ti.init` mid-render | the render job counter (`render_job_holding_the_arch`) does not cover the whole job, so a change slipped past `render_is_active()`. |
| Daemon refuses a client over `ALGAN_RENDER_DEVICE` | `STARTUP_ENV_ADOPTED` did not reach `describe_env_mismatch`, or the client is also differing on a variable that is genuinely not adoptable -- read the whole report, not the first line. |

---

## 5. Numbers to record

The CPU column is measured; fill in the CUDA one and keep both, because the
argument for "switch at the top of a script, not between renders" rests on the
gap between a re-init and a normal render.

| | CPU (measured) | CUDA |
| --- | --- | --- |
| `ti.init` alone | 0.19 s | |
| next render after a re-init, minus a steady-state render | +4.0 s (trivial scene, warm offline cache) | |
| steady-state render, same scene | 0.13 s | |
| compiled specializations dropped by the re-init | 22 -> 0 | |
| VRAM held after `import algan` with `ALGAN_RENDER_DEVICE=cpu` | n/a | |
| `tests/fast` | pass | |
| `tests/full_renders` deviation vs the same machine on the base branch | unchanged | |

A CUDA re-init will be much more expensive than the CPU's +4.0 s: the scenes
compile more kernel variants and the megakernels are the expensive ones. That
is expected and is the reason `ensure_taichi_for_render()` compares arches
instead of calling `ti.init` every job. Record the real figure so the docstring
in `taichi_runtime.ensure_taichi_for_render` can cite it instead of the CPU one.
