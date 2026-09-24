# Custom shaders for video authors

Use a shader when the user's requested surface appearance needs one. This
reference covers the public authoring interfaces, not renderer development.
Start with a material when its existing controls express the requested result;
use a custom function or stage when they do not.

## Choose the correct interface

| Interface | Evaluated where | Author supplies |
|---|---|---|
| `mob.set_material(material)` | Built-in in-kernel material path | Material class and parameter values |
| `mob.set_shader(function)` | Per vertex, using PyTorch | A function with nine fixed leading parameters |
| `mob.set_fragment_shader(stage_or_list)` | Per rendered hit, in the compiled shader pipeline | `FragmentStage` objects or recognized built-in lighting functions |

Install the chosen configuration before spawn. Animate parameters registered
on the Mob afterwards. A later `set_shader`, `set_material`, or
`set_fragment_shader` does not append a layer to the previous configuration.
To compose operations, pass one ordered fragment-stage list.

A plain Python/PyTorch function is not an arbitrary fragment shader; a GLSL or
WGSL string is not this API either. Do not patch the renderer, call `ti.init`, or
register internal pipeline IDs from a video script.

## PyTorch vertex-shader contract

The first **nine** parameters, in this order, are:

```python
memory, vertex_location, vertex_normal, albedo_color,
camera_location, light_origin, light_color,
light_intensity, ambient_light_intensity
```

Declare all nine even when unused. Parameters after them are exposed as
animatable Mob attributes. Give each custom parameter an explicit default.
Use distinctive names that do not collide with ordinary Mob attributes.

`albedo_color` is `[..., 4]`: **RGB plus glow**, not RGBA. The return value must
preserve the broadcast leading dimensions and four-channel layout. Work on
`[..., :3]` for RGB and preserve `[..., 3:4]` for glow unless the requested
operation deliberately changes it. Normal Mob opacity is handled separately.

Here is a complete vertex-stage function with an animatable phase. The modulation
is an arithmetic demonstration, not a proposed look for the user's scene:

```python
import torch

def vertex_modulation(
    memory, vertex_location, vertex_normal, albedo_color,
    camera_location, light_origin, light_color,
    light_intensity, ambient_light_intensity,
    modulation_frequency=4.0, modulation_phase=0.0,
):
    angle = (vertex_location[..., 0:1] * modulation_frequency
             + modulation_phase)
    weight = 0.5 + 0.5 * torch.cos(angle)
    rgb = albedo_color[..., :3] * weight
    glow = albedo_color[..., 3:4].expand_as(rgb[..., :1])
    return torch.cat((rgb, glow), dim=-1)
```

Install and animate it with:

```python
ball = Sphere()
ball.set_shader(vertex_modulation)
ball.spawn()
with Seq(runtime=2.0, easing=easings.identity):
    ball.modulation_phase = 2 * PI
```

There is no implicit shader time parameter. Make time-varying controls explicit
as animatable parameters, or drive those parameters with an updater. Trigonometric
phases are radians; Mob rotation arguments are degrees.

Use torch operations and preserve device/dtype. Inputs and parameters can have
frame and vertex batch dimensions; do not flatten away those dimensions, call
`.item()` on them, or manufacture CPU tensors in the callback. This example
ignores the light inputs; it is not a substitute for a physically based lighting
model. The built-in shader functions are useful contract references when writing
a lighting-aware vertex shader.

The custom vertex path interpolates vertex results across faces and has limited
lighting support: plain PointLight input and no received shadows. A scene with
other lights, an environment, or requested shadows can warn that those features
are dropped for that shader. Increasing mesh resolution does not restore missing
lighting features. When the requested effect needs per-fragment variation or
full in-kernel lighting, use a fragment pipeline instead.

Define one function and reuse it across objects. Generating an equivalent new
function per Mob can split otherwise compatible batches. See the runnable
[vertex example](../examples/vertex_shader.py).

## Compiled fragment-stage contract

Use Algan's compiler facade:

```python
from algan.taichi_compat import ti
from algan.rendering.shaders.fragment_shaders import FragmentStage, STAGE_STANDARD
```

Keep compiled functions in a real `.py` file, such as the bundled
`fragment_shader_taichi.py`. **Do not add `from __future__ import annotations`**
to that file: the compiler needs live annotation objects such as `ti.template()`,
not strings. Do not generate a stage with `exec`, a notebook-only transient
lambda, or a function whose source cannot be inspected.

A stage is a `@ti.func` with the following ordered contract. Even unused arguments
must remain in the signature:

```python
@ti.func
def fragment_modulation(
    pos, view_dir, n_interp, face_n, in_rgb, in_glow,
    params: ti.template(), f, prim, off,
    light_pos: ti.template(), light_col: ti.template(), num_lights,
    shadows: ti.template(), vis, cam_pos,
):
    tm = f % params.shape[0]
    frequency = params[tm, prim, off + 0]
    phase = params[tm, prim, off + 1]
    weight = 0.5 + 0.5 * ti.cos(pos[0] * frequency + phase)
    return ti.math.vec4(
        in_rgb[0] * weight,
        in_rgb[1] * weight,
        in_rgb[2] * weight,
        in_glow,
    )

MODULATION = FragmentStage(
    fragment_modulation,
    [
        ('modulation_frequency', 1, 4.0),
        ('modulation_phase', 1, 0.0),
    ],
)
```

This is compiled scalar/vector math. Use `ti` operations inside the stage, not
PyTorch, NumPy, Python file access, or Python objects requiring dynamic callbacks.
`pos` is a world-space hit position; the example's pattern is world-locked. Do
not claim it is an object-space or UV pattern. Use a native UV texture for an
object-attached image or verify a supported coordinate mapping for a custom
material-space effect. The contract does not include arbitrary UV arguments.

The return is `vec4(R, G, B, glow)`. Its fourth component is **not opacity**.
Color stages do not change ray throughput merely by returning another fourth
value. Preserve `in_glow` unless intentionally changing the requested glow.

### Parameter packing

The stage constructor's ordered `param_specs` are `(name, width, default)`.
A scalar uses width 1; an RGB parameter uses width 3. Read at
`params[tm, prim, off + slot]`, where `slot` is the cumulative component offset,
not the parameter's index in the list. For example, a width-3 color followed
by a scalar gain uses offsets 0, 1, 2 for the color and 3 for the gain.

Keep the supplied `off`; another stage may appear before yours. Use
`tm = f % params.shape[0]` so static parameter frames can broadcast correctly.
Do not assume the parameter buffer contains one row for every rendered frame.

A width-3 value must contain three components. A five-component Algan palette
color cannot be passed directly to it: use an RGB tuple or `color[..., :3]`.
Duplicate parameter names across stages are suffixed (`name`, `name_2`, ...).
Distinct names make the script less fragile. Stage topology, parameter widths,
and function definitions are configuration, not animation targets.

### Compose with built-in lighting

```python
ball = Sphere()
ball.set_fragment_shader([MODULATION, STAGE_STANDARD])
ball.roughness = 0.4
ball.metalness = 0.0
ball.spawn()
with Seq(runtime=2.0, easing=easings.identity):
    ball.modulation_phase = 2 * PI
```

The list runs left to right: each stage receives the preceding stage's output
color and glow. Placing a color transform before a lighting stage differs from
placing it after lighting. Use the order needed by the user, not an automatic
recipe. A recognized built-in function such as `phong_shader` resolves to its
compiled stage when passed to `set_fragment_shader`.

Built-in stage objects include `STAGE_UNLIT`, `STAGE_LAMBERT`, `STAGE_PHONG`,
`STAGE_STANDARD`, `STAGE_PHYSICAL`, and `STAGE_MANIM`. Shipped custom stages
include `cosine_color`, `fresnel_rim`, and `glass_ball`; these are available tools,
not recommended aesthetic defaults. Inspect their parameter specifications
before animating them. Use one shared stage definition for multiple objects.

Custom pipelines compile on first use, and changing the function/pipeline can
require compilation again. A syntax check does not compile a stage. Render a
small representative scene on the intended backend, then test a multi-frame
animation of its parameters before starting the full export.

See the complete [fragment example](../examples/fragment_shader_taichi.py).

## Custom ray continuation: optional advanced authoring

A stage can supply `scatter=some_ti_function` to control how rays continue after
the pipeline shades a hit. This is separate from recoloring the surface. Prefer
ordinary material transmission/metalness/roughness when those controls express
the request. Do not write custom scattering merely to add a color pattern.

For a requested nonstandard continuation, inspect the installed scatter contract
in `algan/rendering/raytracing/shading_taichi.py` and the complete
`forced_mirror_scatter` example in `fragment_shaders.py`. Copy the **current
signature and return contract**, then implement only the requested behavior.
The last stage with a scatter function supplies the pipeline's scatter.

Both renderers support the shared custom pipeline/scatter interfaces in the
source-checked version. The deterministic renderer can follow multiple returned
branches; the path tracer samples a continuation rather than performing the
same split. Arbitrary custom scattering is not automatically a physically valid
BSDF with matching sampling, energy, or noise behavior. Validate it in the final
renderer and disclose approximations. An older `set_fragment_shader` docstring
saying the path tracer ignores pipelines is stale in this source snapshot.

## Shader verification checklist

Check the exact signature, declared defaults, channel widths, and source-file
requirements before rendering. Then test parameter values at the beginning,
middle, and end of the intended animation. Include more than one frame and more
than one object when the production scene needs batching. Verify that required
lighting, shadows, transparency, and reflection interactions actually appear;
absence of an exception is not proof that the intended path ran.

A mathematical tensor test can check shape, broadcasting, input mutation, and
channel preservation for a vertex function. It cannot prove the final rendered
lighting. An AST/syntax check cannot prove a fragment stage compiles on the
selected backend. Report these validation levels separately.

Source basis: [material and shader contracts](api-sources.md#materials-and-shaders).
