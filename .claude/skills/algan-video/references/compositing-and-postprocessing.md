# Backgrounds, post-processing, and transparent export

Implement the requested processing and compositing contract. Do not add bloom,
a color grade, a background treatment, or tone mapping as a creative rule.
Algan has defaults; inspect them when the user specifies exact pixel behavior.

## Static and time-dependent backgrounds

Set a background on the Scene or for one render:

```python
Scene.set_background(Color([0.1, 0.1, 0.1]))
Scene.save_video('renders/clip.mp4', background=Color([0.1, 0.1, 0.1]))
```

A background assignment is Scene configuration affecting the whole render, not
a keyframe from that point onward. Do not write sequential `set_background` calls
expecting a timed color transition. For a changing backdrop, use supported
time-dependent background logic or appropriate animated geometry.

A color has five channels. Multiplying an Algan color constant by a scalar also
changes its opacity. To change only RGB, build the intended RGB color explicitly
or restore its opacity; do not accidentally request a transparent export while
trying to dim a background.

An image path supplies a screen background. A 3D environment map, set through
`Scene.set_environment_map`, serves a different purpose in lighting and ray
transport; a screen plate does not automatically appear in object reflections.

A torch background callback has signature `(x, y, time)` and returns the native
five-channel color layout. Coordinates and time arrive as broadcastable tensors;
build the full broadcast shape, not just one row or column:

```python
import torch

def user_background(x, y, time):
    base = torch.zeros_like(x + y + time)
    level = base + 0.1
    return torch.cat((level, level, level, base, base + 1.0), dim=-1)
```

This is a neutral constant-output contract example. Implement the user's actual
function with tensor math. A procedural background is routed as opaque because
its alpha is not known before evaluation. Use an explicit transparent background
color for alpha export rather than relying on a callback's last channel to
select the container.

## Post-processing callbacks

`Scene.save_video(..., post_processes=())` omits the post-processing passes;
the default includes bloom. Supply an ordered tuple to specify passes, preserving
any required default explicitly when replacing the tuple.

```python
from algan.rendering.post_processing.bloom import bloom_filter

def user_pass(frames, memory=None):
    result = frames.clone()
    # Implement the requested RGB transformation here.
    # Preserve all trailing channels unless their semantics require a change.
    return result

Scene.save_video('renders/processed.mp4',
                 post_processes=(bloom_filter, user_pass))
```

Algan passes `memory=` as a keyword. A callback accepting only `frames` fails when
rendering reaches it; use `memory=None` or accept appropriate keyword arguments.
Frames are torch tensors on the render device. Preserve shape, device, and dtype.
Do not assume the working buffer is an 8-bit RGBA image, or that its fourth
component is always coverage alpha. Operate on RGB and preserve other channels
unless you have verified the exact pipeline contract.

A no-op callback is not a useful production effect; the snippet shows the calling
convention only. Do not insert it into a final video unnecessarily. For tuned
bloom use `functools.partial(bloom_filter, ...)` with supported parameters such as
`strength` and `glow_spread`, using values established for the requested output.
`mob.glow` controls glow rather than opacity or physical emission equivalently.

Video supersampling and FXAA are separate from post-processing callbacks:
`SETTINGS.video.set(supersampling=..., fxaa=...)`. Renderer tone mapping is
configured through `SETTINGS.raytracing`, including `tonemapping`,
`tonemap_method`, and `tonemap_exposure`. Do not enable a nonlinear tone curve
when the user needs authored colors or a linear compositing contract without
accounting for the difference.

## Ordinary transparent output

```python
Scene.set_background(TRANSPARENT)
Scene.save_video('renders/overlay.mov')
Scene.save_frame('renders/overlay.png', at=0.5)
```

Use an alpha-capable container/codec. An MP4 extension does not provide an alpha
channel. When an extension is omitted, the transparent default can choose MOV
instead of MP4. Check the actual returned path and encoded stream, not the name
you expected to receive. Explicit `codec=` wins over automatic codec selection.

A pixel-format label alone does not prove that the file contains useful alpha.
Inspect a decoded frame and composite over contrasting backgrounds. Verify fully
opaque areas, partially covered edges, intentionally translucent objects, and
fully transparent regions. A viewer showing black behind an overlay is not proof
that its alpha was lost, and a black-looking render is not proof that it is correct.

## Additive glow and coverage in one overlay

When the user needs additive light and normal geometry occlusion in a single
transparent clip, use Algan's explicit premultiplied-over mode:

```python
Scene.set_background(TRANSPARENT)
Scene.set_premultiplied_over()
Scene.save_video('renders/glow_overlay.mov')
```

This mode is Scene-local, defaults to disabled, and is ignored for an opaque
background. It also affects frame export. It requires linear color space,
`tonemapping=False`, and the renderer's post-process tone-map stage enabled
(the source-checked defaults). Do not change unrelated experimental switches;
if these settings were overridden, restore the documented compatible values or
explain the conflict before rendering.

The output's stored RGB is **sRGB-encoded linear-premultiplied RGB**. Decode the
RGB directly into linear light before compositing; do not multiply it by alpha
again, and do not apply an alpha divide/multiply around the input color decoding.
For an opaque linear-light backdrop:

```text
C_out = C_clip + (1 - alpha_clip) * C_backdrop
```

For a transparent backdrop, the resulting alpha is:

```text
alpha_out = alpha_clip + (1 - alpha_clip) * alpha_backdrop
```

The glow may carry nonzero RGB at zero alpha. Preserve those values through
loading, color conversion, and merging. An ordinary display viewer or a pipeline
that discards RGB under zero alpha may show the intermediate incorrectly. Merely
selecting an application's alpha-mode label is not proof that its color-processing
order matches this contract.

MOV defaults to ProRes 4444 in this special mode. The source samples are still
quantized to eight bits before encoding; the codec does not make this an HDR or
higher-source-precision export. RGB above the supported range is clipped and
lossy encoding can introduce additional error. PNG preserves the encoded samples
but still needs the same interpretation.

This is a compositing layer, not universal equality with re-rendering the scene
over every background. The layer contains its own bloom; background-dependent
bloom and nonlinear processing do not generally commute with compositing. Colored
refraction of external footage cannot be represented by one coverage-alpha value.
Disclose those limitations when they affect the user's intended use.

Custom post-processing in this mode must preserve coverage alpha and the linear
premultiplied-light contract. Validate an opaque patch, a partial-alpha edge, and
a nonzero-RGB/zero-alpha glow patch over known backdrops before production delivery.
Do not claim an editor-specific setup was tested unless it actually was.

Source basis: [background and compositing sources](api-sources.md#backgrounds-and-compositing).
