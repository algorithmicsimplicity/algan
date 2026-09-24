# Geometry, layout, text, and assets

Use objects and assets to implement the user's specified content. This reference
does not prescribe composition, typography, or visual style.

## Geometry and coordinate conventions

Native objects include 2D shapes such as `Circle`, `Square`, `Rectangle`, and
`Triangle`, and 3D objects such as `Sphere`, `Cylinder`, `Torus`, and `Cube`.
`Surface` supports parametric surfaces, and `TriangleMesh` supports explicit
triangle geometry. Inspect the installed constructor when building a custom
surface, mesh, or Bézier path rather than inventing field names or array layouts.
A closed filled Bézier circuit is not automatically an open 3D stroked curve;
verify that the selected class represents the requested geometry.

`Line` is flat 2D geometry and can disappear edge-on after a 3D rotation. Use
`Line3D` for box edges, spatial axes and connecting wires that need visible
thickness from changing camera angles:

```python
edge = Line3D(start=LEFT + DOWN, end=RIGHT + UP,
              radius=0.015, resolution=8, color=BLUE)
edge.spawn()
```

Its radius is in world units, and resolution controls the cylinder tessellation.
Use enough segments for the intended shot; each edge is a real 3D surface with
a geometry cost. This choice addresses the geometry's viewing-angle behavior.

World axes are `RIGHT` (+x), `UP` (+y), and `OUT` (+z), with opposite constants
`LEFT`, `DOWN`, and `IN`. `ORIGIN` is the zero point. `OUT` points toward the
viewer for the default camera, not for every camera pose. Every root angle
argument is in degrees: rotations, adapted Manim shapes (`Arc(angle=90)`), and
`RegularPolygon(start_angle=...)`, `Line(path_arc=...)` and
`Wiggle(rotation_angle=...)`. A root angle that looks like radians warns.

Use `.move(vector)` for a displacement and `.move_to(point)` for a destination.
Use `.scale(factor)` or `.scale_to_width(...)` / `.scale_to_height(...)` for size.
Do not confuse a model's file units with screen pixels or world coordinates.

### Outlines on 2D shapes

Filled native 2D shapes are drawn with an outline by default (`stroke_width=5`,
`stroke_color=WHITE`), laid inside the fill. `stroke_width` is screen-space: it
is measured in pixels of a PREVIEW-height (396 px) frame and scales with the
output height, not with the Mob's size or distance. A small or distant filled
shape can therefore render as mostly outline. Pass `stroke_width=0` when the
specification has no outline, and choose the stroke explicitly when it has one.
`filled=False` gives an outline-only shape.

Mob `opacity` applies to fill and outline together. For a translucent fill with
an opaque outline, set the two separately on one Mob:

```python
box = RoundedRectangle(width=3, height=2, color=BLUE, fill_opacity=0.2,
                       stroke_color=WHITE, stroke_opacity=1, stroke_width=3)
box.spawn()
box.fill_opacity = 0.6        # animatable, like any attribute
box.color = RED               # recolors; the 0.6 fill opacity is kept
```

Once set, `fill_opacity` / `stroke_opacity` survive later `color` /
`stroke_color` assignments whose alpha is 1 (every named color); a color with its
own alpha below 1, such as `RED.set_opacity(0.3)`, replaces it. `opacity`
multiplies both. Both keywords work on native shapes and on the root adapters of
Manim shapes.

### Lit and unlit geometry

Flat 2D shapes, `Text`/`Tex` and `ImageMob` show their own colors. 3D geometry,
including `Line3D`, `Sphere`, `Prism`, `Surface` and `TriangleMesh`, is shaded by
the Scene's lights through its material, so a colored `Line3D` or textured
`Surface` can look dark in a dimly lit scene. When the specification calls for a
3D object that shows its own color regardless of lighting, construct it with
`unlit=True` (for example `Line3D(start=a, end=b, radius=0.01, unlit=True)`).
That is the same shading as `set_material(UnlitMaterial())` before spawning, and
reaches the object's parts and any children added to it later. A later
`set_material(...)` before spawning replaces it. `ImageMob` is already unlit.

## Grouping and placement

```python
with Off():
    parts = Group([Square(), Circle()])
    parts.arrange_in_line(RIGHT, buffer=0.4)
    parts.move_to(ORIGIN)
    parts.spawn()
```

A parent transform carries its descendants. Use `.add_children([child])` to
establish a hierarchy, but use a dependent updater when only some relationship
should persist. Do not animate a Python list as though it were a Group.

Useful public layout operations:

| Method | Meaning |
|---|---|
| `a.move_next_to(b, RIGHT, buffer=...)` | Place boundaries beside each other |
| `a.align_with(b, direction, ...)` | Align along the requested axis; inspect anchor options |
| `group.arrange_in_line(direction, buffer=...)` | Lay out children along a line |
| `group.arrange_in_grid(n)` | Arrange children in a grid |
| `mob.move_to_screen_position(x, y)` | Fractional screen location: bottom-left `(0, 0)`, top-right `(1, 1)` |
| `mob.move_to_screen_edge(UP)` | Place against an edge of the current camera's frame |
| `mob.move_to_screen_corner((UP, LEFT))` | Place against two frame edges |
| `mob.fit_to_screen((x0, y0), (x1, y1))` | Fit and position a hierarchy in a fractional screen rectangle |

Measurements such as `get_center`, `get_width`, and `get_height` describe the
state at the point they are queried. Layout methods do not automatically track
later target or camera changes. For a persistent relationship, evaluate it in an
updater or attach the object to the appropriate parent.

For a user-requested camera-fixed overlay, attach the caption to the camera and
place it during setup:

```python
with Off():
    caption = Text('User-provided caption', font_size=32)
    Scene.get_camera().add_children([caption])
    caption.move_to_screen_position(0.2, 0.1)
    caption.spawn()
```

The string, size, and location are placeholders, not recommended design values.
Set the final aspect ratio before computing screen-relative layout. Check the
actual projected result after camera changes rather than relying on the default
camera's approximate world-space bounds.

## Text and equations

`Text` uses system-font text when Pango is available. `Tex` compiles LaTeX math.
Use the user's exact wording and requested installed font family:

```python
label = Text('User text', font_size=48)
formula = Tex(r'\frac{a}{b}', font_size=48)
```

`Tex` is already in math mode: do not wrap the expression in dollar signs. Use
raw Python strings for LaTeX backslashes. Check optional text dependencies before
promising an equation or a specific typeface. Do not distribute font files.

Text supports `font`, `weight`, `slant`, and substring style maps such as
`color_map={'term': BLUE}`. Geometry scaling can be animated after spawn; it is
different from reconstructing glyphs with a new font size or a new font family.
`Text(..., height=h)` / `width=w` scale the finished text to a world-unit size,
which is easier than deriving a size from `font_size`.

Every glyph is a Bezier circuit. Text that is still or only moves rigidly
(translation, including a moving camera) has its outlines built once per render
batch; text that rotates, scales, morphs or is written on is rebuilt every frame.
If profiling still shows glyph outlines dominating, see
[performance](performance.md#many-glyphs-of-text) for rasterizing text blocks
into an `ImageMob`.

For semantic formula parts, supply separate strings and use `get_segment`:

```python
formula = Tex(r'a', r'+b', r'=c').spawn()
with Lag(0.3, runtime=1.0):
    for i in range(len(formula.tex_strings)):
        formula.get_segment(i).opacity = 0.5
```

Do not iterate `formula.children` expecting one child per supplied string.
Glyphs are packed; semantic segments are exposed separately.
`character_mobs` exposes visible glyphs, not spaces. Glyph indices are not
necessarily string-character offsets, especially for LaTeX.

Text writing can be invoked with `Text(...).spawn(False).write(runtime=...)`.
`spawn(False)` avoids an ordinary appearance animation before the writing.
`write` is a convenience API with its own timing options and does not replace
lifespan management. Use it only when that reveal is requested.

For changing numbers:

```python
counter = DecimalNumber(0.0, decimal_places=2).spawn()
with Seq(runtime=2.0):
    counter.value = 10.0
```

Do not create a new Text object every frame to update a numerical display.
Verify digit width changes, alignment, signs, and decimal precision against the
user's specification.

## Images and texture layouts

Use `ImageMob('assets/image.png')` for an image-bearing plane. It accepts image
arrays in conventional `[height, width, channels]` form as well, with 3, 4 or 5
channels (RGB, glow, opacity), so an array generated in Python can carry its own
glow. The plane is unlit, one world unit tall with the image's aspect ratio, and
`set_color_by_image(array_or_path)` replaces its picture (inside `Off()` for an
instant change). Paths resolve against the working directory and then the script
directory; explicit project asset paths make deliveries easier to reproduce.

There are two distinct texture entry points:

| Entry point | Input convention |
|---|---|
| Material `map`, `normal_map`, `roughness_map`, `metalness_map` | File path or image `[H, W, C]`; Algan forwards it to compatible UV geometry |
| Native `Surface.color_texture` and property textures | Tensor in UV order `[W, H, C]`; not a filename |

Native surface color textures have **five** channels: red, green, blue, glow,
and opacity. The last channel must be populated for a visible texture:

```python
import torch

texture = torch.zeros(16, 8, 5)
texture[..., :3] = 1.0
texture[..., 4] = 1.0
surface = Sphere(color_texture=texture)
```

Do not treat channel 3 as alpha in this native array. Do not transpose channels
when changing H/W orientation. For loading an image onto a Surface, use
`surface.set_color_by_image(path)` or the material image slot rather than passing
a path into `color_texture`.

Material image sampling requires UVs: Surfaces and a `TriangleMesh` constructed
with UVs support it. Do not assume a Cube or every arbitrary Mob has usable UVs.
Unsupported maps can warn and be ignored; a successful render alone does not
prove the texture was used.

`roughness_map` reads the image's green channel and `metalness_map` its blue
channel; a single-channel image supplies its value directly. This matters when
using packed material-property images. `normal_map` is a tangent-space normal
map, not a replacement vertex-position array.

Native property names include `roughness_texture`, `reflectivity_texture`
(metalness), `refractive_index_texture`, `normal_texture`, and `glow_texture`.
These low-level texture names do not imply matching public scalar setters on a
Mob; scalar transport controls come from materials. Native scalar maps have a
last dimension of one; normal maps have three. Check the installed Surface API
for mapping options and shape requirements before constructing custom maps.

A Surface's `color_texture` can be animated by assigning another compatible
texture after spawning. Keep texture dimensions compatible for interpolation.
Material property maps are static, unlike the material's scalar attributes.
Do not generalize the animated color map to every map slot. Native procedural
helpers such as `get_checkerboard` and `get_stripes` can generate texture arrays
when that pattern is requested; their output is not a default art direction.

For a UV-preserving surface shape change:

```python
world = ImageMob('assets/image.png').spawn()
with Seq(runtime=2):
    world.set_shape_to(Sphere(radius=2, add_to_scene=False))
```

Inspect seams, poles, aspect ratio, and orientation. This is a Surface operation,
not a promise that any imported mesh can morph with arbitrary topology.

## Importing a 3D model

```python
model = Model3D('assets/model.glb', fit_to_size=3).spawn()
```

`fit_to_size` recenters and scales the bounding-box **diagonal** to that many
world units; it is not a width or height argument. Standard loading includes
glTF/glB and common mesh formats such as OBJ, PLY, STL, DAE, and OFF. FBX requires
an optional binding and native Assimp. Keep referenced textures and related
files in the asset manifest; do not deliver just an OBJ whose materials point
to missing files.

Imported materials, UVs, textures, and normals depend on the asset and the loader
options. Inspect warnings and rendered samples. The public part interface is:

```python
print(model.node_names)
part = model.get_part('NameFromTheActualAsset')
```

`get_part` can return one mesh or a list. Handle both forms explicitly; do not
call `.rotate` on a list. Parts remain in the model hierarchy, so a parent move
also carries them.

For imported clips, inspect `model.animation_names`, then use the actual name:

```python
model.play_animation('ActualClipName', runtime=2.0, loop=2)
```

The runtime is per loop. `fps` controls the bake sampling rate. Supported imported
motion is rigid node animation; skeletal skinning that deforms vertices between
bones is not applied. Do not promise full character-animation playback from a
successful model import. Verify a representative moving pose before building
the rest of the video around it.

## Manim geometry compatibility

Use `import algan.manim as mn` when intentionally accessing the compatibility
namespace. It supplies wrapped geometry, not Manim's Scene/renderer/animation
engine. Continue to use Algan's spawn, timeline, materials, and export APIs.

The boundary changes conventions. Root `Arc(angle=90)` takes degrees;
`mn.Arc(angle=PI / 2)` takes radians.

Several names exported at the root are themselves adapters of Manim geometry
(on Algan 0.0.2: `RoundedRectangle`, `Star`, `Annulus`, `Arc`, `Arrow`,
`Ellipse`, `Sector`, `AnnularSector`, `DashedLine`, `Elbow`, `Cross`; check
`type(obj).__module__` for others). Their constructors take Manim's keywords
converted to Algan's units (degrees, Algan stroke widths), plus Algan's `opacity`,
`glow`, `unlit`, `name`, `location` and `basis`, and `fill_opacity` /
`stroke_opacity`. The corresponding `mn` classes keep Manim's radians and stroke
widths. Do not mix these merely because two classes have the same name. Root
spellings include `SVGMob`, `MobMatrix`, and `MobTable`, rather than their
`Mobject`-named counterparts.

Verify a less common class through the installed namespace and signature. Do
not install a separate Manim engine to resolve a guessed Algan symbol. Graphs,
plots, and annotations can use available adapted geometry, but their arrays,
constructor parameters, and resulting Mob hierarchy must be checked, not inferred
from unrelated Manim examples.

Source basis: [geometry and asset sources](api-sources.md#geometry-text-and-assets).
