===================
Images and Textures
===================

Algan can texture 3-D surfaces with images or numpy/torch arrays, and use texture
maps to drive material properties like roughness, reflectivity, index of
refraction, and surface normals.

There are four ways to use images in Algan:

1. **ImageMob:** A flat, textured plane for displaying photos or graphics on screen.
2. ** :class:`~.Surface` texture maps:** Per-texel material and color maps sampled inside the GPU raytracer.
3. **2-D shape texture grids:** Color gradients and image fills across 2-D shapes (:class:`~.Circle`, :class:`~.Square`, text glyphs).
4. **Background images:** A static backdrop behind your entire scene (see :doc:`backgrounds_and_post_processing`).

.. note::

    The Three.js material classes accept Three.js's image slots (``map``,
    ``normalMap``, ``roughnessMap``, ...) for API parity but **do not sample them**.
    Texturing goes through :class:`~.Surface`, as described below.

Showing an Image
================

:class:`~.ImageMob` takes an image file path or an RGBA array and gives you a flat
textured surface:

.. algan:: TexturesImageMob

    from algan import *

    photo = ImageMob('world_map.png').scale(2).spawn()
    with Seq(run_time=2):
        photo.rotate(30, UP)
        photo.rotate(-30, UP)

    Scene.save_video()

Image paths are resolved against the working directory and then against the
directory holding your script, so an image sitting beside your ``.py`` file loads
regardless of where you launch Python from. The same resolution applies to
:meth:`~algan.scene.Scene.set_background_color`,
:meth:`~algan.scene.Scene.set_environment_map` and
:class:`~algan.mobs.three_d_models.model_mob.ThreeDModelMob`.

Instead of a path you can pass a ``[H, W, 4]`` or ``[H, W, 5]`` tensor, which is how
you texture something with data you computed rather than loaded.

.. important::

    The per-material texture arguments on :class:`~algan.mobs.surfaces.surface.Surface`
    (``color_texture``, ``roughness_texture``, ``normal_texture`` and the rest)
    take **tensors only**. Handing one a file path raises ``TypeError``. Load the
    image yourself first, with :func:`~algan.utils.file_utils.get_image`, or use
    :meth:`~algan.mobs.surfaces.surface.Surface.set_color_by_image`, which takes a
    path and orients the image onto the surface's ``(u, v)`` axes for you.

Reshaping a textured surface
============================

:class:`~.ImageMob` is itself a :class:`~.Surface`, so you can change its *shape*
while it keeps its texture. That is how you wrap a map onto a globe:

.. algan:: TexturesReshaping

    from algan import *

    # Start as a flat plane colored by our image file.
    world = ImageMob('world_map.png').scale(2).spawn()
    world.wait()

    with Seq(run_time_unit=5, rate_func=rate_funcs.identity):
        for shape in (Sphere(radius=2, add_to_scene=False),
                      Cylinder(radius=1, height=2, add_to_scene=False)):
            # Change the surface shape; the texture comes along.
            world.set_shape_to(shape)
            world.rotate(360, UP)
            world.rotate(360, RIGHT)

    Scene.save_video()

:meth:`~.Surface.set_shape_to` re-maps the surface's intrinsic (UV) coordinates onto
a new shape, and the texture follows them. Any :class:`~.Surface` works as a
target. Build the target with ``add_to_scene=False``, as above: it only says what
shape to become and is never drawn, and without the flag Algan registers it as a
Mob you meant to show and warns that you never spawned it.

.. note::

    A low-resolution surface is automatically resized to a higher-resolution grid
    when the target shape needs one, so morphing a flat plane into a sphere does not
    come out faceted.

Texturing Any Surface
=====================

:class:`~.Surface` and everything built on it (:class:`~.Sphere`,
:class:`~.Cylinder`, :class:`~.Cone`, :class:`~.Torus`, your own surface functions)
take texture arguments at construction:

.. algan:: TexturesColorTexture

    from algan import *
    import torch

    # A 16x16 checkerboard as an RGB + glow + opacity texture.
    checker = torch.zeros(16, 16, 5)
    grid = (torch.arange(16).view(-1, 1) + torch.arange(16).view(1, -1)) % 2
    checker[..., 0] = grid          # red channel
    checker[..., 2] = 1 - grid      # blue channel
    checker[..., 4] = 1.0           # opacity

    globe = Sphere(radius=1.5, color_texture=checker).spawn()
    with Seq(run_time=3):
        globe.rotate(360, UP)

    Scene.save_video()

The available texture arguments:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Argument
     - Shape
     - What it drives
   * - ``color_texture``
     - ``[W, H, 5]``
     - Base color: red, green, blue, glow, opacity.
   * - ``roughness_texture``
     - ``[W, H, 1]``
     - How blurred reflections are, per texel.
   * - ``reflectivity_texture``
     - ``[W, H, 1]``
     - Metalness, per texel.
   * - ``refractive_index_texture``
     - ``[W, H, 1]``
     - Index of refraction, per texel.
   * - ``normal_texture``
     - ``[W, H, 3]``
     - Tangent-space normal map; perturbs the shading normal.
   * - ``glow_texture``
     - ``[W, H, 1]``
     - Glow strength, per texel.

Color and the three material property maps are sampled **bilinearly per fragment,
inside the ray tracing kernel**, for both flat and curved (PN) triangles. A property
without a map keeps the ordinary per-vertex value, and maps of different resolutions
are resampled to a common one.

Wrapping around a closed surface
--------------------------------

On a surface that closes on itself -- a :class:`~.Sphere`, a :class:`~.Cylinder`
and a :class:`~.Cone` close around ``u``, a :class:`~.Torus` around both -- the
map **wraps**: the last column of texels is a neighbour of the first, and Algan
blends across that meridian the same way it blends anywhere else. So the
checkerboard above meets itself where the sphere comes back around, and a map
whose two edges are meant to join (an equirectangular world map, a tiling
pattern) joins seamlessly.

Each texel column therefore spans exactly ``1 / W`` of the way around, not
``1 / (W - 1)``: with a 16-wide map, column 0 is centred at the seam and column
8 faces the other side, whichever direction the surface is spun. Algan works out
which axes close from the geometry, so a surface of your own written with
:class:`~.Surface` and a ``coord_function`` wraps too, without saying so.

An **open** surface -- a flat plane, an :class:`~.ImageMob`, the pole-to-pole
``v`` axis of a sphere -- has no far side to blend into, so its first and last
texels sit exactly on its two edges and the edge value carries beyond them.

Building a map from world positions
-----------------------------------

A texture is written in ``(u, v)``, which is the wrong language for a map whose
content depends on *where the surface is*: "everything above the equator",
"redder the further from the origin". :meth:`~algan.mobs.surfaces.surface.Surface.get_texture_locations`
translates -- it hands back the world position of every texel, laid out exactly
like the map itself, so the map can be written as arithmetic on 3-D coordinates:

.. algan:: TexturesByWorldPosition
    :save_last_frame:

    from algan import *

    globe = Sphere(radius=1.5)
    xyz = globe.get_texture_locations((256, 256))
    globe.color_texture = BLUE.mult_opacity((xyz[..., 1:2] > 0).float())
    globe.spawn()

    Scene.save_video()

The positions come from the surface's **current mesh**, not from its coordinate
function, so they are right for a shape that has been deformed, morphed with
:meth:`~algan.mobs.surfaces.surface.Surface.set_shape_to`, or built by assigning
``surface.grid.location`` directly. They also account for the wrapping above and
for the curvature between grid vertices, which is what keeps a boundary like the
one in that example straight rather than scalloped once the map out-resolves the
grid.

The ``resolution`` argument is what sizes a map that does not exist yet, as
above. Leave it out once the surface has a ``color_texture`` and you get that
texture's resolution; the material maps are constructor arguments, so size those
from a surface of the same shape:

.. code-block:: python

    probe = Sphere(radius=1.5)
    height = probe.get_texture_locations((128, 128))[..., 1:2]
    sphere = Sphere(radius=1.5, roughness_texture=(height / 1.5).clamp(0, 1))

Because a texture is carried in ``(u, v)``, colors derived this way travel with
the surface when it later moves: they record where it *was* when you asked. Take
the positions again inside an
:meth:`~algan.animatable_base.animatable.Animatable.add_updater` callback for a
texture that stays locked to world space.

Animating a texture
-------------------

A texture map is an ordinary animatable attribute, so you animate it the way you
animate a color or a location: **assign a new one**. Algan interpolates the old
texture to the new one per texel over the current context's duration.

.. algan:: TexturesAnimatedTexture

    from algan import *
    import torch

    def stripes(horizontal):
        index = torch.arange(32)
        bands = (index.view(-1, 1) if horizontal else index.view(1, -1)) // 4 % 2
        texture = torch.zeros(32, 32, 5)
        texture[..., 0] = bands.expand(32, 32)          # red
        texture[..., 2] = 1 - bands.expand(32, 32)      # blue
        texture[..., 4] = 1.0                           # opacity
        return texture

    globe = Sphere(radius=1.5, color_texture=stripes(True)).spawn()
    with Seq(run_time=3):
        globe.color_texture = stripes(False)   # cross-fades, texel by texel

    Scene.save_video()

Normal maps
-----------

A ``normal_texture`` is a tangent-space normal map with components in ``[-1, 1]``:
x along increasing ``u``, y along increasing ``v``, z along the smooth surface
normal, so ``(0, 0, 1)`` means "unperturbed".

.. important::

    Under the default vertex-shaded pipeline, lighting is baked at the vertices, so a
    normal map only affects things evaluated per fragment: mirror reflections,
    refraction, ray-traced shadows, and fragment shading. If a normal map appears to
    do nothing to the diffuse shading, that is why -- see
    :doc:`shaders_and_materials`.

Glow maps
---------

``glow_texture`` is the exception to the per-fragment rule: glow is consumed by the
glow accumulator per *vertex*, so the map is baked down to the surface grid
resolution. Raise ``grid_width`` / ``grid_height`` if you need more detail from it.

That bake is also the exception to the wrapping above -- it lands the map's two
edges on the same grid column of a closed surface, so a glow map whose edges do
not already agree shows its seam. Make the first and last column of a
``glow_texture`` match if you need it to wrap.

Coloring a 2-D Shape
====================

Algan's 2-D shapes -- :class:`~.Square`, :class:`~.Circle`, :class:`~.Polygon`,
:class:`~.Line`, the glyphs of :class:`~.Text` and :class:`~.Tex` -- are not
meshes. They are cubic bezier circuits (:class:`~.BezierCircuitCubic`), evaluated
analytically by the renderer, so there are no vertices to hang colors off.

Instead a circuit carries a **texture grid**: a rectangular grid of color
samples laid across the shape's own frame, which the renderer interpolates
bilinearly per fragment. It defaults to a single texel -- one flat color, which
is all a shape needs most of the time -- so painting anything across a shape
starts by asking for a grid:

.. algan:: TexturesCircuitGradient
    :save_last_frame:

    from algan import *
    import torch

    square = Square(texture_grid_width=64, texture_grid_height=64, border_width=0)
    square.set_color_by_function(
        lambda uv: torch.cat((uv[..., :1], 1 - uv[..., :1], uv[..., 1:]), -1)
    )
    square.spawn()

    Scene.save_video()

``texture_grid_width`` and ``texture_grid_height`` are the number of color
samples along each axis, and they are the resolution of everything painted on the
shape. Both default to ``1``; giving only the width squares the grid up.

The ``(u, v)`` domain
---------------------

:meth:`~.BezierCircuitCubic.set_color_by_function` hands your function a
``[..., 2]`` tensor of ``(u, v)`` coordinates -- the same convention
:class:`~.Surface` uses, so a color function written for one works on the other.
``u`` runs from 0 to 1 along the circuit's first basis row and ``v`` along its
second, which for an upright 2-D shape means ``u`` left to right and ``v`` top to
bottom. Return RGB, RGBA, or Algan's five-channel RGB + glow + alpha; the
function is called once on the whole grid, so write it with tensor operations.

.. note::

    Both basis rows are as long as the distance from the circuit's centre to its
    furthest control point, so the domain covers the square that *circumscribes*
    the shape. A :class:`~.Square` therefore occupies the middle of it rather
    than all of it: a gradient across ``u`` has already run through part of its
    range by the time it reaches the square's left edge, and finishes the rest
    beyond its right one. :meth:`~.BezierCircuitCubic.get_base_grid` returns the
    grid if you want to look at it.

Painting an image on a shape
----------------------------

:meth:`~.BezierCircuitCubic.set_color_by_image` takes the same paths and arrays
:class:`~.ImageMob` does and resamples them onto the grid, with the image's top
left at ``(u, v) == (0, 0)``:

.. algan:: TexturesCircuitImage
    :save_last_frame:

    from algan import *

    circle = Circle(texture_grid_width=128, texture_grid_height=128, border_width=0)
    circle.set_color_by_image('world_map.png')
    circle.spawn()

    Scene.save_video()

.. important::

    A circuit has no separate texture map: the texture grid **is** the
    resolution. That is the difference from
    :meth:`~algan.mobs.surfaces.surface.Surface.set_color_by_image`, which keeps
    the image at its own resolution however coarse the surface's grid is.

Both methods are recorded as animations, like any other attribute write, so the
colors cross-fade over the current context's duration:

.. algan:: TexturesCircuitCrossFade

    from algan import *
    import torch

    def cool(uv):
        return torch.cat((torch.zeros_like(uv[..., :1]), uv[..., 1:],
                          1 - uv[..., 1:]), -1)

    def hot(uv):
        return torch.cat((torch.ones_like(uv[..., :1]), 1 - uv[..., 1:],
                          torch.zeros_like(uv[..., :1])), -1)

    square = Square(texture_grid_width=64, border_width=0).scale(2)
    square.set_color_by_function(cool)
    square.spawn()

    with Seq(run_time=2):
        square.set_color_by_function(hot)   # cross-fades from whatever it was

    Scene.save_video()

On a filled circuit these color the fill and leave ``border_color`` alone. On an
unfilled one, where the stroke is all there is, they color the stroke. And on a
multi-circuit Mob (e.g. a :class:`~.Text`, a :class:`~.Tex`), which take the same
grid arguments and pass them down to their packed glyphs, each circuit is
colored over its own frame, so the pattern repeats per glyph:

.. algan:: TexturesTextGradient
    :save_last_frame:

    from algan import *
    import torch

    text = Text('Algan', texture_grid_width=16, texture_grid_height=16)
    for glyph in text.character_mobs:
        glyph.set_color_by_function(
            lambda uv: torch.cat(
                (uv[..., 1:], 1 - uv[..., 1:], torch.zeros_like(uv[..., :1])), -1
            )
        )
    text.spawn()

    Scene.save_video()

Coloring along a line
---------------------

A :class:`~.Line` is one-dimensional, and its texture grid follows: its control
points are collinear, so the second basis row is synthesized perpendicular to the
path and carries none of the shape's extent. ``texture_grid_height`` therefore
defaults to a single row and ``texture_grid_width`` alone is the number of color
samples *along* the line.

:meth:`Line.set_color_by_function <algan.mobs.shapes_2d.Line.set_color_by_function>`
drops the second coordinate to match, handing your function a single ``t``
running from 0 at :meth:`~.Line.get_start` to 1 at :meth:`~.Line.get_end`:

.. algan:: TexturesLineGradient
    :save_last_frame:

    from algan import *
    import torch

    line = Line(LEFT * 4, RIGHT * 4, border_width=30, texture_grid_width=64)
    line.set_color_by_function(
        lambda t: torch.cat((t, torch.zeros_like(t), 1 - t), -1)
    )
    line.spawn()

    Scene.save_video()

.. note::

    A straight line's frame is pinned to its geometry: **the first basis row
    points from the line's centre toward its start**. That is a guarantee, so
    ``t == 1 - u`` and you never have to work out which end of a line ``u == 0``
    sits at.

Choosing a resolution
=====================

A surface's texture detail is limited by two independent things: the resolution of
the image you supply, and (for glow) the surface's own grid resolution.
:class:`~.Surface` sizes its grid automatically between ``min_grid_resolution`` and
``max_grid_resolution``, and dices curved triangles at render time to whichever of
``render_tolerance`` (a fraction of screen height, so a surface that fills the frame
gets more triangles than one in the distance) and ``render_tolerance_pixels`` (an
absolute pixel count, which takes over at high resolutions) is finer.

Textures also cost render memory. If a heavily textured scene runs out of it, reduce
the texture resolution before reducing anything else; see
:doc:`performance_and_quality`.

See Also
========

- :doc:`../galleries/mob_gallery` -- :class:`~.ImageMob`, :class:`~.Surface` and
  the shapes these textures go on.
- :doc:`three_d_models` -- imported models bring their own textures and materials.
- :doc:`shaders_and_materials` -- what each material property does.
- :doc:`reflections_and_glass` -- the reflection and refraction those maps drive.
- :doc:`lighting_and_shadows` -- the lights a normal map perturbs.
- :doc:`backgrounds_and_post_processing` -- an image behind the whole scene.
- :doc:`renderer_limitations` -- which maps are sampled per fragment and which
  are baked, and what a circuit's color grid cannot do.
- :doc:`performance_and_quality` -- what texture resolution costs in render
  memory.
