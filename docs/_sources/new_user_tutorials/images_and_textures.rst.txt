===================
Images and Textures
===================

Algan provides utilities to color mobs based on image files or RGBA arrays,
known as texturing. Currently texturing is only implemented for 3d objects
inheriting from :class:`.Surface` .

Texturing Surfaces
******************

The :class:`.Surface` class is used to make arbitrary curved surfaces, both
:class:`.Sphere` and :class:`.Cylinder` are created by inheriting from it.
Surfaces can be shaped by providing a function which maps intrinsic (UV) coordinates
to world coordinates.
:class:`ImageMob` also inherits from :class:`.Surface`, but allows for coloring
based on a provided image file path.

.. algan:: TexturingManifolds

    from algan import *

    # Make a flat (plane) surface, colored by our image file.
    mob = ImageMob('world_map.jpg').scale(2).spawn()
    mob.wait()

    with Seq(run_time_unit=10, rate_func=rate_funcs.identity):
        for shape in (Sphere(radius=2), Cylinder(radius=1, height=2)):
            # Change the surface shape.
            mob.set_shape_to(shape)
            mob.rotate(360, UP)
            mob.rotate(360, RIGHT)

    render_to_file()
