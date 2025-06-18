=======================
Lights, Camera, Action!
=======================

If you have been paying close attention to the video's output by Algan,
then you may have noticed that 3-D objects actually have a sheen,
as if being lighted from a light source above.

By default, Algan spawns a light source (of class :class:`~.PointLight` ) above and to the the
right of the :class:`~.Camera` . The :class:`~.Camera` itself is spawned at `ORIGIN+OUT*7` , and
is rotated to look towards the `ORIGIN` . All 3-D objects (specifically, those
inheriting from :class:`~.Surface` ), will,
at render time, change their color depending on the angle at which rays from the
light source hit their surface. This gives the lighting effect you see.

Both the :class:`~.Camera` and :class:`~.PointLight` are implemented as :class:`~.Mob` s,
which means you can animate them just as you would any other :class:`~.Mob`. You can also
add more light sources using :meth:`.SCene.add_light_sources` .

Example: Animating Lights and Camera
------------------------------------

.. algan:: LightsAndCameraExample

    from algan import *

    c1 = GREEN
    c2 = RED
    mobs = Group([Cylinder(color=c1 * (1-t) + t * c2).rotate(90, OUT) for t in
                  torch.linspace(0.,1,9)]).arrange_in_grid(3, buffer=1).scale(0.7).spawn()

    r = 5

    # Get the active camera used by the scene.
    # Now we can animate it just like any other Mob.
    camera = Scene.get_camera()

    with Sync(run_time=r, rate_func=rate_funcs.identity):
        mobs.rotate(360, RIGHT)
        camera.orbit_around_point(ORIGIN, 360, UP)

    # Get the only light source in the scene.
    light_source = Scene.get_light_sources()[0]

    with Seq(run_time=r):
        light_source.rotate_around_point(ORIGIN, 360, OUT)

    # Add a new light source. This one shines BLUE light.
    # Make sure to spawn it!
    Scene.add_light_source(PointLight(location=ORIGIN + DOWN*10 + OUT * 20,
                                      color=BLUE).spawn())

    mobs.wait(r)

    render_to_file()
