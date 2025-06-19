====================
Importing From Manim
====================

Algan provides a bunch of built-in Mobs, but our collection is nowhere as extensive
as `Manim's <https://docs.manim.community/en/stable/>`_. So we provide functionality
to import Mobs directly from Manim.
This way you can make use of the vast collection of Mobs defined by Manim.

.. important::

    You must install the Manim package to your Python environment in order to use Manim Mobs.

.. note::

    You can only import mobs from Manim, but not animations. If there is a Manim animation
    you wish to use you will need to recreate it yourself in Algan.


The ManinMob
------------

Importing is straightforward, just create a :class:`~.ManimMob` object and pass in a Manim Mobject as the parameter.

.. algan:: ImportingManimMob

    from algan import *
    import manim as mn

    # Let's grab a complex plane from the manim library.
    mob = ManimMob(mn.ComplexPlane().add_coordinates()).spawn()

    # Now we have a mob that we can animate using Algan!
    with Seq(run_time=5):
        mob.scale(0.5)
        mob.rotate(90, OUT)

    render_to_file()

.. important::

    Do not use both `from algan import *` and `from manim import *`, as this will lead to
    clashing definitions. You must give them one of them a name, as in this example where we
    name manim as mn.

