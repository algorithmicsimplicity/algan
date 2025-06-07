=============
Batching Mobs
=============

In order for Algan to work, it keeps track of all changes made to mobs. This introduces quite a bit of overhead computation
for each Mob. You won't notice it for small scenes with a few dozen mobs, but if you want to have scenes with thousands
of Mobs it can become excruciatingly slow. For this reason, if the computation time of your animations is to slow,
you will want to batch your mobs to reduce overhead.

.. important::

    Currently batching is only supported for Mobs composed of TrianglePrimitives.

.. note::

    Batching Mobs has no effect on rendering time, only animation time.

Example Mob Batching
----------------

You can batch mobs by simply passing a list of locations to a Mob's constructor. Here's an example

.. algan:: BatchingMobsExample1

    from algan import *

    mobs = Circle().spawn()#TriangleTriangulated(location=[LEFT*0.5, ORIGIN, RIGHT*0.5])
    mobs.move(UP*0.5)

    render_to_file()

By providing a list of 3 different locations to the constructor, we create a batch of 3 different mobs,
at the given locations. All of the mob's other attributes which we didn't specify, such as color, are equal for
all 3 mobs. If we wanted to, we could give them each separate colors by providing a list of 3 colors to the constructor
as well.

The advantage of this batched mob, as compared to making 3 different mobs, is that it will be tracked by Algan as a
single Mob. This means that you only pay the overhead cost once, even when batching 10,000 mobs. It also means
that any operations you apply to the mob, such as the move function, will be batched,
and hence can be efficiently computed on GPUs.


