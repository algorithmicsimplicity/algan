===============
Extending Algan
===============

In Algan, animations are rendered to videos by extracting all of the RenderPrimitives from a scene.
RenderPrimitives are a special type of mob that define a render() function, which accepts various information
about the scene and camera, and returns a collection of fragments (pixel ID, fragment color, fragment depth).
These fragments are then combined from all primitives to form the final image.

Currently, Algan only supports two types of render primitives: triangles and cubic bezier circuits. If you want
to render something that cannot be created by combining these primitives, you will need to implement your own custom
rendering behaviour by creating a new class which inherits from RenderPrimitive. You can check out the source code
for TrianglePrimitive and BezierCircuitCubicPrimitive to see how to do this.

Once you have created a new primitive, you can also create a new Mob class to automatically handle animation of its
properties, see BezierCircuitCubic source for an example of how we add border_width and border_color as animatable
attributes, thereby inheriting all of the functionality to automatically animate changes in them.
