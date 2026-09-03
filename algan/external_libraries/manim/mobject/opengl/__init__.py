"""Stand-ins for Manim's OpenGL Mobject tree, which Algan does not vendor.

Upstream carries a parallel ~7000-line hierarchy (``OpenGLMobject`` and
friends) that exists to serve ``manim --renderer=opengl``. The vendored subset
has no renderer at all, so none of it can ever run; what the geometry modules
actually mention is a metaclass plus a handful of classes used in
``isinstance`` checks and annotations.

The metaclass here is the real one minus its renderer branch. The classes are
inert: nothing is ever an instance of them, which is exactly the answer those
``isinstance`` checks want under the Cairo-shaped code path, and constructing
one says why it cannot work.
"""
