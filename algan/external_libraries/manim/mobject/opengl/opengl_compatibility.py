from __future__ import annotations

from abc import ABCMeta

#from algan.external_libraries.manim import config
#from algan.external_libraries.manim.mobject.opengl.opengl_mobject import OpenGLMobject
#from algan.external_libraries.manim.mobject.opengl.opengl_point_cloud_mobject import OpenGLPMobject
#from algan.external_libraries.manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject

from ...constants import RendererType

__all__ = ["ConvertToOpenGL"]


class ConvertToOpenGL(ABCMeta):
    """Metaclass for swapping (V)Mobject with its OpenGL counterpart at runtime
    depending on config.renderer. This metaclass should only need to be inherited
    on the lowest order inheritance classes such as Mobject and VMobject.
    """

    _converted_classes = []

    def __new__(mcls, name, bases, namespace):  # noqa: B902
        return super().__new__(mcls, name, bases, namespace)

    def __init__(cls, name, bases, namespace):  # noqa: B902
        super().__init__(name, bases, namespace)
        cls._converted_classes.append(cls)
