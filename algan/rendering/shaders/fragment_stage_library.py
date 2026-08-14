"""Ready-to-use fragment stages.

Where :mod:`algan.constants.material_presets` collects material *presets*, this
collects composable :class:`~algan.rendering.shaders.fragment_shaders.FragmentStage`
looks. Every stage here is **additive**: it adds to the colour it is handed
rather than replacing it, so it layers over a lit base material::

    from algan import *

    ball = Sphere(radius=1, color=BLUE_E)
    ball.set_fragment_shader([standard_shader, fresnel_rim])   # lit, then rimmed
    ball.rim_color = (0.40, 0.90, 1.00)   # width-3 RGB, not a 5-channel Color
    ball.rim_power = 3.0
    ball.spawn()

Each stage's parameters become ordinary animatable attributes of the Mob, so
``ball.rim_gain = 2`` animates like any other attribute.

Why these are stages rather than material parameters
----------------------------------------------------
A Fresnel rim is an authoring control, not a BSDF term: Three.js, Unreal, Unity
and Blender all leave it out of the physically-based material and expose it
through a shader graph or a custom shader. Algan's fragment pipeline is that
escape hatch, and keeping the rim here leaves
:class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial` a faithful mirror
of its Three.js counterpart.

Note that a *physically* lit glass ball -- dark body, bright rim -- does not need
any of this: give it ``MeshPhysicalMaterial(transmission=...)`` and an
environment map bright in the directions the camera cannot see (see
:meth:`~algan.scene.Scene.set_environment_map`). These stages are for when you
want the look without authoring an environment, or want it stylised.
"""

import taichi as ti

from algan.rendering.shaders.fragment_shaders import FragmentStage

__all__ = ["fresnel_rim", "glass_ball"]


@ti.func
def _stage_fresnel_rim(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                       params: ti.template(), f, prim, off,
                       light_pos: ti.template(), light_col: ti.template(),
                       num_lights, shadows: ti.template(), vis):
    """Add ``rim_color * rim_gain * (1 - |N.V|) ** rim_power``."""
    tm = f % params.shape[0]
    rim = ti.math.vec3(params[tm, prim, off + 0],
                       params[tm, prim, off + 1],
                       params[tm, prim, off + 2])
    power = params[tm, prim, off + 3]
    gain = params[tm, prim, off + 4]

    v = view_dir.normalized()
    n = n_interp.normalized()
    fres = ti.math.clamp(1.0 - ti.abs(n.dot(v)), 0.0, 1.0)
    acc = in_rgb + rim * (gain * ti.pow(fres, ti.max(power, 1e-3)))
    return ti.math.vec4(acc[0], acc[1], acc[2], in_glow)


#: Classic Fresnel rim light: brightens the silhouette, leaves the middle alone.
#: ``rim_power`` controls how tightly it hugs the edge (1 = broad wash, 6 = thin
#: outline); ``rim_gain`` scales it. Compose *after* a lighting stage, e.g.
#: ``mob.set_fragment_shader([standard_shader, fresnel_rim])``.
fresnel_rim = FragmentStage(
    _stage_fresnel_rim,
    [
        ("rim_color", 3, (0.40, 0.70, 1.00)),
        ("rim_power", 1, 3.0),
        ("rim_gain", 1, 1.0),
    ],
)


@ti.func
def _stage_glass_ball(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                      params: ti.template(), f, prim, off,
                      light_pos: ti.template(), light_col: ti.template(),
                      num_lights, shadows: ti.template(), vis):
    """Studio glass-ball edge: two Fresnel lobes, a silhouette ring and two
    screen-space specular blobs, all added to the incoming colour.
    """
    tm = f % params.shape[0]
    rim = ti.math.vec3(params[tm, prim, off + 0],
                       params[tm, prim, off + 1],
                       params[tm, prim, off + 2])
    rim_power = params[tm, prim, off + 3]
    edge = ti.math.vec3(params[tm, prim, off + 4],
                        params[tm, prim, off + 5],
                        params[tm, prim, off + 6])
    edge_power = params[tm, prim, off + 7]
    ring = ti.math.vec3(params[tm, prim, off + 8],
                        params[tm, prim, off + 9],
                        params[tm, prim, off + 10])
    ring_center = params[tm, prim, off + 11]
    ring_width = params[tm, prim, off + 12]
    anisotropy = params[tm, prim, off + 13]
    key = ti.math.vec3(params[tm, prim, off + 14],
                       params[tm, prim, off + 15],
                       params[tm, prim, off + 16])
    key_gain = params[tm, prim, off + 17]
    key_x = params[tm, prim, off + 18]
    key_y = params[tm, prim, off + 19]
    key_angle = params[tm, prim, off + 20]
    key_long = params[tm, prim, off + 21]
    key_wide = params[tm, prim, off + 22]
    fill_gain = params[tm, prim, off + 23]
    fill_x = params[tm, prim, off + 24]
    fill_y = params[tm, prim, off + 25]
    fill_size = params[tm, prim, off + 26]

    v = view_dir.normalized()
    n = n_interp.normalized()
    if n.dot(v) < 0.0:
        # Face the viewer, so the screen-space highlight positions keep their side.
        n = -n
    fres = ti.math.clamp(1.0 - n.dot(v), 0.0, 1.0)

    # Edge anisotropy: real studio lighting makes the top and bottom of a ball
    # brighter than its sides. Weight by |sin(azimuth)|, normalised so the mean
    # over azimuth stays 1 (mean |sin| = 2/pi) -- turning it up therefore
    # redistributes the edge rather than brightening the whole ball.
    planar = ti.sqrt(n[0] * n[0] + n[1] * n[1])
    azimuth = ti.abs(n[1]) / ti.max(planar, 1e-3)
    edge_w = (1.0 - anisotropy * 0.6366198) + anisotropy * azimuth

    acc = in_rgb
    acc += rim * (ti.pow(fres, ti.max(rim_power, 1e-3)) * edge_w)
    acc += edge * (ti.pow(fres, ti.max(edge_power, 1e-3)) * edge_w)
    acc += ring * (ti.exp(-((fres - ring_center)
                            / ti.max(ring_width, 1e-3)) ** 2) * edge_w)

    # On a sphere the shading normal's x/y components *are* the screen offset in
    # units of the radius, so the highlights are placed in those coordinates.
    ca = ti.cos(key_angle)
    sa = ti.sin(key_angle)
    dx = n[0] - key_x
    dy = n[1] - key_y
    along = dx * sa + dy * (-ca)
    across = dx * ca + dy * sa
    acc += key * (key_gain
                  * ti.exp(-(along / ti.max(key_long, 1e-3)) ** 2
                           - (across / ti.max(key_wide, 1e-3)) ** 2))

    ex = n[0] - fill_x
    ey = n[1] - fill_y
    acc += key * (fill_gain
                  * ti.exp(-(ex * ex + ey * ey)
                           / ti.max(fill_size * fill_size, 1e-6)))

    return ti.math.vec4(acc[0], acc[1], acc[2], in_glow)


#: The studio glass-ball look: a dark body with a bright, crisp silhouette and a
#: comma-shaped specular streak, as seen in stock "glass sphere" renders.
#:
#: Two Fresnel lobes shape the falloff (``rim_*`` is the broad one, ``edge_*``
#: the sharp one), ``ring_*`` adds a Gaussian band at a chosen distance from the
#: silhouette (``ring_center`` is in ``1 - |N.V|``, so ~0.64 sits at 93% of the
#: radius), and ``anisotropy`` biases the edge toward the top and bottom without
#: changing its average. The two highlights are positioned in screen space:
#: ``key_x``/``key_y`` are fractions of the radius, right and up from the centre,
#: and ``key_angle`` (radians) tilts the streak.
#:
#: Compose over a lit base or use alone over a flat colour::
#:
#:     ball.set_fragment_shader(glass_ball)
#:     ball.edge_color = (0.6, 0.9, 1.0)
glass_ball = FragmentStage(
    _stage_glass_ball,
    [
        ("rim_color", 3, (0.03, 0.32, 0.52)),
        ("rim_power", 1, 0.75),
        ("edge_color", 3, (0.29, 0.16, 0.05)),
        ("edge_power", 1, 2.6),
        ("ring_color", 3, (0.07, 0.07, 0.07)),
        ("ring_center", 1, 0.66),
        ("ring_width", 1, 0.11),
        ("anisotropy", 1, 0.7),
        ("key_color", 3, (1.00, 1.02, 1.04)),
        ("key_gain", 1, 0.95),
        ("key_x", 1, 0.60),
        ("key_y", 1, 0.42),
        ("key_angle", 1, 0.39),
        ("key_long", 1, 0.24),
        ("key_wide", 1, 0.062),
        ("fill_gain", 1, 0.22),
        ("fill_x", 1, -0.44),
        ("fill_y", 1, 0.40),
        ("fill_size", 1, 0.10),
    ],
)
