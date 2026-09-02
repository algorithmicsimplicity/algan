"""Procedural texture images -- checkerboards, stripes, dots, gradients, noise.

Every function here returns an ``[W, H, 5]`` :class:`~algan.constants.color.Color`
image, which is exactly what a :class:`~algan.mobs.surfaces.surface.Surface`
takes as its ``color_texture`` (and, one channel of it, as a
``roughness_texture`` or a ``glow_texture``)::

    from algan import *

    Sphere(color_texture=get_checkerboard((RED, WHITE))).spawn()

The first axis of the image is the surface's ``u`` parameter and the second its
``v``, matching :attr:`~algan.mobs.surfaces.surface.Surface.color_texture`.
Because the pattern lives in a texture rather than in the mesh, its detail is
independent of the surface's grid resolution: a flat plane sampled at two
vertices per axis carries as fine a checkerboard as a densely tessellated
sphere, and deforming the surface carries the pattern along with it.

Every generator takes a ``texture_resolution`` -- the size of the image in
texels -- separately from the count of pattern cells. The renderer samples the
map bilinearly, so a pattern needs several texels per cell to read as a hard
edge rather than a gradient; the defaults give 32.
"""

from __future__ import annotations

import math

import torch

from algan.constants.color import BLACK, GRAY_D, WHITE, Color, to_color
from algan.errors import AlganConfigurationError

__all__ = [
    "get_bricks",
    "get_checkerboard",
    "get_gradient",
    "get_grid_lines",
    "get_noise",
    "get_polka_dots",
    "get_radial_gradient",
    "get_stripes",
]

#: Texels spent on one pattern cell when ``texture_resolution`` is left to the
#: generator. The renderer samples a color map bilinearly, so a checker square
#: one texel wide would arrive as a smooth gradient rather than a square; 32
#: keeps the interior flat and confines the blend to the cell boundary.
_TEXELS_PER_CELL = 32

#: Bounds on the generated image, per axis. The floor keeps a coarse pattern
#: from being stored at a resolution the bilinear sample would visibly soften;
#: the ceiling keeps a fine one from silently allocating a 4096-square map (a
#: texture is one animatable attribute row, so its texels are paid for on every
#: rendered frame window).
_MIN_TEXTURE_RESOLUTION = 64
_MAX_TEXTURE_RESOLUTION = 1024


def _as_texel(color) -> torch.Tensor:
    """One user-supplied color as a plain ``(5,)`` RGBA+glow row."""
    value = to_color(color)
    if value is None or not torch.is_tensor(value):
        raise AlganConfigurationError(
            f"Expected a color -- a Color such as RED, a hex string, a hex int "
            f"or an RGB tuple -- got {color!r}."
        )
    value = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if value.shape[0] != 5:
        raise AlganConfigurationError(
            f"Expected one color, got a value with {value.shape[0]} channels."
        )
    return value


def _palette(colors, argument: str = "colors") -> torch.Tensor:
    """A color or a sequence of them as a ``(n, 5)`` stack of texels."""
    if colors is None:
        raise AlganConfigurationError(f"`{argument}` needs at least one color.")
    value = to_color(colors)
    # ``to_color`` turns every single-color spelling -- a name, a hex string, a
    # hex int, an (r, g, b) tuple -- into one 5-wide row, so a tensor coming
    # back here is one color (1-D) or a ready-made stack of them (2-D). A list
    # of colors is not a color, and comes back unconverted to be walked below.
    if torch.is_tensor(value):
        texels = value.reshape(1, -1) if value.dim() == 1 else value
        if texels.dim() != 2 or texels.shape[-1] != 5:
            raise AlganConfigurationError(
                f"`{argument}` must be a color or a sequence of colors, got a "
                f"tensor of shape {tuple(value.shape)}."
            )
        return torch.as_tensor(texels, dtype=torch.float32)
    try:
        sequence = list(colors)
    except TypeError:
        raise AlganConfigurationError(
            f"`{argument}` must be a color or a sequence of colors, got {colors!r}."
        ) from None
    if not sequence:
        raise AlganConfigurationError(f"`{argument}` needs at least one color.")
    return torch.stack([_as_texel(color) for color in sequence])


def _count_pair(value, argument: str) -> tuple[int, int]:
    """A cell count given as one int or as ``(u_count, v_count)``."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise AlganConfigurationError(
                f"`{argument}` takes one number or a (u, v) pair, got {value!r}."
            )
        counts = tuple(int(v) for v in value)
    else:
        counts = (int(value), int(value))
    if min(counts) < 1:
        raise AlganConfigurationError(
            f"`{argument}` must be at least 1 along each axis, got {value!r}."
        )
    return counts


def _texture_size(texture_resolution, counts) -> tuple[int, int]:
    """The output image size, defaulted from the pattern's cell counts."""
    if texture_resolution is not None:
        size = _count_pair(texture_resolution, "texture_resolution")
        if min(size) < 2:
            raise AlganConfigurationError(
                f"`texture_resolution` must be at least 2 texels along each "
                f"axis, got {texture_resolution!r}."
            )
        return size
    return tuple(
        min(
            _MAX_TEXTURE_RESOLUTION,
            max(_MIN_TEXTURE_RESOLUTION, count * _TEXELS_PER_CELL),
        )
        for count in counts
    )


def _uv_grid(size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Texel-center ``(u, v)`` coordinates in ``[0, 1)``, each ``[W, H]``."""
    width, height = size
    u = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    v = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    return u.reshape(-1, 1).expand(width, height), v.reshape(1, -1).expand(
        width, height
    )


def _texel_width(size: tuple[int, int]) -> float:
    """One texel as a fraction of the unit square, for edge softening."""
    return 1.0 / min(size)


def _coverage(signed_distance: torch.Tensor, softness: float) -> torch.Tensor:
    """Smoothstep a signed distance field into a 0..1 mask.

    Positive distances are inside the shape. The transition is one texel wide
    rather than a hard threshold, so a circle's edge does not come out as a
    staircase of texels at the sizes these defaults produce.
    """
    if softness <= 0:
        return (signed_distance > 0).float()
    t = (signed_distance / softness + 0.5).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _mix(palette: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Interpolate a palette across ``t`` in ``[0, 1]``, one stop per color."""
    if palette.shape[0] == 1:
        return palette[0].expand(*t.shape, 5).contiguous()
    scaled = t.clamp(0, 1) * (palette.shape[0] - 1)
    low = scaled.floor().clamp(0, palette.shape[0] - 2)
    fraction = (scaled - low).unsqueeze(-1)
    low = low.long()
    return torch.lerp(palette[low], palette[low + 1], fraction)


def _blend(background: torch.Tensor, foreground: torch.Tensor, mask: torch.Tensor):
    """Lay ``foreground`` over ``background`` by a 0..1 per-texel mask."""
    return torch.lerp(background, foreground, mask.unsqueeze(-1))


def _as_texture(image: torch.Tensor) -> Color:
    """Hand a finished ``[W, H, 5]`` image back as a Color."""
    return image.contiguous().as_subclass(Color)


def get_checkerboard(
    colors=(WHITE, BLACK),
    resolution: int | tuple[int, int] = 8,
    *,
    texture_resolution: int | tuple[int, int] | None = None,
) -> Color:
    """Build a checkerboard texture, as an ``[W, H, 5]`` color image.

    The square a texel falls in is colored by ``(u_index + v_index) % len(colors)``,
    so two colors give the usual checkerboard and three or more give diagonal
    stripes.

    Pass the result as a surface's ``color_texture``. The pattern is carried in
    the surface's ``(u, v)`` domain, so it is as fine on a flat two-triangle
    quad as on a densely tessellated sphere, and it follows the surface as it
    deforms.

    Parameters
    ----------
    colors
        The colors to alternate between: a sequence of two or more, or a single
        color for a plain fill. Each is an Algan
        :class:`~algan.constants.color.Color`, a named constant such as
        ``BLUE``, or anything ``Color()`` accepts. Defaults to
        ``(WHITE, BLACK)``.
    resolution
        How many squares the pattern has, as ``(u_squares, v_squares)`` or one
        number for both. Defaults to ``8``, an 8x8 board.
    texture_resolution
        Size of the generated image in texels, as ``(width, height)`` or one
        number for both. Defaults to ``None``, meaning 32 texels per square,
        bounded to ``64``--``1024`` per axis.

    Returns
    -------
    :class:`~algan.constants.color.Color`
        The texture, shape ``[W, H, 5]``: the first axis is ``u``, the second
        ``v``, and the five channels are ``(R, G, B, glow, alpha)``.

    Raises
    ------
    :class:`~algan.errors.AlganConfigurationError`
        If ``colors`` holds no colors, or a count is below its minimum.

    See Also
    --------
    :func:`get_stripes` : Bands rather than squares.
    :attr:`~algan.mobs.surfaces.surface.Surface.color_texture` : Where the image goes.
    :meth:`~algan.mobs.surfaces.surface.Surface.set_checkerboard_colors` : The same
        look painted on the mesh's own vertices instead.

    Examples
    --------
    A checkered sphere, and a flat plane carrying the same board -- the pattern
    comes from the texture, so the two-triangle plane is as detailed as the
    sphere:

    .. algan:: Example1GetCheckerboard
        :save_last_frame:

        from algan import *
        import torch

        def plane(uv):
            xy = (uv - 0.5) * 2.4
            return torch.cat((xy, torch.zeros_like(uv[..., :1])), -1)

        checker = get_checkerboard((RED, WHITE), resolution=6)
        Sphere(radius=1.2, color_texture=checker).move(LEFT * 1.6).spawn()
        Surface(plane, grid_width=2, grid_height=2, color_texture=checker).move(
            RIGHT * 1.6
        ).spawn()

        Scene.save_video()
    """
    palette = _palette(colors)
    counts = _count_pair(resolution, "resolution")
    size = _texture_size(texture_resolution, counts)
    # Integer square indices rather than a threshold on a float coordinate:
    # every texel of a square then lands on exactly the same color, whatever
    # the ratio between the image size and the square count.
    u_index = torch.arange(size[0]) * counts[0] // size[0]
    v_index = torch.arange(size[1]) * counts[1] // size[1]
    index = (u_index.reshape(-1, 1) + v_index.reshape(1, -1)) % palette.shape[0]
    return _as_texture(palette[index])


def get_stripes(
    colors=(WHITE, BLACK),
    resolution: int = 8,
    *,
    angle: float = 0.0,
    texture_resolution: int | tuple[int, int] | None = None,
) -> Color:
    """Build a texture of parallel stripes, as an ``[W, H, 5]`` color image.

    Parameters
    ----------
    colors
        The colors to cycle through across the stripes, or a single color for a
        plain fill. Each is an Algan :class:`~algan.constants.color.Color`, a
        named constant such as ``BLUE``, or anything ``Color()`` accepts.
        Defaults to ``(WHITE, BLACK)``.
    resolution
        How many stripes the pattern has across the whole domain. Defaults to
        ``8``.
    angle
        Direction the stripes run, in degrees, measured from the ``v`` axis
        towards ``u``. Defaults to ``0``, giving stripes that run along ``v``
        and band across ``u`` -- lines of longitude on a sphere. ``90`` gives
        the other axis.
    texture_resolution
        Size of the generated image in texels, as ``(width, height)`` or one
        number for both. Defaults to ``None``, meaning 32 texels per stripe,
        bounded to ``64``--``1024`` per axis.

    Returns
    -------
    :class:`~algan.constants.color.Color`
        The texture, shape ``[W, H, 5]``.

    Raises
    ------
    :class:`~algan.errors.AlganConfigurationError`
        If ``colors`` holds no colors, or a count is below its minimum.

    See Also
    --------
    :func:`get_checkerboard` : Squares rather than bands.

    Examples
    --------
    A barber-pole cylinder:

    .. algan:: Example1GetStripes
        :save_last_frame:

        from algan import *

        Cylinder(
            radius=0.9,
            height=2.0,
            color_texture=get_stripes((RED, WHITE), resolution=12, angle=30),
        ).rotate(20, RIGHT).spawn()

        Scene.save_video()
    """
    palette = _palette(colors)
    count = _count_pair(resolution, "resolution")[0]
    size = _texture_size(texture_resolution, (count, count))
    u, v = _uv_grid(size)
    radians = math.radians(float(angle))
    # The stripes run along ``angle``, so the coordinate that varies ACROSS
    # them is the perpendicular one.
    across = u * math.cos(radians) + v * math.sin(radians)
    index = torch.floor(across * count).long() % palette.shape[0]
    return _as_texture(palette[index])


def get_grid_lines(
    line_color=WHITE,
    background_color=BLACK,
    resolution: int | tuple[int, int] = 8,
    *,
    line_width: float = 0.08,
    texture_resolution: int | tuple[int, int] | None = None,
) -> Color:
    """Build a texture of grid lines over a flat background.

    The lines sit on the cell boundaries, so a graph-paper surface reads its
    own ``(u, v)`` parameterization: latitude and longitude on a sphere, a wire
    grid on a plotted saddle.

    Parameters
    ----------
    line_color
        Color of the lines. An Algan :class:`~algan.constants.color.Color`, a
        named constant such as ``BLUE``, or anything ``Color()`` accepts.
        Defaults to ``WHITE``.
    background_color
        Color of the cells between them, in the same forms. Defaults to
        ``BLACK``. Pass ``TRANSPARENT`` for lines over a see-through surface.
    resolution
        How many cells the grid has, as ``(u_cells, v_cells)`` or one number for
        both. Defaults to ``8``.
    line_width
        Thickness of a line as a fraction of one cell, from ``0`` to ``1``.
        Defaults to ``0.08``.
    texture_resolution
        Size of the generated image in texels, as ``(width, height)`` or one
        number for both. Defaults to ``None``, meaning 32 texels per cell,
        bounded to ``64``--``1024`` per axis.

    Returns
    -------
    :class:`~algan.constants.color.Color`
        The texture, shape ``[W, H, 5]``.

    Raises
    ------
    :class:`~algan.errors.AlganConfigurationError`
        If a color is unreadable, or a count is below its minimum.

    Examples
    --------
    A wireframe globe:

    .. algan:: Example1GetGridLines
        :save_last_frame:

        from algan import *

        Sphere(
            radius=1.5,
            color_texture=get_grid_lines(BLUE_B, BLUE_E, resolution=(12, 6)),
        ).rotate(20, RIGHT).spawn()

        Scene.save_video()
    """
    line = _as_texel(line_color)
    background = _as_texel(background_color)
    counts = _count_pair(resolution, "resolution")
    size = _texture_size(texture_resolution, counts)
    u, v = _uv_grid(size)
    width = float(line_width)
    # Distance to the nearest cell boundary, in cells, along each axis: a texel
    # is on a line when either is inside half the line width.
    du = ((u * counts[0] + 0.5) % 1.0 - 0.5).abs()
    dv = ((v * counts[1] + 0.5) % 1.0 - 0.5).abs()
    softness = _texel_width(size) * max(counts)
    mask = torch.maximum(
        _coverage(width * 0.5 - du, softness),
        _coverage(width * 0.5 - dv, softness),
    )
    image = _blend(background.expand(*u.shape, 5), line.expand(*u.shape, 5), mask)
    return _as_texture(image)


def get_polka_dots(
    dot_color=WHITE,
    background_color=BLACK,
    resolution: int | tuple[int, int] = 8,
    *,
    radius: float = 0.3,
    texture_resolution: int | tuple[int, int] | None = None,
) -> Color:
    """Build a texture of evenly spaced dots over a flat background.

    Parameters
    ----------
    dot_color
        Color of the dots. An Algan :class:`~algan.constants.color.Color`, a
        named constant such as ``BLUE``, or anything ``Color()`` accepts.
        Defaults to ``WHITE``.
    background_color
        Color behind them, in the same forms. Defaults to ``BLACK``.
    resolution
        How many dots the pattern has, as ``(u_dots, v_dots)`` or one number for
        both. Defaults to ``8``.
    radius
        Radius of a dot as a fraction of one cell, from ``0`` to ``0.5`` (at
        ``0.5`` neighbouring dots touch). Defaults to ``0.3``.
    texture_resolution
        Size of the generated image in texels, as ``(width, height)`` or one
        number for both. Defaults to ``None``, meaning 32 texels per dot,
        bounded to ``64``--``1024`` per axis.

    Returns
    -------
    :class:`~algan.constants.color.Color`
        The texture, shape ``[W, H, 5]``.

    Raises
    ------
    :class:`~algan.errors.AlganConfigurationError`
        If a color is unreadable, or a count is below its minimum.

    Examples
    --------
    A spotted ball:

    .. algan:: Example1GetPolkaDots
        :save_last_frame:

        from algan import *

        Sphere(
            radius=1.5,
            color_texture=get_polka_dots(YELLOW, PURPLE, resolution=(10, 5)),
        ).rotate(20, RIGHT).spawn()

        Scene.save_video()
    """
    dot = _as_texel(dot_color)
    background = _as_texel(background_color)
    counts = _count_pair(resolution, "resolution")
    size = _texture_size(texture_resolution, counts)
    u, v = _uv_grid(size)
    du = (u * counts[0]) % 1.0 - 0.5
    dv = (v * counts[1]) % 1.0 - 0.5
    distance = torch.sqrt(du * du + dv * dv)
    softness = _texel_width(size) * max(counts)
    mask = _coverage(float(radius) - distance, softness)
    image = _blend(background.expand(*u.shape, 5), dot.expand(*u.shape, 5), mask)
    return _as_texture(image)


def get_bricks(
    colors=("#9C4A32", "#B35A3F"),
    resolution: int | tuple[int, int] = (6, 12),
    *,
    mortar_color=GRAY_D,
    mortar_width: float = 0.06,
    offset: float = 0.5,
    texture_resolution: int | tuple[int, int] | None = None,
) -> Color:
    """Build a running-bond brick texture, as an ``[W, H, 5]`` color image.

    Every other course is shifted along ``u`` so the joints do not line up,
    which is what makes a brick wall read as one rather than as a grid.

    Parameters
    ----------
    colors
        The colors the bricks cycle through, or a single color for a uniform
        wall. Each is an Algan :class:`~algan.constants.color.Color`, a named
        constant such as ``BLUE``, or anything ``Color()`` accepts. Defaults to
        two clay browns, ``("#9C4A32", "#B35A3F")``.
    resolution
        How many bricks the wall has, as ``(u_bricks, v_courses)`` or one number
        for both. Defaults to ``(6, 12)``, six bricks across each of twelve
        courses.
    mortar_color
        Color of the joints between them. Defaults to ``GRAY_D``.
    mortar_width
        Thickness of a joint as a fraction of one brick, from ``0`` to ``1``.
        Defaults to ``0.06``.
    offset
        How far each alternate course is shifted along ``u``, as a fraction of
        one brick. Defaults to ``0.5``, the half-brick running bond; ``0``
        stacks the courses.
    texture_resolution
        Size of the generated image in texels, as ``(width, height)`` or one
        number for both. Defaults to ``None``, meaning 32 texels per brick,
        bounded to ``64``--``1024`` per axis.

    Returns
    -------
    :class:`~algan.constants.color.Color`
        The texture, shape ``[W, H, 5]``.

    Raises
    ------
    :class:`~algan.errors.AlganConfigurationError`
        If ``colors`` holds no colors, or a count is below its minimum.

    Examples
    --------
    A brick column:

    .. algan:: Example1GetBricks
        :save_last_frame:

        from algan import *

        Cylinder(
            radius=0.9,
            height=2.5,
            color_texture=get_bricks(resolution=(8, 10)),
        ).rotate(15, RIGHT).spawn()

        Scene.save_video()
    """
    palette = _palette(colors)
    mortar = _as_texel(mortar_color)
    counts = _count_pair(resolution, "resolution")
    size = _texture_size(texture_resolution, counts)
    u, v = _uv_grid(size)
    course = torch.floor(v * counts[1])
    shifted = u * counts[0] + (course % 2) * float(offset)
    column = torch.floor(shifted)
    index = (column.long() + course.long()) % palette.shape[0]
    du = (shifted % 1.0 - 0.5).abs()
    dv = ((v * counts[1]) % 1.0 - 0.5).abs()
    softness = _texel_width(size) * max(counts)
    joint = torch.maximum(
        _coverage(du - (0.5 - float(mortar_width) * 0.5), softness),
        _coverage(dv - (0.5 - float(mortar_width) * 0.5), softness),
    )
    image = _blend(palette[index], mortar.expand(*u.shape, 5), joint)
    return _as_texture(image)


def get_gradient(
    colors=(BLACK, WHITE),
    *,
    angle: float = 0.0,
    texture_resolution: int | tuple[int, int] | None = None,
) -> Color:
    """Build a linear color ramp, as an ``[W, H, 5]`` color image.

    The colors are spread evenly along the ramp, one stop each, and interpolated
    between -- all five channels of them, so a ramp from an opaque color to
    ``TRANSPARENT`` fades a surface out along an axis.

    Parameters
    ----------
    colors
        The color stops, in order along the ramp. Each is an Algan
        :class:`~algan.constants.color.Color`, a named constant such as
        ``BLUE``, or anything ``Color()`` accepts. Defaults to
        ``(BLACK, WHITE)``.
    angle
        Direction the ramp runs, in degrees, measured from the ``u`` axis
        towards ``v``. Defaults to ``0``, running along ``u``.
    texture_resolution
        Size of the generated image in texels, as ``(width, height)`` or one
        number for both. Defaults to ``None``, meaning ``256`` square.

    Returns
    -------
    :class:`~algan.constants.color.Color`
        The texture, shape ``[W, H, 5]``.

    Raises
    ------
    :class:`~algan.errors.AlganConfigurationError`
        If ``colors`` holds no colors.

    See Also
    --------
    :func:`get_radial_gradient` : The same ramp spread out from a point.

    Examples
    --------
    A sunset-graded sphere:

    .. algan:: Example1GetGradient
        :save_last_frame:

        from algan import *

        Sphere(
            radius=1.5,
            color_texture=get_gradient((PURPLE_E, RED, YELLOW), angle=90),
        ).rotate(20, RIGHT).spawn()

        Scene.save_video()
    """
    palette = _palette(colors)
    size = _texture_size(texture_resolution, (8, 8))
    u, v = _uv_grid(size)
    radians = math.radians(float(angle))
    along = u * math.cos(radians) + v * math.sin(radians)
    # Normalize over the projection of the unit square onto the ramp direction
    # so the full palette is spent whatever the angle is.
    span = abs(math.cos(radians)) + abs(math.sin(radians))
    low = min(0.0, math.cos(radians)) + min(0.0, math.sin(radians))
    return _as_texture(_mix(palette, (along - low) / span))


def get_radial_gradient(
    colors=(WHITE, BLACK),
    *,
    center: tuple[float, float] = (0.5, 0.5),
    radius: float = 0.5,
    texture_resolution: int | tuple[int, int] | None = None,
) -> Color:
    """Build a radial color ramp, as an ``[W, H, 5]`` color image.

    The first color sits at the center and the last at ``radius`` and beyond.

    Parameters
    ----------
    colors
        The color stops, from the center outwards. Each is an Algan
        :class:`~algan.constants.color.Color`, a named constant such as
        ``BLUE``, or anything ``Color()`` accepts. Defaults to
        ``(WHITE, BLACK)``.
    center
        Where the ramp starts, as ``(u, v)`` in ``[0, 1]``. Defaults to
        ``(0.5, 0.5)``, the middle of the domain.
    radius
        Distance from the center at which the last color is reached, in the
        same ``[0, 1]`` units. Defaults to ``0.5``.
    texture_resolution
        Size of the generated image in texels, as ``(width, height)`` or one
        number for both. Defaults to ``None``, meaning ``256`` square.

    Returns
    -------
    :class:`~algan.constants.color.Color`
        The texture, shape ``[W, H, 5]``.

    Raises
    ------
    :class:`~algan.errors.AlganConfigurationError`
        If ``colors`` holds no colors.

    Examples
    --------
    A torus lit from the middle of its texture domain:

    .. algan:: Example1GetRadialGradient
        :save_last_frame:

        from algan import *

        Torus(
            ring_radius=1.4,
            tube_radius=0.5,
            color_texture=get_radial_gradient((YELLOW, RED_E)),
        ).rotate(60, RIGHT).spawn()

        Scene.save_video()
    """
    palette = _palette(colors)
    size = _texture_size(texture_resolution, (8, 8))
    u, v = _uv_grid(size)
    distance = torch.sqrt((u - float(center[0])) ** 2 + (v - float(center[1])) ** 2)
    scale = float(radius)
    if scale <= 0:
        raise AlganConfigurationError(f"`radius` must be positive, got {radius!r}.")
    return _as_texture(_mix(palette, distance / scale))


def get_noise(
    colors=(BLACK, WHITE),
    resolution: int | tuple[int, int] = 8,
    *,
    octaves: int = 4,
    persistence: float = 0.5,
    seed: int | None = None,
    texture_resolution: int | tuple[int, int] | None = None,
) -> Color:
    """Build a fractal value-noise texture, as an ``[W, H, 5]`` color image.

    Smooth random values are drawn on a lattice and summed over successively
    finer, weaker octaves -- clouds, marble, rust, a rough-looking roughness
    map. The lattice wraps, so the image tiles seamlessly on an axis where the
    surface closes on itself.

    Parameters
    ----------
    colors
        The color stops the noise value is mapped through, from lowest to
        highest. Each is an Algan :class:`~algan.constants.color.Color`, a named
        constant such as ``BLUE``, or anything ``Color()`` accepts. Defaults to
        ``(BLACK, WHITE)``, a greyscale field.
    resolution
        Lattice size of the coarsest octave, as ``(u_cells, v_cells)`` or one
        number for both. Defaults to ``8``.
    octaves
        How many times the lattice is doubled and added in at half the weight.
        Defaults to ``4``. ``1`` gives plain smooth blobs.
    persistence
        How much weight each octave keeps relative to the one before, from ``0``
        to ``1``. Defaults to ``0.5``; higher is rougher.
    seed
        Seed for the random lattice, so a texture can be reproduced. Defaults to
        ``None``, meaning a fresh pattern on every call.
    texture_resolution
        Size of the generated image in texels, as ``(width, height)`` or one
        number for both. Defaults to ``None``, meaning 32 texels per lattice
        cell, bounded to ``64``--``1024`` per axis.

    Returns
    -------
    :class:`~algan.constants.color.Color`
        The texture, shape ``[W, H, 5]``.

    Raises
    ------
    :class:`~algan.errors.AlganConfigurationError`
        If ``colors`` holds no colors, or a count is below its minimum.

    Examples
    --------
    A cloudy planet:

    .. algan:: Example1GetNoise
        :save_last_frame:

        from algan import *

        Sphere(
            radius=1.5,
            color_texture=get_noise((BLUE_E, WHITE), resolution=6, seed=7),
        ).rotate(20, RIGHT).spawn()

        Scene.save_video()
    """
    palette = _palette(colors)
    counts = _count_pair(resolution, "resolution")
    size = _texture_size(texture_resolution, counts)
    octaves = max(1, int(octaves))
    # Drawn on the CPU and moved: torch's default device may be CUDA, and a
    # seeded generator has to be built for the device it draws on, so seeding
    # on CPU is the one spelling that reproduces a texture on every machine.
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
    u, v = _uv_grid(size)
    total = torch.zeros(size, dtype=torch.float32)
    weight_sum = 0.0
    weight = 1.0
    for octave in range(octaves):
        cells = (counts[0] * 2**octave, counts[1] * 2**octave)
        # One extra row and column, copied from the first, so the lattice wraps
        # and the image tiles.
        lattice = torch.rand(cells, generator=generator, device="cpu").to(u.device)
        lattice = torch.cat((lattice, lattice[:1]), 0)
        lattice = torch.cat((lattice, lattice[:, :1]), 1)
        total = total + weight * _sample_lattice(lattice, u, v, cells)
        weight_sum += weight
        weight *= float(persistence)
    return _as_texture(_mix(palette, total / weight_sum))


def _sample_lattice(lattice, u, v, cells) -> torch.Tensor:
    """Smoothly interpolate a ``(cells + 1)`` value lattice at ``(u, v)``."""
    x = u * cells[0]
    y = v * cells[1]
    x0 = x.floor()
    y0 = y.floor()
    # Smoothstep the cell-local coordinate rather than lerping it directly:
    # plain bilinear interpolation leaves visible creases on the lattice lines.
    fx = (x - x0).clamp(0, 1)
    fy = (y - y0).clamp(0, 1)
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    x0 = x0.long().clamp(0, cells[0] - 1)
    y0 = y0.long().clamp(0, cells[1] - 1)
    c00 = lattice[x0, y0]
    c10 = lattice[x0 + 1, y0]
    c01 = lattice[x0, y0 + 1]
    c11 = lattice[x0 + 1, y0 + 1]
    return torch.lerp(torch.lerp(c00, c10, fx), torch.lerp(c01, c11, fx), fy)
