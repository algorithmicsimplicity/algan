"""Color representation and Algan's named color palette.

Algan stores color in **five channels** -- red, green, blue, glow, alpha -- so
that emissive strength travels with a color the way opacity does. RGB and alpha
are in ``[0, 1]``; glow is an additive brightness fed to the bloom accumulator.

:class:`Color` accepts the forms you would expect -- a hex string such as
``"#58C4DD"``, or an ``(r, g, b)`` tuple -- with ``glow`` and ``opacity`` as
separate arguments. It subclasses :class:`torch.Tensor`, so a color takes part
in ordinary tensor arithmetic and can carry batch dimensions.
:meth:`Color.add_defaults` pads a bare RGB or RGBA tensor out to the five-channel
layout, which is how the API accepts the shorter forms wherever a color array is
expected.

The module then defines the palette exported by ``from algan import *``:
``BLACK``, ``WHITE``, the greys, and the ``BLUE``/``RED``/``GREEN``/``YELLOW``
families with their ``_A`` (lightest) through ``_E`` (darkest) shades. Both
British and American spellings of grey are provided. Two special values are worth
knowing: ``TRANSPARENT`` (invisible, and the background color that produces an
alpha-channel video) and ``GLOW`` (black with full glow).
"""

from __future__ import annotations

import torch

from algan.errors import InvalidColorError
from algan.settings._startup import _ANIMATION_DEVICE
from algan.utils.tensor_utils import broadcast, cast_to_tensor, unsqueeze_left

CSS_COLORS: dict[str, str] = {
    "black": "#000000",
    "white": "#FFFFFF",
    "red": "#FF0000",
    "green": "#008000",
    "blue": "#0000FF",
    "yellow": "#FFFF00",
    "cyan": "#00FFFF",
    "magenta": "#FF00FF",
    "gray": "#808080",
    "grey": "#808080",
    "darkgray": "#A9A9A9",
    "darkgrey": "#A9A9A9",
    "lightgray": "#D3D3D3",
    "lightgrey": "#D3D3D3",
    "orange": "#FFA500",
    "pink": "#FFC0CB",
    "purple": "#800080",
    "brown": "#A52A2A",
    "gold": "#FFD700",
    "silver": "#C0C0C0",
    "maroon": "#800000",
    "navy": "#000080",
    "olive": "#808000",
    "lime": "#00FF00",
    "teal": "#008080",
    "aqua": "#00FFFF",
    "coral": "#FF7F50",
    "salmon": "#FA8072",
    "violet": "#EE82EE",
    "indigo": "#4B0082",
    "turquoise": "#40E0D0",
    "transparent": "#00000000",
}


def _parse_color_string(s: str) -> tuple[tuple[float, float, float], float]:
    """Parse a hex string or standard CSS color name into ((r, g, b), opacity)."""
    raw = s.strip()
    key = raw.lower().replace(" ", "").replace("_", "").replace("-", "")
    if key in CSS_COLORS:
        raw = CSS_COLORS[key]

    hex_str = raw
    if hex_str.startswith("#"):
        hex_str = hex_str[1:]
    elif hex_str.lower().startswith("0x"):
        hex_str = hex_str[2:]

    if not hex_str or not all(c in "0123456789abcdefABCDEF" for c in hex_str):
        raise InvalidColorError(
            f"Invalid color string: {s!r}. Expected a hex color ('#RRGGBB', '#RGB', '#RRGGBBAA') "
            f"or a standard CSS color name (e.g. 'red', 'navy', 'coral')."
        )

    if len(hex_str) == 3 or len(hex_str) == 4:
        hex_str = "".join(c * 2 for c in hex_str)

    if len(hex_str) == 6:
        val = int(hex_str, 16)
        r = ((val >> 16) & 0xFF) / 255.0
        g = ((val >> 8) & 0xFF) / 255.0
        b = (val & 0xFF) / 255.0
        return (r, g, b), 1.0
    elif len(hex_str) == 8:
        val = int(hex_str, 16)
        r = ((val >> 24) & 0xFF) / 255.0
        g = ((val >> 16) & 0xFF) / 255.0
        b = ((val >> 8) & 0xFF) / 255.0
        a = (val & 0xFF) / 255.0
        return (r, g, b), a
    else:
        raise InvalidColorError(
            f"Invalid hex color length for {s!r}: expected 3, 4, 6, or 8 hex digits, got {len(hex_str)}."
        )


class Color(torch.Tensor):
    def __new__(
        cls,
        rgb: str | tuple[float, ...] | list[float] | torch.Tensor,
        glow=0,
        opacity=1,
        *args,
        **kwargs,
    ):
        if isinstance(rgb, str):
            rgb_tuple, extracted_opacity = _parse_color_string(rgb)
            rgb = rgb_tuple
            if opacity == 1 and extracted_opacity != 1.0:
                opacity = extracted_opacity
        elif isinstance(rgb, (tuple, list)):
            if len(rgb) == 4:
                rgb, opacity = tuple(rgb[:3]), rgb[3]
            elif len(rgb) == 5:
                rgb, glow, opacity = tuple(rgb[:3]), rgb[3], rgb[4]
            elif len(rgb) == 3:
                rgb = tuple(rgb)
        elif isinstance(rgb, torch.Tensor):
            t = rgb.reshape(-1)
            if t.numel() == 5:
                rgb, glow, opacity = (
                    (float(t[0]), float(t[1]), float(t[2])),
                    float(t[3]),
                    float(t[4]),
                )
            elif t.numel() == 4:
                rgb, opacity = (float(t[0]), float(t[1]), float(t[2])), float(t[3])
            elif t.numel() == 3:
                rgb = (float(t[0]), float(t[1]), float(t[2]))
        return (
            super()
            .__new__(cls, (*rgb, glow, opacity), *args, **kwargs)
            .to(_ANIMATION_DEVICE)
        )

    def __init__(self, rgb, glow=0, opacity=1, *args, **kwargs):
        pass  # super().__init__((red, green, blue, glow, opacity))

    def __eq__(self, other):
        eq = super().__eq__(other)
        if self.numel() == 5 and hasattr(other, "numel") and other.numel() == 5:
            eq = eq.all()
        return eq.as_subclass(torch.Tensor)

    @property
    def opacity(self):
        opacity = self.data[..., -1:].as_subclass(torch.Tensor)
        # if opacity.numel() == 1:
        #    opacity = opacity.item()
        return opacity

    @opacity.setter
    def opacity(self, value):
        self.data[..., -1:] = value

    @property
    def glow(self):
        glow = self.data[..., -2:-1].as_subclass(torch.Tensor)
        # if glow.numel() == 1:
        #    glow = glow.item()
        return glow

    def is_transparent(self):
        return self.opacity < 1

    @glow.setter
    def glow(self, value):
        self.data[..., -2:-1] = value

    @property
    def rgb(self):
        return self.data[..., :3]

    @rgb.setter
    def rgb(self, value):
        self.data[..., :3] = value

    def set_rgb(self, rgb):
        out = self.prep_set(rgb)
        out.rgb = rgb
        return out

    def mult_rgb(self, other):
        orgb = other.rgb if isinstance(other, Color) else other
        out = self.new_empty()
        out.data = self.data.clone()
        out.rgb = self.rgb * orgb
        return out

    def prep_set(self, value):
        value = cast_to_tensor(value)
        out = self.new_empty()
        out.data = self.data.clone()
        # A named color is a single ``[5]`` row, so painting a grid of values
        # onto it -- one per vertex, one per texel -- has nothing to broadcast
        # against until the leading axes exist. Give it the value's, but only
        # when the value really carries a grid: a scalar opacity arrives padded
        # to ``[1, 1, 1]``, and inflating a named color to match it would
        # change the shape every existing caller gets back.
        if any(size > 1 for size in value.shape[:-1]):
            out = unsqueeze_left(out, value)
        out = broadcast(out, value, [-1]).contiguous()
        return out

    def set_opacity(self, opacity):
        """Return a copy of this color with its opacity replaced.

        Opacity is one of the five channels, so ordinary color arithmetic
        moves it along with the others -- ``BLUE * 0.5`` halves the alpha as
        well as the brightness, and renders half-transparent. This sets the
        alpha channel alone and leaves red, green, blue and glow as they are.

        Parameters
        ----------
        opacity
            The new opacity, in ``[0, 1]``: 0 is invisible, 1 fully opaque.
            A tensor is broadcast against this color, giving one opacity per
            row -- per vertex or per texel.

        Returns
        -------
        :class:`Color`
            A new color. The color it was called on is left unchanged, so
            the named palette constants stay safe to reuse.
        """
        out = self.prep_set(opacity)
        out.opacity = opacity
        return out

    def mult_opacity(self, opacity):
        return self.set_opacity(self.opacity * opacity)

    def set_glow(self, glow):
        out = self.prep_set(glow)
        out.glow = glow
        return out

    def convert_to_uint8(self):
        return (self * 255).to(torch.uint8)

    def new_empty(self, *args, **kwargs):
        """Return a new opaque black :class:`Color` on this color's device.

        Overrides :meth:`torch.Tensor.new_empty`, which would otherwise hand
        back an uninitialized tensor of the requested size: a color's row is
        always the fixed ``[R, G, B, glow, opacity]``, so the size arguments
        are accepted for signature compatibility and ignored, and the row is
        zeroed rather than left as whatever the allocator returned. Keyword
        arguments are forwarded to the :class:`Color` constructor.
        """
        return Color((0, 0, 0), **kwargs).to(self.device).as_subclass(Color)

    @staticmethod
    def add_defaults(color):
        """Widen RGB or RGBA to Algan's ``[R, G, B, glow, opacity]``.

        Only 3 and 4 channels are widened. A width that is neither is not a
        color missing its extra channels, and padding it anyway meant the
        error it eventually caused reported a shape the caller never wrote --
        ``ImageMob(torch.zeros(8, 8, 2))`` was told its texture had shape
        ``(8, 8, 4)``.
        """
        if color.shape[-1] == 3:
            color = torch.cat((color, torch.ones_like(color[..., :1])), -1)
        if color.shape[-1] == 4:
            color = torch.cat(
                (color[..., :-1], torch.zeros_like(color[..., :1]), color[..., -1:]), -1
            )
        return color


def to_color(value):
    """Coerce a user-supplied color into something Algan can store.

    The colors Algan hands out are :class:`Color` constants, but the ones
    users reach for first are the ones every other graphics library takes: a
    hex string, a CSS name, a hex int, an RGB triple. Materials have accepted
    all of those since they were written -- ``MeshStandardMaterial(color=
    0x8B5A2B)`` is how the shipped presets are spelled -- while ``Square(
    color="#ff0000")`` raised ``AttributeError: 'str' object has no attribute
    'reshape'`` from deep inside the timeline. This is the one place that
    decides, so both spellings mean the same thing.

    Anything already tensor-shaped is returned untouched: a per-row color
    buffer is a legitimate value and must not be collapsed to one color.

    Parameters
    ----------
    value
        A :class:`Color`, a hex string (``"#ff0000"``) or CSS name
        (``"red"``), a hex int (``0xff0000``), an RGB/RGBA/RGBA+glow sequence
        of floats in ``[0, 1]``, a tensor, or ``None``.

    Returns
    -------
    :class:`Color` or the value unchanged

    Raises
    ------
    :class:`~algan.errors.InvalidColorError`
        If ``value`` is a string that names no color, or a bool.
    """
    if value is None:
        return value
    if isinstance(value, Color) or torch.is_tensor(value):
        # An RGB or RGBA buffer is a color that is merely missing Algan's
        # extra channels; pad it rather than making the caller know the
        # layout. Anything already 5 wide (or not shaped like a color at all)
        # is left exactly as it is, including a per-row buffer.
        if value.shape and value.shape[-1] in (3, 4):
            return Color.add_defaults(value)
        return value
    if isinstance(value, bool):
        # bool is an int subclass, and True as a color is a mistake, not black.
        raise InvalidColorError(
            f"Invalid color value: {value!r}. Use a Color such as RED, a hex "
            f"string ('#ff0000'), a hex int (0xff0000) or an RGB tuple."
        )
    if isinstance(value, int):
        return Color("#%06X" % (value & 0xFFFFFF))
    if isinstance(value, str):
        return Color(value)
    if (
        isinstance(value, (tuple, list))
        and 3 <= len(value) <= 5
        and all(isinstance(channel, (int, float)) for channel in value)
    ):
        return Color(tuple(float(channel) for channel in value))
    return value


def color_to_texture_map(color):
    return lambda coords: color.view(([1] * (coords.dim() - 1)) + [-1]).expand(
        list(coords.shape[:-1]) + [-1]
    )


GLOW = Color((0, 0, 0), 1, 0)
TRANSPARENT = Color((0, 0, 0), 0, 0)
# REDS = [Color(*[__ / 255 for __ in _]) for _ in ((249, 113, 123), (225, 69, 81), (213, 27, 41), (172, 13, 24), (139, 0, 10))]
# YELLOWS = [Color(*[__ / 255 for __ in _]) for _ in ((255,230, 116), (231, 202, 71), (219, 184, 28), (177, 147, 13), (142, 116, 0))]
# BLUES = [Color(*[__ / 255 for __ in _]) for _ in ((110, 92, 178), (82, 62, 159), (59+20, 35+20, 151+20), (43, 22, 122), (20, 11, 98))]
# GREENS = [Color(*[__ / 255 for __ in _]) for _ in ((112, 212, 96), (77, 191, 59), (45, 181, 23), (29, 146, 11), (16, 118, 0))]

GRAY_A = Color("#DDDDDD")
GREY_A = Color("#DDDDDD")
GRAY_B = Color("#BBBBBB")
GREY_B = Color("#BBBBBB")
GRAY_C = Color("#888888")
GREY_C = Color("#888888")
GRAY_D = Color("#444444")
GREY_D = Color("#444444")
GRAY_E = Color("#222222")
GREY_E = Color("#222222")
BLACK = Color("#000000")
WHITE = Color("#FFFFFF")
LIGHTER_GRAY = Color("#DDDDDD")
LIGHTER_GREY = Color("#DDDDDD")
LIGHT_GRAY = Color("#BBBBBB")
LIGHT_GREY = Color("#BBBBBB")
GRAY = Color("#888888")
GREY = Color("#888888")
DARK_GRAY = Color("#444444")
DARK_GREY = Color("#444444")
DARKER_GRAY = Color("#222222")
DARKER_GREY = Color("#222222")
BLUE_A = Color("#C7E9F1")
BLUE_B = Color("#9CDCEB")
BLUE_C = Color("#58C4DD")
BLUE_D = Color("#29ABCA")
BLUE_E = Color("#236B8E")
PURE_BLUE = Color("#0000FF")
BLUE = Color("#58C4DD")
DARK_BLUE = Color("#236B8E")
TEAL_A = Color("#ACEAD7")
TEAL_B = Color("#76DDC0")
TEAL_C = Color("#5CD0B3")
TEAL_D = Color("#55C1A7")
TEAL_E = Color("#49A88F")
TEAL = Color("#5CD0B3")
GREEN_A = Color("#C9E2AE")
GREEN_B = Color("#A6CF8C")
GREEN_C = Color("#83C167")
GREEN_D = Color("#77B05D")
GREEN_E = Color("#699C52")
PURE_GREEN = Color("#00FF00")
GREEN = Color("#83C167")
YELLOW_A = Color("#FFF1B6")
YELLOW_B = Color("#FFEA94")
YELLOW_C = Color("#FFFF00")
YELLOW_D = Color("#F4D345")
YELLOW_E = Color("#E8C11C")
YELLOW = Color("#FFFF00")
GOLD_A = Color("#F7C797")
GOLD_B = Color("#F9B775")
GOLD_C = Color("#F0AC5F")
GOLD_D = Color("#E1A158")
GOLD_E = Color("#C78D46")
GOLD = Color("#F0AC5F")
RED_A = Color("#F7A1A3")
RED_B = Color("#FF8080")
RED_C = Color("#FC6255")
RED_D = Color("#E65A4C")
RED_E = Color("#CF5044")
PURE_RED = Color("#FF0000")
RED = Color("#FC6255")
MAROON_A = Color("#ECABC1")
MAROON_B = Color("#EC92AB")
MAROON_C = Color("#C55F73")
MAROON_D = Color("#A24D61")
MAROON_E = Color("#94424F")
MAROON = Color("#C55F73")
PURPLE_A = Color("#CAA3E8")
PURPLE_B = Color("#B189C6")
PURPLE_C = Color("#9A72AC")
PURPLE_D = Color("#715582")
PURPLE_E = Color("#644172")
PURPLE = Color("#9A72AC")
PINK = Color("#D147BD")
LIGHT_PINK = Color("#DC75CD")
ORANGE = Color("#FF862F")
LIGHT_BROWN = Color("#CD853F")
DARK_BROWN = Color("#8B4513")
GRAY_BROWN = Color("#736357")
GREY_BROWN = Color("#736357")
CYAN = Color("#00FFFF")
MAGENTA = Color("#FF00FF")
