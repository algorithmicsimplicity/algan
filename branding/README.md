# Algan branding

The logo is rendered by Algan itself. It is Manim's mark lifted into 3-D: the circle, square
and triangle become a chrome sphere, a blue cube and a gold cone, next to the wordmark whose
capital A carries a wave crossbar (the wave is the motif of the demo video).

| File | What it is |
| --- | --- |
| `logo_scene.py` | The ray-traced scene. `python branding/logo_scene.py BANNER_4K` renders the README banner, `ICON_4K` the square icon. Output lands in `algan_outputs/branding/`. |
| `make_logo_svg.py` | The flat vector set (docs sidebar wordmark, icon, favicon) built from Franklin Gothic Heavy outlines with fontTools. |
| `demo_video.py` | The 16-second demo video that ends on the logo. `python branding/demo_video.py HD` renders it; `STILLS` renders key-moment previews. |

Shipped assets live in `docs/source/_static/`:

- `algan-banner.png` (1920x720): README hero and the docs' social card. Downscaled from the 4K master.
- `algan-icon.png` (1024x1024): square rendered icon.
- `algan-demo.mp4` (1080p30, 16 s): the demo video, embedded on the docs landing page.
- `algan-logo-sidebar.svg`, `algan-logo-sidebar-dark.svg`: furo sidebar wordmarks.
- `algan-icon-light.svg`, `algan-icon-dark.svg`, `algan-favicon.svg`, `favicon.ico`.

The demo video, beat by beat: "Algorithmic Animation" writes itself on two lines, filling the frame; a colour wave, a glow
wave and a mirror wave sweep through it (all three are continuous spatial waves from
`wave_color`, which colours Bézier text per texel, not per glyph; the mirror wave fades a plain
copy of the text to reveal a chrome copy underneath, which reflects a gallery hidden behind
the camera: a grid backdrop, the three logo solids oversized, and a dozen other built-in mobs); every glyph except A-l-g-a-n dissolves and the survivors slide together;
the A morphs into a lambda as the wave takes its crossbar; the solids fly in and the camera
cranes up to the banner framing.

Design notes that are easy to lose:
- Typeface: Franklin Gothic Heavy (`FRAHV.TTF`, ships with Windows), used by the video, the
  ray-traced logo and the flat SVGs alike. It was chosen over Open Sans Bold because thick
  strokes are what make the reflected gallery readable through the mirror letters; Arial
  Black is heavier still but so wide the banner overflows the frame. Open Sans has no weight
  above Semibold installed here, so Pango's HEAVY silently renders the same as BOLD.
- The banner layout is centred on the camera from the measured word width, so a change of
  face does not push the cone off the right edge.

- The wordmark is `Λlgan` in Open Sans: the Greek capital lambda is the A without its crossbar,
  and the wave is drawn across it as a separate stroke.
- Flat text facing the camera reflects the region just above the horizon *behind* the camera,
  with the top of a letter mapping to the lowest elevation. The studio environment map in
  `studio_env` puts a bright band there, cut off above, which is what gives the letters their
  chrome horizon.
- The floor is a perfect mirror (roughness 0) with low metalness. A glossy floor picks up a
  grey haze at grazing angles from the prefiltered environment.
- In the video the floor is paper-thin and stops short of the camera: a slab's front face is a
  little vertical mirror, and a floor running behind the camera hides the gallery from the
  letters. The camera starts at the letters' height for the same reason: reflected rays that
  dip even slightly hit the floor before they reach anything behind the camera.
- `Color(GOLD, glow=1.0)` ignores its keywords when given a Color; use `GOLD.set_glow(1.0)`.
