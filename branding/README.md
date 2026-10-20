# Algan branding

The logo is rendered by Algan itself. It is Manim's mark lifted into 3-D: the circle, square
and triangle become a chrome sphere, a blue cube and a gold cone, rising left to right beside
the wordmark, whose capital A carries a wave for a crossbar (the wave is the motif of the demo
video, and a lambda, and y = sin x).

| File | What it is |
| --- | --- |
| `brand.py` | What the logo is made of, shared by the two scenes: the typeface, the colours and materials, the studio environment, the crossbar tube and the layout of the wordmark and the solids. |
| `logo_scene.py` | The ray-traced logo. `python branding/logo_scene.py BANNER_4K` renders the README banner (the full lockup), `ICON_4K` the square icon (the symbol alone). Output lands in `algan_outputs/branding/`. |
| `demo_video.py` | The 16-second demo video that ends on the logo. `python branding/demo_video.py HD` renders it; `STILLS` renders key-moment previews. Its y = sin x label is typeset with LaTeX. |
| `make_logo_svg.py` | The flat vector set, in three tiers: the full lockup, the wordmark alone for the docs sidebar, and the symbol for icons and the favicon. Writes straight into `docs/source/_static/`, `favicon.ico` included. |
| `fonts/` | `AlganFranklin-Black.ttf`, the logo's typeface, its licence (`OFL.txt`) and `build_font.py`, which made it. |

Shipped assets live in `docs/source/_static/`:

- `algan-banner.png` (1920x720): README hero and the docs' social card. Downscaled from the 4K master.
- `algan-icon.png` (1024x1024): the rendered symbol, for avatars.
- `algan-demo.mp4` (1080p30, 16 s): the demo video, embedded on the docs landing page.
- `algan-logo.svg`, `algan-logo-dark.svg`: the full flat lockup.
- `algan-logo-sidebar.svg`, `algan-logo-sidebar-dark.svg`: furo sidebar wordmarks.
- `algan-icon-light.svg`, `algan-icon-dark.svg`, `algan-favicon.svg`, `favicon.ico`.

The demo video, beat by beat, with the camera moving from the first frame: "Algorithmic
animation" writes itself on two lines while the camera drifts sideways; one blue wave sweeps
through and leaves the letters a mirror (a `wave_color` that fades a plain copy of the text to
reveal a chrome copy underneath), and as the camera orbits they reflect three large solids and a
grid standing behind it; the letters outside A-L-G and A-N fade out in a wave, left to right,
while the survivors slide into "Algan" and the gallery fades out of their reflections, so the
mirror settles into the logo's chrome; the A's crossbar lifts off as a straight
bar, bends into y = sin x on faint axes and settles back as the wave, leaving a lambda; the
sphere rolls in and the cube and cone drop onto the mirror while the camera cranes up to the
banner framing; a tagline fades in, and the video fades to black so its loop on the docs page
has a clean seam.

Design notes that are easy to lose:
- Typeface: Libre Franklin Black (SIL OFL 1.1), the open revival of Franklin Gothic, which the
  logo was first drawn in but which ships only with Windows. `fonts/build_font.py` adds the
  Greek capital lambda Libre Franklin lacks (the A's outline with the crossbar cut away, so the
  two swap cleanly in the video), tightens the tracking by 2% of the em and the lambda's advance
  by a further 4% (its diagonal otherwise leaves a gap before the l), and renames the family
  "Algan Franklin" so an installed Libre Franklin can never stand in for it. `brand.py`
  registers the file with Pango per process, so nothing needs installing.
- The wordmark floats just clear of the floor, its g's descender hovering over the mirror. A
  baseline on the floor sinks the g into it and makes the reflection illegible.
- Sizes come from the font's own proportions (`brand.solid_sizes`): the sphere is as tall as a
  lowercase letter, the cone as a capital, the cube between. Positions are the original logo's
  grouping (`brand.solid_layout`): the sphere in front, just clear of the n, the cube behind it
  and the cone in front of the cube, overlapping it from the camera's view. The banner is
  centred on the camera from the measured layout, so a change of face or size cannot push the
  cone off the edge.
- One blue: the wave's `#4da3ff` is the brand colour; the cube is a deeper shade of its hue. The
  cube is lacquered rather than glass: in a black studio a glass cube has nothing behind it to
  refract and renders as a dark box.
- Flat text facing the camera reflects the region just below the horizon *behind* the camera,
  the top of a letter seeing the lowest elevation. `studio_env` puts a bright band there, cut
  off above, which is what gives the letters their chrome horizon; a broad, dimmer band round
  the sides and back and two softboxes give the curved solids something to reflect. All of it
  is masked out of the camera's own view, which reaches the horizon: the bands' tails otherwise
  tint the backdrop grey. The backdrop is near black, tinted towards the brand blue.
- The floor is a perfect mirror (roughness 0). A glossy floor picks up a grey haze at grazing
  angles from the prefiltered environment, and highlight blobs from the area light.
- In the video the floor runs well behind the camera, so its front edge never comes into shot,
  and the gallery the chrome title reflects stands on it there.
- `Color(GOLD, glow=1.0)` ignores its keywords when given a Color; use `GOLD.set_glow(1.0)`.
