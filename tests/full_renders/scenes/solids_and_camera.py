"""3-D geometry, the transform hierarchy, and the camera.

Two geometry families are on screen at once and must stay distinguishable:

* **Analytic surfaces** (``Sphere``, ``Cylinder``, ``Cone``, ``Torus``,
  ``Surface``) tessellate to *logical PN triangles* that are diced per frame
  from the camera, so they exercise auto-resolution, adaptive subdivision and
  the crack-free boundary snap.
* **Flat meshes** (``Cube``, ``Prism``, the Platonic solids, ``Arrow3D``,
  ``Line3D``, ``Dot3D``, ``ConvexHull3D``) go straight into the flat-triangle
  BVH, so their hard-edged silhouettes are a reference next to the smooth ones.

The scene then drives them through the transform hierarchy (a parent Group and
its children animating in the same block), the movement helpers, screen-relative
layout, and every camera motion.  Every transform returns the scene to a known
layout so neighbouring columns never collide.
"""

from algan import *

# Pinned so the render does not depend on the host's fonts;
# tests/conftest.py registers the vendored faces.
FONT = "Algan Test Sans"

Scene.set_background(DARKER_GRAY)


def saddle(uv):
    """A hyperbolic paraboloid: curvature of both signs in one patch."""
    u = uv[..., :1] * 2 - 1
    v = uv[..., 1:] * 2 - 1
    return u * 1.1 * RIGHT + v * 1.1 * UP + (u * u - v * v) * 0.5 * OUT


# --------------------------------------------------------------------------
# Act 1 -- analytic surfaces above, flat meshes below.
# --------------------------------------------------------------------------
with Off():
    AmbientLight(color=WHITE, intensity=0.45).spawn(animate=False)
    key_light = DirectionalLight(
        location=RIGHT * 5 + UP * 6 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=0.8,
    ).spawn(animate=False)
    HemisphereLight(
        color=BLUE_A,
        ground_color=MAROON_E,
        intensity=0.3,
    ).spawn(animate=False)

    title = Text(
        "SOLIDS AND CAMERA",
        font_size=44,
        weight="BOLD",
        color=WHITE,
        font=FONT,
    ).move(UP * 2.85)

    curved = Group(
        Sphere(radius=0.55).set_material(
            MeshStandardMaterial(color=BLUE, roughness=0.45)
        ),
        Cylinder(radius=0.45, height=1.0, show_ends=True).set_material(
            MeshLambertMaterial(color=GREEN)
        ),
        Cone(base_radius=0.55, height=1.1, show_base=True).set_material(
            MeshPhongMaterial(color=ORANGE, shininess=55)
        ),
        Torus(ring_radius=0.55, tube_radius=0.22).set_material(
            MeshStandardMaterial(color=TEAL, roughness=0.4, metalness=0.2)
        ),
        # An odd grid makes the checkerboard read as diagonal banding, which is
        # far easier to eyeball than a per-vertex checker at auto resolution.
        Surface(
            saddle,
            color=MAROON,
            checkered_color=YELLOW,
            grid_width=9,
            grid_height=9,
        )
        .set_material(MeshStandardMaterial(roughness=0.55))
        .scale(0.55),
    ).arrange_in_line(RIGHT, buffer=0.85)
    curved.move(UP * 1.5 - curved.get_center())
    curved_center = curved.get_center()

    curved_labels = Group(
        Text("sphere", font_size=21, color=GRAY_A, font=FONT),
        Text("cylinder", font_size=21, color=GRAY_A, font=FONT),
        Text("cone", font_size=21, color=GRAY_A, font=FONT),
        Text("torus", font_size=21, color=GRAY_A, font=FONT),
        Text("parametric", font_size=21, color=GRAY_A, font=FONT),
    )
    for solid, label in zip(curved, curved_labels):
        label.move_to(solid.get_center() + DOWN * 1.05)

    flat = Group(
        Cube(size=0.85).set_material(MeshBasicMaterial(color=RED)),
        Prism(width=1.0, height=0.65, depth=0.65).set_material(
            MeshLambertMaterial(color=PURPLE)
        ),
        Tetrahedron(edge_length=1.05).set_material(MeshStandardMaterial(color=BLUE_B)),
        Octahedron(edge_length=0.8).set_material(MeshStandardMaterial(color=GREEN_B)),
        Icosahedron(edge_length=0.52).set_material(MeshPhongMaterial(color=GOLD)),
        Dodecahedron(edge_length=0.4).set_material(
            MeshStandardMaterial(color=MAROON_B)
        ),
    ).arrange_in_line(RIGHT, buffer=0.75)
    flat.move(DOWN * 1.0 - flat.get_center())
    flat_center = flat.get_center()

    flat_labels = Group(
        Text("cube", font_size=20, color=GRAY_A, font=FONT),
        Text("prism", font_size=20, color=GRAY_A, font=FONT),
        Text("tetra", font_size=20, color=GRAY_A, font=FONT),
        Text("octa", font_size=20, color=GRAY_A, font=FONT),
        Text("icosa", font_size=20, color=GRAY_A, font=FONT),
        Text("dodeca", font_size=20, color=GRAY_A, font=FONT),
    )
    for solid, label in zip(flat, flat_labels):
        label.move_to(solid.get_center() + DOWN * 0.95)

with Seq():
    title.spawn()
    with Lag(0.16, run_time=1.5):
        for solid in curved:
            solid.spawn()
    with Sync(run_time=0.6):
        curved_labels.spawn()
    with Lag(0.12, run_time=1.2):
        for solid in flat:
            solid.spawn()
    with Sync(run_time=0.6):
        flat_labels.spawn()

# --------------------------------------------------------------------------
# Act 2 -- parent and children transform inside the same block.
# --------------------------------------------------------------------------
with Seq():
    with Sync(run_time=2.4):
        # The Group tilts about its own centre while each member also spins
        # about its own axis: the descendant bases have to compose, not
        # overwrite, and the labels must not follow (they are not children).
        flat.rotate(14, OUT, about=flat_center)
        flat[0].rotate(140, UP + RIGHT)
        flat[1].rotate(-110, UP)
        flat[2].rotate(200, RIGHT + OUT)
        flat[3].rotate(160, UP)
        flat[4].rotate(-180, UP + OUT)
        flat[5].rotate(130, RIGHT)
        curved[0].rotate(220, UP)
        curved[1].rotate(75, RIGHT)
        curved[2].rotate(-60, RIGHT)
        curved[3].rotate(140, UP + OUT)
        curved[4].rotate(-70, RIGHT)
        key_light.move(LEFT * 9)
    with Sync(run_time=1.6):
        flat.rotate(-14, OUT, about=flat_center)
        # Travel out along a curved path and back, so the row is restored.
        curved[0].move_to(curved[0].get_center(), 180, arc_normal=OUT)
        curved[3].scale(1.3)
        key_light.move(RIGHT * 9)
    with Sync(run_time=0.8):
        curved[3].scale(1 / 1.3)
    Scene.wait(0.2)

# --------------------------------------------------------------------------
# Act 3 -- an axis triad built from Arrow3D/Line3D/Dot3D, a convex hull, and
# the camera moving around them.
# --------------------------------------------------------------------------
with Sync(run_time=0.8):
    curved_labels.despawn()
    flat_labels.despawn()
    flat.despawn()

with Off():
    triad = Group(
        Arrow3D(
            start=ORIGIN, end=RIGHT * 1.1, shaft_radius=0.05, color=RED
        ).set_material(MeshBasicMaterial(color=RED)),
        Arrow3D(
            start=ORIGIN, end=UP * 1.1, shaft_radius=0.05, color=GREEN
        ).set_material(MeshBasicMaterial(color=GREEN)),
        Arrow3D(
            start=ORIGIN, end=OUT * 1.1, shaft_radius=0.05, color=BLUE
        ).set_material(MeshBasicMaterial(color=BLUE)),
        # NOT a rendering artifact, though it reads as one: this line is
        # coaxial with the red arrow and ends exactly at its tip, so its
        # 0.03 radius shows past the cone's apex and again at the head's
        # shoulder, where the cone has tapered to about the same width. A
        # supersampled reference renders both the same way. Shorten the line
        # or thin it if the white on the red arrow is ever unwanted.
        Line3D(start=LEFT * 1.1, end=RIGHT * 1.1, radius=0.03, color=GRAY_A),
        Dot3D(point=ORIGIN, radius=0.14, color=WHITE),
    )
    hull = ConvexHull3D(
        RIGHT * 0.65,
        LEFT * 0.65,
        UP * 0.65,
        DOWN * 0.65,
        OUT * 0.65,
        IN * 0.65,
        RIGHT * 0.4 + UP * 0.4 + OUT * 0.4,
    ).set_material(MeshStandardMaterial(color=TEAL_B, roughness=0.4))
    triad.move(LEFT * 2.6 + DOWN * 1.1)
    hull.move(RIGHT * 2.6 + DOWN * 1.1)
    camera_label = Text(
        "Arrow3D / Line3D / Dot3D / ConvexHull3D  +  camera orbit",
        font_size=23,
        color=TEAL_A,
        font=FONT,
    ).move(DOWN * 3.05)

with Seq():
    with Sync(run_time=0.8):
        triad.spawn()
        hull.spawn()
        camera_label.spawn()
    with Sync(run_time=2.0):
        triad.rotate(120, UP + RIGHT)
        hull.rotate(-150, UP + OUT)
        Scene.get_camera().rotate(9, UP, about=ORIGIN)
    with Sync(run_time=1.8):
        Scene.get_camera().rotate(-9, UP, about=ORIGIN)
        Scene.get_camera().orbit(4, RIGHT, about=ORIGIN)
    with Sync(run_time=1.2):
        Scene.get_camera().orbit(-4, RIGHT, about=ORIGIN)
    Scene.wait(0.2)

# --------------------------------------------------------------------------
# Act 4 -- screen-relative layout, a colour wave over a curved surface, and
# geometry leaving the frame.
# --------------------------------------------------------------------------
with Sync(run_time=0.8):
    triad.despawn()
    hull.despawn()
    camera_label.despawn()

with Off():
    layout_label = Text(
        "fit_to_screen  +  wave_color  +  move_off_screen",
        font_size=23,
        color=TEAL_A,
        font=FONT,
    ).move(DOWN * 3.05)

with Seq():
    layout_label.spawn()
    with Sync(run_time=1.4):
        # Fit a named rectangle of the frame, then park the rest on a grid of
        # screen positions the camera resolves at record time.
        curved[4].fit_to_screen(
            bottom_left=(0.06, 0.3),
            top_right=(0.40, 0.76),
        )
        curved[0].move_center_to_screen_position((0.60, 0.70))
        curved[1].move_center_to_screen_position((0.85, 0.70))
        curved[2].move_center_to_screen_position((0.60, 0.32))
        curved[3].move_center_to_screen_position((0.85, 0.32))
    with Sync(run_time=1.8):
        # A colour wave over PN-tessellated geometry: the sphere is a single
        # flat colour, so the travelling band is unambiguous.
        curved[0].wave_color(YELLOW, direction=RIGHT)
        curved[1].rotate(180, UP)
        curved[3].rotate(180, RIGHT)
    with Sync(run_time=1.4):
        curved[0].move_off_screen(RIGHT, despawn=False)
        curved[1].move_off_screen(UP, despawn=False)
        curved[4].scale(0.6)
    Scene.wait(0.3)
