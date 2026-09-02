"""Every material class, every light type, shadows, glow and transmission.

The scene is built out of identical spheres so that *only* the shading differs
between columns: any change to a shader, to the light transport, or to the
material parameter plumbing shows up as one column changing while its
neighbours stay put.

It is also the only scene that turns shadows on and that puts refractive and
mirror-like surfaces in the frame, which forces the render onto the wavefront
tracer's secondary-ray paths (reflection, refraction and shadow continuations)
rather than primary visibility alone.
"""

from algan import *

# Pinned so the render does not depend on the host's fonts;
# tests/conftest.py registers the vendored faces.
FONT = "Algan Test Sans"

Scene.set_background(DARKER_GRAY)
# Restored by the full-render harness after this scene renders.
SETTINGS.raytracing.set(shadows=True)

# --------------------------------------------------------------------------
# Act 1 -- the material zoo, two labelled rows of identical spheres.
# --------------------------------------------------------------------------
with Off():
    ambient = AmbientLight(color=WHITE, intensity=0.35).spawn(animate=False)
    key_light = DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=1.0,
        shadow_angle=0.4,
    ).spawn(animate=False)

    title = Text(
        "MATERIALS AND LIGHTING",
        font_size=42,
        weight="BOLD",
        color=WHITE,
        font=FONT,
    ).move(UP * 2.9)

    lit = Group(
        Sphere(radius=0.5).set_material(MeshBasicMaterial(color=BLUE)),
        Sphere(radius=0.5).set_material(MeshLambertMaterial(color=GREEN)),
        Sphere(radius=0.5).set_material(
            MeshPhongMaterial(color=ORANGE, specular=WHITE, shininess=80)
        ),
        Sphere(radius=0.5).set_material(
            MeshStandardMaterial(color=RED, roughness=0.2, metalness=0.75)
        ),
        Sphere(radius=0.5).set_material(MeshToonMaterial(color=TEAL, bands=4)),
        Sphere(radius=0.5, color=WHITE).set_material(MeshNormalMaterial()),
    ).arrange_in_line(RIGHT, buffer=0.62)
    lit.move(UP * 1.3 - lit.get_center())

    lit_labels = Group(
        Text("Basic", font_size=21, color=GRAY_A, font=FONT),
        Text("Lambert", font_size=21, color=GRAY_A, font=FONT),
        Text("Phong", font_size=21, color=GRAY_A, font=FONT),
        Text("Standard", font_size=21, color=GRAY_A, font=FONT),
        Text("Toon", font_size=21, color=GRAY_A, font=FONT),
        Text("Normal", font_size=21, color=GRAY_A, font=FONT),
    )
    for mob, label in zip(lit, lit_labels):
        label.move_to(mob.get_center() + DOWN * 0.85)

    exotic = Group(
        Sphere(radius=0.5).set_material(MeshMatcapMaterial(color=GOLD)),
        Sphere(radius=0.5).set_material(MeshDepthMaterial(near=4.0, far=11.0)),
        Sphere(radius=0.5).set_material(
            MeshPhysicalMaterial(
                color=BLUE_A,
                roughness=0.1,
                clearcoat=0.85,
                transmission=0.5,
                ior=1.45,
            )
        ),
        Sphere(radius=0.5, color=GREEN_A).set_material(GLASS),
        Sphere(radius=0.5).set_material(MIRROR),
        Sphere(radius=0.5).set_material(COPPER),
    ).arrange_in_line(RIGHT, buffer=0.62)
    exotic.move(DOWN * 0.5 - exotic.get_center())

    exotic_labels = Group(
        Text("Matcap", font_size=21, color=GRAY_A, font=FONT),
        Text("Depth", font_size=21, color=GRAY_A, font=FONT),
        Text("Physical", font_size=21, color=GRAY_A, font=FONT),
        Text("GLASS", font_size=21, color=GRAY_A, font=FONT),
        Text("MIRROR", font_size=21, color=GRAY_A, font=FONT),
        Text("COPPER", font_size=21, color=GRAY_A, font=FONT),
    )
    for mob, label in zip(exotic, exotic_labels):
        label.move_to(mob.get_center() + DOWN * 0.85)

with Seq():
    title.spawn()
    with Lag(0.14, runtime=1.3):
        for mob in lit:
            mob.spawn()
    with Sync(runtime=0.4):
        lit_labels.spawn()
    with Lag(0.14, runtime=1.3):
        for mob in exotic:
            mob.spawn()
    with Sync(runtime=0.4):
        exotic_labels.spawn()

# --------------------------------------------------------------------------
# Act 2 -- material parameters are ordinary animatable attributes.
# --------------------------------------------------------------------------
with Seq():
    with Sync(runtime=2.0):
        lit[2].shininess = 12
        lit[3].roughness = 0.85
        lit[3].metalness = 0.15
        exotic[2].clearcoat = 0.1
        exotic[2].transmission = 0.05
        key_light.move(LEFT * 9)
        for mob in lit:
            mob.rotate(150, UP)
        for mob in exotic:
            mob.rotate(-150, UP)
    with Sync(runtime=1.4):
        key_light.move(RIGHT * 9)
        lit[3].roughness = 0.2
        lit[3].metalness = 0.75
    Scene.wait(0.2)

# --------------------------------------------------------------------------
# Act 3 -- neutral probes in front of a wall, lit by each light type in turn
# so the shadows they cast are the only thing distinguishing them.
# --------------------------------------------------------------------------
with Sync(runtime=0.7):
    lit.despawn()
    lit_labels.despawn()
    exotic.despawn()
    exotic_labels.despawn()
    title.despawn()

with Off():
    # The wall sits behind the probes only: nothing above it casts onto it, so
    # every shadow in frame belongs to a probe.
    wall = (
        Prism(width=17.0, height=5.2, depth=0.3)
        .set_material(MeshLambertMaterial(color=GRAY_D))
        .move(IN * 2.4 + DOWN * 0.9)
    )
    probes = Group(
        *[
            Sphere(radius=0.6).set_material(
                MeshStandardMaterial(color=WHITE, roughness=0.6)
            )
            for _ in range(4)
        ]
    ).arrange_in_line(RIGHT, buffer=1.5)
    probes.move(DOWN * 0.9 - probes.get_center())
    light_label = Text(
        "point / spot / rect-area / hemisphere lights  +  shadows",
        font_size=23,
        color=TEAL_A,
        font=FONT,
    ).move(DOWN * 3.15)

with Seq():
    with Off():
        ambient.despawn()
        key_light.despawn()
        point_light = PointLight(
            location=LEFT * 3.6 + UP * 0.6 + OUT * 2.2,
            color=YELLOW,
            intensity=2.2,
            decay=1.0,
        ).spawn(animate=False)
        spot_light = SpotLight(
            location=LEFT * 1.2 + UP * 2.4 + OUT * 2.2,
            target=LEFT * 1.2 + DOWN * 0.9,
            color=BLUE_A,
            intensity=5.0,
            cone_angle=22.0,
            penumbra=0.35,
        ).spawn(animate=False)
        rect_light = RectAreaLight(
            location=RIGHT * 1.2 + UP * 1.8 + OUT * 2.2,
            target=RIGHT * 1.2 + DOWN * 0.9,
            color=GREEN_A,
            intensity=3.0,
            width=1.8,
            height=1.0,
        ).spawn(animate=False)
        hemisphere = HemisphereLight(
            color=MAROON_A,
            ground_color=BLUE_E,
            intensity=0.5,
        ).spawn(animate=False)
    with Sync(runtime=0.7):
        wall.spawn()
        probes.spawn()
        light_label.spawn()
    with Sync(runtime=1.8):
        point_light.move(RIGHT * 2.4)
        spot_light.move(RIGHT * 2.4)
        rect_light.move(RIGHT * 2.4)
    with Sync(runtime=1.4):
        point_light.move(LEFT * 2.4)
        spot_light.move(LEFT * 2.4)
        rect_light.move(LEFT * 2.4)
    Scene.wait(0.2)

# --------------------------------------------------------------------------
# Act 4 -- emissive glow through the bloom post-process, and opacity.
# --------------------------------------------------------------------------
with Sync(runtime=0.7):
    probes.despawn()
    wall.despawn()
    hemisphere.despawn()
    rect_light.despawn()
    spot_light.despawn()
    point_light.despawn()
    light_label.despawn()

with Off():
    AmbientLight(color=WHITE, intensity=0.45).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=1.0,
    ).spawn(animate=False)
    emitters = Group(
        Sphere(radius=0.6).set_material(MeshBasicMaterial(color=YELLOW)),
        Sphere(radius=0.6).set_material(MeshBasicMaterial(color=TEAL)),
        Sphere(radius=0.6).set_material(MeshStandardMaterial(color=RED)),
        Sphere(radius=0.6).set_material(MeshStandardMaterial(color=BLUE)),
    ).arrange_in_line(RIGHT, buffer=1.5)
    emitters.move(-emitters.get_center())
    glow_label = Text(
        "glow + bloom + tonemapping                    opacity",
        font_size=23,
        color=TEAL_A,
        font=FONT,
    ).move(DOWN * 3.15)

with Seq():
    with Sync(runtime=0.6):
        emitters.spawn()
        glow_label.spawn()
    with Sync(runtime=1.8):
        emitters[0].glow = 1.0
        emitters[1].glow = 2.5
        emitters[2].opacity = 0.2
        emitters[3].opacity = 0.55
    with Sync(runtime=1.2):
        emitters[0].glow = 0.0
        emitters[1].glow = 0.0
        emitters[2].opacity = 1.0
        emitters[3].opacity = 1.0
    Scene.wait(0.3)
