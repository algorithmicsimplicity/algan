from algan import *

Scene.set_background_color(DARKER_GRAY)

with Off():
    AmbientLight(color=WHITE, intensity=0.35).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 3,
        target=ORIGIN,
        color=WHITE,
        intensity=1.1,
    ).spawn(animate=False)
    title = Text(
        "GEOMETRY IN MOTION",
        font_size=50,
        weight="BOLD",
        color=WHITE,
    ).move(UP * 1.58)
    solids = (
        Group(
            Sphere(radius=0.43).set_material(MeshBasicMaterial(color=BLUE)),
            Cylinder(radius=0.34, height=0.85, show_ends=True).set_material(
                MeshLambertMaterial(color=GREEN)
            ),
            Cone(base_radius=0.42, height=0.85, show_base=True).set_material(
                MeshPhongMaterial(color=ORANGE, shininess=55)
            ),
            Torus(major_radius=0.4, minor_radius=0.14).set_material(
                MeshStandardMaterial(color=TEAL, roughness=0.35, metalness=0.2)
            ),
            Cube(side_length=0.76).set_material(MeshBasicMaterial(color=RED)),
        )
        .arrange_in_line(RIGHT, buffer=0.36)
        .move(DOWN * 0.05)
    )
    solid_labels = Group(
        Text("sphere", font_size=21, color=GRAY_A),
        Text("cylinder", font_size=21, color=GRAY_A),
        Text("cone", font_size=21, color=GRAY_A),
        Text("torus", font_size=21, color=GRAY_A),
        Text("cube", font_size=21, color=GRAY_A),
    )
    for solid, label in zip(solids, solid_labels):
        label.move_to(solid.location + DOWN * 0.7)

with Seq():
    title.spawn()
    with Lag(0.17, run_time=1.5):
        for solid in solids:
            solid.spawn()
    with Lag(0.12, run_time=0.75):
        for label in solid_labels:
            label.spawn()
    with Sync(run_time=2.1):
        solids[0].rotate(220, UP)
        solids[1].move_between_points(LEFT * 0.9 + DOWN * 0.3, LEFT * 0.9 + UP * 0.6)
        solids[2].rotate(80, RIGHT)
        solids[3].rotate(130, UP + OUT)
        solids[4].rotate(100, UP + OUT)
        Scene.get_camera().rotate(16, UP, about_point=ORIGIN)
    Scene.wait(0.25)

with Sync():
    solids.despawn()
    solid_labels.despawn()

with Off():
    polyhedra = (
        Group(
            Tetrahedron(edge_length=1.0).set_material(MeshBasicMaterial(color=BLUE)),
            Octahedron(edge_length=0.95).set_material(MeshLambertMaterial(color=GREEN)),
            Icosahedron(edge_length=0.65).set_material(
                MeshPhongMaterial(color=ORANGE, shininess=45)
            ),
        )
        .arrange_in_line(RIGHT, buffer=0.55)
        .move(LEFT * 0.75 + DOWN * 0.2)
    )
    axes = Group(
        Arrow3D(
            start=LEFT * 0.55,
            end=RIGHT * 0.55,
            thickness=0.025,
            color=RED,
        ).set_material(MeshBasicMaterial(color=RED)),
        Arrow3D(
            start=DOWN * 0.55,
            end=UP * 0.55,
            thickness=0.025,
            color=GREEN,
        ).set_material(MeshBasicMaterial(color=GREEN)),
        Arrow3D(
            start=IN * 0.55,
            end=OUT * 0.55,
            thickness=0.025,
            color=BLUE,
        ).set_material(MeshBasicMaterial(color=BLUE)),
    ).move(RIGHT * 1.9 + DOWN * 1.05)
    phase_label = Text(
        "polyhedra + parent transforms + camera orbit",
        font_size=29,
        color=GRAY_A,
    ).move(UP * 0.92)

with Seq():
    with Sync():
        polyhedra.spawn()
        axes.spawn()
        phase_label.spawn()
    with Sync(run_time=2.2):
        polyhedra.rotate(12, OUT)
        polyhedra[0].rotate(-130, UP + RIGHT)
        polyhedra[1].rotate(145, RIGHT + OUT)
        polyhedra[2].rotate(170, UP)
        axes.rotate(45, UP + RIGHT)
        Scene.get_camera().rotate(-18, UP, about_point=ORIGIN)
    with Sync(run_time=1.0):
        polyhedra.scale(0.92)
        polyhedra.move(UP * 0.05)
        axes.scale(0.9)
        phase_label.color = TEAL_A
    Scene.wait(0.3)

Scene.save_frame("algan_outputs/checkpoints/geometry_and_camera.png")
