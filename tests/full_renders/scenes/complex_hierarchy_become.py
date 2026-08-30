"""Arbitrary hierarchy-to-hierarchy ``become`` across primitive families.

The source is a three-branch tree containing nested Groups, cubic-bezier
circuits, logical-PN Surfaces, a triangle-mesh Polyhedron and an ImageMob. It
becomes a four-branch tree with different nesting. Renderer-facing primitives
are paired independently of those Group boundaries: every non-image leaf uses a
geometric morph, while the ImageMob alone cross-dissolves. Visible outline frames
make the otherwise implicit target hierarchy readable after the transformation.

The final parent rotation is part of the regression: cross-kind replacements
must be spliced into the live hierarchy so a later transform of the returned
root still reaches every replacement descendant.
"""

from algan import *

# Pinned so the render does not depend on the host's fonts;
# tests/conftest.py registers the vendored faces.
FONT = "Algan Test Sans"

CHECKPOINT_TIMES = {
    "source_tree": 0.8,
    "mid_hierarchy_morph": 2.3,
    "resolved_target_tree": 3.9,
    "replacement_parent_tilt": 4.8,
    "resolved_final": 6.8,
}


def framed(content, *, buffer=0.2):
    """Give a Group a visible boundary without flattening its child tree."""
    frame = SurroundingRectangle(
        content,
        color=GRAY_B,
        border_width=2,
        filled=False,
        corner_radius=0.1,
        buffer=buffer,
        opacity=0.7,
    )
    return Group(frame, content)


Scene.set_background(DARKER_GRAY)

with Off():
    AmbientLight(color=WHITE, intensity=0.55).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=1.0,
    ).spawn(animate=False)

    title = Text(
        "ARBITRARY HIERARCHY MORPH",
        font_size=38,
        weight="BOLD",
        color=WHITE,
        font=FONT,
    ).move(UP * 3.05)
    subtitle = Text(
        "3 branches  ->  4 branches   |   primitives cross group boundaries",
        font_size=21,
        color=GRAY_A,
        font=FONT,
    ).move(UP * 2.55)
    legend = Group(
        Text("BEZIER", font_size=19, color=BLUE_A, font=FONT),
        Text("SURFACE", font_size=19, color=TEAL_A, font=FONT),
        Text("MESH", font_size=19, color=ORANGE, font=FONT),
        Text("IMAGE", font_size=19, color=PURPLE_A, font=FONT),
    ).arrange_in_line(RIGHT, buffer=0.65)
    legend.move(DOWN * 3.05)

    # Source branch 1: one framed Group holding a circuit and an image Surface.
    source_left_content = Group(
        Circle(radius=0.5, color=BLUE_A).move(LEFT * 3.8 + UP * 0.45),
        ImageMob("assets/world_map.jpg").scale(0.48).move(LEFT * 3.2 + DOWN * 0.5),
    )
    source_left = framed(source_left_content)

    # Source branch 2: a framed Group inside another framed Group.
    source_center_inner_content = Group(
        Square(size=0.75, color=BLUE_A).move(LEFT * 0.5 + UP * 0.55),
        Sphere(radius=0.4, color=TEAL_A).move(RIGHT * 0.48 + UP * 0.55),
    )
    source_center_inner = framed(source_center_inner_content, buffer=0.16)
    source_center_content = Group(
        source_center_inner,
        Circle(radius=0.42, color=BLUE_A).move(DOWN * 0.72),
    )
    source_center = framed(source_center_content, buffer=0.24)

    # Source branch 3 is a mesh primitive directly under the root.
    source_right = (
        Tetrahedron(edge_length=1.25, color=ORANGE)
        .rotate(24, RIGHT)
        .rotate(-28, UP)
        .move(RIGHT * 3.35 + UP * 0.1)
    )
    source_tree = Group(source_left, source_center, source_right)

    # Target branch 1 has three leaves instead of two. Pairing is global across
    # the hierarchy, so this branch need not inherit only source-left leaves.
    target_left_content = Group(
        Sphere(radius=0.43, color=TEAL_A).move(LEFT * 3.85 + UP * 0.48),
        Square(size=0.72, color=BLUE_A).move(LEFT * 3.05 + UP * 0.48),
        Tetrahedron(edge_length=0.9, color=ORANGE)
        .rotate(22, RIGHT)
        .rotate(25, UP)
        .move(LEFT * 3.45 + DOWN * 0.55),
    )
    target_left = framed(target_left_content)

    # Target branch 2 retains an outer Group but changes both its depth and leaf
    # count, exercising target-tree reconstruction after primitive matching.
    target_center_inner_content = Group(
        Tetrahedron(edge_length=0.95, color=ORANGE)
        .rotate(-20, RIGHT)
        .rotate(25, UP)
        .move(LEFT * 0.52 + UP * 0.58)
    )
    target_center_inner = framed(target_center_inner_content, buffer=0.16)
    target_center_lower_content = Group(
        Cylinder(radius=0.29, height=0.72, color=TEAL_A).move(
            RIGHT * 0.18 + DOWN * 0.64
        ),
        RegularPolygon(5, radius=0.4, color=BLUE_A).move(RIGHT * 0.88 + DOWN * 0.64),
    )
    target_center_lower = framed(target_center_lower_content, buffer=0.15)
    target_center_content = Group(target_center_inner, target_center_lower)
    target_center = framed(target_center_content, buffer=0.22)

    # The right side changes from one direct mesh into a nested Group plus an
    # extra root leaf. Surplus targets start as their own collapsed geometry at
    # the nearest existing source point and fade up as they grow outward, rather
    # than duplicating an already-visible source.
    target_right_content = Group(
        Sphere(radius=0.42, color=TEAL_A).move(RIGHT * 3.0 + UP * 0.42),
        Square(size=0.72, color=BLUE_A).move(RIGHT * 3.72 + DOWN * 0.38),
    )
    target_right = framed(target_right_content, buffer=0.2)
    target_extra = (
        Cylinder(radius=0.28, height=0.68, color=TEAL_A)
        .rotate(18, RIGHT)
        .move(RIGHT * 3.15 + DOWN * 1.62)
    )
    # Reference geometry: this tree only says what to become, and is never
    # drawn itself. Its members have a parent, so flagging the root is enough.
    target_tree = Group(
        target_left, target_center, target_right, target_extra, add_to_scene=False
    )

with Seq():
    with Sync(run_time=0.6):
        title.spawn()
        subtitle.spawn()
        legend.spawn()
        source_tree.spawn()
    Scene.wait(0.4)
    with Sync(run_time=2.6):
        source_tree = source_tree.become(target_tree, minimize_movement=True)
    Scene.wait(0.6)

    # Prove that target-class replacements now belong to the returned root.
    with Sync(run_time=1.2):
        source_tree.rotate(18, UP)
    with Sync(run_time=1.2):
        source_tree.rotate(-18, UP)
    Scene.wait(0.4)
