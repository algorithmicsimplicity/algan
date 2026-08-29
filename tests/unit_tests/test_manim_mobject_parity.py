from __future__ import annotations

import inspect

import numpy as np

import algan
import algan.manim as mn


def test_all_manim_0201_mobjects_are_exported():
    assert mn.MANIM_COMMUNITY_VERSION == "0.20.1"
    assert len(mn.MANIM_MOBJECT_NAMES) == 188
    assert len(set(mn.MANIM_MOBJECT_NAMES)) == 188
    assert mn.missing_manim_mobjects(vars(mn)) == ()
    mn.validate_manim_mobject_parity(vars(mn))
    assert all(
        inspect.isclass(getattr(mn, name)) and issubclass(getattr(mn, name), algan.Mob)
        for name in mn.MANIM_MOBJECT_NAMES
    )


def test_manim_backed_graphing_and_mutation_api():
    axes = algan.Axes(
        x_range=[-1, 1, 1],
        y_range=[-1, 1, 1],
        add_to_scene=False,
    )
    graph = axes.plot(lambda x: x * x, x_range=[-1, 1])
    assert isinstance(graph, mn.ParametricFunction)
    assert algan.Arc(add_to_scene=False).get_render_primitives() is not None
    arrow = algan.Arrow(algan.LEFT, algan.RIGHT, add_to_scene=False)
    assert arrow.add_tip() is arrow
    assert arrow.has_tip()

    with algan.Off():
        assert graph.scale(1.5) is graph
        assert graph.rotate(45) is graph


def test_native_surface_accepts_manim_parametric_api():
    surface = algan.Surface(
        lambda u, v: np.array([u, v, u * v]),
        u_range=(-1, 1),
        v_range=(0, 2),
        resolution=(3, 4),
        checkerboard_colors=False,
        add_to_scene=False,
    )
    assert (surface.grid_width, surface.grid_height) == (4, 5)
    assert surface.grid.location.shape[-2:] == (20, 3)
    assert surface.get_unit_normals().shape[-2:] == (20, 3)
    u_values, v_values = surface._get_u_values_and_v_values()
    assert len(u_values) == 4
    assert len(v_values) == 5


def test_native_3d_geometry_families_build_renderable_meshes():
    objects = [
        algan.Sphere(resolution=(4, 3), add_to_scene=False),
        algan.Cylinder(resolution=(4, 3), add_to_scene=False),
        algan.Cone(resolution=(4, 3), add_to_scene=False),
        algan.Torus(resolution=(4, 3), add_to_scene=False),
        algan.Tetrahedron(add_to_scene=False),
        algan.Octahedron(add_to_scene=False),
        algan.Icosahedron(add_to_scene=False),
        algan.Dodecahedron(add_to_scene=False),
        algan.ConvexHull3D(
            [-1, -1, 0],
            [1, -1, 0],
            [0, 1, 0],
            [0, 0, 1],
            add_to_scene=False,
        ),
    ]
    for mob in objects:
        assert isinstance(mob, algan.Mob)
        assert mob.get_bounding_box().shape[-1] == 3
    assert objects[0].get_render_primitives() is not None
    for mob in objects[4:]:
        assert mob.get_render_primitives() is not None

    arrow = algan.Arrow3D(add_to_scene=False)
    arrow_primitives = arrow.get_render_primitives()
    assert arrow_primitives is not None
    # Shaft, its two end discs, head, its base disc. The arrow aggregates its
    # whole subtree because the discs are children of the shaft and the head
    # and are not Scene actors, so nothing else would ask them to build.
    assert len(arrow_primitives) == 5


def test_point_cloud_and_image_apis_are_native():
    cloud = mn.DotCloud(
        points=[[0, 0, 0], [1, 0, 0]],
        radius=0.03,
        add_to_scene=False,
    )
    assert cloud.get_num_points() == 2
    old_children = tuple(cloud.children)
    cloud.filter_out(lambda points: points[..., 0] < 0.5)
    assert cloud.get_num_points() == 1
    assert tuple(cloud.children) != old_children
    assert len(cloud.get_descendants()) > 0

    first = mn.DotCloud(
        points=[[0, 0, 0]],
        color=algan.RED,
        add_to_scene=False,
    )
    second = mn.DotCloud(
        points=[[0, 0, 0]],
        color=algan.BLUE,
        add_to_scene=False,
    )
    cloud.set_points([[0, 0, 0]])
    assert cloud.interpolate_color(first, second, 0.5) is cloud

    pixels = np.zeros((2, 3, 4), dtype=np.uint8)
    pixels[..., 3] = 128
    image = mn.ImageMobject(
        pixels,
        scale_to_resolution=None,
        add_to_scene=False,
    )
    assert image.get_pixel_array().shape == (2, 3, 4)
    assert image.set_opacity(0.5) is image
    assert np.all(image.get_pixel_array()[..., 3] == 64)

    abstract = mn.AbstractImageMobject(
        scale_to_resolution=None,
        add_to_scene=False,
    )
    with np.testing.assert_raises(NotImplementedError):
        abstract.get_pixel_array()


def test_labeled_dot_has_manim_0201_buff_parameter():
    assert "buff" in inspect.signature(algan.LabeledDot).parameters
    label = algan.Circle(radius=0.2, add_to_scene=False)
    dot = algan.LabeledDot(label, buff=0.3, add_to_scene=False)
    rendered_label = dot.get_manim_mobject().submobjects[0]
    expected = 0.3 + np.linalg.norm([rendered_label.width, rendered_label.height]) / 2
    assert abs(float(dot.get_manim_mobject().radius) - expected) < 1e-6


def test_group_add_is_variadic_and_chainable():
    group = algan.Group(add_to_scene=False)
    result = group.add(
        algan.Circle(add_to_scene=False),
        algan.Square(add_to_scene=False),
    )
    assert result is group
    assert len(group) == 2


def test_native_vector_style_and_surrounding_rectangle_api():
    circle = algan.Circle(
        fill_opacity=0.2,
        stroke_color=algan.RED,
        stroke_width=4,
        add_to_scene=False,
    )
    square = algan.Square(side_length=2, add_to_scene=False)
    surrounding = algan.SurroundingRectangle(
        circle,
        square,
        buff=(0.2, 0.3),
        corner_radius=0.15,
        add_to_scene=False,
    )
    assert abs(float(circle.color[..., -1].reshape(-1)[0]) - 0.2) < 1e-6
    assert float(circle.border_width.reshape(-1)[0]) == 2
    assert surrounding.corner_radius == 0.15
    assert surrounding.control_points.location.shape[-2] > 16


def test_renderer_specific_surface_equivalents():
    surface = mn.OpenGLSurface(
        lambda u, v: (u, v, u * v),
        resolution=(3, 3),
        add_to_scene=False,
    )
    mesh = mn.OpenGLSurfaceMesh(
        surface,
        resolution=(2, 2),
        add_to_scene=False,
    )
    assert surface.get_unit_normals().shape[-1] == 3
    assert len(mesh) > 0


def test_svg_and_vector_field_families_convert_nested_geometry(tmp_path):
    svg_path = tmp_path / "triangle.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<path d="M 1 1 L 9 1 L 5 9 Z" fill="#58C4DD" '
        'stroke="#FFFFFF"/></svg>'
    )

    def field(point):
        return np.array([-point[1], point[0], 0.0])

    objects = [
        algan.SVGMobject(svg_path, add_to_scene=False),
        mn.ArrowVectorField(
            field,
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            add_to_scene=False,
        ),
        mn.StreamLines(
            field,
            x_range=[-1, 1, 2],
            y_range=[-1, 1, 2],
            noise_factor=0,
            virtual_time=0.1,
            dt=0.1,
            max_anchors_per_line=4,
            add_to_scene=False,
        ),
    ]
    for mob in objects:
        assert len(mob.get_descendants()) > 0
        assert any(
            hasattr(descendant, "get_render_primitives")
            and descendant.get_render_primitives() is not None
            for descendant in mob.get_descendants()
        )
