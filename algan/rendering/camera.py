import math
import warnings

from algan.animation_timeline.animation_contexts import Off, Sync
from algan.constants.spatial import *  # CAMERA_ORIGIN
from algan.animatable_base.mob import Mob
from algan.utils.tensor_utils import (
    broadcast_gather,
    unsquish,
    dot_product,
)
from algan.geometry.geometry import intersect_line_with_plane_colinear
from algan.errors import AlganConfigurationError, ApproximationWarning
import torch.nn.functional as F


class Camera(Mob):
    def __init__(
        self, orthographic=False, screen_distance=5, screen_scale=2.5,
        fov=None, near=0.0, far=0.0, *args, **kwargs
    ):
        # fov (vertical, degrees) is an alternative way to specify the
        # perspective: it fixes the camera-to-screen distance for the given
        # screen size. near/far are the clip distances (0 disables each);
        # near is a plane, far is a distance along the ray.
        screen_distance = self._validated_positive(
            "screen_distance", screen_distance
        )
        screen_scale = self._validated_positive("screen_scale", screen_scale)
        if fov is not None:
            fov = self._validated_fov(fov)
            screen_distance = screen_scale / math.tan(
                math.radians(fov) * 0.5)
        self._near = self._validated_clip("near", near)
        self._far = self._validated_clip("far", far)
        self._validate_clip_order(self._near, self._far)
        # Camera ownership is managed by Scene; tolerate the common generic
        # Mob kwargs without passing duplicates into the base constructor.
        kwargs.pop("add_to_scene", None)
        kwargs.pop("init", None)
        super().__init__(add_to_scene=False, init=False, *args, **kwargs)
        self.animatable_attrs.remove("color")
        with Off(animation_manager=self.animation_manager):
            self.orthographic = orthographic
            self.screen = Mob(
                scene=self.scene,
                location=self.location + screen_distance * self.get_forward_direction(),
                add_to_scene=False,
                init=False,
            )
            self.screen.scale(torch.tensor((1 / screen_scale, 1 / screen_scale, 1)))
            self.screen_scale_factor = screen_scale
            self.screen.is_primitive = True
            self.is_primitive = True
            self.add_children(self.screen)
            self.screen_distance = screen_distance
            self.corner_x_coords = torch.tensor([-1, -1, 1, 1]).view(-1, 1, 1, 1)
            self.corner_y_coords = torch.tensor([-1, 1, 1, -1]).view(-1, 1, 1, 1)
            self.spawn(animate=False)
        if orthographic:
            with Off(animation_manager=self.animation_manager):
                self.set_near_orthographic()

    @property
    def pixel_height(self):
        """Current normalized screen height represented by one output pixel."""
        return 2.0 / self.scene.num_pixels_screen_height

    @property
    def pixel_width(self):
        """Current normalized screen width represented by one output pixel."""
        return 2.0 / self.scene.num_pixels_screen_width

    @staticmethod
    def _as_finite_number(name, value):
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise AlganConfigurationError(
                f"{name} must be a finite number"
            ) from exc
        if not math.isfinite(result):
            raise AlganConfigurationError(f"{name} must be a finite number")
        return result

    @classmethod
    def _validated_positive(cls, name, value):
        result = cls._as_finite_number(name, value)
        if result <= 0:
            raise AlganConfigurationError(f"{name} must be positive")
        return result

    @classmethod
    def _validated_fov(cls, fov):
        value = cls._as_finite_number("fov", fov)
        if not math.isfinite(value) or not 0.0 < value < 180.0:
            raise AlganConfigurationError("fov must be finite and in (0, 180)")
        return value

    @classmethod
    def _validated_clip(cls, name, value):
        value = cls._as_finite_number(name, value)
        if value < 0:
            raise AlganConfigurationError(
                f"{name} clip distance must be finite and non-negative"
            )
        return value

    @staticmethod
    def _validate_clip_order(near, far):
        if near > 0 and far > 0 and near >= far:
            raise AlganConfigurationError(
                "near clip distance must be less than far clip distance"
            )

    def set_to_orthographic(self):
        """Use Algan's legacy near-orthographic perspective approximation.

        True parallel-ray orthographic projection is not implemented by the
        current renderer.  This compatibility method is retained with an
        explicit warning; new code should call :meth:`set_near_orthographic`
        so the approximation is visible at the call site.
        """
        warnings.warn(
            "Camera.set_to_orthographic() uses a far-distance perspective "
            "approximation, not true parallel-ray projection. Use "
            "set_near_orthographic() to opt into that approximation explicitly.",
            ApproximationWarning,
            stacklevel=2,
        )
        return self.set_near_orthographic()

    def set_near_orthographic(self, distance=1e5):
        """Flatten perspective by moving the camera far from its screen."""
        distance = self._validated_positive("distance", distance)
        self.orthographic = True
        return self._set_distance_to_screen(distance, preserve_mode=True)

    def get_fov(self):
        """The camera's vertical field of view in degrees (like Three.js's
        ``PerspectiveCamera.fov``), derived from the screen size and the
        camera-to-screen distance."""
        d = (
            (self.screen.location - self.location)
            .norm(p=2, dim=-1)
            .flatten()[0]
            .item()
        )
        return math.degrees(2.0 * math.atan(self.screen_scale_factor / max(d, 1e-9)))

    def set_fov(self, fov):
        """Set the vertical field of view (degrees). The camera stays where it
        is; its screen moves along the forward axis so that the given angle is
        spanned (small fov = telephoto, large fov = wide angle). Animatable.

        Parameters
        ----------
        fov
            Vertical field of view in degrees, in (0, 180).
        """
        fov = self._validated_fov(fov)
        d = self.screen_scale_factor / math.tan(math.radians(fov) * 0.5)
        self.orthographic = False
        self.screen.move_to(self.location + self.get_forward_direction() * d)
        return self

    fov = property(get_fov, set_fov)

    def get_near(self):
        """Near clip distance (world units from the camera along its forward
        axis); geometry closer than this is not rendered. 0 = disabled."""
        return getattr(self, "_near", 0.0)

    def set_near(self, near):
        """Set the near clip plane distance (0 disables near clipping)."""
        near = self._validated_clip("near", near)
        self._validate_clip_order(near, self.far)
        self._near = near
        return self

    near = property(get_near, set_near)

    def get_far(self):
        """Far clip distance (world units of ray travel from the camera);
        geometry farther than this shows the background/environment instead.
        0 = disabled."""
        return getattr(self, "_far", 0.0)

    def set_far(self, far):
        """Set the far clip distance (0 disables far clipping)."""
        far = self._validated_clip("far", far)
        self._validate_clip_order(self.near, far)
        self._far = far
        return self

    far = property(get_far, set_far)

    def set_distance_to_screen(self, distance):
        """Moves the camera focus to be the given distance away from its screen, thereby changing the perspective.

        Parameters
        ----------
        distance
            The camera focus will be set to be this distance away from the screen.
        """
        return self._set_distance_to_screen(distance, preserve_mode=False)

    def _set_distance_to_screen(self, distance, *, preserve_mode):
        distance = self._validated_positive("distance", distance)
        if not preserve_mode:
            self.orthographic = False
        self.set_non_recursive(
            location=self.screen.location - self.get_forward_direction() * distance
        )
        return self

    def set_euler_angles(self, angle_1, angle_2, angle_3):
        with Sync(animation_manager=self.animation_manager):
            self.rotate(angle_1, RIGHT, about_point=ORIGIN)
            self.rotate(angle_2, UP, about_point=ORIGIN)
            self.rotate(angle_3, OUT, about_point=ORIGIN)
        return self

    def get_render_screen_basis(self):
        """Per-frame screen basis used by the renderers to project the scene.

        Derived from the camera's own basis -- which is purely rotational,
        the camera mob is never scaled -- with the screen's in-plane scale
        applied along the screen's *local* axes. The screen mob's stored
        basis cannot be used directly: its non-uniform scale is applied
        along world axes (basis = rotation @ scale), so once the camera
        rotates the rows skew and shrink/stretch, which makes the projection
        anisotropic -- the image visibly squashes with the orbit angle
        (e.g. a sphere renders as an ellipse).
        """
        basis = unsquish(self.basis, -1, 3).clone()
        basis[..., :2, :] = basis[..., :2, :] / self.screen_scale_factor
        return basis

    def retroactive_center(self, mob, **kwargs):
        with self.retroactive():
            self.move_to_make_mob_center_of_view(mob, **kwargs)
        return self

    def move_to_make_mob_center_of_view(self, mob, buffer_portion=0.7):
        f = self.get_forward_direction()
        r = self.get_right_direction()
        u = self.get_upwards_direction()
        # mob_boundary_points = [mob.get_boundary_in_direction(_) + _ * buffer for _ in [u, r, -u, -r]]
        mob_boundary_points = [mob.get_boundary_in_direction(_) for _ in [-r, u, r, -u]]
        mob_boundary_points = torch.stack(mob_boundary_points)
        # mobl = sum(mob_boundary_points) / len(mob_boundary_points)
        mobl = 0.5 * ((mob_boundary_points).amax(0) + (mob_boundary_points).amin(0))
        mob_boundary_points = (mob_boundary_points - mobl) * (1 + buffer_portion) + mobl
        selfl = self.location

        with Sync(animation_manager=self.animation_manager):
            self.move_to(mobl - f * dot_product(mobl - selfl, f))
            selfl = self.location

            corner_rays = F.normalize(self.get_corner_pixels() - selfl, dim=-1, p=2)
            edge_plane_rays = torch.stack(
                (corner_rays, torch.cat((corner_rays[1:], corner_rays[:1]))), 1
            )
            up_plane = edge_plane_rays[1]
            right_plane = edge_plane_rays[0]

            vertical_move, vertical_dist = intersect_line_with_plane_colinear(
                -f, selfl, up_plane[0], up_plane[1], mob_boundary_points[1]
            )
            horizontal_move, horizontal_dist = intersect_line_with_plane_colinear(
                -f, selfl, right_plane[0], right_plane[1], mob_boundary_points[0]
            )

            return self.move(
                (mob_boundary_points[1] - vertical_move)
                if vertical_dist <= horizontal_dist
                else (mob_boundary_points[0] - horizontal_move)
            )

    def project_point_onto_screen_border(self, point, direction):
        corner_rays = F.normalize(
            (self.get_corner_pixels()) - self.location, dim=-1, p=2
        )
        edge_plane_rays = torch.stack(
            (corner_rays, torch.cat((corner_rays[1:], corner_rays[:1])))
        )
        intersection_points, intersection_distances = (
            torch.stack(_)
            for _ in zip(
                *[
                    intersect_line_with_plane_colinear(
                        direction,
                        self.location,
                        edge_plane_rays[0, i],
                        edge_plane_rays[1, i],
                        point,
                    )
                    for i in range(edge_plane_rays.shape[1])
                ]
            )
        )
        intersection_distances = intersection_distances.nan_to_num(
            nan=1e12, posinf=1e12, neginf=1e12
        )
        m = (
            intersection_distances.sign()
            == torch.cat(
                (intersection_distances[2:], intersection_distances[:2])
            ).sign()
        ).float()
        diff_sign_intersection_distances = intersection_distances.clone()
        diff_sign_intersection_distances[diff_sign_intersection_distances < 0] = 1e12
        intersection_distances = (
            intersection_distances.abs() * m
            + (1 - m) * diff_sign_intersection_distances
        )
        closest_ind = intersection_distances.argmin(0, keepdim=True)
        closest_point = broadcast_gather(
            intersection_points, 0, closest_ind, keepdim=False
        )
        return closest_point

    def get_corner_pixels(self):
        b = unsquish(self.screen.basis, -1, 3)
        b = b / b.norm(p=2, dim=-1, keepdim=True).square().clamp_min(1e-6)
        aspect_ratio = (
            self.scene.video_settings.resolution[0]
            / self.scene.video_settings.resolution[1]
        )
        return (
            self.screen.location
            + b[..., 0, :] * self.corner_x_coords * aspect_ratio
            + b[..., 1, :] * self.corner_y_coords
        )
