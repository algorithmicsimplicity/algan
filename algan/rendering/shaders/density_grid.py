"""Density grids: interiors whose density varies from place to place.

A :class:`DensityGrid` handed to
:class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial` as
``density=`` scales the material's scattering and absorption coefficients by a
3-D scalar field, so one closed Mob can hold a puff of smoke, a cloud or a
patch of fog that thins out towards its edges, instead of a medium of the same
density everywhere. The path tracer renders it (see
``algan.rendering.raytracing.pt_volume_taichi`` for the transport).

The grid is laid over a box in the frame of the Mob that ``set_material`` was
called on. The box moves, turns and scales with that Mob, so an animated cloud
carries its density with it.

Internals. ``set_material`` leaves one :class:`_DensityBinding` on every Mob of
the subtree: the grid, the map from the owner's local frame to the box's
``[0, 1]^3`` coordinates, a positive integer *key* and the owner. The geometry
Mobs write the key into their triangles' material block
(``RayTracedTrianglePrimitive.declare_density_volume``); the render loop
snapshots each live owner's per-frame world-to-box map beside the camera
(:func:`_density_volume_snapshot`), and the path tracer looks the key up in
the table it builds from that snapshot (``pt_volume.prepare_volumes``).
"""

from __future__ import annotations

import itertools
import math
import weakref

import torch

__all__ = ["DensityGrid"]

#: Keys travel through a float32 material slot, which holds every integer
#: below 2**24 exactly.
_KEY_LIMIT = 1 << 24
_KEYS = itertools.count(1)
#: Every binding still referenced by some Mob, so the render loop can find the
#: owners without walking the scene.
_BINDINGS = weakref.WeakSet()

#: Samples one grid may hold. The kernel indexes the samples with 32-bit
#: integers; this keeps a grid, its majorant and its neighbours well inside
#: that, and a 256**3 grid well inside it.
_MAX_SAMPLES = 1 << 26
#: Density samples per coarse majorant cell along each axis, and the most
#: coarse cells along one axis.
_MAJORANT_CELL = 4
_MAJORANT_MAX_CELLS = 64


def _parse_resolution(resolution):
    if isinstance(resolution, bool):
        raise TypeError("resolution must be an integer or three integers")
    if isinstance(resolution, int):
        resolution = (resolution,) * 3
    resolution = tuple(resolution)
    if len(resolution) != 3 or not all(
        isinstance(n, int) and not isinstance(n, bool) for n in resolution
    ):
        raise TypeError("resolution must be an integer or three integers")
    if min(resolution) < 1:
        raise ValueError(f"resolution must be at least 1 per axis, got {resolution}")
    return resolution


def _parse_bounds(bounds):
    if bounds is None:
        return None
    try:
        t = torch.as_tensor(bounds).detach().cpu().to(torch.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"bounds must be two corners, ((x0, y0, z0), (x1, y1, z1)): {error}"
        ) from None
    if t.shape != (2, 3):
        raise ValueError(
            "bounds must be two corners, ((x0, y0, z0), (x1, y1, z1)), "
            f"got shape {tuple(t.shape)}"
        )
    if not bool(torch.isfinite(t).all()):
        raise ValueError("bounds must be finite")
    if not bool((t[1] > t[0]).all()):
        raise ValueError(
            "bounds' second corner must exceed its first along every axis, "
            f"got {t.tolist()}"
        )
    return t


def _hash_unit(ix, iy, iz, seed):
    """A uniform value in ``[0, 1)`` per integer lattice point.

    Wellons' lowbias32 over a 32-bit combine of the coordinates, in int64
    arithmetic: the products wrap, and only their low 32 bits are kept, so the
    result is the same on every device.
    """
    mask = 0xFFFFFFFF
    h = (
        (ix * 0x8DA6B343) ^ (iy * 0xD8163841) ^ (iz * 0xCB1AB31F) ^ (seed * 0x165667B1)
    ) & mask
    h = h ^ (h >> 16)
    h = (h * 0x7FEB352D) & mask
    h = h ^ (h >> 15)
    h = (h * 0x846CA68B) & mask
    h = h ^ (h >> 16)
    return h.to(torch.float64) * (1.0 / 4294967296.0)


class DensityGrid:
    """Make a 3-D density field for a scattering interior: smoke, clouds, fog.

    Pass one to :class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial`
    as ``density=``. The material's ``sigma_s`` and absorption then hold where
    the density is 1 and scale with it everywhere else: twice as thick where it
    is 2, empty where it is 0. Color, ``g`` and the surface stay the
    material's.

    The samples fill a box in the frame of the Mob the material is set on.
    Without ``bounds`` that box is the Mob's bounding box, measured when
    ``set_material`` is called. Sample ``values[i, j, k]`` sits at the centre of
    cell ``(i, j, k)`` when the box is cut into ``nx * ny * nz`` equal cells,
    ``i`` along the Mob's own right axis, ``j`` along its up axis and ``k``
    along its forward axis. Between samples the density is interpolated
    linearly, inside the box past the outermost samples it holds their value,
    and outside the box it is 0.

    Only a closed interior has a density -- the same requirement as ``sigma_s``
    -- and the grid only shows inside it: a box larger than the Mob is harmless,
    and a box smaller than the Mob leaves the rest of the interior empty.

    Animation
    ---------
    Construction is immediate and records nothing. The samples themselves
    cannot be animated, but the field moves, turns and scales with the Mob,
    so an animated cloud carries its density along. ``mob.sigma_s`` stays
    animatable and scales the whole field at once.

    Parameters
    ----------
    values
        Density samples, shape ``(nx, ny, nz)``, each finite and non-negative.
        Any array-like is accepted and copied. 1 means the material's own
        coefficients.
    bounds
        The box's two corners, ``((x0, y0, z0), (x1, y1, z1))``, in the Mob's
        own coordinates: offsets from its ``location`` along its right, up and
        forward axes, in its own units, as it stands when ``set_material`` is
        called. Defaults to None, using the Mob's bounding box.

    Raises
    ------
    ValueError
        If ``values`` is not 3-D, is empty, holds a negative or non-finite
        value or more than 2**26 samples, or if ``bounds`` is not two finite
        corners with the second above the first on every axis.

    See Also
    --------
    :meth:`DensityGrid.from_function` : Build a grid by evaluating a function.
    :meth:`DensityGrid.noise` : Smooth random values for clouds and smoke.

    Notes
    -----
    Requires path tracing (``samples_per_pixel > 1``); the deterministic
    renderer ignores the interior altogether, as it does ``sigma_s``. Render
    time grows with the densest parts of the grid, since the tracker takes
    more steps through them. The grid lives on the owner Mob, so a
    :func:`~algan.utils.mob_utils.batch_mobs` pack of several owners is
    refused at render time; give each its own Mob instead.

    Examples
    --------
    .. algan:: DensityGridSmokePuff

        from algan import *

        SETTINGS.raytracing.set(samples_per_pixel=64, max_bounces=16)
        puff = DensityGrid.from_function(
            lambda p: (1 - p.norm(dim=-1)).clamp(min=0) * DensityGrid.noise(p) * 4
        )
        cloud = Sphere(radius=1.5)
        cloud.set_material(MeshPhysicalMaterial(
            color=WHITE, transmission=1, ior=1, roughness=0,
            sigma_s=4, density=puff,
        ))
        cloud.spawn()
        cloud.rotate(90, UP)
        Scene.save_video()
    """

    def __init__(self, values, *, bounds=None):
        t = torch.as_tensor(values, dtype=torch.float32).detach().cpu()
        if t.dim() != 3:
            raise ValueError(
                f"values must have shape (nx, ny, nz), got {tuple(t.shape)}"
            )
        if t.numel() == 0:
            raise ValueError(f"values must not be empty, got {tuple(t.shape)}")
        if t.numel() > _MAX_SAMPLES:
            raise ValueError(
                f"values holds {t.numel()} samples, more than the {_MAX_SAMPLES} "
                "a grid may hold; sample it more coarsely"
            )
        if not bool(torch.isfinite(t).all()) or bool((t < 0).any()):
            raise ValueError("values must be finite and non-negative")
        self._values = t.clone().contiguous()
        self._bounds = _parse_bounds(bounds)
        self._packed = None

    @property
    def values(self) -> torch.Tensor:
        """The density samples, shape ``(nx, ny, nz)``. A copy; read-only."""
        return self._values.clone()

    @property
    def resolution(self) -> tuple[int, int, int]:
        """The number of samples along each axis, ``(nx, ny, nz)``."""
        return tuple(int(n) for n in self._values.shape)

    @property
    def bounds(self) -> torch.Tensor | None:
        """The box's corners, shape ``(2, 3)``, or None for the Mob's bounding box."""
        return None if self._bounds is None else self._bounds.clone()

    def __repr__(self):
        nx, ny, nz = self.resolution
        return f"DensityGrid({nx}x{ny}x{nz}, max={float(self._values.max()):.4g})"

    @classmethod
    def from_function(cls, function, resolution=32, *, bounds=None) -> DensityGrid:
        """Make a grid by evaluating ``function`` at every sample point.

        The points are given in box coordinates: -1 on the box's lower face
        and +1 on its upper face along each axis, so ``p.norm(dim=-1)`` is 0 at
        the box's centre and 1 on the sphere that touches its faces, whatever
        size the box turns out to be.

        Parameters
        ----------
        function
            Called once with a tensor of points, shape ``(nx, ny, nz, 3)``;
            must return the densities, shape ``(nx, ny, nz)``, finite and
            non-negative.
        resolution
            Samples per axis: one integer for all three, or ``(nx, ny, nz)``.
            Defaults to 32.
        bounds
            As for :class:`DensityGrid`. Defaults to None, the Mob's bounding
            box.

        Returns
        -------
        :class:`DensityGrid`
            The sampled grid.

        Raises
        ------
        ValueError
            If ``resolution`` is below 1, or the returned densities have the
            wrong shape or a negative or non-finite value.
        """
        nx, ny, nz = _parse_resolution(resolution)
        axes = [
            (torch.arange(n, dtype=torch.float32) + 0.5) / n * 2.0 - 1.0
            for n in (nx, ny, nz)
        ]
        points = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        values = torch.as_tensor(function(points), dtype=torch.float32)
        if values.shape == (nx, ny, nz, 1):
            values = values[..., 0]
        if tuple(values.shape) != (nx, ny, nz):
            raise ValueError(
                f"function must return densities of shape {(nx, ny, nz)}, "
                f"got {tuple(values.shape)}"
            )
        return cls(values, bounds=bounds)

    @staticmethod
    def noise(points, *, frequency=4.0, octaves=4, seed=0) -> torch.Tensor:
        """Get smooth random values in ``[0, 1]`` at ``points``: the makings of a cloud.

        Fractal value noise: ``octaves`` layers of smoothly interpolated random
        lattice values, each at twice the frequency and half the weight of the
        one before. Values average 0.5. Shape a cloud by multiplying it with a
        falloff, or carve wisps by subtracting a threshold::

            DensityGrid.from_function(
                lambda p: (DensityGrid.noise(p, seed=3) - 0.45).clamp(min=0) * 8
            )

        The same points, frequency, octaves and seed always give the same
        values, on every device.

        Parameters
        ----------
        points
            Positions, shape ``(..., 3)``, usually the box coordinates
            :meth:`from_function` hands its function.
        frequency
            Lattice cells per unit of position in the first layer. Defaults
            to 4, about four blobs across a box from -1 to 1 counting both
            halves.
        octaves
            Number of layers, at least 1. Defaults to 4.
        seed
            Integer choosing the random pattern. Defaults to 0.

        Returns
        -------
        torch.Tensor
            The values, shape ``points.shape[:-1]``, float32.

        Raises
        ------
        ValueError
            If ``points`` does not end in 3 coordinates, ``frequency`` is not
            positive and finite, or ``octaves`` is below 1.
        """
        p = torch.as_tensor(points).detach().cpu().to(torch.float64)
        if p.dim() == 0 or p.shape[-1] != 3:
            raise ValueError(f"points must have shape (..., 3), got {tuple(p.shape)}")
        frequency = float(frequency)
        if not (math.isfinite(frequency) and frequency > 0.0):
            raise ValueError(f"frequency must be positive and finite, got {frequency}")
        octaves = int(octaves)
        if octaves < 1:
            raise ValueError(f"octaves must be at least 1, got {octaves}")
        seed = int(seed)
        total = torch.zeros(p.shape[:-1], dtype=torch.float64)
        weight = 1.0
        weights = 0.0
        for octave in range(octaves):
            q = p * (frequency * 2.0**octave)
            cell = torch.floor(q)
            frac = q - cell
            # Quintic fade: the interpolant's first two derivatives vanish at
            # lattice points, so no grid shows through the layers.
            fade = frac * frac * frac * (frac * (frac * 6.0 - 15.0) + 10.0)
            i = cell.to(torch.int64)
            # Kept to 32 bits: a wider seed would overflow the int64 hash.
            octave_seed = (seed * 0x9E3779B1 + octave * 0x7F4A7C15) & 0xFFFFFFFF
            value = torch.zeros_like(total)
            for corner in range(8):
                ox, oy, oz = (corner >> 2) & 1, (corner >> 1) & 1, corner & 1
                lattice = _hash_unit(
                    i[..., 0] + ox, i[..., 1] + oy, i[..., 2] + oz, octave_seed
                )
                w = (
                    (fade[..., 0] if ox else 1.0 - fade[..., 0])
                    * (fade[..., 1] if oy else 1.0 - fade[..., 1])
                    * (fade[..., 2] if oz else 1.0 - fade[..., 2])
                )
                value = value + w * lattice
            total = total + weight * value
            weights += weight
            weight *= 0.5
        return (total / weights).to(torch.float32)

    def _majorant_grid(self):
        """``(majorant, (mx, my, mz))``: an upper bound per coarse cell.

        Coarse cell ``c`` along an axis covers box coordinates
        ``[c / m, (c + 1) / m)``. The trilinear interpolant there blends only
        samples ``floor(c n / m - 1/2)`` to ``ceil((c + 1) n / m - 1/2)``
        (clamped to the grid), so their maximum bounds it. The range is widened
        by one sample each way, so a point the tracker's cell walk assigns to
        a neighbouring cell by rounding is still bounded.
        """
        values = self._values
        dims = []
        for axis in range(3):
            n = int(values.shape[axis])
            m = max(1, min(_MAJORANT_MAX_CELLS, math.ceil(n / _MAJORANT_CELL)))
            dims.append(m)
            slabs = []
            for c in range(m):
                lo = max(0, math.floor(c * n / m - 0.5) - 1)
                hi = min(n - 1, math.ceil((c + 1) * n / m - 0.5) + 1)
                slabs.append(values.narrow(axis, lo, hi - lo + 1).amax(axis))
            values = torch.stack(slabs, axis)
        return values.contiguous(), tuple(dims)

    def _packed_grid(self):
        """``(density samples, majorant, (nx, ny, nz), (mx, my, mz))``, cached."""
        if self._packed is None:
            majorant, coarse = self._majorant_grid()
            self._packed = (
                self._values.reshape(-1),
                majorant.reshape(-1),
                self.resolution,
                coarse,
            )
        return self._packed


def _new_key():
    key = next(_KEYS)
    if key >= _KEY_LIMIT:
        raise RuntimeError(
            "too many density grids have been set in this process "
            f"({_KEY_LIMIT - 1}); restart it"
        )
    return key


def _frame(mob):
    """``(location (3,), basis (3, 3))`` of ``mob`` now, float64.

    The basis rows are the Mob's right, up and forward axes, scaled, so a
    point with local coordinates ``l`` sits at ``location + l @ basis``.
    """
    location = mob.location.detach().reshape(-1, 3)[-1].cpu().to(torch.float64)
    basis = mob.basis.detach().reshape(-1, 9)[-1].cpu().to(torch.float64)
    return location, basis.reshape(3, 3)


def _local_points(mob, location, basis):
    """Every boundary location of ``mob``'s subtree, in its local frame.

    The same points :meth:`~.Mob.get_bounding_box` measures (children marked
    ``exclude_from_boundary`` are skipped), taken at their current values.
    """
    points = []

    def collect(node):
        if not node.exclude_from_boundary:
            points.append(node.location.detach().reshape(-1, 3).cpu().to(torch.float64))
        for child in node.children:
            collect(child)

    collect(mob)
    world = torch.cat(points, 0)
    return torch.linalg.solve(basis.T, (world - location).T).T


class _DensityBinding:
    """A grid bound to an owner Mob's frame (see the module docstring).

    ``local_to_box`` is a ``(3, 4)`` float64 affine map from the owner's local
    coordinates to the box's ``[0, 1]^3``: ``u = A @ l + b``.
    """

    __slots__ = ("grid", "local_to_box", "key", "_owner", "__weakref__")

    def __init__(self, grid, local_to_box, owner):
        self.grid = grid
        self.local_to_box = local_to_box
        self.key = _new_key()
        self._owner = weakref.ref(owner)
        _BINDINGS.add(self)

    @property
    def owner(self):
        return self._owner()

    def __deepcopy__(self, memo):
        """A clone of the owner owns a clone of the binding, with its own key.

        Both start in the same place and move independently from then on, so
        each needs its own per-frame transform. A copy that shares its source's
        data (a view, ``clone_data=False``) shares the binding as well: it
        draws through its source, never by itself.
        """
        if id(self) in memo:
            return memo[id(self)]
        if not memo.get("___clone_data___", True):
            memo[id(self)] = self
            return self
        owner = self.owner
        new_owner = memo.get(id(owner)) if owner is not None else None
        local_to_box = self.local_to_box
        if new_owner is None:
            # A part cloned without the Mob that owns the grid: the clone root
            # takes over, with the box re-expressed in its frame as both stand
            # now, so the clone starts with the field exactly where its source
            # had it.
            root_id = memo.get("___copy_root_id___")
            source_root = None
            if owner is not None:
                source_root = next(
                    (m for m in owner.get_descendants() if id(m) == root_id), None
                )
            new_owner = memo.get(root_id)
            if source_root is not None:
                local_to_box = _rebase(local_to_box, owner, source_root)
        if new_owner is None:
            memo[id(self)] = self
            return self
        clone = _DensityBinding(self.grid, local_to_box.clone(), new_owner)
        memo[id(self)] = clone
        return clone


def _rebase(local_to_box, owner, mob):
    """``local_to_box`` re-expressed in ``mob``'s local frame, both as they stand."""
    owner_location, owner_basis = _frame(owner)
    location, basis = _frame(mob)
    # world = location + l @ basis; owner local = (world - owner_location) @ inv(owner_basis)
    inv_owner = torch.linalg.inv(owner_basis)
    to_owner = basis @ inv_owner  # l (row) -> owner local (row): l @ to_owner + shift
    shift = (location - owner_location) @ inv_owner
    a = local_to_box[:, :3]
    b = local_to_box[:, 3]
    return torch.cat((a @ to_owner.T, (a @ shift + b).unsqueeze(1)), 1)


def _bind_density(grid, owner):
    """The binding ``set_material`` leaves on ``owner``'s subtree for ``grid``."""
    if not isinstance(grid, DensityGrid):
        raise TypeError(
            f"density must be a DensityGrid or None, got {type(grid).__name__}"
        )
    location, basis = _frame(owner)
    if abs(float(torch.linalg.det(basis))) < 1e-12:
        raise ValueError(
            "cannot place a density grid on a Mob scaled to zero along some axis"
        )
    if grid._bounds is not None:
        lower, upper = grid._bounds[0], grid._bounds[1]
    else:
        local = _local_points(owner, location, basis)
        lower = local.amin(0)
        upper = local.amax(0)
        # A flat Mob has no extent along some axis; give the box a sliver so
        # the map stays finite (its interior is empty anyway).
        pad = max(float((upper - lower).max()) * 1e-6, 1e-9)
        lower = lower - pad
        upper = upper + pad
    scale = 1.0 / (upper - lower)
    local_to_box = torch.cat((torch.diag(scale), (-lower * scale).unsqueeze(1)), 1).to(
        torch.float64
    )
    return _DensityBinding(grid, local_to_box, owner)


def _density_key(mob) -> int:
    """The key ``mob``'s triangles carry: 0 for a homogeneous interior."""
    binding = getattr(mob, "_density_binding", None)
    return 0 if binding is None else int(binding.key)


def _world_to_box(local_to_box, location, basis):
    """Per-frame world-to-box maps, ``(T, 12)`` float32: 3x3 row-major, then shift.

    ``location`` is ``(T, 3)`` and ``basis`` ``(T, 3, 3)``, float64. A frame
    whose basis is singular (the Mob scaled to nothing) maps every point to
    ``(-1, -1, -1)``, outside the box, so its interior is empty that frame.
    """
    a = local_to_box[:, :3]
    b = local_to_box[:, 3]
    det = torch.linalg.det(basis)
    ok = det.abs() > 1e-30
    safe = torch.where(ok[:, None, None], basis, torch.eye(3, dtype=basis.dtype))
    inv = torch.linalg.inv(safe)  # world row -> local row: (w - loc) @ inv
    # u = A @ ((w - loc) @ inv)^T + b = (A @ inv^T) @ w - (A @ inv^T) @ loc + b
    m = a.unsqueeze(0) @ inv.transpose(-1, -2)
    t = b.unsqueeze(0) - (m @ location.unsqueeze(-1)).squeeze(-1)
    m = torch.where(ok[:, None, None], m, torch.zeros_like(m))
    t = torch.where(ok[:, None], t, torch.full_like(t, -1.0))
    return torch.cat((m.reshape(-1, 9), t), 1).to(torch.float32)


def _density_volume_snapshot(scene, window_start_time, window_end_time, num_frames):
    """The render loop's per-batch snapshot of every live density grid.

    Returns None when no grid is live in the window, else ``(entries, xforms)``:
    ``entries`` a tuple of ``(key, DensityGrid)`` and ``xforms`` a
    ``(num_frames, len(entries), 12)`` float32 CPU tensor of world-to-box maps
    (:func:`_world_to_box`), read from each owner's materialized per-frame
    location and basis -- the rows the camera snapshot beside it was read
    from.
    """
    from algan.errors import AlganConfigurationError

    owners = []
    for binding in list(_BINDINGS):
        owner = binding.owner
        if owner is None or getattr(owner, "_density_binding", None) is not binding:
            continue
        if getattr(owner, "scene", None) is not scene:
            continue
        try:
            spawn_time = float(owner.lifespan.start())
            despawn_time = float(owner.lifespan.end())
        except (AttributeError, TypeError, ValueError):
            continue
        # The render loop's lifespan-overlap test: start < 0 means never
        # spawned, end < 0 means never despawned.
        if spawn_time < 0 or spawn_time > window_end_time:
            continue
        if 0 <= despawn_time < window_start_time:
            continue
        owners.append((binding, owner))
    if not owners:
        return None
    owners.sort(key=lambda pair: pair[0].key)
    entries = []
    xforms = []
    for binding, owner in owners:
        location = owner.location.detach().cpu().to(torch.float64)
        basis = owner.basis.detach().cpu().to(torch.float64)
        if location.shape[-2] != 1:
            raise AlganConfigurationError(
                f"{owner._describe()} packs {location.shape[-2]} Mobs into one "
                "and owns a DensityGrid. A density grid follows one Mob's "
                "frame, so a pack of several cannot own one; leave those Mobs "
                "unpacked"
            )
        location = location.reshape(-1, 3)
        basis = basis.reshape(-1, 3, 3)
        rows = max(int(location.shape[0]), int(basis.shape[0]))
        location = location.expand(rows, 3)
        basis = basis.expand(rows, 3, 3)
        xform = _world_to_box(binding.local_to_box, location, basis)
        if rows != num_frames:
            if rows != 1:
                raise RuntimeError(
                    f"density grid owner materialized {rows} frames for a "
                    f"{num_frames}-frame window"
                )
            xform = xform.expand(num_frames, 12)
        entries.append((binding.key, binding.grid))
        xforms.append(xform)
    return tuple(entries), torch.stack(xforms, 1).contiguous()
