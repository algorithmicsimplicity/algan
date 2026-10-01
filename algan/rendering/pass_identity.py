"""Which Mob a rendered pixel belongs to, for the object-id compositing pass.

The renderer can name, per pixel, the primitive that is visible there: a global
triangle SURFACE id (a value of the merged scene's ``tri_obj``) or a merged
circuit index (a column of ``circuit_meta``). While an object-id pass is armed
-- ``scene._aux_id_registry`` set to a dict for the duration of one render --
the render loop stamps every primitive with the ``Mob.id`` of the actor that
built it, and the scene merge turns those stamps into two host tables:

``merged["tri_obj_source_ids"]``
    int32 numpy array, global surface id -> source ``Mob.id`` (``-1``: none).
``merged["circuit_source_ids"]``
    int32 numpy array, merged circuit index -> source ``Mob.id`` (``-1``: none).

and ``scene._aux_id_registry`` ends up mapping each of those ids to its actor.
This module is the host-side rest of the chain:

* :func:`resolve_pass_owner` turns an actor into the ``(pass_id, owner)`` the
  pass writes -- an authored :attr:`~algan.animatable_base.mob.Mob.pass_index`,
  or an automatic id naming the object the actor is part of.
* :func:`pass_id_table` does that for a whole registry at once, as a dense
  lookup table the per-pixel id image can be gathered through.
* :func:`pass_id_color` / :func:`pass_id_from_color` / :func:`pass_id_colors` /
  :func:`pass_ids_from_colors` are a bijection between 24-bit pass ids and
  24-bit RGB, so an id image can be stored as an ordinary 8-bit colour image
  and decoded exactly.
* :func:`describe_owner` is the JSON sidecar entry for one owner.

Nothing here is star-exported (``algan/__init__.py`` never imports it), and it
imports no viewer or kernel code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from algan.animatable_base.mob import Mob

#: First automatic pass id. Authored ``pass_index`` values occupy
#: ``1 .. AUTO_ID_BASE - 1`` (``Mob.pass_index`` rejects anything larger), and a
#: Mob with none resolves to ``AUTO_ID_BASE + owner.id``, so the two ranges can
#: never collide. 0 is the background.
AUTO_ID_BASE = 65536

#: Pass ids and their colours are 24-bit: one byte each of R, G and B.
PASS_ID_BITS = 24
_ID_MASK = (1 << PASS_ID_BITS) - 1

#: The colour bijection multiplies by this odd constant modulo 2**24 -- the
#: 24-bit golden-ratio multiplier, floor(2**24 / phi) -- which is invertible
#: because it is odd, maps 0 to 0, and scatters consecutive ids across all three
#: channels (ids 1, 2, 3 land on (158, 55, 121), (60, 110, 242), (218, 166, 107)).
_COLOR_MULTIPLIER = 0x9E3779
#: Its inverse modulo 2**24, which undoes it exactly.
_COLOR_INVERSE = pow(_COLOR_MULTIPLIER, -1, 1 << PASS_ID_BITS)


# ---------------------------------------------------------------------------
# Which Mob a primitive's source belongs to
# ---------------------------------------------------------------------------

_CONTAINER_CLASSES = None


def _container_classes():
    """The classes whose instances are pure grouping nodes, resolved once.

    Exact classes, never ``isinstance``: a subclass exists because it means
    something (``Paragraph`` is a Group that is one paragraph, a user's
    ``class Molecule(Group)`` is one molecule), so only the anonymous grouping
    types themselves are transparent:

    * ``Mob`` itself -- the structural node composites hang their parts from
      (``TriangulatedBezierCircuit.tiles``, the wrappers Manim conversions put
      between a composite and its sub-paths);
    * ``Group``, whose whole job is to hold other Mobs;
    * the Manim-compatibility ``VGroup``, ``Group`` and ``VDict`` wrappers and
      the point-cloud ``PGroup`` family, which are the same thing in Manim's
      spelling (their OpenGL names alias these classes);
    * ``OpenGLSurfaceGroup``, Manim's group of parametric surfaces.

    Looked up lazily so this module imports nothing from the Mob tree at load
    time; every one of those modules is loaded by ``import algan`` anyway.
    """
    global _CONTAINER_CLASSES
    if _CONTAINER_CLASSES is None:
        from algan.animatable_base.mob import Mob
        from algan.mobs import manim_compat, opengl_compat, point_cloud
        from algan.mobs.group import Group

        found = {Mob, Group}
        registry = getattr(manim_compat, "_MANIM_WRAPPER_REGISTRY", {})
        found.update(
            registry[name] for name in ("VGroup", "Group", "VDict") if name in registry
        )
        for module, names in (
            (point_cloud, ("PGroup", "OpenGLPGroup")),
            (opengl_compat, ("OpenGLSurfaceGroup",)),
        ):
            found.update(
                getattr(module, name) for name in names if hasattr(module, name)
            )
        _CONTAINER_CLASSES = frozenset(found)
    return _CONTAINER_CLASSES


def is_container(mob: Mob) -> bool:
    """Whether ``mob`` is a pure grouping node, transparent to ownership."""
    return type(mob) in _container_classes()


def auto_owner(mob: Mob) -> Mob:
    """The object ``mob`` is automatically reported as part of.

    The highest non-container on the chain ``mob -> parents[0] -> parents[0]
    ...``: a glyph pack reports its ``Text``, a cap its ``Cylinder``, a
    triangulated fill its ``TriangulatedBezierCircuit``, a tick its
    ``NumberLine``, while ``Group(a, b)`` reports ``a`` and ``b`` separately.
    If every node on the chain is a container, the last drawn Mob ``mob``
    stands for (``mob`` itself, a history clone's original): a stand-in for
    a container keeps the identity it had, since the container has none of
    its own to lend.

    Pass the actor the render registered (``scene._aux_id_registry[id]``), not
    a packed view: a view shares its pack's id but has no ``parents``.
    """
    drawn, seen = mob, set()
    step = mob
    while step is not None and id(step) not in seen:
        seen.add(id(step))
        if not is_container(step):
            drawn = step
        step = getattr(step, "_pass_identity", None)
    owner = None
    seen = set()
    node = owner_identity(mob)
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if not is_container(node):
            owner = node
        parents = pass_parents(node)
        # A parent spliced out by a become() is followed to the Mob that took
        # its place: the hierarchy the author sees in the end.
        node = owner_identity(parents[0]) if parents else None
    return drawn if owner is None else owner


def owner_identity(mob: Mob) -> Mob:
    """:func:`render_identity`, then an ownership-only link if one is set.

    A composite a ``become()`` retired whose result already carries a nearer
    tag cannot stand for that result outright -- its other members would take
    that tag -- but it is still the same object, so ``become`` records the
    result in ``_pass_owner_identity`` (``mob_morph._hand_on_structure_identity``)
    and only :func:`auto_owner` follows it.
    """
    mob = render_identity(mob)
    successor = getattr(mob, "_pass_owner_identity", None)
    return mob if successor is None else render_identity(successor)


def pass_parents(mob: Mob):
    """The parents identification continues through from ``mob``.

    Its ``parents``; or, for a ``become()`` root that was taken out of its
    parents without a Mob standing in its place -- a Group that became one of
    its own members, say -- the parents it had, which ``become`` records in
    ``_pass_parents`` (``mob_morph._keep_pass_parents``) so the members it
    still holds keep reaching the tags and the owner above it. The
    ``pass_index`` search also walks ``_pass_tag_parents``, a tags-only route
    (:func:`_resolved_pass_index`).
    """
    return getattr(mob, "parents", None) or getattr(mob, "_pass_parents", None) or ()


def resolve_pass_owner(mob: Mob) -> tuple[int, Mob]:
    """``(pass_id, owner)`` for the geometry ``mob`` built.

    If ``mob`` or an ancestor sets :attr:`~algan.animatable_base.mob.Mob.pass_index`
    (nearest first, breadth-first over ``parents``), that value and the Mob that
    set it. Otherwise ``AUTO_ID_BASE + owner.id`` and ``owner`` =
    :func:`auto_owner` of ``mob``. ``pass_id`` is always positive, so 0 stays
    free for the background.
    """
    index, setter = _resolved_pass_index(render_identity(mob))
    if index is not None:
        return int(index), setter
    owner = auto_owner(mob)
    return AUTO_ID_BASE + int(owner.id), owner


def _resolved_pass_index(mob: Mob):
    """``Mob._resolved_pass_index`` through :func:`render_identity` links.

    The same nearest-first, breadth-first search over ``parents``, except that
    every node visited is first followed to the Mob it stands for, so a tag on
    a group still reaches a child whose parent a become() replaced, that a
    become() root nothing stands in for continues through the parents it had
    (:func:`pass_parents`), and that a node's ``_pass_tag_parents`` -- the
    source an unspliced dissolve replacement fades in for -- are searched
    after its parents: tags reach through them, ownership does not.
    """
    queue = [mob]
    seen = set()
    while queue:
        node = render_identity(queue.pop(0))
        if id(node) in seen:
            continue
        seen.add(id(node))
        index = getattr(node, "_pass_index", None)
        if index is not None:
            return index, node
        queue.extend(pass_parents(node))
        queue.extend(getattr(node, "_pass_tag_parents", None) or ())
    return None, None


def render_identity(mob: Mob) -> Mob:
    """The Mob ``mob`` is drawing on behalf of, for identification.

    Several internal Mobs render frames of a Mob the author holds: the hidden
    clone :meth:`~algan.animatable_base.mob.Mob.detach_history` hands the
    earlier frames to, a triangle-soup stand-in, a dissolve's replacement,
    and a source a ``become`` spliced out of its parents in favour of its
    replacement. Each records whom it stands for in ``_pass_identity``
    (``mob_morph._link_stand_in``); following that chain to its end names
    the Mob in the final hierarchy. :func:`auto_owner` and the ``pass_index``
    search apply it at every node they visit, so one object keeps one ID --
    and its ``pass_index`` -- across a ``become``, and children keep theirs.
    """
    seen = set()
    while id(mob) not in seen:
        seen.add(id(mob))
        successor = getattr(mob, "_pass_identity", None)
        if successor is None:
            break
        mob = successor
    return mob


def pass_id_table(registry: dict) -> tuple[np.ndarray, dict]:
    """Resolve a whole render's registry into a dense lookup table.

    ``registry`` is the ``{Mob.id: actor}`` dict the render filled
    (``scene._aux_id_registry``). Returns ``(lut, owners)``: ``lut`` is an int32
    array with ``lut[source + 1]`` the pass id of source ``Mob.id`` ``source``
    and ``lut[0] == 0``, so a ``-1`` ("no source") entry of
    ``tri_obj_source_ids`` / ``circuit_source_ids`` lands on the background;
    ``owners`` maps each pass id used to its owner, for :func:`describe_owner`.

    Compose it with a source table on the host -- ``lut[tri_obj_source_ids +
    1]`` is the pass id per global surface -- and upload only the result.

    Safe to call mid-render: the registry is copied first (one atomic step
    under the GIL), because the prefetch worker may be adding the next batch's
    actors to it while the current batch renders.
    """
    registry = dict(registry)
    size = (max(registry) + 2) if registry else 1
    lut = np.zeros(size, dtype=np.int32)
    owners = {}
    for mob_id, actor in registry.items():
        pass_id, owner = resolve_pass_owner(actor)
        lut[int(mob_id) + 1] = pass_id
        owners.setdefault(pass_id, owner)
    return lut, owners


def describe_owner(owner: Mob) -> dict:
    """A JSON-serialisable description of one owner, for the pass's sidecar.

    ``{"class", "mob_id", "label", "name", "pass_index"}``: the class name, the
    ``Mob.id``, the label the viewer uses for it (``"Cube #8"``, or
    ``"title (Text #7)"`` when named -- the rule of
    :func:`algan.viewer.pixels.mob_label`), its name (``None`` if unnamed) and
    its own authored ``pass_index`` (``None`` if it set none).
    """
    name = _mob_name(owner)
    base = f"{type(owner).__name__} #{getattr(owner, 'id', '?')}"
    mob_id = getattr(owner, "id", None)
    pass_index = getattr(owner, "pass_index", None)
    return {
        "class": type(owner).__name__,
        "mob_id": None if mob_id is None else int(mob_id),
        "label": base if name is None else f"{name} ({base})",
        "name": name,
        "pass_index": None if pass_index is None else int(pass_index),
    }


def _mob_name(mob):
    """The Mob's authored name as a string, or ``None`` (unset or ``"_"``).

    Mirrors ``algan.viewer.pixels.mob_label``'s rule without importing the
    viewer package, whose ``__init__`` starts its web server machinery.
    """
    name = getattr(mob, "name", None)
    if not name or name == "_":
        return None
    return str(name)


# ---------------------------------------------------------------------------
# Pass id <-> colour
# ---------------------------------------------------------------------------


def pass_id_color(pass_id: int) -> tuple[int, int, int]:
    """The 8-bit RGB colour a pass id is written as; 0 is black.

    A bijection on ``[0, 2**24)``: :func:`pass_id_from_color` inverts it
    exactly. Consecutive ids land on unrelated colours, so neighbouring objects
    stay distinguishable by eye.
    """
    pass_id = int(pass_id)
    if not 0 <= pass_id <= _ID_MASK:
        raise ValueError(
            f"pass id {pass_id} is outside the 24-bit range [0, {_ID_MASK}]."
        )
    code = (pass_id * _COLOR_MULTIPLIER) & _ID_MASK
    return (code >> 16) & 0xFF, (code >> 8) & 0xFF, code & 0xFF


def pass_id_from_color(rgb) -> int:
    """The pass id an 8-bit RGB colour encodes; black is 0 (background)."""
    r, g, b = (int(channel) for channel in rgb)
    if not all(0 <= channel <= 0xFF for channel in (r, g, b)):
        raise ValueError(f"colour {(r, g, b)} has a channel outside [0, 255].")
    code = (r << 16) | (g << 8) | b
    return (code * _COLOR_INVERSE) & _ID_MASK


def pass_id_colors(ids: torch.Tensor) -> torch.Tensor:
    """:func:`pass_id_color` over a whole tensor, on its own device.

    ``ids`` is an integer tensor of any shape; the result is ``uint8`` with a
    trailing axis of 3 (R, G, B). Negative ids -- the ``-1`` a source table
    uses for "no source" -- are written black like the background, and ids at
    or above ``2**24`` wrap modulo ``2**24`` (automatic ids reach that only past
    sixteen million Mobs).
    """
    ids = ids.to(torch.int64)
    code = ((ids & _ID_MASK) * _COLOR_MULTIPLIER) & _ID_MASK
    code = torch.where(ids > 0, code, torch.zeros_like(code))
    return torch.stack(
        ((code >> 16) & 0xFF, (code >> 8) & 0xFF, code & 0xFF), dim=-1
    ).to(torch.uint8)


def pass_ids_from_colors(rgb: torch.Tensor) -> torch.Tensor:
    """:func:`pass_id_from_color` over a ``[..., 3]`` colour tensor (int64 out)."""
    rgb = rgb.to(torch.int64)
    code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    return (code * _COLOR_INVERSE) & _ID_MASK
