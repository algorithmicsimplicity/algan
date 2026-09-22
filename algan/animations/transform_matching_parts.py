"""Match authored terms or normalized geometry before using Algan's morphs."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import NoExtra, Off, Seq, Sync
from algan.errors import AlganConfigurationError
from algan.mobs.group import Group
from algan.mobs.text import Tex
from algan.utils.api_renames import _renamed_keywords

__all__ = ["TransformMatchingShapes", "TransformMatchingTex"]


def _tex_parts(root):
    """Read token metadata against live Algan geometry, including packed glyphs."""
    parts = []
    seen = set()

    def visit(mob):
        if id(mob) in seen:
            return
        seen.add(id(mob))
        if isinstance(mob, Tex):
            if not mob.latex:
                raise TypeError(
                    "TransformMatchingTex requires LaTeX; use TransformMatchingShapes for Text."
                )
            for key, start, end in zip(
                mob._matching_tex_keys, mob.segment_starts, mob.segment_ends
            ):
                parts.extend(
                    (key, glyph) for glyph in mob.character_mobs[int(start) : int(end)]
                )
            return

        # Read the retained metadata, never a delegated Manim query: those
        # queries can return freshly converted Mobs instead of the live rows.
        backing = mob.__dict__.get("manim_mobject")
        keys = mob.__dict__.get("_matching_tex_keys")
        if keys is not None:
            batch = mob.__dict__.get("_matching_tex_batch")
            children = (
                mob.submobjects
                if batch is None
                else [batch[i] for i in range(len(keys))]
            )
            parts.extend(zip(keys, children))
        elif backing is not None and hasattr(backing, "tex_string"):
            parts.append((backing.tex_string, mob))
        else:
            children = mob.get_non_component_children()
            if mob._morph_family is not None and not getattr(mob, "empty", False):
                raise TypeError(
                    "TransformMatchingTex requires Tex, MathTex, or groups of LaTeX Mobs."
                )
            for child in children:
                visit(child)

    visit(root)
    return parts


def _shape_parts(root):
    """Split containers and packs, but keep a shape's own geometry together."""
    parts = []
    seen = set()

    def visit(mob):
        if id(mob) in seen:
            return
        seen.add(id(mob))
        if isinstance(mob, Tex):
            parts.extend(mob.character_mobs)
            parts.extend(mob.image_mobs)
        elif mob._morph_family is not None and not getattr(mob, "empty", False):
            parts.extend([mob[i] for i in range(len(mob))] if len(mob) > 1 else [mob])
        else:
            for child in mob.get_non_component_children():
                visit(child)

    visit(root)
    return [(_shape_key(part), part) for part in parts]


def _shape_key(mob):
    """Translation/positive-uniform-scale invariant, with ordered topology."""
    point_sets = []
    topology = []

    def collect(node):
        family = node._morph_family
        if family == "image":
            raise TypeError(
                "TransformMatchingShapes requires vector or mesh geometry, not images."
            )
        if hasattr(node, "_face_primitive_mobs"):
            for face in node._face_primitive_mobs():
                collect(face)
            return
        owner = node.__dict__.get("control_points")
        if owner is None:
            owner = node.__dict__.get("grid")
        if owner is not None and not getattr(node, "empty", False):
            points = owner.location.detach().reshape(-1, 3).double().cpu()
            point_sets.append(points)
            topology.append(
                (
                    family,
                    len(points),
                    getattr(node, "grid_width", None),
                    getattr(node, "grid_height", None),
                )
            )
        for child in node.get_non_component_children():
            collect(child)

    collect(mob)
    if not point_sets:
        raise TypeError(f"{type(mob).__name__} has no geometry for shape matching.")
    points = torch.cat(point_sets)
    low, high = points.amin(0), points.amax(0)
    extent = high - low
    # Like Manim, use unit height and three decimal places. Horizontal lines
    # and points need a finite fallback instead of dividing by zero.
    scale = extent[1] if extent[1] > 1e-8 else extent.max()
    if scale <= 1e-8:
        scale = 1.0
    normalized = torch.round((points - (low + high) / 2) / scale * 1000).to(torch.int64)
    return tuple(topology), normalized.numpy().tobytes()


def _buckets(parts):
    buckets = {}
    for key, part in parts:
        buckets.setdefault(key, []).append(part)
    return buckets


def _match_buckets(source, target, key_map):
    """Consume explicit mappings first; preserve insertion order for ties."""
    source, target = dict(source), dict(target)
    mapped = {}
    for source_key, target_key in key_map.items():
        if source_key not in source or target_key not in target:
            raise AlganConfigurationError(
                "Every key_map entry must name a source part and a target part."
            )
        mapped.setdefault(target_key, []).extend(source.pop(source_key))
    pairs = [(parts, target.pop(key)) for key, parts in mapped.items()]
    for key in list(source):
        if key in target:
            pairs.append((source.pop(key), target.pop(key)))
    return (
        pairs,
        [part for parts in source.values() for part in parts],
        [part for parts in target.values() for part in parts],
    )


def _copy_parts(parts, scene):
    # Packed views share a lifespan. Give each selected part independent rows
    # before spawning, fading, morphing or despawning any of them.
    return Group(
        [part.clone(add_to_scene=False, spawn=False) for part in parts],
        scene=scene,
        add_to_scene=False,
    )


def _transform_matching(
    mob,
    target_mob,
    *,
    tex,
    key_map,
    transform_mismatches,
    fade_transform_mismatches,
    runtime,
):
    if not isinstance(mob, Mob) or not isinstance(target_mob, Mob):
        raise TypeError("Matching transforms require two Algan Mobs.")
    if mob.scene is not target_mob.scene:
        raise AlganConfigurationError(
            "Matching transforms require source and target in the same Scene."
        )
    if transform_mismatches and fade_transform_mismatches:
        raise AlganConfigurationError(
            "Choose only one of transform_mismatches and fade_transform_mismatches."
        )
    if not mob.is_spawned() or mob.is_despawned():
        raise AlganConfigurationError(
            "Spawn the source before using a matching transform."
        )
    if any(node.data_sub_inds is not None for node in mob.get_descendants()):
        raise AlganConfigurationError(
            "Use a whole source Mob, not a packed slice, for a matching transform."
        )
    if key_map is None:
        key_map = {}
    if not isinstance(key_map, Mapping):
        raise TypeError("key_map must be a mapping.")
    if tex:
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in key_map.items()
        ):
            raise TypeError("TeX key_map entries must map strings to strings.")
        collect = _tex_parts
    else:
        if any(
            not isinstance(key, Mob) or not isinstance(value, Mob)
            for key, value in key_map.items()
        ):
            raise TypeError(
                "Shape key_map entries must map source Mob exemplars to target Mob exemplars."
            )
        key_map = {_shape_key(key): _shape_key(value) for key, value in key_map.items()}
        collect = _shape_parts

    am = mob.animation_manager
    # Constructing this context validates runtime before any lifecycle edits.
    with Seq(runtime=runtime, animation_manager=am):
        pairs, unmatched_source, unmatched_target = _match_buckets(
            _buckets(collect(mob)), _buckets(collect(target_mob)), key_map
        )
        parent_slots = mob._capture_parent_slots(mob)
        with (
            Off(spawn_at_end=False, animation_manager=am),
            NoExtra(priority_level=1, animation_manager=am),
        ):
            result = target_mob.clone(add_to_scene=False, spawn=False)
            transitions = [
                (_copy_parts(a, mob.scene), _copy_parts(b, mob.scene), "auto")
                for a, b in pairs
            ]
            outgoing = (
                _copy_parts(unmatched_source, mob.scene) if unmatched_source else None
            )
            incoming = (
                _copy_parts(unmatched_target, mob.scene) if unmatched_target else None
            )
            if (
                unmatched_source
                and unmatched_target
                and (transform_mismatches or fade_transform_mismatches)
            ):
                transitions.append(
                    (
                        outgoing,
                        incoming,
                        "dissolve" if fade_transform_mismatches else "auto",
                    )
                )
                outgoing = incoming = None
            transient = [source for source, _, _ in transitions]
            transient.extend(part for part in (outgoing, incoming) if part is not None)
            incoming_opacities = []
            if incoming is not None:
                incoming_opacities = [
                    (node, node.opacity.clone()) for node in incoming.get_descendants()
                ]
                mob._zero_hierarchy_opacity(incoming)
            for part in transient:
                mob._register_hierarchy_for_render(part)
                part.spawn(animate=False)
            mob.despawn(animate=False)

        cleanup = []
        with Sync(animation_manager=am) as motion:
            # Even an empty-to-empty transform occupies one animation slot.
            motion.wait()
            for source, target, strategy in transitions:
                cleanup.append(source.become(target, strategy=strategy))
            if outgoing is not None:
                outgoing.opacity = 0
                cleanup.append(outgoing)
            if incoming is not None:
                for node, opacity in incoming_opacities:
                    node.set_non_recursive(opacity=opacity)
                cleanup.append(incoming)

        with (
            Off(spawn_at_end=False, animation_manager=am),
            NoExtra(priority_level=1, animation_manager=am),
        ):
            for part in cleanup:
                part.despawn(animate=False)
            mob._fill_captured_parent_slots(parent_slots, result)
            mob._register_hierarchy_for_render(result)
            result.spawn(animate=False)
    return result


@_renamed_keywords(mobject="mob", target_mobject="target_mob")
def TransformMatchingTex(
    mob: Mob,
    target_mob: Mob,
    *,
    key_map: Mapping[str, str] | None = None,
    transform_mismatches: bool = False,
    fade_transform_mismatches: bool = False,
    runtime: float | None = None,
) -> Mob:
    r"""Rearrange an equation while keeping matching LaTeX terms together.

    Match exact source strings in native :class:`~algan.mobs.text.Tex`,
    :class:`~algan.mobs.manim_compat.MathTex`,
    Manim-compatible LaTeX Mobs, or groups of those objects. Separate constructor
    arguments and double-brace groups define the terms; a single ungrouped
    expression is one term. This compares text, not algebraic equivalence.
    Repeated terms form one matching group; Algan's ordinary morph handles
    differing glyph counts within it. Unmatched terms fade in or out in place.

    Animation
    ---------
    Recorded over the current context's runtime (1 second by default), including
    descendants. Use ``with Seq(runtime=3): ...`` or supply ``runtime`` to change
    timing; ``with Off(): ...`` makes the replacement immediate. Spawn the whole
    source first. The returned replacement has the target's hierarchy and text
    indexing, and takes the source's parent slots. Use it for later animations.
    The supplied target stays unchanged. Existing source updaters are not copied.

    Parameters
    ----------
    mob
        Spawned source equation or group of equations. Packed slices cannot be
        used as the source because they share their owner's lifespan.
    target_mob
        Target equation or group, in the same Scene; need not be spawned.
    key_map
        Explicit source-string to target-string mappings, such as
        ``{"x": "a"}``. These take precedence over automatic matches. Every
        entry must identify existing terms. Defaults to ``None`` (exact matches).
    transform_mismatches
        Morph the remaining unmatched terms into one another. Defaults to
        ``False`` (fade them independently).
    fade_transform_mismatches
        Cross-dissolve remaining terms while fitting their positions and sizes.
        Mutually exclusive with ``transform_mismatches``. Defaults to ``False``.
    runtime
        Total animation time, in seconds. Defaults to ``None``, inheriting the
        enclosing context's timing.

    Returns
    -------
    :class:`~.Mob`
        A spawned copy of the target. Keep this replacement for later animation.

    Raises
    ------
    TypeError
        If inputs are not LaTeX Mobs or groups, or mapping keys are not strings.
    AlganConfigurationError
        If Scenes differ, the source is unspawned or a packed slice, a mapping
        names a missing term, or both mismatch policies are enabled.

    See Also
    --------
    TransformMatchingShapes : Match glyph or shape geometry instead of LaTeX.

    Examples
    --------
    Make x and y exchange positions without changing which glyph is which:

    .. algan:: Example1TransformMatchingTex

        from algan import *

        equation = Tex("x", "+", "y", font_size=72).spawn()
        equation = TransformMatchingTex(
            equation, Tex("y", "+", "x", font_size=72), runtime=2
        )
        equation.move(UP)
        Scene.save_video()
    """
    return _transform_matching(
        mob,
        target_mob,
        tex=True,
        key_map=key_map,
        transform_mismatches=transform_mismatches,
        fade_transform_mismatches=fade_transform_mismatches,
        runtime=runtime,
    )


@_renamed_keywords(mobject="mob", target_mobject="target_mob")
def TransformMatchingShapes(
    mob: Mob,
    target_mob: Mob,
    *,
    key_map: Mapping[Mob, Mob] | None = None,
    transform_mismatches: bool = False,
    fade_transform_mismatches: bool = False,
    runtime: float | None = None,
) -> Mob:
    """Move matching shapes or text glyphs to their places in a new arrangement.

    Compare ordered geometry after centering, scaling to unit height, and
    rounding to three decimal places. Horizontal geometry uses its largest
    extent instead of height. Position, positive uniform scale and color do not
    affect identity; rotation, point order and topology do. Containers are
    traversed and packed glyphs are matched individually. Images are unsupported.
    Repeated shapes form one matching group, whose morph handles unequal counts.
    Unmatched shapes fade in or out in place by default.

    Animation
    ---------
    Recorded over the current context's runtime (1 second by default), including
    descendants. Use ``with Seq(runtime=3): ...`` or ``runtime`` to change timing;
    ``with Off(): ...`` makes the replacement immediate. Spawn the whole source
    first. Keep the returned replacement for later animation: it has the target's
    hierarchy and occupies the source's parent slots. The supplied target stays
    unchanged. Existing source updaters are not copied.

    Parameters
    ----------
    mob
        Spawned source Mob, text or group. Packed slices cannot be used as the
        source because they share their owner's lifespan.
    target_mob
        Target Mob in the same Scene; need not be spawned.
    key_map
        Map source shape exemplars to target shape exemplars, for example
        ``{source_square: target_circle}``. Their normalized geometry supplies
        the keys, and explicit mappings take precedence over automatic matches.
        Every key must occur in its respective hierarchy. Defaults to ``None``.
    transform_mismatches
        Morph remaining unmatched shapes into one another. Defaults to ``False``
        (fade them independently).
    fade_transform_mismatches
        Cross-dissolve remaining shapes while fitting their positions and sizes.
        Mutually exclusive with ``transform_mismatches``. Defaults to ``False``.
    runtime
        Total animation time, in seconds. Defaults to ``None``, inheriting the
        enclosing context's timing.

    Returns
    -------
    :class:`~.Mob`
        A spawned copy of the target. Keep this replacement for later animation.

    Raises
    ------
    TypeError
        If inputs or mapping exemplars are not Mobs, or have unsupported geometry.
    AlganConfigurationError
        If Scenes differ, the source is unspawned or a packed slice, an explicit
        shape key is absent, or both mismatch policies are enabled.

    See Also
    --------
    TransformMatchingTex : Preserve authored LaTeX term identity.

    Examples
    --------
    Move the same letters into a different word:

    .. algan:: Example1TransformMatchingShapes

        from algan import *

        word = Text("STOP", font_size=72).spawn()
        word = TransformMatchingShapes(word, Text("POST", font_size=72), runtime=2)
        Scene.save_video()
    """
    return _transform_matching(
        mob,
        target_mob,
        tex=False,
        key_map=key_map,
        transform_mismatches=transform_mismatches,
        fade_transform_mismatches=fade_transform_mismatches,
        runtime=runtime,
    )
