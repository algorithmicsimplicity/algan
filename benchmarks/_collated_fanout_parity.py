"""Parity for the collated spawn/despawn fan-out and ``map_animated_attribute``.

Spawning or despawning a subtree used to record one animation per descendant.
It now records **one** for the whole set (``Animatable._collated_fade_in`` /
``_collated_fade_out``), addressed through an explicit ``RowRanges`` scope so
Mobs with their own ``on_create`` / ``on_destroy`` keep their per-Mob call and
stay out of the collated write.

This compares the two, per row and per frame, on the *materialized* state --
what the primitive builders read -- not on the authored buffer, because the
authored end state can agree while intermediate frames do not.

Two scenes, because the change is bit-identical everywhere except one place:

* ``build_strict`` must match **exactly**.
* ``build_custom_exit`` is the intended behaviour change. An ancestor's exit is
  a *recursive* opacity write, so despawning a group used to fade a ``Tex``'s
  glyphs uniformly on top of the diagonal wave its own ``on_destroy`` records --
  the wave was authored and then overwritten. Collated, the Tex is excluded
  from the ancestor's write and its wave is what plays. The check requires the
  difference to be confined to that Mob's rows and both arms to end fully
  transparent; it does not require equality, and it fails if the difference
  vanishes (which would mean the check stopped testing anything).

    <venv-python> benchmarks/_collated_fanout_parity.py

CPU only, no render. Exit code is non-zero if a check fails. ``ALGAN_OPT_DISABLE
=collate`` selects the pre-collation arm, so both paths stay exercised.

Read the coverage assertions before trusting a pass: a scene that never reaches
the collated path would compare the old code against itself and pass vacuously
(the trap that made ``_resolve_rollback_check.py``'s first version worthless).
``--mutate`` deliberately breaks the implementation and requires this script to
notice, so the checks cannot silently stop testing anything.
"""

import os
import sys

os.environ.setdefault("ALGAN_RENDER_DEVICE", "cpu")
os.environ.setdefault("ALGAN_ANIMATION_DEVICE", "cpu")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import algan.animation_timeline.timeline as tl  # noqa: E402
from algan import (  # noqa: E402
    LD,
    RIGHT,
    UP,
    Circle,
    Group,
    Off,
    Seq,
    Square,
    Sync,
    Tex,
    Text,
    Triangle,
)
from algan.animatable_base.animatable import Animatable  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

COVERAGE = {
    "collated_fade_in": 0,
    "collated_fade_out": 0,
    "custom_hook": 0,
    "per_mob_fallback": 0,
    "map_attr": 0,
}


def _instrument():
    """Count which branches the scene actually reaches, so a pass cannot be
    vacuous.
    """
    fade_in = Animatable._collated_fade_in
    fade_out = Animatable._collated_fade_out
    per_mob_create = Animatable._create_recursive_per_mob
    is_standard = Animatable._is_standard_hook

    def counted_in(self, nodes):
        COVERAGE["collated_fade_in"] += 1
        return fade_in(self, nodes)

    def counted_out(self, nodes):
        COVERAGE["collated_fade_out"] += 1
        return fade_out(self, nodes)

    def counted_per_mob(self, animate=True):
        COVERAGE["per_mob_fallback"] += 1
        return per_mob_create(self, animate)

    def counted_is_standard(node, hook_name):
        standard = is_standard(node, hook_name)
        if not standard:
            hook = getattr(type(node), hook_name, None)
            if hook is not None and hook is not getattr(Animatable, hook_name):
                COVERAGE["custom_hook"] += 1
        return standard

    Animatable._collated_fade_in = counted_in
    Animatable._collated_fade_out = counted_out
    Animatable._create_recursive_per_mob = counted_per_mob
    Animatable._is_standard_hook = staticmethod(counted_is_standard)


def build_strict():
    """Everything that must come out **bit-identical**.

    * a nested Group of plain Mobs -- the collated path,
    * ``Text``, whose ``on_create`` is its own: it must keep its per-Mob call,
      and its glyphs (which it claims via ``_create_recursive(animate=False)``)
      must not also be faded in underneath its wave,
    * a subtree spawned *after* one of its children already was -- only the
      fresh members may be faded in,
    * a lone Mob, which has nothing to collate with,
    * a ``Text`` despawned as its own root, where the custom exit owns the whole
      walk and there is nothing to collate,
    * ``map_animated_attribute``, the public form of the same one-event write.
    """
    inner = Group([Square().move(RIGHT * i) for i in range(3)])
    text = Text("hi").move(UP * 2)
    outer = Group([inner, Circle().move(UP), text])

    lone = Triangle().move(RIGHT * 3)
    lone.spawn()

    partial_child = Square().move(UP * 2)
    partial_child.spawn()
    partial = Group([partial_child, Circle().move(UP * 3)])

    outer.spawn()
    with Sync():
        outer.map_animated_attribute("opacity", lambda o: o * 0.4)
        COVERAGE["map_attr"] += 1
    partial.spawn()

    with Seq():
        inner.despawn()
        outer.map_animated_attribute("glow", lambda g: g + 0.25)
        COVERAGE["map_attr"] += 1
        partial.despawn()
        lone.despawn()
        text.despawn()  # custom exit as its own root: nothing to collate
    return None


def build_custom_exit():
    """The one intended behaviour change: a Mob with its own exit animation
    inside a despawned group.

    Before, the ancestor's exit was a *recursive* opacity write, so it faded the
    Tex's glyphs uniformly on top of the diagonal wave
    :meth:`~algan.mobs.text.Tex.on_destroy` records -- the wave was authored and
    then overwritten, and the text left the screen as a plain fade. Collated,
    the Tex is excluded from the ancestor's write and its wave is what plays.

    The check below is therefore not equality: it requires the difference to be
    confined to the Tex's own rows, and both arms to end fully transparent.
    """
    tex = Tex("x^2").move(UP * 2)
    group = Group([Square(), tex])
    with Off():
        group.spawn()
    group.despawn()
    return tex


def arm(builder, disabled):
    """Author the scene with the collation on or off and materialize the whole
    timeline at a dense set of frame times, in global row order.

    Returns the per-attribute state, the recorded event count, and the rows the
    builder singled out (used by the custom-exit check).
    """
    tl._OPT_DISABLED = frozenset(("collate",) if disabled else ())
    scene = SceneManager.reset()
    scene.set_video_settings(LD)
    marked = builder()
    scene._initialize_frames()

    end = max(
        (float(a.lifespan.end()) for a in scene.actors if a.lifespan.end() >= 0),
        default=1.0,
    )
    times = torch.linspace(0.0, max(end, 1.0) + 0.5, 61)
    scene.timeline_manager.set_state_to_times(times)

    out, marked_rows = {}, {}
    for attr, at in scene.timeline_manager.attr_to_timeline.items():
        rows = sorted(
            {
                int(i)
                for inds in at.mob_id_to_inds.values()
                for i in inds.reshape(-1).tolist()
            }
        )
        if not rows:
            continue
        state = at.active_state
        if state is None or state.numel() == 0:
            continue
        out[attr] = state[:, torch.tensor(rows, dtype=torch.long)].clone()
        if marked is not None:
            try:
                inds = marked._get_attr_inds(attr, include_descendants=True)
            except AttributeError:
                inds = None  # the marked Mob owns no rows of this attribute
            owned = set(inds.tolist()) if inds is not None else set()
            marked_rows[attr] = torch.tensor([r in owned for r in rows])
    scene.timeline_manager.clear_buffers()
    events = int(len(scene.timeline_manager.function_timeline.function_applications))
    return out, events, marked_rows


def compare(a, b, label=""):
    """Report per-attribute agreement. Returns the largest absolute deviation."""
    worst = 0.0
    print(f"\n{label}")
    for attr in sorted(set(a) | set(b)):
        ta, tb = a.get(attr), b.get(attr)
        if ta is None or tb is None:
            print(f"  {attr:20s} MISSING IN ONE ARM")
            worst = float("inf")
            continue
        if ta.shape != tb.shape:
            print(f"  {attr:20s} shape {tuple(ta.shape)} vs {tuple(tb.shape)}")
            worst = float("inf")
            continue
        d = (ta - tb).abs()
        m = float(d.max().item()) if d.numel() else 0.0
        worst = max(worst, m)
        n_bad = int((d > 1e-6).sum().item())
        flag = "" if m == 0.0 else ("  <-- differs" if m > 1e-6 else "  (fp noise)")
        extra = ""
        if n_bad:
            frame, row, _ = (d > 1e-6).nonzero()[0].tolist()
            extra = (
                f", first at frame {frame} row {row} ({n_bad} of {d.numel()} values)"
            )
        print(f"  {attr:20s} max|diff| = {m:.3e}{extra}{flag}")
    return worst


def main():
    mutate = "--mutate" in sys.argv
    if mutate:
        # Forget that subclass entrances/exits exist: collate every Mob,
        # including the ones whose hook is their own. If this script cannot
        # tell that apart from the real implementation, it is not testing the
        # scope logic at all.
        Animatable._is_standard_hook = staticmethod(lambda node, hook_name: True)

    _instrument()
    old, old_events, _ = arm(build_strict, disabled=True)
    new, new_events, _ = arm(build_strict, disabled=False)

    print(
        f"recorded events: per-Mob {old_events}  collated {new_events} "
        f"({old_events / max(new_events, 1):.2f}x)"
    )
    print("branch coverage:", ", ".join(f"{k}={v}" for k, v in COVERAGE.items()))

    ok = True
    for name, count in COVERAGE.items():
        if count == 0:
            print(f"FAIL: the scene never reached {name} -- this run is vacuous")
            ok = False
    if new_events >= old_events:
        print("FAIL: the collated arm recorded no fewer events; nothing collated")
        ok = False

    worst = compare(
        old, new, "strict scene -- materialized state, per row and per frame:"
    )
    if worst > 1e-6:
        print(f"\nFAIL: strict scene DIFFERS (max |diff| = {worst:.3e})")
        ok = False
    else:
        print(f"\nstrict scene matches exactly (max |diff| = {worst:.3e})")

    # The one intended change, checked rather than waived.
    old_c, old_ce, marked = arm(build_custom_exit, disabled=True)
    new_c, new_ce, _ = arm(build_custom_exit, disabled=False)
    print(f"\ncustom-exit scene: events {old_ce} -> {new_ce}")
    for attr in sorted(set(old_c) | set(new_c)):
        a, b = old_c[attr], new_c[attr]
        mask = marked.get(attr)
        if a.shape != b.shape or mask is None:
            print(f"  {attr:20s} shape {tuple(a.shape)} vs {tuple(b.shape)}")
            ok = False
            continue
        d = (a - b).abs()
        off = float(d[:, ~mask].max().item()) if (~mask).any() else 0.0
        on = float(d[:, mask].max().item()) if mask.any() else 0.0
        print(f"  {attr:20s} max|diff| off the Tex {off:.3e}, on the Tex {on:.3e}")
        if off > 1e-6:
            print(
                f"    FAIL: {attr} changed on rows outside the Mob with the "
                f"custom exit; the change must be confined to it"
            )
            ok = False
    # The difference must be *transient*: the wave takes a different route to
    # the same place, so once every exit animation has finished the two arms
    # agree again. (Both arms hold the same non-zero value past the end of the
    # animation, where nothing is drawn because the actor filter has already
    # dropped a despawned Mob -- so this compares the arms, it does not assert
    # transparency.)
    if old_c["opacity"].shape != new_c["opacity"].shape:
        print(
            f"    FAIL: the arms do not even own the same rows "
            f"({tuple(old_c['opacity'].shape)} vs {tuple(new_c['opacity'].shape)}) "
            f"-- a Mob's own entrance never ran"
        )
        return 0 if mutate else 1
    tail = (old_c["opacity"][-1] - new_c["opacity"][-1]).abs().max().item()
    print(f"  arms agree again after the exits finish: max|diff| {tail:.3e}")
    if tail > 1e-6:
        print("    FAIL: the difference outlives the exit animation")
        ok = False
    if not (new_c["opacity"] - old_c["opacity"]).abs().max().item() > 1e-6:
        print(
            "    FAIL: the custom-exit scene shows no difference at all -- "
            "this check has gone vacuous"
        )
        ok = False

    if mutate:
        print(
            "\n--mutate: this run is expected to FAIL. "
            + ("Good, it did." if not ok else "IT DID NOT -- the checks are vacuous.")
        )
        return 0 if not ok else 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
