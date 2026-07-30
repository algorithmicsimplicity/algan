import pytest

from algan import (
    AnimationManager,
    AudioManager,
    Camera,
    Circle,
    Group,
    Mob,
    Off,
    Scene,
    SceneManager,
    TimelineManager,
)


def _empty_scene(scene):
    scene.camera = None
    scene.light_sources = []


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    SceneManager.reset()
    yield
    SceneManager.reset()


def test_scene_owned_managers_are_regular_instances():
    assert TimelineManager() is not TimelineManager()
    assert AnimationManager() is not AnimationManager()
    assert AudioManager() is not AudioManager()

    first = Scene(scene_initializer=_empty_scene)
    second = Scene(scene_initializer=_empty_scene)
    assert first.timeline_manager is not second.timeline_manager
    assert first.animation_manager is not second.animation_manager
    assert first.audio_manager is not second.audio_manager
    second.terminate()
    first.terminate()


def test_scene_context_pushes_and_pops_active_scene():
    manager = SceneManager.instance()
    outer = Scene(scene_initializer=_empty_scene)
    assert manager.current_scene is outer

    with Scene(scene_initializer=_empty_scene) as inner:
        assert manager.current_scene is inner
        mob = Mob()
        assert mob.scene is inner

    assert manager.current_scene is outer
    outer.terminate()


def test_explicit_scene_remains_isolated_while_another_scene_is_active():
    manager = SceneManager.instance()
    outer = Scene(scene_initializer=_empty_scene)

    with Scene(scene_initializer=_empty_scene) as inner:
        mob = Circle(scene=outer)
        camera = Camera(scene=outer)
        mob.spawn(animate=False)

        assert manager.current_scene is inner
        assert mob.scene is outer
        assert camera.scene is outer
        assert all(descendant.scene is outer for descendant in mob.get_descendants())
        assert all(descendant.scene is outer for descendant in camera.get_descendants())
        assert mob.lifespan is outer.timeline_manager.get_lifespan(mob.id)
        assert mob.id not in inner.timeline_manager.mob_id_to_lifespan

    outer.terminate()


def test_reset_replaces_only_one_scenes_managers():
    outer = Scene(scene_initializer=_empty_scene)
    with Scene(scene_initializer=_empty_scene) as inner:
        old_outer_managers = (
            outer.timeline_manager,
            outer.animation_manager,
            outer.audio_manager,
        )
        inner_managers = (
            inner.timeline_manager,
            inner.animation_manager,
            inner.audio_manager,
        )

        outer.reset()

        assert outer.timeline_manager is not old_outer_managers[0]
        assert outer.animation_manager is not old_outer_managers[1]
        assert outer.audio_manager is not old_outer_managers[2]
        assert (
            inner.timeline_manager,
            inner.animation_manager,
            inner.audio_manager,
        ) == inner_managers
        assert SceneManager.instance().current_scene is inner

    outer.terminate()


def test_covered_scene_cannot_be_terminated_out_of_order():
    outer = Scene(scene_initializer=_empty_scene)
    inner = Scene(scene_initializer=_empty_scene)
    with pytest.raises(RuntimeError, match="current active Scene"):
        outer.terminate()
    inner.terminate()
    outer.terminate()


def test_unqualified_mob_uses_current_scene_not_context_manager_scene():
    outer = Scene(scene_initializer=_empty_scene)
    with Scene(scene_initializer=_empty_scene) as inner:
        with Off(animation_manager=outer.animation_manager):
            mob = Mob()

        assert mob.scene is inner
        assert mob.animation_manager is inner.animation_manager

    outer.terminate()


def test_scene_context_pops_after_exception():
    manager = SceneManager.instance()
    outer = Scene(scene_initializer=_empty_scene)

    with pytest.raises(RuntimeError, match="boom"):
        with Scene(scene_initializer=_empty_scene) as inner:
            assert manager.current_scene is inner
            raise RuntimeError("boom")

    assert manager.current_scene is outer
    outer.terminate()


def test_inactive_scene_render_temporarily_activates_it(monkeypatch):
    from algan.utils import algan_utils

    manager = SceneManager.instance()
    outer = Scene(scene_initializer=_empty_scene)
    inner = Scene(scene_initializer=_empty_scene)
    observed = {}

    def fake_render(scene, *args, **kwargs):
        observed["scene"] = scene
        observed["active"] = manager.current_scene
        observed["animation_manager"] = scene.animation_manager
        return "rendered"

    monkeypatch.setattr(algan_utils, "_render_scene_to_file", fake_render)
    assert outer.save_video() == "rendered"
    assert observed == {
        "scene": outer,
        "active": outer,
        "animation_manager": outer.animation_manager,
    }
    assert manager.current_scene is inner

    inner.terminate()
    outer.terminate()


def test_empty_group_slice_keeps_owning_scene():
    outer = Scene(scene_initializer=_empty_scene)
    with Scene(scene_initializer=_empty_scene):
        group = Group(scene=outer, add_to_scene=False)
        empty_view = group[:]
        assert empty_view.scene is outer
        assert empty_view not in outer.actors

    outer.terminate()
