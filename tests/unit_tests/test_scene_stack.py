"""The active-Scene stack under the patterns explicit-Scene code reaches for.

``Scene()`` makes itself current on construction, so a Scene built inside a
``with`` block sits above the block's Scene when the block ends.
"""

import pytest

from algan import Scene, SceneManager


def _empty_scene(scene):
    scene.camera = None
    scene.light_sources = []


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    SceneManager.reset()
    yield
    SceneManager.reset()


def test_a_scene_built_inside_a_with_block_is_popped_with_it():
    manager = SceneManager.instance()
    below = manager.current_scene
    with Scene(scene_initializer=_empty_scene) as outer:
        helper = Scene(scene_initializer=_empty_scene)
        assert manager.current_scene is helper
    assert manager.scene_stack == (below,)
    assert outer._terminated
    assert helper._terminated


def test_an_error_in_the_block_is_not_masked_by_the_stack():
    manager = SceneManager.instance()
    below = manager.current_scene
    with (  # noqa: PT012 -- the error must be raised inside the Scene block
        pytest.raises(ValueError, match="user error"),
        Scene(scene_initializer=_empty_scene),
    ):
        Scene(scene_initializer=_empty_scene)
        raise ValueError("user error")
    assert manager.scene_stack == (below,)


def test_an_entered_scene_above_is_still_refused():
    # Only Scenes nobody entered are popped on the way out; one whose own
    # block is still open is a genuine ordering mistake.
    outer = Scene(scene_initializer=_empty_scene).__enter__()
    inner = Scene(scene_initializer=_empty_scene).__enter__()
    with pytest.raises(RuntimeError, match="current active Scene"):
        outer.__exit__(None, None, None)
    inner.__exit__(None, None, None)
    outer.__exit__(None, None, None)


def test_entering_a_covered_scene_says_why():
    a = Scene(scene_initializer=_empty_scene)
    b = Scene(scene_initializer=_empty_scene)
    with pytest.raises(RuntimeError, match="current on construction"):
        a.__enter__()
    b._terminate()
    a._terminate()
