"""The profiler's prefetch-wait hook must survive being installed.

``RenderLoopMixin._await_prefetched_batch`` exists so the profiler can time
the render thread's wait on the prefetch worker. ``StageTimers.wrap_function``
replaces a class attribute with a plain function, so the method has to be an
ordinary instance method: as a ``staticmethod`` the wrapped call received
``self`` as an extra argument and a profiled multi-batch render died on its
first prefetched batch (Kaggle T4, `nn_scene_UHD.py`, 2026-09-12) -- a
failure no single-batch scene can see.
"""

from concurrent.futures import Future

from algan.render_loop import RenderLoopMixin
from algan.utils import profiling_utils


def test_prefetch_wait_is_timed_when_called_through_the_hook():
    original = RenderLoopMixin._await_prefetched_batch
    timers = profiling_utils.TIMERS
    try:
        timers.wrap_function(
            RenderLoopMixin, "_await_prefetched_batch", "wait for prefetched batch"
        )
        timers.reset()
        done = Future()
        done.set_result(("primitives", 7, "state"))

        class Loop(RenderLoopMixin):
            pass

        assert Loop()._await_prefetched_batch(done) == ("primitives", 7, "state")
        assert timers.counts["wait for prefetched batch"] == 1
    finally:
        RenderLoopMixin._await_prefetched_batch = original
        timers.reset()
