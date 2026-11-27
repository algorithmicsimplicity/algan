"""Timing keywords a context cannot take name the spelling that works.

``Lag(lag_ratio=0.5)`` and ``Off(runtime=2)`` failed with Python's "got
multiple values for keyword argument", which names an internal base class,
and ``Off(run_time=2)`` suggested ``Off(runtime=2)``, which fails too.
"""

import pytest

from algan import Lag, Off
from algan.animation_timeline.animation_contexts import _TimingParameterError


def test_lag_lag_ratio_says_ratio():
    with pytest.raises(_TimingParameterError, match="'ratio'") as error:
        Lag(lag_ratio=0.5)
    assert isinstance(error.value, TypeError)
    assert "Lag(0.5)" in str(error.value)


@pytest.mark.parametrize("name", ["runtime", "run_time", "duration"])
def test_off_runtime_says_off_is_instant(name):
    with pytest.raises(_TimingParameterError, match="instantly"):
        Off(**{name: 2})


def test_plain_lag_and_off_still_build():
    Lag(0.3)
    Lag(ratio=0.3, runtime=2)
    Off()
