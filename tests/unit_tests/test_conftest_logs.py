"""Where the conftest pipes a run's log, and when it does not.

Opening a suite's ``pytest.log`` truncates it, so ``--collect-only`` used to
wipe the last real run's log, and two ``pytest-xdist`` workers would each
truncate the other's.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def _config(collectonly=False, workerinput=None):
    config = SimpleNamespace(option=SimpleNamespace(collectonly=collectonly))
    if workerinput is not None:
        config.workerinput = workerinput
    return config


def test_a_collection_writes_no_log():
    log_path = sys.modules["conftest"]._log_path
    assert log_path(_config(collectonly=True), Path("suite")) is None


def test_one_process_logs_to_pytest_log_and_each_of_several_workers_to_its_own():
    log_path = sys.modules["conftest"]._log_path
    suite = Path("suite")
    assert log_path(_config(), suite) == suite / "pytest.log"
    one = {"workerid": "gw0", "workercount": 1}
    assert log_path(_config(workerinput=one), suite) == suite / "pytest.log"
    names = {
        log_path(_config(workerinput={"workerid": w, "workercount": 2}), suite)
        for w in ("gw0", "gw1")
    }
    assert names == {suite / "pytest.gw0.log", suite / "pytest.gw1.log"}
