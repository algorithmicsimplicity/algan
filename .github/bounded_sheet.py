import os
import signal
import subprocess
import sys

commands = [
    ([sys.executable, '-m', 'pytest', '-q', 'tests/unit_tests/test_sheet_stream.py'], 180),
    ([sys.executable, 'benchmarks/_sheet_stream_check.py', '--scene', 'explainer', '--quality', 'PREVIEW', '--frames', '60', '--pairs', '2', '--arms', 'reference,all', '--require-mps'], 600),
]
for command, deadline in commands:
    print('BOUNDED_SHEET_START', command, flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        result = process.wait(timeout=deadline)
    except subprocess.TimeoutExpired:
        print('BOUNDED_SHEET_TIMEOUT', deadline, flush=True)
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        sys.exit(124)
    print('BOUNDED_SHEET_EXIT', result, flush=True)
    if result:
        sys.exit(result)
