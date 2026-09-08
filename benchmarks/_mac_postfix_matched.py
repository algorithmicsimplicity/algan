"""One-VM CPU/MPS comparison, followed by a separate warm GPU profile.

The CPU blocks bracket MPS to expose VM timing drift. Each block is a fresh
process, retaining compiler caches between its renders. Controls have no
native instrumentation; the final MPS render enables detailed profiling.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", default="UHD")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    output = Path("algan_outputs/postfix_matched")
    output.mkdir(parents=True, exist_ok=True)
    child = Path(__file__).with_name("_mac_postfix_profile.py")

    def snapshot(label):
        record = {"label": label, "time": time.time(), "platform": platform.platform()}
        if sys.platform == "darwin":
            for name, command in (
                ("vm_stat", ["vm_stat"]),
                ("swap", ["sysctl", "vm.swapusage"]),
                ("thermal", ["pmset", "-g", "therm"]),
            ):
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                record[name] = {"returncode": result.returncode, "stdout": result.stdout,
                                "stderr": result.stderr}
        with (output / "system.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print("MATCHED_SYSTEM " + json.dumps(record), flush=True)

    blocks = [("cpu_before", "cpu", 2, 450), ("mps", "mps", 4, 1150),
              ("cpu_after", "cpu", 2, 450)]
    if args.cpu_only:
        blocks = [("cpu_smoke", "cpu", 3, 600)]
    outcomes = []
    for tag, device, runs, timeout in blocks:
        snapshot(tag + "_before")
        command = [sys.executable, "-u", str(child), "--child", device, "--tag", tag,
                   "--quality", args.quality, "--arena-mib", "1720", "--runs", str(runs)]
        if device == "mps":
            command += ["--profile-run", "4", "--full-profile", "--native-graphs", "--native-sample"]
        elif args.cpu_only:
            command += ["--profile-run", "3", "--full-profile"]
        print("MATCHED_START " + json.dumps({"tag": tag, "command": command}), flush=True)
        started = time.perf_counter()
        # Keep the original child output even if it exceeds its time budget.
        with (output / (tag + ".log")).open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=dict(os.environ))
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
                code = 124
        print((output / (tag + ".log")).read_text(), flush=True)
        outcomes.append({"tag": tag, "exit_code": code, "process_wall": time.perf_counter()-started})
        (output / "outcomes.json").write_text(json.dumps(outcomes, indent=2))
        snapshot(tag + "_after")
        # Still collect the closing CPU control if the GPU profile fails.
        print("MATCHED_END " + json.dumps(outcomes[-1]), flush=True)
    return int(any(row["exit_code"] for row in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
