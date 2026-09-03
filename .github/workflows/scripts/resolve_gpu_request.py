"""Resolve one GPU-harness run request into GitHub Actions step outputs.

`run_on_mac.yaml` has two entry points -- `workflow_dispatch` inputs, and a
committed request file for branches whose workflow file has not reached the
default branch yet -- and everything downstream needs exactly one answer. This
turns whichever fired into a single set of `name=value` lines on stdout, for
the caller to append to `$GITHUB_OUTPUT`.

Dispatch inputs arrive in the environment as `IN_*` and win when a command is
present; otherwise the request file is read. Multi-line values are emitted with
the heredoc form `$GITHUB_OUTPUT` requires.

Kept out of the YAML because a shell-and-`jq` version of this was the part of
the harness most likely to be wrong in a way that only shows up as an empty
matrix at 20 minutes in; here it is testable
(`tests/unit_tests/test_gpu_harness.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

# label -> the matrix entry it stands for. `label` is what names the job, the
# artifact and the summary section, so it is the user-facing handle.
ARMS: dict[str, dict[str, str]] = {
    "mac-mps": {"os": "macos-latest", "device": "mps", "label": "mac-mps"},
    "mac-cpu": {"os": "macos-latest", "device": "cpu", "label": "mac-cpu"},
    "linux-cpu": {"os": "ubuntu-latest", "device": "cpu", "label": "linux-cpu"},
}

DEFAULTS = {
    "arms": "mac-mps",
    "env": "",
    "latex": "false",
    "wheel": "33342025517",
    "artifacts": "",
    "timeout": "60",
}


def _truthy(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return (
        "true" if str(value).strip().lower() in ("true", "1", "yes", "on") else "false"
    )


def resolve(env: dict[str, str], request: dict | None) -> dict[str, str]:
    """Merge the two entry points into the final parameter set."""
    request = request or {}

    def pick(input_name: str, file_key: str, default: str) -> str:
        value = env.get(input_name, "")
        if value.strip():
            return value
        if file_key in request and str(request[file_key]).strip():
            return str(request[file_key])
        return default

    command = pick("IN_COMMAND", "command", "")
    if not command.strip():
        raise SystemExit(
            "no command: dispatch with a `command` input, or commit one to "
            ".github/gpu-run/mac.json"
        )

    arms_raw = pick("IN_ARMS", "arms", DEFAULTS["arms"])
    if isinstance(request.get("arms"), list) and not env.get("IN_ARMS", "").strip():
        arms_raw = ",".join(request["arms"])
    arms = [a.strip() for a in arms_raw.replace("\n", ",").split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")
    if not arms:
        raise SystemExit("no arms selected")

    extra_env = pick("IN_ENV", "env", DEFAULTS["env"])
    if isinstance(request.get("env"), dict) and not env.get("IN_ENV", "").strip():
        extra_env = "\n".join(f"{k}={v}" for k, v in request["env"].items())

    artifacts = pick("IN_ARTIFACTS", "artifacts", DEFAULTS["artifacts"])
    if (
        isinstance(request.get("artifacts"), list)
        and not env.get("IN_ARTIFACTS", "").strip()
    ):
        artifacts = "\n".join(request["artifacts"])

    timeout = pick("IN_TIMEOUT", "timeout_minutes", DEFAULTS["timeout"]).strip()
    if not timeout.isdigit():
        raise SystemExit(f"timeout_minutes must be a whole number, got {timeout!r}")

    # `latex` is a checkbox, and an unticked checkbox arrives as the string
    # "false" rather than as an empty value -- so it cannot go through `pick`,
    # which would take that "false" as an answer and never consult the file.
    latex = env.get("IN_LATEX", "").strip()
    if not latex and "latex" in request:
        latex = _truthy(request["latex"])
    latex = _truthy(latex or DEFAULTS["latex"])

    wheel = pick("IN_WHEEL", "taichi_wheel_run_id", DEFAULTS["wheel"]).strip()

    return {
        "command": command,
        "env": extra_env,
        "latex": latex,
        "wheel": wheel or "none",
        "artifacts": artifacts,
        "timeout": timeout,
        "matrix": json.dumps([ARMS[a] for a in arms]),
    }


def format_outputs(values: dict[str, str]) -> str:
    """Render as `$GITHUB_OUTPUT` lines, heredoc'ing anything multi-line."""
    lines: list[str] = []
    for key, value in values.items():
        if "\n" in value:
            marker = f"ghadelim_{uuid.uuid4().hex}"
            lines.append(f"{key}<<{marker}")
            lines.append(value)
            lines.append(marker)
        else:
            lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", default=".github/gpu-run/mac.json")
    args = parser.parse_args(argv)

    path = Path(args.request_file)
    request = None
    if path.exists():
        request = json.loads(path.read_text(encoding="utf-8"))

    sys.stdout.write(format_outputs(resolve(dict(os.environ), request)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
