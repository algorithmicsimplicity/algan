#!/usr/bin/env python3
"""
Dynamic Local GitHub Actions CI Runner for Algan.

Discovers, parses, and executes CI checks defined in `.github/workflows/*.{yaml,yml}`
on the local machine. Automatically adapts when workflows are modified, added,
or removed without needing manual updates.

By default, setup/provisioning steps (such as installing uv, system apt-get packages,
setting up python, creating virtual environments, and artifact uploads) are skipped,
running only the actual check actions (linting, formatting, testing, doc building, lockfile checks).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import itertools
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# Terminal colors and formatting
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    @classmethod
    def disable(cls) -> None:
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.WHITE = ""


if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    Style.disable()


def strip_ansi(s: str) -> str:
    """Strips ANSI escape sequences from a string for true width calculation."""
    return re.sub(r"\033\[[0-9;]*m", "", s)


def format_cell(text: str, width: int, align: str = "left") -> str:
    """Pads a string considering ANSI color sequences."""
    vis_len = len(strip_ansi(text))
    padding = max(0, width - vis_len)
    if align == "right":
        return " " * padding + text
    return text + " " * padding


@dataclass
class Step:
    name: str
    run: Optional[str] = None
    uses: Optional[str] = None
    with_args: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    if_cond: Optional[str] = None
    working_directory: Optional[str] = None
    continue_on_error: bool = False
    is_setup: bool = False
    source_file: Optional[Path] = None


@dataclass
class Job:
    id: str
    name: str
    runs_on: str
    needs: List[str] = field(default_factory=list)
    strategy_matrix: Dict[str, List[Any]] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    steps: List[Step] = field(default_factory=list)
    if_cond: Optional[str] = None


@dataclass
class Workflow:
    path: Path
    filename: str
    name: str
    env: Dict[str, str] = field(default_factory=dict)
    jobs: Dict[str, Job] = field(default_factory=dict)


@dataclass
class StepResult:
    workflow_name: str
    job_name: str
    step_name: str
    command: str
    status: str  # "PASSED", "FAILED", "SKIPPED"
    duration: float
    error_message: Optional[str] = None


class StepClassifier:
    """Classifies workflow steps into setup/provisioning vs actual verification checks."""

    SETUP_ACTIONS = {
        "actions/checkout",
        "actions/setup-python",
        "actions/setup-node",
        "actions/setup-go",
        "actions/setup-java",
        "actions/cache",
        "astral-sh/setup-uv",
        "actions/upload-artifact",
        "actions/download-artifact",
        "codecov/codecov-action",
    }

    SETUP_COMMAND_PREFIXES = (
        "apt-get",
        "sudo apt-get",
        "apt",
        "sudo apt",
        "brew",
        "choco",
        "yum",
        "dnf",
        "pacman",
        "apk",
        "uv venv",
        "uv sync",
        "uv pip install",
        "pip install",
        "python -m venv",
        "virtualenv",
        "npm install",
        "yarn install",
        "pnpm install",
    )

    @classmethod
    def is_setup_step(cls, step: Step) -> bool:
        # Check action (uses:)
        if step.uses:
            action_base = step.uses.split("@")[0].strip().lower()
            if any(action_base.startswith(sa) for sa in cls.SETUP_ACTIONS):
                return True
            if action_base.startswith("./.github/actions/"):
                subpath = action_base[len("./.github/actions/") :]
                if subpath.startswith(("install-", "setup-", "provision-")):
                    return True

        # Check run commands
        if step.run:
            cmd = step.run.strip()
            first_line = cmd.splitlines()[0].strip() if cmd else ""
            clean_first_line = first_line.lower()

            # If it starts with a known package manager or venv creation
            for prefix in cls.SETUP_COMMAND_PREFIXES:
                if (
                    clean_first_line.startswith(prefix)
                    or f"sudo {prefix}" in clean_first_line
                ):
                    return True

            # If step name strongly indicates setup and is not a known check
            name_lower = step.name.lower()
            if any(
                keyword in name_lower
                for keyword in [
                    "install system dependencies",
                    "install latex",
                    "install ffmpeg",
                    "install dependencies",
                    "install uv",
                    "create venv",
                    "set up python",
                    "install the project",
                    "install project",
                ]
            ):
                # Unless it's running pytest / ruff / sphinx / uv lock
                if not any(
                    check_tool in clean_first_line
                    for check_tool in ["pytest", "ruff", "sphinx", "uv lock"]
                ):
                    return True

        return False


class ExpressionEvaluator:
    """Evaluates GitHub Actions expression syntax: ${{ ... }}."""

    @staticmethod
    def evaluate(
        text: str,
        matrix_context: Dict[str, Any],
        env_context: Dict[str, str],
        repo_root: Path,
        workflow_name: str,
    ) -> str:
        if not text or "${{" not in text:
            return text

        def replacer(match: re.Match[str]) -> str:
            expr = match.group(1).strip()

            # Handle logical fallbacks first: ${{ github.head_ref || github.run_id }}
            if "||" in expr:
                parts = [p.strip() for p in expr.split("||")]
                for part in parts:
                    val = ExpressionEvaluator.evaluate(
                        f"${{{{ {part} }}}}",
                        matrix_context,
                        env_context,
                        repo_root,
                        workflow_name,
                    )
                    if val:
                        return val
                return ""

            # Handle matrix: ${{ matrix.python-version }} or ${{ matrix.var }}
            if expr.startswith("matrix."):
                key = expr[len("matrix.") :].strip()
                if key in matrix_context:
                    return str(matrix_context[key])
                return ""

            # Handle github context
            if expr == "github.workspace":
                return str(repo_root)
            if expr == "github.workflow":
                return workflow_name
            if expr.startswith("github."):
                sub = expr[len("github.") :].strip()
                if sub == "ref":
                    return "refs/heads/master"
                if sub == "sha":
                    return "HEAD"
                if sub == "run_id":
                    return "local_run"
                return ""

            # Handle env context: ${{ env.VAR }}
            if expr.startswith("env."):
                key = expr[len("env.") :].strip()
                return env_context.get(key, os.environ.get(key, ""))

            # Handle secrets: ${{ secrets.TOKEN }}
            if expr.startswith("secrets."):
                key = expr[len("secrets.") :].strip()
                return os.environ.get(key, "")

            return ""

        pattern = r"\$\{\{\s*(.*?)\s*\}\}"
        return re.sub(pattern, replacer, text)


class WorkflowParser:
    """Discovers and parses workflow YAML files into Workflow objects."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.workflows_dir = repo_root / ".github" / "workflows"

    def discover_workflows(self) -> List[Path]:
        if not self.workflows_dir.exists():
            return []

        workflow_files = []
        for ext in ("*.yaml", "*.yml"):
            for path in sorted(self.workflows_dir.glob(ext)):
                workflow_files.append(path)

        return workflow_files

    def parse_workflow(self, file_path: Path) -> Optional[Workflow]:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is not installed. Please install PyYAML (e.g., `uv pip install pyyaml`) "
                "or run using `uv run python scripts/run_ci.py`."
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(
                f"{Style.RED}[ERROR]{Style.RESET} Failed to parse {file_path}: {e}",
                file=sys.stderr,
            )
            return None

        if not isinstance(data, dict) or "jobs" not in data:
            return None

        workflow_name = str(data.get("name", file_path.stem))
        workflow_env = {str(k): str(v) for k, v in data.get("env", {}).items()}

        workflow = Workflow(
            path=file_path,
            filename=file_path.name,
            name=workflow_name,
            env=workflow_env,
        )

        jobs_data = data.get("jobs", {})
        if not isinstance(jobs_data, dict):
            return workflow

        for job_id, job_spec in jobs_data.items():
            if not isinstance(job_spec, dict):
                continue

            job_name = str(job_spec.get("name", job_id))
            runs_on = str(job_spec.get("runs-on", "ubuntu-latest"))

            # Needs dependencies
            needs_raw = job_spec.get("needs", [])
            if isinstance(needs_raw, str):
                needs = [needs_raw]
            elif isinstance(needs_raw, list):
                needs = [str(n) for n in needs_raw]
            else:
                needs = []

            # Matrix strategy
            strategy_matrix: Dict[str, List[Any]] = {}
            strategy = job_spec.get("strategy", {})
            if isinstance(strategy, dict):
                matrix = strategy.get("matrix", {})
                if isinstance(matrix, dict):
                    for m_key, m_val in matrix.items():
                        if isinstance(m_val, list):
                            strategy_matrix[str(m_key)] = m_val
                        else:
                            strategy_matrix[str(m_key)] = [m_val]

            # Job env
            job_env = {str(k): str(v) for k, v in job_spec.get("env", {}).items()}

            # Job steps
            steps_data = job_spec.get("steps", [])
            steps: List[Step] = []
            if isinstance(steps_data, list):
                for step_spec in steps_data:
                    if not isinstance(step_spec, dict):
                        continue

                    raw_run = step_spec.get("run")
                    raw_uses = step_spec.get("uses")
                    raw_name = step_spec.get("name")

                    if not raw_name:
                        if raw_run:
                            first_line = raw_run.strip().splitlines()[0]
                            raw_name = first_line[:60]
                        elif raw_uses:
                            raw_name = raw_uses
                        else:
                            raw_name = "Unnamed step"

                    step_env = {
                        str(k): str(v) for k, v in step_spec.get("env", {}).items()
                    }
                    with_args = (
                        step_spec.get("with", {})
                        if isinstance(step_spec.get("with"), dict)
                        else {}
                    )

                    step = Step(
                        name=str(raw_name),
                        run=str(raw_run) if raw_run is not None else None,
                        uses=str(raw_uses) if raw_uses is not None else None,
                        with_args=with_args,
                        env=step_env,
                        if_cond=str(step_spec.get("if")) if "if" in step_spec else None,
                        working_directory=step_spec.get("working-directory"),
                        continue_on_error=bool(
                            step_spec.get("continue-on-error", False)
                        ),
                        source_file=file_path,
                    )
                    step.is_setup = StepClassifier.is_setup_step(step)
                    steps.append(step)

            job = Job(
                id=job_id,
                name=job_name,
                runs_on=runs_on,
                needs=needs,
                strategy_matrix=strategy_matrix,
                env=job_env,
                steps=steps,
                if_cond=str(job_spec.get("if")) if "if" in job_spec else None,
            )
            workflow.jobs[job_id] = job

        return workflow

    def load_all_workflows(self) -> List[Workflow]:
        workflow_paths = self.discover_workflows()
        workflows = []
        for path in workflow_paths:
            wf = self.parse_workflow(path)
            if wf and wf.jobs:
                workflows.append(wf)
        return workflows


class LocalCIRunner:
    """Executes CI checks locally, managing environments, matrix expansion, and reports."""

    def __init__(
        self,
        repo_root: Path,
        skip_setup: bool = True,
        fail_fast: bool = False,
        dry_run: bool = False,
        matrix_all: bool = False,
        verbose: bool = False,
    ):
        self.repo_root = repo_root
        self.skip_setup = skip_setup
        self.fail_fast = fail_fast
        self.dry_run = dry_run
        self.matrix_all = matrix_all
        self.verbose = verbose
        self.results: List[StepResult] = []

    def _normalize_command(self, cmd: str) -> str:
        """Normalizes multiline shell commands (e.g. joining lines ending in \\)."""
        lines = [line.strip() for line in cmd.strip().splitlines() if line.strip()]
        if not lines:
            return ""

        joined_lines = []
        current_chunk = []

        for line in lines:
            if line.endswith("\\"):
                current_chunk.append(line[:-1].rstrip())
            else:
                current_chunk.append(line)
                joined_lines.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            joined_lines.append(" ".join(current_chunk))

        return " && ".join(joined_lines) if len(joined_lines) > 1 else joined_lines[0]

    def _get_matrix_combinations(
        self, matrix_def: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generates matrix combinations. Returns [ {} ] if matrix is empty."""
        if not matrix_def:
            return [{}]

        keys = list(matrix_def.keys())
        value_lists = [matrix_def[k] for k in keys]

        if not self.matrix_all:
            first_comb = {keys[i]: value_lists[i][0] for i in range(len(keys))}
            return [first_comb]

        combinations = []
        for combo in itertools.product(*value_lists):
            combinations.append({keys[i]: combo[i] for i in range(len(keys))})
        return combinations

    def _execute_command(
        self,
        cmd: str,
        env: Dict[str, str],
        working_dir: Path,
        step_label: str,
    ) -> Tuple[bool, float, Optional[str]]:
        """Executes a single normalized shell command and streams output."""
        start_time = time.time()

        if self.dry_run:
            print(
                f"{Style.DIM}  [DRY-RUN] Would run: {Style.RESET}{Style.CYAN}{cmd}{Style.RESET}"
            )
            return True, 0.0, None

        merged_env = os.environ.copy()
        merged_env.update(env)
        merged_env["LOCAL_CI"] = "1"

        print(f"{Style.DIM}  > {cmd}{Style.RESET}")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(working_dir),
                env=merged_env,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    print(f"    {line}", end="")
                process.stdout.close()

            process.wait()
            duration = time.time() - start_time

            if process.returncode == 0:
                return True, duration, None
            else:
                err = f"Command exited with code {process.returncode}"
                return False, duration, err

        except Exception as e:
            duration = time.time() - start_time
            return False, duration, str(e)

    def run_step(
        self,
        step: Step,
        workflow: Workflow,
        job: Job,
        matrix_context: Dict[str, Any],
        step_idx: int,
        total_steps: int,
    ) -> StepResult:
        matrix_str = (
            f" ({', '.join(f'{k}={v}' for k, v in matrix_context.items())})"
            if matrix_context
            else ""
        )
        display_job_name = f"{job.name}{matrix_str}"

        interpolated_name = ExpressionEvaluator.evaluate(
            step.name, matrix_context, step.env, self.repo_root, workflow.name
        )

        if self.skip_setup and step.is_setup:
            if self.verbose:
                print(
                    f"{Style.DIM}  [{step_idx}/{total_steps}] [SKIP SETUP] {interpolated_name}{Style.RESET}"
                )
            return StepResult(
                workflow_name=workflow.name,
                job_name=display_job_name,
                step_name=interpolated_name,
                command=step.uses or step.run or "",
                status="SKIPPED",
                duration=0.0,
            )

        effective_env = workflow.env.copy()
        effective_env.update(job.env)
        effective_env.update(step.env)

        work_dir = self.repo_root
        if step.working_directory:
            interpolated_dir = ExpressionEvaluator.evaluate(
                step.working_directory,
                matrix_context,
                effective_env,
                self.repo_root,
                workflow.name,
            )
            work_dir = self.repo_root / interpolated_dir

        if step.uses and not step.run:
            print(
                f"{Style.BLUE}  [{step_idx}/{total_steps}] Action: {Style.BOLD}{interpolated_name}{Style.RESET} {Style.DIM}({step.uses}){Style.RESET}"
            )
            return StepResult(
                workflow_name=workflow.name,
                job_name=display_job_name,
                step_name=interpolated_name,
                command=step.uses,
                status="PASSED",
                duration=0.0,
            )

        raw_cmd = step.run or ""
        interpolated_cmd = ExpressionEvaluator.evaluate(
            raw_cmd, matrix_context, effective_env, self.repo_root, workflow.name
        )
        normalized_cmd = self._normalize_command(interpolated_cmd)

        print(
            f"{Style.CYAN}  [{step_idx}/{total_steps}] Running: {Style.BOLD}{interpolated_name}{Style.RESET}"
        )

        success, duration, err = self._execute_command(
            normalized_cmd, effective_env, work_dir, interpolated_name
        )

        status = "PASSED" if success else "FAILED"
        status_color = Style.GREEN if success else Style.RED
        print(
            f"  {status_color}[{status}]{Style.RESET} {interpolated_name} ({duration:.2f}s)\n"
        )

        return StepResult(
            workflow_name=workflow.name,
            job_name=display_job_name,
            step_name=interpolated_name,
            command=normalized_cmd,
            status=status,
            duration=duration,
            error_message=err,
        )

    def run_job(
        self, job: Job, workflow: Workflow, matrix_context: Dict[str, Any]
    ) -> bool:
        matrix_str = (
            f" [matrix: {', '.join(f'{k}={v}' for k, v in matrix_context.items())}]"
            if matrix_context
            else ""
        )
        print(
            f"\n{Style.BOLD}{Style.WHITE}--- Job: {job.name}{matrix_str} ---{Style.RESET}"
        )

        steps_to_run = job.steps
        if self.skip_setup:
            non_setup_steps = [s for s in job.steps if not s.is_setup]
            if not non_setup_steps:
                print(
                    f"{Style.DIM}  All steps in this job are setup steps. Skipping job.{Style.RESET}"
                )
                return True
            steps_to_run = non_setup_steps

        all_passed = True
        total = len(steps_to_run)

        for idx, step in enumerate(steps_to_run, 1):
            res = self.run_step(step, workflow, job, matrix_context, idx, total)
            self.results.append(res)

            if res.status == "FAILED":
                all_passed = False
                if not step.continue_on_error and self.fail_fast:
                    print(
                        f"{Style.RED}[FAIL-FAST] Stopping on first failure.{Style.RESET}"
                    )
                    return False

        return all_passed

    def run_workflow(
        self, workflow: Workflow, target_job_id: Optional[str] = None
    ) -> bool:
        print(
            f"\n{Style.BOLD}{Style.MAGENTA}======================================================{Style.RESET}"
        )
        print(
            f"{Style.BOLD}{Style.MAGENTA}Workflow: {workflow.name} ({workflow.filename}){Style.RESET}"
        )
        print(
            f"{Style.BOLD}{Style.MAGENTA}======================================================{Style.RESET}"
        )

        job_order = self._resolve_job_order(workflow.jobs)
        workflow_passed = True

        for job_id in job_order:
            if (
                target_job_id
                and job_id != target_job_id
                and workflow.jobs[job_id].name != target_job_id
            ):
                continue

            job = workflow.jobs[job_id]
            matrix_combos = self._get_matrix_combinations(job.strategy_matrix)

            for matrix_context in matrix_combos:
                job_passed = self.run_job(job, workflow, matrix_context)
                if not job_passed:
                    workflow_passed = False
                    if self.fail_fast:
                        return False

        return workflow_passed

    def _resolve_job_order(self, jobs: Dict[str, Job]) -> List[str]:
        """Resolves execution order of jobs according to `needs:` dependencies."""
        visited = set()
        order: List[str] = []

        def visit(jid: str) -> None:
            if jid in visited or jid not in jobs:
                return
            job = jobs[jid]
            for dep in job.needs:
                visit(dep)
            visited.add(jid)
            order.append(jid)

        for jid in jobs:
            visit(jid)

        return order

    def print_summary(self) -> int:
        """Prints a structured summary table and returns exit code (0 or 1)."""
        print(
            f"\n{Style.BOLD}{Style.WHITE}========================================================================================{Style.RESET}"
        )
        print(
            f"{Style.BOLD}{Style.WHITE}                                LOCAL CI CHECK SUMMARY                                  {Style.RESET}"
        )
        print(
            f"{Style.BOLD}{Style.WHITE}========================================================================================{Style.RESET}"
        )

        executed_results = [r for r in self.results if r.status != "SKIPPED"]
        if not executed_results:
            print(f"{Style.YELLOW}No checks were executed.{Style.RESET}")
            return 0

        # Table headers: Workflow (18), Job (20), Check / Step (34), Status (8), Time (7)
        w_wf, w_job, w_step, w_status, w_time = 18, 20, 34, 8, 7
        total_width = w_wf + w_job + w_step + w_status + w_time + 4

        header = (
            f"{format_cell('Workflow', w_wf)} "
            f"{format_cell('Job', w_job)} "
            f"{format_cell('Check / Step', w_step)} "
            f"{format_cell('Status', w_status)} "
            f"{format_cell('Time', w_time, 'right')}"
        )
        print(f"{Style.DIM}{header}{Style.RESET}")
        print(f"{Style.DIM}{'-' * total_width}{Style.RESET}")

        passed_count = 0
        failed_count = 0
        total_time = 0.0

        for r in executed_results:
            total_time += r.duration
            if r.status == "PASSED":
                status_disp = f"{Style.GREEN}PASSED{Style.RESET}"
                passed_count += 1
            else:
                status_disp = f"{Style.RED}FAILED{Style.RESET}"
                failed_count += 1

            wf_disp = r.workflow_name[: w_wf - 1]
            job_disp = r.job_name[: w_job - 1]
            step_disp = r.step_name[: w_step - 1]
            time_disp = f"{r.duration:.2f}s"

            row = (
                f"{format_cell(wf_disp, w_wf)} "
                f"{format_cell(job_disp, w_job)} "
                f"{format_cell(step_disp, w_step)} "
                f"{format_cell(status_disp, w_status)} "
                f"{format_cell(time_disp, w_time, 'right')}"
            )
            print(row)

        print(f"{Style.DIM}{'-' * total_width}{Style.RESET}")
        print(
            f"Total: {len(executed_results)} checks | "
            f"{Style.GREEN}{passed_count} passed{Style.RESET} | "
            f"{Style.RED if failed_count else Style.DIM}{failed_count} failed{Style.RESET} | "
            f"Total time: {total_time:.2f}s"
        )

        if failed_count > 0:
            print(
                f"\n{Style.BOLD}{Style.RED}FAILED: Some CI checks failed.{Style.RESET}"
            )
            return 1
        else:
            print(
                f"\n{Style.BOLD}{Style.GREEN}SUCCESS: All CI checks passed!{Style.RESET}"
            )
            return 0


def list_workflows(workflows: List[Workflow], skip_setup: bool) -> None:
    """Lists discovered workflows, jobs, and steps with their classification."""
    print(
        f"\n{Style.BOLD}{Style.WHITE}Discovered CI Workflows & Actions:{Style.RESET}\n"
    )

    for wf in workflows:
        print(
            f"{Style.BOLD}{Style.MAGENTA}Workflow: {wf.name}{Style.RESET} ({Style.DIM}{wf.filename}{Style.RESET})"
        )
        for job_id, job in wf.jobs.items():
            matrix_info = (
                f" [matrix: {job.strategy_matrix}]" if job.strategy_matrix else ""
            )
            needs_info = f" [needs: {job.needs}]" if job.needs else ""
            print(
                f"  {Style.CYAN}Job: {job.name} ({job_id}){matrix_info}{needs_info}{Style.RESET}"
            )

            for step in job.steps:
                if step.is_setup:
                    tag = (
                        f"{Style.DIM}[SETUP - SKIPPED]{Style.RESET}"
                        if skip_setup
                        else f"{Style.YELLOW}[SETUP]{Style.RESET}"
                    )
                else:
                    tag = f"{Style.GREEN}[CHECK ACTION]{Style.RESET}"

                cmd_preview = ""
                if step.run:
                    first_line = step.run.strip().splitlines()[0]
                    cmd_preview = f" -> {first_line[:50]}"
                elif step.uses:
                    cmd_preview = f" -> uses: {step.uses}"

                print(f"    {tag} {step.name}{Style.DIM}{cmd_preview}{Style.RESET}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dynamic Local GitHub Actions CI Runner for Algan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ci.py                 # Run all CI checks (skipping setup)
  python scripts/run_ci.py -w code_quality # Run only code quality checks
  python scripts/run_ci.py -w docs         # Run documentation build check
  python scripts/run_ci.py -w test         # Run test suite check
  python scripts/run_ci.py --list          # List all discovered workflows and steps
  python scripts/run_ci.py --dry-run       # Preview commands without executing
        """,
    )

    parser.add_argument(
        "workflows",
        nargs="*",
        help="Optional workflow name(s) or filename(s) to run (e.g. code_quality, test, docs.yaml)",
    )
    parser.add_argument(
        "-w",
        "--workflow",
        dest="workflow_filter",
        help="Filter by workflow name or filename",
    )
    parser.add_argument(
        "-j",
        "--job",
        dest="job_filter",
        help="Filter by job ID or job name",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List discovered workflows, jobs, and steps without running",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without running them",
    )
    parser.add_argument(
        "--all-steps",
        "--no-skip-setup",
        dest="skip_setup",
        action="store_false",
        default=True,
        help="Run all steps including setup/provisioning steps (default: skip setup steps)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop execution immediately upon the first failed step",
    )
    parser.add_argument(
        "--matrix-all",
        action="store_true",
        default=False,
        help="Run all matrix combinations instead of the default/first one",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    current_path = Path(__file__).resolve().parent
    repo_root = current_path.parent if current_path.name == "scripts" else current_path
    if not (repo_root / ".github").exists() and (Path.cwd() / ".github").exists():
        repo_root = Path.cwd()

    parser_obj = WorkflowParser(repo_root)
    discovered_workflows = parser_obj.load_all_workflows()

    if not discovered_workflows:
        print(
            f"{Style.RED}[ERROR]{Style.RESET} No GitHub workflow files found in {repo_root / '.github' / 'workflows'}",
            file=sys.stderr,
        )
        return 1

    if args.list:
        list_workflows(discovered_workflows, args.skip_setup)
        return 0

    filters = []
    if args.workflow_filter:
        filters.append(args.workflow_filter.lower())
    if args.workflows:
        filters.extend([w.lower() for w in args.workflows])

    target_workflows = []
    if filters:
        for wf in discovered_workflows:
            wf_name_lower = wf.name.lower()
            wf_file_lower = wf.filename.lower()
            wf_stem_lower = wf.path.stem.lower()

            if any(
                f in wf_name_lower or f in wf_file_lower or f == wf_stem_lower
                for f in filters
            ):
                target_workflows.append(wf)

        if not target_workflows:
            print(
                f"{Style.RED}[ERROR]{Style.RESET} No workflows matched filter: {filters}",
                file=sys.stderr,
            )
            print(
                f"Available workflows: {', '.join(w.filename for w in discovered_workflows)}"
            )
            return 1
    else:
        target_workflows = discovered_workflows

    runner = LocalCIRunner(
        repo_root=repo_root,
        skip_setup=args.skip_setup,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
        matrix_all=args.matrix_all,
        verbose=args.verbose,
    )

    for wf in target_workflows:
        runner.run_workflow(wf, target_job_id=args.job_filter)

    return runner.print_summary()


if __name__ == "__main__":
    sys.exit(main())
