---
name: ox-agents-in-parallel-worktrees
description: "Running several Ox Alpha agents at once on algan: git worktrees, venvs, brief structure, and the failure modes that actually happened"
metadata:
  node_type: memory
  type: feedback
  modified: 2026-08-26
---

Ox Alpha (`opencode run --auto --variant max --model opencode/x-preview-f-free`)
is capable enough to be given a real optimization task and trusted to report
honestly — including reporting that the task was not worth doing, which it did
twice in one session. What it needs is isolation and a brief that makes the
measurement the deliverable.

**Why:** three agents on disjoint targets finish in roughly the time one would,
and the reports are the durable artifact even when the code is not.

## Setup

One **git worktree per agent**, each with its own `.venv`:

```
git worktree add -b perf/<task> ../algan_wt_<name> <base-commit>
cd ../algan_wt_<name> && uv sync && uv pip install pytest parameterized
```

**`uv sync` installs CPU-only torch** (`2.7.1+cpu`) because `pyproject.toml`
pins no CUDA index — the main `.venv` got its `+cu128` build by hand. Installing
it per worktree means a 3 GiB download each, and three of them in parallel
crawled for over an hour. **Junction the main venv's install instead** — it is
instant and read-only:

```bash
SP=<wt>/.venv/Lib/site-packages; MAIN=/d/algan/.venv/Lib/site-packages
for pkg in torch torchvision functorch torchgen nvidia triton; do
  rm -rf "$SP/$pkg"
  cmd //c mklink //J "$(cygpath -w "$SP/$pkg")" "$(cygpath -w "$MAIN/$pkg")"
done
```

(PowerShell's `Stop-Process`/`Remove-Item` refuse `cmd /c rmdir /S /Q` style
arguments here; do the removal from bash.)

Drive each with `scratch_perf/r2/ox/run_ox.sh <brief> <log>`, which pings the
endpoint first and retries the transient "Endpoint is unavailable".

## Failure modes that actually happened

**An agent ran renders inside another agent's worktree.** The brief said "run
the same test on a pristine `master` worktree"; the agent found a sibling
worktree and used it — while another agent was editing it. **Every brief must
name the one directory the agent may touch and say the others are in use right
now.** If a verification step wants a pristine checkout and there is none, tell
it to say so rather than go looking.

**Swapping torch under a running agent invalidates its A/B.** A CPU render and a
CUDA render of the same scene are not byte-identical. Finish the venv setup
before launching, or warn the agent in the brief that both arms of any
comparison must print the same device line.

**Four GPU-using processes on one card makes every wall-clock number noise**, and
a killed pytest leaves orphaned render subprocesses holding VRAM. Check
`nvidia-smi` and kill orphans before believing a timing. Briefs should say so and
ask for **counts** — rays, launches, calls — as the primary evidence.

## What makes a brief work

* **Lead with the measurement that motivates it**, with real numbers and where
  they came from. The agents reason from it and will tell you when the premise
  is wrong.
* **"Measure first, do not implement until Part 1 is done."** Both negative
  results this session came from agents that followed this and found the premise
  false. Say explicitly that an honest negative is a good outcome.
* **Spell out the repo's own traps** — `ti.static` gates are baked at compile
  time so every A/B arm is its own process; an H.264 re-encode turns a
  single-channel difference into thousands of pixels so comparisons must be
  `libx264rgb -qp 0`; `available_memory_override` must be pinned in both arms or
  the batch window moves and legitimately moves pixels.
* **Ask for the verification output verbatim**, and for an explicit list of what
  was *not* verified. Both were honoured.

## Integrating several agents' work

Collect `git diff` from each worktree, apply them to one integration tree, and
verify the *combination* — each agent only verified its own change. Archive
every patch (including ones you do not ship) and every report under
`scratch_perf/`, then reset the agent trees onto the integration commit so the
next round's diffs compose. See `[[t4-round2-findings]]`.
