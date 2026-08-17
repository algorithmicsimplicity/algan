<!--
Write this description yourself, in your own words, from what you set out to
do. Do not paste a generated summary: they restate the diff rather than the
change, and they invent novelty ("X was impossible before") for things that
were merely undocumented. The diff is already in the PR; this is for what the
diff cannot show. See CLAUDE.md, "Pull requests".

Delete any section that genuinely does not apply.
-->

## What and why

<!--
A couple of sentences: what the change does, what was true before it, and the
problem or gap it closes. Behaviour, not files. Name private helpers only if a
reviewer needs them to follow the argument.
-->

## Rendered output

<!--
The first thing a reviewer here needs to know. One of:

- Unchanged, and how you know -- which suites you ran, on which device.
- Changed, deliberately: which baselines you regenerated, on which device
  (`expected_outputs_cpu/` and `expected_outputs_cuda/` are separate sets and
  the CUDA one has to be regenerated on a CUDA machine, so a change that moves
  output needs both), and why the new frames are the correct ones. Say that you
  looked at them. Never re-baseline to turn a red test green.

Tessellation, projection and level-criterion changes are invisible to `--fast`
-- `tests/fast/scene.py` has no PN geometry -- so they need
`pytest -q tests/full_renders`.
-->

## Verification

<!--
Which suites ran, on what hardware, and what they said. For example:

- `pytest -q --fast` (CPU) -- pass
- `pytest -q` (CPU) -- pass, except <pre-existing failure and the evidence it
  is pre-existing>
- an A/B parity script, for an optimization

Say plainly what you could not check here -- CUDA behaviour from a CPU-only
machine, for one -- rather than leaving it implied.
-->

## Docs

<!--
Public API touched? Then docstrings follow DOCSTRINGS.md, and the tutorial and
reference pages move in this same PR. Say which pages changed.
-->
