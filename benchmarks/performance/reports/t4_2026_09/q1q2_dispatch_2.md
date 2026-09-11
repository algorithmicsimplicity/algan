# Q1/Q2 control: is the branch a regression with the toggle *off*?

**No. Branch-off is indistinguishable from master.** The regression is the
feature, not the branch.

This closes the gap left open by `q1q2_dispatch_1.md`. With
`device_dispatch=False` the branch still reads
`SETTINGS.raytracing.device_dispatch` on every `ManualMemory.get_tensor` call,
and `current_pointer` / `current_reverse_pointer` are now properties with a
tracker check on every write. Those are unconditional, so the `off` arm of the
first round was not master and its 9.82 s could in principle have been a
regression the A/B could not see.

It is not.

| arm | warm RUN 2 | median |
| --- | --- | ---: |
| master (`f2073d7`) | 9.75 / 10.03 / 9.95 | **9.95** |
| branch, `ALGAN_DEVICE_DISPATCH=0` | 9.93 / 9.70 / 10.02 | **9.93** |
| branch, `ALGAN_DEVICE_DISPATCH=1` | 10.27 | **10.27** |

Master and branch-off overlap completely (9.75–10.03 against 9.70–10.02); the
median gap is 0.02 s, two thousandths of the render. The always-on settings
read and the property indirection cost nothing measurable.

The single `on` arm reproduces the first round's result in an independent
session: **+3.4%** against branch-off, **+3.2%** against master.

## How the master arm ran

The notebook clones one branch, and a cross-session comparison is not a reading
on this box (13% drift at PREVIEW), so master had to run *inside the same
session*. Each master step swaps the source tree in place and restores it:

```
git fetch -q origin master && git checkout -q FETCH_HEAD -- algan/ \
  && (python benchmarks/performance/nn_ablation.py base UHD; rc=$?; \
      git checkout -q HEAD -- algan/; exit $rc)
```

The install is editable, so the next `python` imports whatever is on disk. The
restore runs whether the render passed or failed, so a failure cannot poison
later steps. Files that exist only on the branch stay on disk during the master
arm; master's `raster_pipeline.py` does not import them, and
`ALGAN_DEVICE_DISPATCH` is unset for those steps, so master's `environment.py`
never sees an undeclared name.

Two independent confirmations that master's code actually ran: the master arm
logs `ALGAN_DEVICE_DISPATCH=(default)` rather than a value, and its first render
cost 66.7 s against ~21 s for a cache-warm branch step — a different source tree
is a different kernel cache key, so it recompiled.

## Output parity, master included

Every arm of this session and the last produced the same UHD video:

```
1c89e4c1712a4cf6   master, branch dd0, branch dd1
```

So the whole branch — not just the toggle — is byte-identical to master on this
scene.

## What this leaves

The verdict on Q1/Q2 is unchanged and now has two sessions behind it: correct,
and slower when enabled. The cost is the queue sizing described in
`q1q2_dispatch_1.md`, not anything the branch does unconditionally — which
means the feature could be withdrawn or fixed without touching the rest of the
branch.

Transcript: `q1q2-control-1.txt`. Kaggle T4, one session, seven UHD arms,
`codex/q1q2-device-dispatch-20260911` @ `445bc7e` against `origin/master`
@ `f2073d7`.
