# OpenCode in a cloud session — handoff notes

Written 2026-08-21 from Claude Code on the web, over three sessions. Nothing here
is about algan itself; it is a record so the next session does not have to
rediscover it. The container is ephemeral, which is why this file is in git.

**Status: working.** OpenCode installs, its agent loop runs, and **Ox Alpha Free
runs as a tool-using subagent** in a cloud session — via the CLI and via the
headless HTTP API. No credential of any kind is needed, Ox Alpha is completely free
for this week.

## 1. Install

The documented installer (`curl -fsSL https://opencode.ai/install | bash`) fetches
from `opencode.ai`, so it only works once that host is allowlisted. Install from
npm instead — `registry.npmjs.org` is on the default allowlist, so this works at
any access level:

```bash
npm install -g opencode-ai
opencode --version     # 1.18.20 as of 2026-08-21
```

Takes about 12 seconds, 3 packages. The base image already has node v22.22.2,
npm 10.9.7, and bun, so there is nothing to install first. The binary lands at
`/opt/node22/bin/opencode`, which is already on `PATH`.

## 2. Running Ox Alpha Free — the model ID

Zen's free models need **no API key and no `opencode auth login`**.

```bash
opencode run --variant max --model opencode/x-preview-f-free "Reply with one word: PONG"
```

**Always pass `--variant max`.** See §2.2 — the default is not the model's top
reasoning effort, and the flag is silently ignored if you misspell it.

**The model ID is `opencode/x-preview-f-free`.** Its display name is
"Ox Alpha Free (Unlimited)". There is no `opencode/ox-alpha-free` — that guess is
what failed in the earlier sessions, and it failed with a generic `UnknownError`
that gave no hint the name was wrong.

The ID is a codename, so do not infer it. Read it off the catalog:

```bash
curl -sS https://models.dev/api.json | python3 -c "
import json,sys
for mid,m in sorted(json.load(sys.stdin)['opencode']['models'].items()):
    print(f\"{mid:42s} {m.get('name')}\")
"
```

Public references spell the model "Ox Alpha", "0x Alpha", and `stealth/ox-alpha`
(the OpenRouter/Kilo ID) interchangeably. In the catalog it appears as
`opencode/x-preview-f-free` and `opencode-go/ox-alpha-free` — the `opencode-go`
provider is the one where the obvious spelling is real, which is probably where the
wrong guess came from.

It was announced as a stealth model free for one week (1M context, multimodal, zero
data retention). It was still live on 2026-08-21. If it lapses, the other `*-free`
IDs in §2.1 need no key either.

### 2.1 Why `opencode models` shows only seven

`opencode models | grep '^opencode/'` lists exactly seven:

```
big-pickle  hy3-free  mimo-v2.5-free  muse-spark-1.2-contributor-free
nemotron-3-ultra-free  nemotron-3.5-lightning-free  x-preview-f-free
```

**Earlier notes claimed this seven-item list was OpenCode's offline fallback,
served because the catalog hosts were blocked. That is wrong.** With all three
hosts reachable, `~/.cache/opencode/models.json` is the live 4.2 MB models.dev
catalog containing **93** Zen models — and `opencode models` still shows seven, as
does `GET /config/providers`.

The seven are the subset usable **without authentication**. Asking for a paid one
fails at the server, not the client:

```
$ opencode run --model opencode/claude-opus-5 "hi"
"name": "UnknownError" ... "Unexpected server error."
```

Two consequences. First, a short list is not evidence that the catalog is blocked —
check the hosts directly instead. Second, `UnknownError` from Zen means "not
available to you", and reads identically whether the model is paid, retired, or
misspelled; it is not a signal you can debug a model ID from.

### 2.2 Reasoning effort — set it to `max` for best performance

Ox Alpha is a reasoning model, and **OpenCode does not run it at full effort
unless you ask.** The catalog entry declares:

```json
"reasoning": true,
"reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}]
```

So there are three levels and `max` is the top one.

```bash
opencode run --variant max --model opencode/x-preview-f-free "..."
```

Over the HTTP API it is a **sibling of `model`, not a field inside it** —
`model` is declared `additionalProperties: false`, so nesting it there is
rejected:

```json
{"model": {"providerID": "opencode", "modelID": "x-preview-f-free"},
 "variant": "max",
 "parts": [{"type": "text", "text": "..."}]}
```

Both forms were verified end to end. Two traps:

* **A bad variant fails silently.** `opencode run --variant bogus ...` returns a
  normal answer with no error and no warning — the flag is not validated
  against the model's declared values. A typo therefore costs you full effort
  and tells you nothing. There is no run-time echo of the effort actually
  used, so the only defence is getting the spelling right: `max`.
* **`--variant` is not `--thinking`.** `--variant` sets the effort; `--thinking`
  only *displays* reasoning blocks. Neither implies the other, and without
  `--thinking` the log shows tool calls and prose only, which makes a run at
  the wrong effort indistinguishable from one at the right effort.

The one substantial task in §4 was run at the **default** effort, not `max`, so
that assessment is a floor on the model's ability rather than a ceiling.

## 3. Driving the agent

Both paths were verified end-to-end against Ox Alpha in the third session. The
agent advertises ten tools: `bash`, `edit`, `glob`, `grep`, `read`, `skill`,
`task`, `todowrite`, `webfetch`, `write`.

**CLI, one shot.** `--auto` auto-approves permissions, which is what makes it work
non-interactively:

```bash
opencode run --auto --variant max --model opencode/x-preview-f-free "your prompt"
```

Run against this repo with a read-only question, it printed its tool calls as it
went and answered correctly:

```
✱ Grep "class ManualMemory"            1 match
✱ Glob "**/memory_utils.py"            1 match
✱ Grep "def scope|scope\("             in algan/utils/memory_utils.py · 8 matches
algan/utils/memory_utils.py
`scope` (ManualMemory.scope, line 792)
```

`--auto` means it can also write and run shell commands. If the task is meant to be
read-only, say so in the prompt *and* check `git status --porcelain` before and
after — that is how the run above was confirmed non-destructive.

**Headless HTTP API** — the real programmatic-control path:

```bash
opencode serve --port 4096 &
SID=$(curl -sS --noproxy '*' -X POST http://127.0.0.1:4096/session \
        -H 'content-type: application/json' -d '{"title":"driven"}' \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
curl -sS --noproxy '*' --max-time 240 -X POST "http://127.0.0.1:4096/session/$SID/message" \
  -H 'content-type: application/json' \
  -d '{"model":{"providerID":"opencode","modelID":"x-preview-f-free"},
       "variant":"max",
       "parts":[{"type":"text","text":"your prompt"}]}'
```

The response is JSON with a `parts` array; assistant text is the `type: "text"`
entries. Give the message call a generous `--max-time` — it blocks for the whole
agent turn, tool calls included. `GET /config/providers` lists what OpenCode thinks
is configured. The server warns `OPENCODE_SERVER_PASSWORD is not set; server is
unsecured` — fine on loopback, set it if the port is ever exposed. Pass
`--noproxy '*'` on loopback curls; `NO_PROXY` already covers `127.0.0.1`, but being
explicit avoids surprises.


## 4. What Ox Alpha is actually capable of

Assessed on 2026-08-21 by giving it one real task from this repo and reviewing
the result against the source. The task was **work-queue item 3, §I self-shadow
rejection by identity**: a designed-but-unbuilt renderer change requiring a new
ndarray and a packed id threaded through five `@ti.func` signatures shared with
a megakernel, across 22 call sites in Taichi kernel code. It was given the
design docs to read and no hints. **It ran at default effort — see §2.2 — so
this is a floor, not a ceiling.** Its own account is `OX_REPORT.md`; the
implementation is commit `ccab393`.

**Result: a competent, mergeable-quality implementation, honestly reported.**

What it did well, in descending order of how much it surprised me:

* **It caught two places where the design doc had gone stale, without being
  told to look.** §I says the source id is available in
  `raster_shadow_event_build` — a function deleted on 2026-08-19 by the
  sheet-resolve flip. It found where events are built now instead. §I also says
  `event_msk` "uses only its low 4 bits, leaving 28"; in fact bits 8+ carry the
  material pipeline id, and a literal implementation corrupts it scene-wide. It
  grepped `_USER_PIPELINE_BASE` on its own initiative, found the collision,
  packed at bits 16–31 instead, added a host-side guard, and flagged both
  deviations in code comments and in its report. **This is the thing to hire it
  for: it treats a spec as a claim to check, not an instruction to follow.**
* **It got the one stated correctness trap right.** "Reject same mesh AND
  near-zero `t`, never same mesh" — a concave solid must still shadow itself.
  The predicate is exactly right.
* **Judgement above the spec.** It centralised the acceptance floor into one
  helper rather than inlining it three times, gated with `ti.static` per this
  repo's conventions, degraded per-event instead of failing hard, and threaded
  `_collect_hits` — which the design's five-function list omits but the gather
  shadow mode needs.
* **Every factual claim in its report held up.** Test counts, the exact
  pre-existing failure (40 channel values, frame 6), "15 ruff findings, all
  pre-existing" — all re-run and confirmed exactly. It verified its own
  byte-identity claim by checking the failure magnitude was *unchanged* rather
  than just that the suite still ran.
* **It was straight about what it had not done**, unprompted: *"the feature
  should be treated as plumbed but pixel-unverified."* It even diagnosed the
  flaw in its own test scene — the block floated 0.47 units above the slab, so
  the scene never exercised contact at all — which is the same fault an
  independent review found separately.

Where it fell short:

* **It never demonstrated the feature changes a pixel**, and stopped without
  closing that gap. Independent instrumentation later showed the mechanism does
  engage (18,900 of 18,900 shadow events packed, kernel gate set) but moves
  zero pixels on every scene tried — so its work was sound and its verification
  was not.
* **One latent defect it did not catch**: with the gate on, the trace kernel
  masks `pid_e` to its low byte for *every* event, including ones the host
  declined to pack. Its own code comment justifies the mask by the host guard,
  which only covers the packed events. Narrow (needs ≥250 user fragment
  pipelines) but real.
* **It missed a design-level inconsistency.** Item 3 names both lost contact
  shadows and grazing-light acne; §I relaxes only *cross-mesh* hits, and acne
  is a mesh shadowing itself, which by design keeps the old epsilon. §I cannot
  do half of what the item claims. It implemented §I faithfully without
  questioning whether §I addresses the symptom.
* **It ran out of steps before writing its report** and needed
  `opencode run --continue` to finish. The continuation resumed with full
  context and respected "do not change code" (the diff md5 was unchanged).

**How to use it.** Give it a well-scoped task with the reference material named
and a stated pre-existing failure it must not chase. It is strong at reading a
large unfamiliar codebase, at threading a mechanical change through many call
sites correctly, and at telling you what it did not verify. It is weak at
designing the experiment that would prove its own work — specify the
verification, do not leave it to invent one. Budget for a continuation call,
and read `git diff` yourself: the report is accurate but it is not a review.

## 4.1 A long prompt hangs the run — put the brief in a file

Found on 2026-08-21 while driving four tasks through `opencode run`. **A prompt
of roughly 9 KB or more hangs before the first step and never recovers.** It is
not a crash and not a timeout: the process sits at 1% CPU with the log stopping
after `init` / `cleanup` and no `loop ... step=1` line ever appearing. Two runs
were killed at 47 and 12 minutes in that state, having written nothing.

It is the prompt *size*, not the model, the network or the task: a one-line ping
and a two-sentence tool-using question both answered normally in the same
minutes that a long prompt hung, on the same model ID and the same `--variant
max`. Prompts around 6-7 KB did work earlier in the same session, so the
threshold is somewhere above that and is not worth locating precisely.

The workaround is reliable and costs nothing:

```bash
cp brief.md /tmp/ox_brief.md
opencode run --auto --variant max --model opencode/x-preview-f-free \
  "Read the file /tmp/ox_brief.md and carry out the task it describes, in full,
   including every verification step it specifies. It is a complete brief;
   follow it."
```

Ox reads the file as its first tool call and proceeds normally. Write briefs to
a file by default rather than deciding each time whether this one is short
enough.

**How to tell a hang from slow thinking**, since `opencode run` buffers all its
output until the process exits and shows you nothing in the meantime: tail
`/root/.local/share/opencode/log/opencode.log`. A healthy run emits
`message=loop ... step=N` lines that keep climbing. A hung one stops at
`message=init` and the `cleanup prune=7.days` line a minute later, and never
logs a step. Check that before waiting another half hour.

**UPDATE: REVIEW FROM A SECOND TASK**
I used it for the read-only audit and the bulk of the implementation, and it earned it: its call-site inventory corrected my sketch in several places (the Monte Carlo megakernel has no refraction at all; two orchestrators tracer.py imports don't exist; const_fill/gen_fused are inert in refraction batches). It also flagged the per-hit IOR interpolation as a caveat — which is exactly what the control frame later tripped on.

Two things to add to scripts/ox_alpha/ox_alpha_opencode_agent.md for the next session:

It ran out of steps twice, both times just before the report — same as last session, so budget two continuation calls, not one.
It declared its adversarial diff re-read complete with "no defects found" while a real one was sitting in the diff: sca_width shadowed — the imported helper passed as a width to _alloc_wavefront_state, which would have crashed the sparse sheet route on every render, gate or no gate. It ran ruff and an import check and neither could see it. §4's "read git diff yourself: the report is accurate but it is not a review" is still the right advice, and this is a sharper example than the last one.

The verification split worked: it did the plumbing, I designed and ran the experiments, and the experiments are what found both the shadowing bug and the edge artifact.

**UPDATE: REVIEW FROM A THIRD TASK**
It did nearly all of it: the kernel, the wiring, the settings/env registration, the harness and test coverage, and every measurement. It reads a large unfamiliar codebase well and reports honestly — its claims re-ran exactly as written when I checked them independently.

Two things worth recording for the next session. It needs continuations, often — the step budget per opencode run invocation is small (~15–25 tool calls), and one invocation burned itself entirely on reading a report. Short, directive prompts get more work per call. And the review is still yours: my read of its first diff found a latent defect it had not (the kernel left rank uninitialised for any stream whose first band flag is clear — unreachable through compact_sheets, but _conflict_rank is now a directly-callable helper) and one small regression (its torch arm re-allocated an arange the caller deliberately shares, falsifying the comment that documents the sharing). Handing both back with a test requirement worked well — it fixed them and wrote the regression case.

## 5. Settled, and still open

Settled — do not re-test:

- Install from npm works in a fresh container (~12s, `opencode --version` → 1.18.20).
- The agent loop works in a fresh container, tools and all (§6).
- Ox Alpha Free is `opencode/x-preview-f-free`, runs with no credential, and works
  as a tool-using subagent through both the CLI and the HTTP API (§3, §5).
- Allowlisting `opencode.ai`, `models.dev` and `models.opencode.ai` is sufficient;
  inference reached no other host.
- The seven-model list is the unauthenticated-usable subset, **not** an offline
  fallback (§3.1).
- `ANTHROPIC_API_KEY` is not set in the environment by default.
- A prompt of ~9 KB or more hangs the run before its first step; pass the brief
  as a file instead (§4.1). Confirmed by killing two hung runs and re-running
  the identical task from a file, which worked.
- `opencode run` shows nothing until it exits, so
  `/root/.local/share/opencode/log/opencode.log` is how you tell progress from a
  hang: look for `message=loop ... step=N`.

- Ox Alpha's quality on a substantial task: assessed in §7. It implemented a real
  designed-but-unbuilt renderer change competently and reported it honestly.
- `--variant max` works on both the CLI and the HTTP API, and a misspelled
  variant is accepted silently (§3.2).

Still open:

- The TUI (`opencode` with no subcommand) — only `run` and `serve` were exercised.
- **What `max` is actually worth.** Every task in §7 ran at the default effort.
  Nothing has been run twice to compare `max` against it, so the gain is assumed,
  not measured. Re-running the §I task at `max` is the obvious experiment: the
  three things it missed (pixel-verifying its own work, the `pid_e` masking
  asymmetry, and that §I cannot address the acne half of the item) are a ready
  scorecard.
- Whether the effort actually reaches the provider. `--variant bogus` is accepted
  without complaint, and nothing in the output reports the effort used, so there
  is no positive confirmation that `max` changes anything.
- Whether Ox Alpha is still free after its announced one-week window.

**UPDATE: REVIEW FROM A FOURTH TASK (tonemapping investigation, 2026-08-22)**

Two runs, both at `--variant max`, both driven from a file per §4.1.

The **read-only audit** is the best result this agent has produced here. One
invocation, ~56 steps, no continuation needed, and a 307-line report
(`scratch_ox_tonemap_audit.md`) whose every checkable claim held up when I
re-ran it independently. It was more precise than my own reading in one place I
had got sloppy: I described the neutral curve's compression as "starting at
0.76", and it pointed out that 0.76 is compared against the peak *after* the
pedestal subtraction, so in input terms the onset is at 0.80. It also flagged a
divergence I had not considered (the torch path computes in the buffer's dtype,
so f16 under `ALGAN_HDR_BUFFER_F16=1`, while the Taichi kernel is always f32),
and it labelled its own limits without being asked: "§3's comparison is source
reading, not execution; ULP-level claims are reasoned, not measured."

**The sharp lesson is what it did not find.** I asked it (question 3) whether
the three tonemap implementations agree *with each other*. It answered that
exactly and correctly: algebraically identical, divergences only at ULP level.
Meanwhile the AgX output matrix was **transposed in both implementations** —
neutral grey renders as saturated magenta — and it did not notice, even though
its §5 transcribed the matrix coefficients into the report. It checked internal
consistency because internal consistency was what I asked for, and two wrong
implementations agree perfectly.

So: **ask it to check against an external invariant, not just for
self-consistency.** "Do these agree?" and "is either right?" are different
questions and it will only answer the one posed. The invariant here was a
one-liner — a colour-space conversion between two spaces sharing a white point
must map white to white, so every matrix row must sum to 1 — and had the brief
said "verify each matrix preserves white" it would almost certainly have caught
it. This is the same shape as §4's "it treats a spec as a claim to check":
that instinct is real, but it fires on the claims you put in front of it, not
on the ones you leave out.

Mechanics worth recording:

- A **container restart is not a fresh container.** After one mid-session,
  `/opt/node22/bin/opencode` was still installed and working and
  `/root/.local/share/opencode/log/` still had its history. Only the running
  process died. Do not reinstall reflexively — check `opencode --version`
  first.
- The background `opencode run` process does die with the container, and it
  dies silently having written nothing. `git status --porcelain` before and
  after is the cheap way to find out whether it got anywhere.
- `grep -c "message=loop" /root/.local/share/opencode/log/opencode.log` counts
  steps **cumulatively across invocations**, not per run. Note the count when
  you launch, or you will misread a fresh run as a long one.

**UPDATE: REVIEW FROM A FIFTH TASK (split-sum glossy reflections, 2026-08-22)**

Two runs, both `--variant max`, both driven from a file. Both landed
mergeable work; the shape of the session is the thing worth recording.

**Budget for an hour, not for twenty minutes.** The first run took **63
minutes** and did not touch a file for the first **45** of them — 60+ steps of
reading and cross-checking before the first edit. §4.1's hang test (log stops
at `init`, no `loop ... step=N`) says nothing about this case: the steps *were*
climbing the whole time, one every 10–120 seconds. So the liveness check is
"is the step counter still moving", not "has it edited anything yet", and the
patience is worth it — see the next paragraph for what it bought.

**It checks the premise you hand it, when the premise is checkable.** The brief
told it to replace a forward-nearest ownership rule with the inverse one and to
verify the swap reproduced today's behaviour for the four counts render
baselines pin. It wrote a throwaway script, measured that the two rules
**disagree on 208 of the 256 possible coverage masks at n=8**, and kept the
forward rule below the ceiling and the inverse one above it rather than doing
what it was told. That is the §4 instinct firing on exactly the claim the brief
put in front of it. It also found, unprompted, that the old table left one of
the eight positions unowned, so 8 taps had always fired 7 rays — and asked
before touching it, because the brief forbade changing behaviour at 8.

**And it still needs reviewing.** A mid-run read of its diff caught
`return _AA_SEC_MAX` in a module that never imports the name — a NameError on
the clamp path. It fixed that itself before finishing (its own test covered
it), so the lesson is narrower than last session's: read the diff *when it
finishes*, not while it works, or you will re-report things it is about to
catch.

**Two sessions can share one tree if they stay in disjoint files.** This whole
task ran with a parallel Claude session implementing a renderer feature in the
same checkout, which committed mid-run. Nothing collided: its edit tool is
targeted string replacement, and it re-reads before each edit. It flagged the
overlap in its own report without being asked, and correctly declined to
`git stash` for failure triage because that would have discarded the other
session's uncommitted work. Do keep the file sets disjoint anyway — and note
that `git status` is not a progress bar for it, since it may read for 45
minutes before writing anything.

**The verification split held again.** It measured what the brief asked and
reported it verbatim; the two defects the session actually found — a pyramid
that dropped odd rows, and a bilinear fetch that derived its second tap from
an already-clamped index — came from *looking at the rendered frames*, which
neither of us would have found from the source. Specify the experiment; look at
the picture yourself.
