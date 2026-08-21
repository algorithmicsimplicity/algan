# OpenCode in a cloud session — handoff notes

Written 2026-08-21 from Claude Code on the web, over three sessions. Nothing here
is about algan itself; it is a record so the next session does not have to
rediscover it. The container is ephemeral, which is why this file is in git.

**Status: working.** OpenCode installs, its agent loop runs, and **Ox Alpha Free
runs as a tool-using subagent** in a cloud session — via the CLI and via the
headless HTTP API. No credential of any kind is needed. The one thing that had to
change was the environment's egress allowlist (§2); the two earlier failures were
that plus a wrong model ID (§3).

| Session | Result |
| --- | --- |
| 2026-08-21 (first) | Install + agent loop verified. All three Zen hosts 403. Egress setting changed mid-session; did not take effect. |
| 2026-08-21 (second) | Fresh container, setting changed *between* sessions. **Still 403** — the edit had been made in the wrong control. Agent loop re-verified. |
| 2026-08-21 (third) | Allowlist live. **Ox Alpha Free ran**, CLI and HTTP API, with tools, against this repo. See §3. |

---

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

## 2. The egress allowlist

This is the step that decides whether the session can do the job at all, and it is
the step that cost two sessions. Check it first:

```bash
for h in api.anthropic.com github.com example.com opencode.ai models.opencode.ai models.dev; do
  printf "%-24s " "$h"
  curl -sS -o /dev/null -w "%{http_code}\n" --max-time 15 "https://$h/"
done
```

Read the whole row, not just the Zen hosts — the controls are what tell you *which*
policy is in force:

| `example.com` | `opencode.ai` | Meaning |
| --- | --- | --- |
| 403 | 000 | stock **Trusted** list. No custom allowlist saved. |
| 403 | 200 | **Custom** list, and it includes Zen. This is the working state. |
| 200 | 200 | **Full** access. Also fine. |

GitHub and MCP connector traffic bypass the allowlist entirely, so `github.com`
answering proves nothing about your edit. `example.com` is the honest control.

A `000` is the proxy refusing CONNECT, not DNS — `getent hosts opencode.ai` resolves
fine either way. Confirm with `curl -sS "$HTTPS_PROXY/__agentproxy/status"`, whose
`recentRelayFailures` reads `gateway answered 403 to CONNECT (policy denial or
upstream failure)`. If it is denied, **stop and tell the user**; this is an
organization policy decision and must not be routed around.

### Hosts Zen needs

| Host | Purpose | Needed for |
| --- | --- | --- |
| `opencode.ai` | the Zen inference gateway | **required** — inference |
| `models.dev` | model catalog source | the real model list (§3) |
| `models.opencode.ai` | catalog mirror | belt and braces |

`models.dev` is not optional in practice: without it OpenCode ships a much shorter
built-in list and you cannot discover the Ox Alpha model ID, which is not the one
you would guess.

### Where the setting actually is

Getting this wrong looks exactly like the feature being broken — it is what made
the second session fail. **It is not under Settings.** Per
[the docs](https://code.claude.com/docs/en/cloud-environments#configure-your-environment):

> On claude.ai/code, select the cloud icon showing the current environment's name,
> in the row above the message box. **There's no settings page or direct URL for
> the selector.**

So:

1. At claude.ai/code, click the **cloud icon** in the row *above the message box*
   (it shows the current environment's name, e.g. "Default").
2. Hover the environment in the list → click the **gear icon** on the right.
   (**Default** is editable in place; you do not have to create a new environment.)
3. Set **Network access**: `None`, `Trusted` (the default allowlist), `Full` (any
   domain), or `Custom` (your own list).
   - For a one-off experiment, **`Full` is the least fiddly** and needs no host list.
   - For `Custom`, put one bare hostname per line in **Allowed domains** and tick
     **"Also include default list of common package managers"** — without it you
     lose npm, PyPI and everything else the build needs. `*.` prefixes match
     subdomains, so `*.opencode.ai` covers `models.opencode.ai`.
4. Save, then **start a new session**. The policy is bound at container
   provisioning: in the first session the setting was changed mid-session and the
   local proxy even restarted (its port moved `32869` → `39187`), but all three
   hosts still 403'd.

**Settings → Capabilities → domain allow list is a different control** and does not
feed this. That was the second session's actual mistake. The docs are explicit:
"Each environment has its own allowed-domains list; there's no organization-level
allowlist that admins can push to every member's environments." The **setup
script** plays no part in egress either.

**Identify the environment in force** via the claude-code-remote MCP tools:
`get_session` (with `session_id` omitted) reports `environment_id`, and
`list_environments` names it. If the account has one environment still described as
"trusted network access", nothing was saved.

## 3. Running Ox Alpha Free — the model ID

Zen's free models need **no API key and no `opencode auth login`**.

```bash
opencode run --model opencode/x-preview-f-free "Reply with exactly one word: PONG"
```

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
IDs in §3.1 need no key either.

### 3.1 Why `opencode models` shows only seven

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

## 4. Other providers — what is and is not usable

Measured across the three sessions; the last column is from the third, with the
custom allowlist live. Re-check rather than trusting it blindly — a different
allowlist changes these answers.

| Provider | Result |
| --- | --- |
| **OpenCode Zen** | **Works, free, no credential.** Seven models, Ox Alpha among them. |
| **Amazon Bedrock** | Auto-detected from `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in the environment, and advertises 118 models, so it *looks* configured — but inference fails with `The security token included in the request is invalid.` Those are not model credentials. Ignore it. |
| **GitHub Copilot** | Auto-detected from `GITHUB_TOKEN`, advertises 36 models; `api.githubcopilot.com` was blocked and the token is the wrong kind anyway. |
| **Anthropic** | `api.anthropic.com` is allowlisted at every access level (it returns a real `authentication_error` for a dummy key, not a proxy denial). Set `ANTHROPIC_API_KEY` as an environment variable on the environment and OpenCode picks it up with no other config. **It is not set by default**, so this needs the user to add it. |
| **OpenAI / OpenRouter** | Blocked under both the trusted and the custom list used here; add the host if you want them. OpenRouter's free tier needs an account key regardless. |

Claude Code's own auth in these sessions is an OAuth token passed via file
descriptor, not an API key, so there is nothing to hand to OpenCode from the
environment.

## 5. Driving the agent

Both paths were verified end-to-end against Ox Alpha in the third session. The
agent advertises ten tools: `bash`, `edit`, `glob`, `grep`, `read`, `skill`,
`task`, `todowrite`, `webfetch`, `write`.

**CLI, one shot.** `--auto` auto-approves permissions, which is what makes it work
non-interactively:

```bash
opencode run --auto --model opencode/x-preview-f-free "your prompt"
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
       "parts":[{"type":"text","text":"your prompt"}]}'
```

The response is JSON with a `parts` array; assistant text is the `type: "text"`
entries. Give the message call a generous `--max-time` — it blocks for the whole
agent turn, tool calls included. `GET /config/providers` lists what OpenCode thinks
is configured. The server warns `OPENCODE_SERVER_PASSWORD is not set; server is
unsecured` — fine on loopback, set it if the port is ever exposed. Pass
`--noproxy '*'` on loopback curls; `NO_PROXY` already covers `127.0.0.1`, but being
explicit avoids surprises.

## 6. Testing the agent loop without any provider

Still useful for isolating an OpenCode problem from a network one. **Both files are
committed in `opencode_probe/`**, so this is a copy-and-run:

```bash
cp opencode_probe/opencode.json .
python3 opencode_probe/mockserver.py &      # listens on 127.0.0.1:8791
opencode run --auto --model mocklocal/mock-1 "run the probe"
kill %1
```

Expect `OPENCODE_TOOL_EXECUTED` (the tool really ran) followed by
`LOOP_VERIFIED: ...` (its result went back to the model). The server appends every
request it saw to `requests.jsonl` next to itself. Verified from scratch in two
separate fresh containers, so this is a reliable ~2-minute readiness check. Two
details that make it work first try:

- **Read the tool name off the request.** The mock should call whichever tool the
  client actually declared (prefer `bash`) rather than a hardcoded guess. Log each
  request's `tools` and `messages[].role` to a file — that log *is* the evidence.
- **OpenCode probes before it dispatches.** The first two requests arrive with an
  empty `tools` list; only the third carries the real ten and the fourth carries
  the tool result. A mock that keys off "has a `role: "tool"` message" handles this
  correctly; one that keys off request count does not.

Observed request sequence when the loop is healthy:

```
tools: []                          roles: []
tools: []                          roles: [system, user, user]
tools: [bash, edit, glob, grep, read, skill, task, todowrite, webfetch, write]
                                   roles: [system, user]
tools: [ ...same ten... ]          roles: [system, user, assistant, tool]
```

## 7. Settled, and still open

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

Still open:

- The TUI (`opencode` with no subcommand) — only `run` and `serve` were exercised.
- Ox Alpha's quality on a substantial task. What is verified is that it drives the
  loop, picks sensible tools, and answers a small codebase question correctly; it
  has not been given real work.
- Whether Ox Alpha is still free after its announced one-week window.
