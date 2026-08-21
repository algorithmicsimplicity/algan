# OpenCode in a cloud session — handoff notes

Written 2026-08-21 from a Claude Code on the web session, after establishing what
works and what is blocked. Nothing here is about algan itself; it is a record so
the next session does not have to rediscover it. The container is ephemeral, which
is why this file is in git.

**Status.** OpenCode installs and its agent loop runs correctly in a cloud session.
Ox Alpha Free has *never* been reached: every host that serves OpenCode Zen is
denied by the environment's egress policy. That is the only blocker — no credential
is needed for Zen's free models.

| Session | Result |
| --- | --- |
| 2026-08-21 (first) | Install + agent loop verified. All three Zen hosts 403. Egress setting changed mid-session; did not take effect. |
| 2026-08-21 (second) | Fresh container, after the setting was changed *between* sessions. **Still 403.** Agent loop re-verified end-to-end. See §2.1. |

---

## 1. Install

The documented installer (`curl -fsSL https://opencode.ai/install | bash`) **does not
work**: `opencode.ai` is denied by the egress proxy. Install from npm instead —
`registry.npmjs.org` is on the default allowlist.

```bash
npm install -g opencode-ai
opencode --version     # 1.18.20 as of 2026-08-21
```

Takes about 12 seconds, 3 packages. The base image already has node v22.22.2,
npm 10.9.7, and bun, so there is nothing to install first. The binary lands at
`/opt/node22/bin/opencode`, which is already on `PATH`.

## 2. Check the egress allowlist before anything else

This is the step that decides whether the session can do the job at all. One curl:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" --max-time 20 https://opencode.ai/zen
```

- `200`/`3xx`/`4xx` → the allowlist includes it, go to section 3.
- `000` → still blocked. Confirm with
  `curl -sS "$HTTPS_PROXY/__agentproxy/status"`, whose `recentRelayFailures` will
  read `gateway answered 403 to CONNECT (policy denial or upstream failure)` for
  `opencode.ai:443`. **Stop here and tell the user** — this is an organization
  policy denial and must not be retried or routed around.

### Hosts Zen needs

| Host | Purpose |
| --- | --- |
| `opencode.ai` | the Zen inference gateway — **required** |
| `models.opencode.ai` | model catalog mirror |
| `models.dev` | model catalog source |

### Where the setting actually is

This is the step that is easy to get wrong, and getting it wrong looks exactly
like the feature being broken. **It is not under Settings.** Per
[the docs](https://code.claude.com/docs/en/cloud-environments#configure-your-environment):

> On claude.ai/code, select the cloud icon showing the current environment's name,
> in the row above the message box. **There's no settings page or direct URL for
> the selector.**

So:

1. At claude.ai/code, click the **cloud icon** in the row *above the message box*
   (it shows the current environment's name, e.g. "Default").
2. Hover the environment in the list → click the **gear icon** on the right.
   (**Default** is editable in place; you do not have to create a new environment.)
3. Set **Network access**. Four levels exist: `None`, `Trusted` (the default
   allowlist), `Full` (any domain), `Custom` (your own list).
   - For a one-off experiment, **`Full` is the least fiddly** and needs no host list.
   - For `Custom`, put one bare hostname per line in **Allowed domains** and tick
     **"Also include default list of common package managers"** — without it you
     lose npm, PyPI and everything else the build needs. `*.` prefixes match
     subdomains, so `*.opencode.ai` covers `models.opencode.ai`.
4. Save, then **start a new session**.

**Settings → Capabilities → domain allow list is a different control** and does not
feed this. The docs are explicit: "Each environment has its own allowed-domains
list; there's no organization-level allowlist that admins can push to every
member's environments." Neither does the **setup script** play any part in egress.

**Confirming the change landed, before wasting a session on it:** changing the
allowed hosts *invalidates the environment's filesystem snapshot* — "The setup
script runs again to rebuild the cache when you change the environment's setup
script or allowed network hosts." So the next session should visibly re-run
provisioning from scratch. A next session that starts instantly from cache is a
strong hint the network change never saved.

Note also that GitHub traffic and MCP connector traffic bypass this allowlist
entirely, which is why `github.com` keeps working at every access level and is
useless as a test of whether your edit applied. Use `example.com` for that.

### The policy is bound at container provisioning

Changing the setting **does not affect a session already running**. In the first
session the setting was changed mid-session; the local proxy even restarted (its
port moved `32869` → `39187`), but all three hosts still returned 403 while control
hosts (`registry.npmjs.org`, `api.anthropic.com`, `github.com`) kept answering
normally. A new session has to be started for a policy change to apply.

## 2.1 A new session was not enough either — check the environment itself

The second session was a fresh container started *after* the egress setting was
changed, and all three Zen hosts were still denied. So "start a new session" is
necessary but not sufficient: verify the change actually landed on the environment
the session runs in.

**Ask OpenCode, not curl.** The clearest diagnostic is the error the provider
returns, which names the host and the reason outright:

```
Error: Forbidden: request blocked: no rule or allowlist entry allows host "opencode.ai"
```

A bare `curl` only gives `000`, which is indistinguishable from a DNS or timeout
failure. (DNS itself resolves fine — `getent hosts opencode.ai` returns Cloudflare
addresses. The denial is at the proxy's CONNECT, not at name resolution.)

**Identify the environment in force.** Via the claude-code-remote MCP tools,
`get_session` (with `session_id` omitted) reports `environment_id`, and
`list_environments` names it. In the second session both agreed:

```
env_01DkL9rDLSP7m6Vm4DtVWDWX   "Default"   "Default - trusted network access"
```

That was the *only* environment on the account, still on stock **Trusted** access —
so the custom allowlist had not been saved anywhere this session could pick up.
The cause was navigational: the edit had been attempted under **Settings →
Capabilities → domain allow list**, which is a different control that does not feed
an environment's egress policy. The real control is the environment selector — see
"Where the setting actually is" in §2.

**Characterize the policy in one command.** Control hosts distinguish "restrictive
default" from "custom list that omitted the Zen hosts":

```bash
for h in api.anthropic.com github.com example.com opencode.ai; do
  printf "%-22s " "$h"
  curl -sS -o /dev/null -w "%{http_code}\n" --max-time 15 "https://$h/"
done
```

In the second session: `404`, `400`, `000`, `000`. A blocked `example.com` with
Anthropic and GitHub answering is exactly the stock trusted list — i.e. no custom
allowlist was in force.

## 3. Running Ox Alpha Free

Zen's free models need **no API key and no `opencode auth login`**. Once the
allowlist is live:

```bash
opencode models | grep '^opencode/'                       # find the real model ID
opencode run --model opencode/ox-alpha-free "say PONG"    # ID is a guess, see below
```

### Two caveats on the model ID

**It was not in the bundled catalog.** `opencode models` listed only seven Zen
models — `big-pickle`, `hy3-free`, `mimo-v2.5-free`,
`muse-spark-1.2-contributor-free`, `nemotron-3-ultra-free`,
`nemotron-3.5-lightning-free`, `x-preview-f-free` — with no Ox Alpha. That list is
OpenCode's **offline fallback**, served because `models.dev` and
`models.opencode.ai` were both blocked and the catalog could not refresh. Guessing
`opencode/ox-alpha-free` returned an `UnknownError` from the server. So: allowlist
the catalog hosts, then re-run `opencode models` and read the ID off that list
rather than assuming the spelling. Public references spell it "Ox Alpha",
"0x Alpha", and `stealth/ox-alpha` (the OpenRouter ID) interchangeably.

**It is time-limited.** Ox Alpha was announced as a stealth model free *for one
week* (1M context, multimodal, zero data retention). That week was already running
on 2026-08-21. If it has lapsed, the other `*-free` Zen models above are the
fallback and need no key either.

## 4. Other providers — what is and is not usable

Measured in the session that wrote this. Re-check rather than trusting it blindly.

| Provider | Result |
| --- | --- |
| **Amazon Bedrock** | Auto-detected from `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in the environment, so it *looks* configured — but inference fails with `The security token included in the request is invalid.` Those are not model credentials. Ignore it. |
| **GitHub Copilot** | Auto-detected from `GITHUB_TOKEN`, but `api.githubcopilot.com` is blocked. Also the wrong kind of token. |
| **OpenAI** | `api.openai.com` blocked. |
| **OpenRouter** | `openrouter.ai` blocked, and its free tier needs an account key anyway. |
| **Anthropic** | **Reachable.** `api.anthropic.com` returns a real `authentication_error` for a dummy key rather than a proxy denial, so it is allowlisted by default. Set `ANTHROPIC_API_KEY` as an environment variable on the environment and OpenCode picks it up with no other config. This is the fallback if Zen stays blocked. |

Claude Code's own auth in these sessions is an OAuth token passed via file
descriptor, not an API key, so there is nothing to hand to OpenCode from the
environment.

## 5. Driving the agent

Both paths were verified end-to-end. The agent advertises ten tools: `bash`,
`edit`, `glob`, `grep`, `read`, `skill`, `task`, `todowrite`, `webfetch`, `write`.

**CLI, one shot.** `--auto` auto-approves permissions, which is what makes it work
non-interactively:

```bash
opencode run --auto --model <provider>/<model> "your prompt"
```

**Headless HTTP API** — this is the real programmatic-control path:

```bash
opencode serve --port 4096 &
SID=$(curl -sS --noproxy '*' -X POST http://127.0.0.1:4096/session \
        -H 'content-type: application/json' -d '{"title":"driven"}' \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
curl -sS --noproxy '*' -X POST "http://127.0.0.1:4096/session/$SID/message" \
  -H 'content-type: application/json' \
  -d "{\"model\":{\"providerID\":\"opencode\",\"modelID\":\"ox-alpha-free\"},
       \"parts\":[{\"type\":\"text\",\"text\":\"your prompt\"}]}"
```

`GET /config/providers` lists what OpenCode thinks is configured. The server warns
`OPENCODE_SERVER_PASSWORD is not set; server is unsecured` — fine on loopback, set
it if the port is ever exposed. Pass `--noproxy '*'` on loopback curls; `NO_PROXY`
already covers `127.0.0.1`, but being explicit avoids surprises.

## 6. Testing the agent loop without any provider

If you need to prove OpenCode works before the network question is settled, point
it at a local OpenAI-compatible endpoint. **Both files are committed in
`opencode_probe/`**, so this is a copy-and-run, not a rewrite:

```bash
cp opencode_probe/opencode.json .
python3 opencode_probe/mockserver.py &      # listens on 127.0.0.1:8791
opencode run --auto --model mocklocal/mock-1 "run the probe"
kill %1
```

Expect `OPENCODE_TOOL_EXECUTED` (the tool really ran) followed by
`LOOP_VERIFIED: ...` (its result went back to the model). The server appends every
request it saw to `requests.jsonl` next to itself.

The config it needs is an `opencode.json` in the working directory:

```json
{
  "provider": {
    "mocklocal": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Mock Local",
      "options": { "baseURL": "http://127.0.0.1:8791/v1", "apiKey": "not-needed" },
      "models": { "mock-1": { "name": "Mock 1", "tool_call": true } }
    }
  }
}
```

Then serve `POST /v1/chat/completions` as SSE returning an OpenAI-style
`tool_calls` delta on the first turn and a plain text delta once a `role: "tool"`
message appears in the history. Run with
`opencode run --auto --model mocklocal/mock-1 "go"`. This confirmed the full loop —
tool advertised, tool call issued, tool executed by OpenCode, result fed back,
final message returned — with no external credential.

Re-verified from scratch in the second session's fresh container, so this is a
reliable ~2-minute readiness check. Two details that make it work first try:

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

## 7. What was never verified

Still true after two sessions:

- **Ox Alpha itself never ran.** Everything in section 3 is untested against the
  live gateway. The model ID is still a guess.
- Whether allowlisting `opencode.ai` is *sufficient*, or whether Zen also reaches
  some other host once inference actually starts.
- The TUI (`opencode` with no subcommand) — only `run` and `serve` were exercised.

What is now settled, and need not be re-tested:

- Install from npm works in a fresh container (~12s, `opencode --version` → 1.18.20).
- The agent loop works in a fresh container, tools and all (§6).
- The seven-model Zen list *is* the offline fallback. If `opencode models` shows
  exactly `big-pickle`, `hy3-free`, `mimo-v2.5-free`,
  `muse-spark-1.2-contributor-free`, `nemotron-3-ultra-free`,
  `nemotron-3.5-lightning-free`, `x-preview-f-free`, the catalog hosts are still
  blocked — no need to check them separately.
- The block is at the gateway, not merely the catalog: a model that *is* in the
  fallback list (`opencode/hy3-free`) fails the same way, so no amount of getting
  the model ID right helps until `opencode.ai` is allowlisted.
- `ANTHROPIC_API_KEY` is **not** set in the environment by default, so the §4
  Anthropic fallback needs the user to add it before it is an option.
