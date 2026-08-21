# OpenCode in a cloud session — handoff notes

Written 2026-08-21 from a Claude Code on the web session, after establishing what
works and what is blocked. Nothing here is about algan itself; it is a record so
the next session does not have to rediscover it. The container is ephemeral, which
is why this file is in git.

**Status when these notes were written:** OpenCode installs and its agent loop runs
correctly in a cloud session. Ox Alpha Free could *not* be reached, because every
host that serves OpenCode Zen was denied by the environment's egress policy. That
was the only blocker — no credential is needed for Zen's free models.

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

They are configured at claude.ai/code → the environment's **Network access** →
**Custom**, keeping "include default list of common package managers" checked. Add
bare hostnames, not full URLs. List `models.opencode.ai` separately from
`opencode.ai` — an apex entry may not cover the subdomain.

### The policy is bound at container provisioning

Changing the setting **does not affect a session already running**. In the session
that produced these notes the setting was changed mid-session; the local proxy even
restarted (its port moved `32869` → `39187`), but all three hosts still returned 403
while control hosts (`registry.npmjs.org`, `api.anthropic.com`, `github.com`) kept
answering normally. A new session has to be started for a policy change to apply.

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
it at a local OpenAI-compatible endpoint. Add an `opencode.json` in the working
directory:

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

## 7. What was never verified

- **Ox Alpha itself never ran.** Everything in section 3 is untested against the
  live gateway.
- Whether allowlisting `opencode.ai` is *sufficient*, or whether Zen also reaches
  some other host once inference actually starts.
- The TUI (`opencode` with no subcommand) — only `run` and `serve` were exercised.
