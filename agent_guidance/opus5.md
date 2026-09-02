# Opus 5

This file provides guidance specific to Opus 5, other agents may ignore.

## Reporting findings
 
When you discover issues that may require the user's attention — unexpected blockers, unintended consequences, unforseen design problems, optimization or coverage tradeoffs — report them to the user with **a clear label, and afterwards point at the label.**
 
Keep a running, session-scoped registry of findings. Each finding gets an integer label the first time you raise it: `(1)`, `(2)`, `(3)`, …
 
Rules that make the labels trustworthy:
 
- **Assign on first mention.** Never raise a problem without a number — an unlabelled problem can't be referenced later, so it will get re-explained.
- **Numbers are permanent.** Never renumber, never reuse a retired number, never restart at 1 for a new file or a new turn. `(3)` means the same thing on message 40 as it did on message 6.
- **Numbers span the session, not the turn.** If you found `(1)`–`(5)` in the previous turn, the next new finding is `(6)`.
- **Count sequentially even if issues are resolved out of order.** Resolving `(2)` doesn't free up the number.
If you have a todo list or scratchpad tool, mirror the registry there (one short line per finding) so it survives context compaction. Don't narrate that you're doing this.
 
### First mention
 
Explain it properly — this is the only time you will. Lead with the label so it's greppable.
 
```
(4) `parseConfig` swallows the error from `readFile` and returns an empty
object, so a missing config silently becomes "all defaults" instead of a
startup failure. src/config.ts:88.
```
 
Include severity, location, and what you'd do about it if that's useful. Be as thorough here as the finding deserves.
 
### Every mention after that
 
Refer to the label. Do not re-describe the issue, do not re-quote the code, do not restate the location.
 
The canonical status line:
 
```
3 outstanding: (1), (4), (7).
```
 
Keep it to one line. Resist the urge to append "…(1), the config error handling, and (4), the retry loop" — the gloss is exactly the repetition this skill exists to prevent. The user can scroll up, and if they can't remember, they'll ask.
 
Use a status line when you finish a file or a phase, when you're about to change direction, or when the user needs to decide something. Don't emit one after every tool call.
 
### Updating a finding
 
State only what changed, on one line, under the original label.
 
| Situation | Write |
|---|---|
| Narrowed or clarified | `(3) narrowed: only the async path is affected, not all callers.` |
| Severity changed | `(3) upgraded to blocking — this runs on every request, not just startup.` |
| Fixed | `(3) fixed.` |
| Turns out to be a non-issue | `(3) withdrawn — the caller already guards this.` |
| Same root cause found elsewhere | `(3) also at api/handlers.go:210.` |
 
Then drop resolved and withdrawn items out of the outstanding count. `(3) fixed. 2 outstanding: (1), (7).`

## Waiting for background work

Never poll. A command started with `run_in_background: true`, and any
subagent, notifies you when it completes: end your turn and wait for that
notification. Do not use `sleep` to wait, in any form -- not in the
foreground, not `sleep N; tail log` in the background, not
`until ...; do sleep ...; done` loops. Each background sleep is a task the
harness has to wake you for, and a chain of them is a polling loop that
looks hung from outside. If you must wait on something the harness cannot
track (a file appearing, a port opening), use the Monitor tool once with an
until-condition, and set a bound on it.

## Acting as a subagent

Inside a subagent, do not start the docs build, the full test suite or the
render suites unless the brief asks for them: several agents sharing the
machine make those take ten times longer, and the lead runs them once after
integration. Run the tests for the files you changed plus `pytest --fast`,
then report what you did not run.