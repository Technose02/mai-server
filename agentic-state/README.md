# agentic-state — working notes for mai-server

Index of analysis / planning notes. Not documentation — living work state.
Code references are to the state analyzed on **2026-10-16** (llama.cpp checkout `b11672-3-g2851b3bcb`).

## Focus area: gw-server ↔ llama-server WebUI integration

| Note | Content |
|---|---|
| [ui-proxy/01-how-it-works.md](ui-proxy/01-how-it-works.md) | How the UI re-presentation at `/chat` is wired today (request flows, auth, ports) |
| [ui-proxy/02-workarounds-and-defects.md](ui-proxy/02-workarounds-and-defects.md) | Inventory of every workaround + defect found (with file:line) |
| [ui-proxy/03-screenshot-triage.md](ui-proxy/03-screenshot-triage.md) | Triage of the Firefox dev-console screenshot (404s / failures, root causes) |
| [ui-proxy/04-upstream-llamacpp-ui-facts.md](ui-proxy/04-upstream-llamacpp-ui-facts.md) | Verified facts about current llama.cpp server/UI serving (upstream ground truth) |
| [ui-proxy/05-refactor-proposals.md](ui-proxy/05-refactor-proposals.md) | Refactoring options, recommendation, migration sketch |
| [ui-proxy/06-probe-api-prefix-results.md](ui-proxy/06-probe-api-prefix-results.md) | ★ Live probe (router-mode, zero RAM): `--api-prefix`+`--api-key` → nothing public on pinned build (#22474 confirmed) |
| [ui-proxy/07-probe-round2-results.md](ui-proxy/07-probe-round2-results.md) | Probe round 2: #22474 verdict (closed *not planned*), prefixed walk WITH key (all green), W8 fallback bugs confirmed live, operational rules |
| [ui-proxy/09-next-session-prompt.md](ui-proxy/09-next-session-prompt.md) | ★ Copy-paste prompt to start the next session with full context |
| [ui-proxy/08-implementation-log.md](ui-proxy/08-implementation-log.md) | ★ Implementation on branch `chat-ui-passthrough`: transparent forwarder + keepalive + catch-all routing, string-surgery layer deleted. Compiles, unit test green, **end-to-end not yet validated** |

## Project map (quick recall)

- Workspace: `gw-server` (axum gateway, default member), `inference-backends`, `managed-process`, `staticmodelconfig`.
- Gateway listens :8443 TLS / :8080 plain (`main.rs`). Spawns/controls two `llama-server` processes:
  - LLM on **11440**, embeddings on **11441**; both get `--api-key <gateway apikey>` (same key!).
- Gateway API namespaces: `/api/v1/*` + `/api/{n}/v1/*` (OpenAI), `/admin/llamacpp/*`, SD routes.
- `--chatui` CLI flag was **dead code** — removed in the `chat-ui-passthrough` branch (see 08).

## Hard working constraints (owner-stated)

- **This is production**: gw-server on :8443 serves live traffic over the internet from this very
  machine (it served the model answering these queries). Never restart/kill it or bind its ports;
  experiment behind feature flags / separate ports only.
- **SSE heartbeats are load-bearing**: TTFT up to ~30 min on local models; owner's www-proxy at work
  kills idle connections. Any redesign must keep periodic bytes on idle streams (see R1b in 05).
- No full-featured reverse-proxy dependency; thin purpose-built forwarding inside gw-server instead.
- The current design is an 11-month-old pragmatic solution — respect its intent, fix its fragility.

## Status / next actions

- [x] Deep analysis of UI proxying as it works today (2026-10-16)
- [x] Verify screenshot root causes against live served bundle (see 03 — 3× stale-minified-pattern,
      1× new upstream feature)
- [x] Probe `--api-prefix` (issue #22474: unfixed, closed not-planned) → design avoids it
- [x] Confirm W8 fallback bugs live (double-prefix + axum double-`Path` 500)
- [x] Implement transparent-forwarding rewrite on branch `chat-ui-passthrough` (see 08) — builds, unit test green
- [ ] End-to-end validation — **needs CLI/env-configurable backend ports first** (hardcoded
      11440/11441 collide with the live instance)
- [ ] Behavioural checks listed in 08 §"NOT done / next session"
- [ ] Decide PR/merge vs. keep on branch (owner evaluates whether the approach is worth it)

## Session-end state (2026-10-16)

- Branch `chat-ui-passthrough`: working tree = the whole rewrite (see 08).
- Production/dev instances untouched all session; probing was read-only except two isolated
  router-mode fixture servers on spare ports (both stopped and verified).
- **Never POST completions to :11434/:11440/:11441** — that is the live session's KV-cache.
