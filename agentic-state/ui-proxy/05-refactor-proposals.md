# 05 — Refactor options & recommendation

Constraints (from owner, 2026-10-16):
1. **Production box**: gw-server :8443 serves live traffic (inc. the model answering this session).
   Migrations must be additive + testable on separate ports; never disrupt the running instance.
2. **SSE heartbeats are load-bearing**: local models have TTFT up to ~30 min; owner is behind a
   www-proxy that kills idle connections. Whatever replaces the current code MUST keep periodic
   bytes flowing on idle streaming responses. (This is a transport concern, not an SSE-content
   concern — see R1b.)
3. **No full-featured reverse-proxy** as the solution. A purpose-built, thin forwarding layer inside
   the gateway is acceptable and desired; pulling in e.g. `rustproxy`/nginx-style config machinery is not.
4. **Primary product = OpenAI-compatible API.** The chat UI is a nice-to-have demo/bonus only.
   `/api/v1/*` + `/api/{n}/v1/*` semantics must not change for API consumers.
5. **Single-apikey security model stays**: owner is the only user; rotates key by restarting the
   whole stack every morning with `-k <newkey>` (gateway + backends share the fresh key by design,
   no hot rotation exists). No secured route may become less protected than today.

## Root cause restated (from 02/03/04)

The gateway encodes *compiled-in knowledge of the upstream UI's URL surface* twice (route whitelist +
path table) and a third time as *string surgery on minified build output*. Upstream is free to change
all three; it demonstrably did (key-order flip, constant folding, new endpoints). The fix direction:
**stop knowing anything about the UI's internals.**

## R1 — Prefix-mounted passthrough + transport-only keepalive  ★ recommended

Idea: make the *upstream itself* subpath-safe (it already is!), then forward without touching content.

### R1a′ — catch-all with PATH interception (refined 2026-10-16 after API-preservation review)

Settles the open design point from R1's checklist and guarantees API-path preservation:

- Backend launched `--api-prefix /chat`. Gateway URL builder for ALL backend calls (API handlers
  included) gains the prefix in ONE place (`LLAMACPP_API_BASE_PATH` equivalent).
- Catch-all `/chat/{*rest}` on the gateway does **path-based interception first, passthrough second**:
  - `POST /chat/v1/chat/completions` (+ `/control`) → routed into the EXISTING
    `post_chat_completions_impl` pipeline (model ensure/switch, `/model`, `/stop`, SD abort,
    parallel param) — UI keeps every gateway feature without any body rewriting; this replaces the
    `./v1/chat/completions` → `./api/1/v1/chat/completions` string hack with pure routing.
  - `POST /chat/v1/embeddings` → same idea (embeddings client/port), if UI ever needs it.
  - everything else (`/chat/_app/*`, props, health, assets, future unknown endpoints) → verbatim
    stream-forward to backend. Unknown = forward, so new upstream UI features work on day one.
- Result: **the OpenAI API surface is literally untouched code** (`/api/v1/*` handlers unchanged);
  the UI no longer rewrites anything; interception is by stable server-API paths, not minified text.

### R1a — (original sketch, superseded by R1a′ above)

- Launch llama-server with `--api-prefix /chat` (PR #14544, env `LLAMA_ARG_API_PREFIX`).
  All API + UI routes are served under `/chat` by the backend itself.
- Gateway: ONE catch-all route `/chat/{*rest}` (any method) → stream-forward to backend
  `http://localhost:11440/chat/{rest}` verbatim (headers, body, query). No path table, no whitelist,
  no content rewriting, no buffering — ETag/304/gzip/immutable pass through untouched.
- Relative `./…` endpoints from the SPA resolve under `/chat/…` and hit the same catch-all → done.
  (Serve/redirect document at `/chat/` with trailing slash so relative resolution stays inside the
  prefix; or require users to bookmark `/chat/`.)
- **Remaining leaks**: a handful of UI constants are root-absolute and do NOT get SvelteKit's base
  prefix (verified in served bundle): `/v1/models`, `/models[/load|/unload|/sse]`, `/tools`,
  `/cors-proxy`. Fix with ~6 explicit alias routes at gateway root → forward to backend's prefixed
  equivalents. These are *server API* paths (stable, documented), not minified-UI internals —
  acceptable coupling. Upstream issue #22474 (public-endpoint allowlist vs `--api-prefix`) must be
  re-checked on the pinned version; workaround: gateway injects the backend key on forwarded
  requests anyway (R1c).

### R1b — generic idle-keepalive stream wrapper (preserves the heartbeat feature, done right)

Replace the completions-specific SSE rebuild with a transport-level `Stream` adapter used on ALL
forwarded responses (or only `text/event-stream`, cheaper): if no bytes flow for N seconds, emit
`: keepalive\n\n`. Properties:
- Bytes from upstream are forwarded **unmodified** (no `:null → :""`, no line re-framing, no forced
  `[DONE]`, non-`data:` events preserved) → upstream resumable-streaming (`X-Conversation-Id`,
  `/v1/stream*`) and any future SSE feature work by construction.
- Works for model-loading waits on ANY endpoint (props during load, router-mode `/models/sse`).
- Content stays parseable by strict OpenAI clients; non-streaming `stream:false` requests pass
  through as real JSON (fixes latent W6 regression).
- Note: llama-server itself has idle SSE keepalives (`--idle-sleep-*`/keepalive behavior in
  server-stream); re-verify whether gateway-level keepalive is still needed end-to-end through the
  www-proxy — likely yes (proxy may measure per-hop silence differently / model-load phase on
  non-SSE requests), but this also means our interval can be relaxed.

### R1c — auth simplification (kills the "paste key into UI" UX + closes a hole)

- Today: assets open, `/chat/props`+`/chat/tools` secured, forwarded UI requests carry NO backend key;
  everything only works because gateway-key == backend-key and users paste it into UI settings.
  (Rotation model: full-stack restart each morning with new `-k`, so gateway/backend keys never
  diverge — see constraint 5. Any cookie/session state may be in-memory; it dies with the process,
  which matches the rotation semantics exactly.)
- Proposal: browser-side auth for the UI via a **gateway-issued cookie** (e.g. open
  `/chat/bootstrap?key=<apikey>` once → sets HttpOnly cookie, redirects to `/chat/`); middleware
  accepts `Bearer` OR that cookie on the UI catch-all. Gateway then **injects** the backend key into
  every forwarded `/chat/*` request server-side. Consequences:
  - whole `/chat` surface protected even when assets are "public" at the backend
  - users never enter a key in the UI; `getAuthHeaders()` empty → backend sees injected gateway key ✓
  - backend can keep (or even rotate) its own key independently — decouples W7

### R1d — what gets deleted

- `responsepayload.rs` entirely (~160 lines of string surgery)
- path match table in `forward_uichat_request_and_process` (~50 lines)
- ~15 route registrations in `openairouter.rs` → 1 catch-all + ~6 aliases
- gzip decompress/recompress-per-request (CPU win on the 9 MB bundle: today every uncached load is
  gunzip + scan + gzip-9 in gateway)
- dead `--chatui` flag
- bonus fix: stale service workers — when UI ever gets disabled, serve upstream's self-unregistering
  `sw.js` semantics (or do it ourselves once)

### Risks / verification checklist for R1

- [ ] `--api-prefix` + `--api-key`: are asset/health endpoints still public on the pinned build?
      (issue #22474 history). If not: gateway injects key on ALL forwarded requests incl. document —
      fine, since injection happens server-side; only the *initial browser navigation* to `/chat/`
      must be authorized via cookie (R1c) or remain public-with-bootstrap-link.
- [ ] `Cross-Origin-Embedder-Policy: require-corp` + COOP on index.html through our forwarding —
      keep headers as-is; verify no CORP breakage for image attachments etc.
- [ ] Service-worker cache: old SW cached the *rewritten* bundle under old URLs → hard-refresh /
      bump via `sw.js` no-cache semantics; document a one-time cleanup for existing users (owner).
- [ ] Resumable streams through catch-all: verify `X-Conversation-Id` round-trip + `/v1/stream`
      SSE passthrough with keepalive wrapper attached.
- [ ] The 6 alias routes: confirm exact set by grepping served bundle for absolute-path string
      literals (`grep -oE '"(/(v1|models|tools|cors-proxy|slots)[^"]*)"' bundle.js | sort -u`) —
      script this as a **compat test** (see R2-test) instead of trusting memory.
- [ ] `/model` & `/stop` chat-command sniffing: keep as-is for now (orthogonal), but move behind a
      small request-mutation step shared by both API and UI-forwarded completions (today it only
      runs on the gateway's own API path — after R1a, UI completions forwarded through the catch-all
      would BYPASS `/model`+`/stop` handling! Decide: keep UI on gateway API paths (`./api/1/v1/…`)
      or run the sniffing in the forwarding layer too. ← important design point, currently open)

## R2 — compat test harness (do regardless of chosen option)

A small test/example that fetches the served bundle from a real backend and asserts every absolute
URL literal it contains is routable through the gateway (and that known-expected rewrites hit).
This converts "silently broken after upstream bump" into "CI/red test after upstream bump".
Cheap (~100 lines), works with R1 (guards the 6 aliases) and would have caught all three observed
breakages.

## Alternatives considered

### R3 — keep rewriting, make it robust (not recommended)

Inject `<base href>` + intercept absolute fetches via a JS shim instead of exact-literal replaces.
Better than status quo but still: content mutation of minified code, buffering, ETag invalidation,
and unknown-unknown endpoints (new features never break *loudly*, they just 404). Rejected as
primary; R1 makes it unnecessary because upstream supports prefixing natively.

### R4 — own frontend against gateway API (deferred option)

Build/ship a custom chat UI (or fork tools/ui pointed at `/api/v1`) served statically by the gateway.
Full control, no coupling — but abandons upstream features (model manager, MCP, PWA, resumable
streams) and creates real maintenance of its own. Only worth it if mai-server wants gateway-specific
UX (model switching UI instead of `/model` text commands, queue status, etc.). Could grow from R1
later: R1's transparent forwarding stays useful for admin/debug access to the vanilla upstream UI.

### R5 — router-mode alignment (future)

llama.cpp router mode (multi-model, load/unload + `/models/sse`) maps naturally onto mai-server's
model management; running one router backend instead of restart-per-model would make the UI's native
model manager work and could replace `/model` command sniffing. Big change; park as
[ui-proxy/06-future-router-mode.md].

## Security parity analysis (per route class, R1a′ design)

| Route class | Today | After R1a′ |
|---|---|---|
| `/api/v1/*`, `/api/{n}/v1/*` (the product) | `check_auth` @ gateway | **untouched code**, `check_auth` @ gateway |
| `/admin/llamacpp/*` | `check_auth` | untouched |
| UI document + assets (`/chat/…`) | open @ gateway, open @ backend allowlist | open @ gateway catch-all → forwarded; backend still enforces its key on non-public paths. Optionally stricter via R1c cookie. |
| props/tools/models/etc. from UI | `/chat/props`,`/chat/tools` behind `check_auth`; browser must carry bearer (key pasted into UI) | same catch-all forwards client `Authorization` verbatim → backend key check enforces; OR (with R1c) gateway cookie checked first + key injected — strictly stronger |
| new/unknown upstream endpoints | 404 (whitelist) | forwarded → still subject to backend api-key unless on backend public allowlist (assets/health only — verified in server-http.cpp) |

Net: no route becomes less protected; the single-key model is preserved everywhere. The UI simply
stops being a special auth world.

**Caveats / fallbacks:**
- issue #22474 class bug (`--api-prefix` + `--api-key` public-allowlist mismatch): if on the pinned
  build assets get 401 at the backend, options: (a) upgrade llama.cpp (check current upstream state),
  (b) gateway injects backend key on forwarded asset requests — requires R1c first (otherwise the
  catch-all would be an unauthenticated key-injection oracle → NOT allowed), (c) drop `--api-key` on
  backend and rely on `--host localhost` bind + gateway auth (weakest; last resort, needs review of
  anything else that can reach :11440 locally).
- Root alias routes (`/v1/models`, `/models*`, `/tools`, `/cors-proxy` → prefixed backend) MUST be
  behind `check_auth` (or R1c cookie) — never open passthrough.
- R1b keepalive must attach only to responses the gateway itself intercepts or to `text/event-stream`
  bodies — never buffer non-SSE responses.

## Recommended sequence

1. R2 compat-test first (measures today's breakage, guards everything after).
2. R1a catch-all + `--api-prefix /chat` behind a feature flag / second gateway port (e.g. 8444) —
   compare side by side with production on 8443. Zero risk to live traffic.
3. R1b keepalive wrapper; validate through the work www-proxy with a long-TTFT model.
4. R1c cookie bootstrap + key injection.
5. Flip default, delete legacy paths (responsepayload.rs, tables, dead flag).
