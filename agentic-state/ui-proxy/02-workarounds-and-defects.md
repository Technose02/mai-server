# 02 — Workaround & defect inventory (with evidence)

Code state 2026-10-16. `llamacppclient/mod.rs` =
`gw-server/src/infrastructure/adapter/localllamacppclient/`.

## Workarounds (deliberate hacks)

**W1 — Whitelist of UI asset routes in axum router** (`openairouter.rs:38-58`).
Every new/renamed upstream asset (hash-named bundles moved from `/bundle.js` to `/_app/immutable/...`
already happened upstream!) requires a code change + redeploy. Two lists exist (open vs secured) with
no explanation of why `props` is secured but e.g. `build.json` isn't.

**W2 — Second path-rewrite match table in the adapter** (`llamacppclient/mod.rs:109-150`).
Duplicates/contradicts W1 (e.g. `/chat/cors-proxy` registered only for HEAD in router; table maps it;
upstream has GET+POST). Unknown paths are forwarded verbatim with only a `warn!` → 404s from backend.

**W3 — String replacement on minified JS/CSS/HTML/JSON bodies**
(`responsepayload.rs::universal_replacements`, ~12 literal patterns).
Matches against *minified build output* of the SvelteKit app (`={LIST:"/v1/models",` etc.).
Any upstream UI rebuild (different bundler mangling, added whitespace, renamed constant like
`CORS_PROXY_ENDPOINT`) silently breaks a feature. This is the single biggest maintenance driver.

**W4 — Buffer + gunzip + gzip-9 every UI response** (`llamacppclient/mod.rs:170-230`).
Kills streaming/HTTP range semantics, adds latency and CPU (`Compression::best()` per request),
breaks ETag/304 interplay (upstream sends ETags; gateway keeps them but changes body/length —
actually keeps upstream ETag while mutating content → cache poisoning risk on revalidation),
and `ResponsePayload::try_from_body` returns 500 on any unexpected `content-encoding` (e.g. `br`,
`deflate`, or zstd once upstream adds it).

**W5 — `/chat` document forces a model load** (`openairouter.rs chat_handler`): opening the page
starts a default LLM with 3-minute blocking wait; if that fails → bare 500. Also races: multiple
parallel page loads each call `ensure_any_languagemodel_is_served`.

**W6 — SSE re-framing + `:null → :""` hack in chat completions** (`llamacppclient/mod.rs:280-375`).
Self-flagged `FIXME: UNSAUBER!!!`. Collateral damage:
- Non-streaming requests are impossible through this path? No — the UI always streams, but any API
  client hitting the *UI-tuned* endpoint gets SSE framing regardless. (API clients normally use
  `/api/v1/chat/completions` → same handler → **also always SSE-framed!**) ← verify: `post_completions`
  → `post_chat_completions_impl` → `process_chat_completions_request` → `post_chat_completions`
  (SSE stream builder, fixed `StatusCode::OK`). Non-stream JSON responses from llama-server get
  line-framed as SSE data and `[DONE]` appended. **Suspected regression for non-streaming API users.**
- Drops SSE `event:`/`id:`/`retry:` fields and multi-line data semantics (each line treated standalone).
- `:null` replacement corrupts legitimate JSON containing `":null"` inside string content chunks.

**W7 — Gateway api-key == backend api-key**. Simplification so the UI's manually-entered key works
against both; also means llama-server's public-endpoint allowlist is load-bearing for security
(assets are open on the *public port too* — mitigated only by binding llama-server to localhost…
verify `--host` args in inference-backends).

**W8 — `api_fallback` double-prefixes `v1`** (`llamacppclient/mod.rs forward_request(request, Some("v1"))`
while incoming path already is `/api/v1/<path>` → backend gets `/v1/v1/<path>`).
Any request that hits the fallback (e.g. UI's `/api/1/v1/props`? no — but e.g. future endpoints,
or `/api/v1/tokenize`) → backend 404 `/v1/v1/tokenize`. **Suspected bug.**

**W9 — Dead `--chatui` flag** (`main.rs:348,403`): parsed into `_provided_llama_cpp_chatui`, no effect.
Historical remnant of an earlier mode where the gateway presumably proxied differently.

**W10 — `/stop` and `/model ` prompts smuggled through user text** (openairouter.rs bottom).
Needed because the UI has no notion of "the gateway may restart the backend". A model switch via
`/model` kills in-flight generations of other users; also a user typing `/model foo` in prose breaks.

## Hard defects observed (screenshot 2026-10-16, gw :8080 `--no-https`, key configured in UI)

| Symptom | Root cause |
|---|---|
| `GET /tools` → 404 | Upstream UI constant `API_TOOLS.LIST = '/tools'` is an **absolute** path (not `./`), so it bypasses the `/chat/…` rewrite (`={LIST:"/tools",EXECUTE:"/tools"}` replacement pattern no longer matches current minified form — verify actual bundle text). Gateway has no open `GET /tools` route → falls to axum 404 (never even reaches backend). |
| `POST /tools` → 404 | Same; router only registers GET `/chat/tools`. Upstream posts tool *executions* to `/tools`. |
| `POST /lookup` (`./v1/streams/lookup`) → 404 | **New upstream feature** (PR #23226 SSE replay buffer, in checkout since ~b11xxx): bundle.js calls `fetch(API_STREAM.LOOKUP)` — note: literally without base prefix (`chat.service.ts:784` uses bare relative fetch!). Gateway never knew about `/v1/streams/*`. |
| `GET /mcp`, `OPTIONS http://127.0.0.1:8000/mcp` → connection refused | UI's MCP discovery probes localhost:8000 directly; unrelated to gateway (would need `--ui-mcp-proxy` + a local MCP server). Cosmetic noise unless MCP wanted. |
| `/props`, `/props?autoload=false`, `/props?model=…`, `/build.json`, `/_app/version.json`, bundle, css, icons → 200 | Current rewrite table happens to cover these. |

Net: the UI *loads* today, but tool-listing/exec and resumable-stream lookup are dead; anything
upstream adds next week breaks again. That is the structural problem: **the gateway reimplements a
subset of "what the UI might request" as compiled-in knowledge.**

## Latent defects (found by code reading, not yet observed)

- W6 non-streaming completions mangled (see above) — verify with `curl -d '{"stream":false,...}'`.
- W8 `/api/v1/<anything-not-completions>` → backend `/v1/v1/...` 404.
- W4 ETag kept while body mutated; browser 304 + mutated-content interplay undefined.
- `ResponsePayload::try_from_body` 500s on unknown content-encodings (upstream gzip-only today).
- `chat_handler_assets` maps *anything* under `/chat/*` to backend — including future gateway-owned
  routes under that prefix (name collision risk with no guard).
- HEAD `/chat/cors-proxy` only; upstream UI issues GET/POST on `/cors-proxy` when MCP proxy enabled.
