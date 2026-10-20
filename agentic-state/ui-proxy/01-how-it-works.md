# 01 — How the llama.cpp WebUI re-presentation works today

Analyzed 2026-10-16. All paths relative to `gw-server/src/`.

## Actors & topology

```
browser ──HTTPS── axum gw-server (:8443 / :8080 plain)
                    │
                    ├── /api/v1/*, /api/{n}/v1/*   → OpenAI-compatible API handlers
                    ├── /chat, /chat/*, /sw.js …   → "UI proxy" (this analysis)
                    ├── /admin/llamacpp/*          → process control
                    │
                    ├── llama-server LLM       @ localhost:11440  (--api-key == gateway key)
                    └── llama-server EMBEDDINGS@ localhost:11441  (--api-key == gateway key)
```

- Both backends are started by the gateway (`LlamaCppControllerAdapter` → `inference_backends::LlamaCppBackendController`)
  with **the same API key** the gateway itself uses (`main.rs` passes `security_config` into run-config
  via `LlamaCppRunConfigDto::map_into_domain(apikey)` / `llamacppconfig.rs:302` `--api-key`).
- llama-server serves its embedded WebUI (SvelteKit SPA, hash-router) at `/`, plus props/tools/assets endpoints.

## The core problem being worked around

maiserver wants to expose the UI at `https://gw/chat` while its own API lives at `/api/v1/*`,
and the backend's UI expects to be served at the **root** of the same origin it calls APIs on.
Two independent mutation layers implement this:

### A) Request-path rewriting (inbound)

`openairouter.rs::create_router` registers a hand-maintained whitelist of UI routes, all funneled
into `chat_handler_assets` → `OpenAiRequestForwardPServiceInPort::forward_ui_request`:

- open routes: `/chat`, `/manifest.webmanifest`, `/chat/bundle.css|js`, `/chat/_app/{*path}`,
  `/chat/favicon.{ico,svg}`, `/chat/apple/{*path}`, `/chat/build.json`, `/chat/pwa-*.png`,
  `/chat/maskable-icon-*.png`, `/sw.js`, `HEAD /chat/cors-proxy` (lines 38–53)
- secured routes: `/chat/props`, `/chat/tools` (lines 57–58)

`LocalLlamaCppClientAdapter::forward_uichat_request_and_process` (`localllamacppclient/mod.rs:109`)
then applies a **second** hand-maintained match table mapping e.g.
`/chat/bundle.css → /bundle.css`, `s.strip_prefix("/chat")` for `_app`/`props`/pwa,
`/chat/apple/apple-* → /apple-*`, unknown → forward verbatim with a `warn!`.

Note: `/chat` (the document) goes through `chat_handler` → `get_chat()` → `request_chat()`, which
first ensures *some* language model is loaded (3-min timeout!), then GETs backend `/`.

### B) Response-body rewriting (outbound) — the fragile core

Same function: whole body is **buffered** (`res_body.collect()`), gunzipped if needed, then
`ResponsePayload::process()` (`localllamacppclient/responsepayload.rs`) applies
`universal_replacements()` — plain `String::replace` on HTML/CSS/JS/JSON payloads:

| replacement | intent |
|---|---|
| `./bundle.` → `/chat/bundle.` | asset URLs in index.html |
| `}/props` → `}/chat/props`, `./props` → `/chat/props` | props endpoint refs in bundle.js |
| `CORS_PROXY_ENDPOINT="/cors-proxy"` → `"/chat/cors-proxy"` | MCP proxy constant |
| `./v1/chat/completions` → `./api/1/v1/chat/completions` | point UI at gateway API (fixed parallel=1) |
| `={LIST:"/v1/models",` → `={LIST:"/api/1/v1/models",` | ditto, minified-form literal |
| `={LIST:"/tools",EXECUTE:"/tools"}` → `={LIST:"/chat/tools",EXECUTE:"/tools"}` | tools list via gateway |
| `/_app/` → `/chat/_app/` | SvelteKit asset base |
| `favicon.ico`→`chat/favicon.ico`, `favicon.svg`→`chat/favicon.svg` | icons |
| `./apple-` → `./chat/apple/apple-`, `href="apple-` → `href="chat/apple/apple-` | PWA splash icons |
| `/build.json` → `/chat/build.json` | version probe |

Then the body is **re-gzipped at `Compression::best()`** and Content-Length/Content-Encoding are
overwritten. Every UI asset request pays full buffer + gunzip + regex-free string surgery + gzip-9,
and every one of these literals is a compile-time bet on llama.cpp's *minified* bundle internals.

### C) SSE special-casing of chat completions

`post_chat_completions` (mod.rs:~230ff) does NOT stream-through. It rebuilds the SSE stream:
- 10 s heartbeat comments while connecting and while waiting for tokens
- parses lines via `FramedRead<LinesCodec>`, drops non-`data:` events
- **`trimmed.replace(":null", ":\"\"")`** — self-documented `FIXME: UNSAUBER!!!` (mod.rs:344)
- normalizes `data:` → `data: `, force-emits `data: [DONE]` if missing

### D) API fallback forwarding

`/api/v1/{*path}` and `/api/{n}/v1/{*path}` (any method) → `forward_api_request` which prefixes
`v1/` onto the incoming path (`forward_request(request, Some("v1"))`).

## Auth model of the UI flow

- Gateway `check_auth` middleware requires `Authorization: Bearer <apikey>` on "secured" routes.
  The **document and assets are open** (no auth), `/chat/props` + `/chat/tools` are secured.
- Browser fetches from bundle.js carry **no** Authorization header unless the user pasted the key
  into the UI's own settings (UI stores `apiKey` in localStorage → `getAuthHeaders()`).
- The gateway forwards UI requests to llama-server **without injecting the backend api-key**
  (`forward_uichat_request_and_process` never sets `Authorization`) — except `/chat/props` and
  `/chat/tools` only work because the *browser* happened to send a valid bearer (user configured it)
  — this is why props works in the screenshot only when a key was entered, and why UX is confusing.
- With `-k <key>` (public runs) the backend has `--api-key <same key>`; UI asset paths are on
  llama-server's public allowlist, but `/props`, `/v1/models`, `/tools`, `/v1/chat/completions` are not.

## Why `/model X` and `/stop` commands exist

UI cannot change server-side model selection, so the gateway sniffs the **last user message text**
for `/model <alias>` and `/stop` (openairouter.rs bottom), mutates the prompt, and starts/stops the
backend process accordingly. Model aliases carry context size (`ContextSizeAwareAlias`, e.g.
`...-iq3@16384`). This is orthogonal to UI serving but part of why the UI's own model manager
(router mode) is not usable here.
