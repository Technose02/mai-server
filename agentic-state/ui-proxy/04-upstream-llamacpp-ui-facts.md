# 04 — Upstream llama.cpp server/UI facts (verified ground truth)

Sources: local checkout `/data0/inference/llama.cpp` @ `b11672-3-g2851b3bcb` (2026-10-16, same binary
the gateway runs) + web research (PRs #14544, #6325, #23226, #24550, #25655; issues #22474, #24573).
Always re-verify against HEAD — this layer is what keeps breaking.

## How the modern server serves its UI

- WebUI = SvelteKit SPA in `tools/ui/` (moved from `tools/server/webui/`), built with
  `adapter-static`, **hash router**, `paths: { relative: true }`, `bundleStrategy: 'single'`.
  A `vite-plugin-relativize-base` rewrites baked `"/_app/` → `"./_app/` in index.html/sw.js so the
  whole dist is relocatable under any subpath (build-time intent: "relocatable under any subdir").
- Assets embedded into the binary via `llama-ui-embed` (`tools/ui/embed.cpp`) → asset table with
  name/etag/type; served by `server-http.cpp`:
  - `/` and `/index.html` → index.html, ETag + 304, `no-cache`, COOP/COEP isolation headers
  - every other asset registered from the table: hashed ones `immutable`, PWA files
    (`sw.js`, `manifest.webmanifest`, `_app/version.json`, `build.json`) `no-cache`
  - gzip build only; if client doesn't accept gzip → **415** (this is what curl/identity probes see)
  - when UI disabled: `/sw.js` returns a self-unregistering worker (stale-service-worker hygiene —
    relevant for us: browsers that visited the UI keep a service worker!)
- `--public_path PATH` can serve files from disk instead of embedded assets.
- **`--api-prefix PREFIX` exists** (PR #14544, env `LLAMA_ARG_API_PREFIX`): server itself mounts ALL
  routes (API + UI + assets) under a prefix. Known wart: public-endpoint allowlist did not account
  for the prefix with `--api-key` (issue #22474 — check current status before relying on it).

## Endpoint surface the current UI touches (from served bundle, b1167x)

- Relative (`./`, resolve against document URL → subpath-friendly):
  `./v1/chat/completions`, `./v1/chat/completions/control`, `./props` (via apiFetchWithParams →
  `new URL(basePath, location.href)`), `./slots`, `./v1/stream`, `./v1/streams/lookup`,
  `./_app/*`, `./manifest.webmanifest`, `./favicon.*`, `./apple-*`, `./build.json`
- **Absolute (root-hoped, NOT subpath-friendly)** — the entire class of our breakage:
  `/v1/models`, `/models`, `/models/load`, `/models/unload`, `/models/sse`, `/tools` (GET+POST),
  `/cors-proxy`
  (`apiFetch` prefixes SvelteKit's `base`, which is `''` in this build — so absolute stays absolute.)
- SSE resumable streaming (PR #23226): client sends `X-Conversation-Id` on chat completions; server
  keeps generation alive across disconnects; UI probes `POST ./v1/streams/lookup` at startup and
  reattaches via `GET /v1/stream?conv_id=…`. **A proxy that mangles SSE framing or drops the header
  breaks this feature; a proxy that just forwards bytes is transparent to it.**

## Auth on backend (`server-http.cpp`)

- Public (no API key needed): `/health`, `/v1/health`, `/` + every embedded asset path
  (`frontend_paths` set built from asset table). Everything else — including `/props`, `/v1/models`,
  `/tools`, `/cors-proxy` — requires `Authorization: Bearer <key>` or `X-Api-Key`.
- Not-yet-ready state: everything except frontend paths → 503 "Loading model" (UI shows loading page).
- CORS now configurable (`--cors-*`, PR #25655); special value `localhost` for origin reflection.
- MCP/tools/cors-proxy endpoints exist but answer **403 "feature disabled"** unless the server was
  launched with the respective flags (`--ui-mcp-proxy`, server tools, MCP servers).

## Model management / multi-model

- Server has a **router mode** (multi-model: `POST /models/load|unload`, `GET /models/sse` events) —
  the UI's model-manager speaks it. mai-server instead restarts one single-model llama-server per
  requested alias (`--alias`, process-level switching) + `/model <x>` chat command.
- `--api-key <key>` on backend; mai-server passes its own gateway key (same value).

## Version drift observed in this project's lifetime

- At the time W1-W3 were written (~late 2025): UI assets were flat `bundle.js`/`bundle.css` at root
  (hence `/chat/bundle.js` routes + `./bundle.` replacement — still present, now dead weight).
- Since then: hashed `_app/immutable/` bundles (PR #24550 static-assets cleanup), key-order changes
  in minified constants, constant folding of `CORS_PROXY_ENDPOINT`, new `/v1/stream*` endpoints,
  router-mode model manager in UI, `--cors-*` hardening. Each step silently broke a pattern.

## Practical notes for this machine

- Production gateway runs on :8443 serving live traffic; LLM backend :11440 (currently
  qwen3.8-flash-next-no-thinking-iq4-xs-max, ctx 262144), embeddings :11441 (bge-m3).
- Backend launched with `--host localhost` (good: not directly reachable externally) and
  `--api-key <gateway key>`; **without** `--ui-mcp-proxy`, so `/cors-proxy` → 403 by design.
- llama.cpp checkout used for builds is pinned via `LLAMACPP_EXECDIR_LLM` env (`.env`:
  `/data0/inference/llama.cpp`; kolibri variant separate).
