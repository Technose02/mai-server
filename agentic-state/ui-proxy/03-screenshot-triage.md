# 03 — Screenshot triage (Firefox devtools, gw :8080, document = /chat)

**VERIFIED 2026-10-16 against the live served bundle** (`/chat/_app/immutable/bundle.BBxhLRGv.js`,
9.0 MB uncompressed, fetched through the running gateway on :8443 read-only — production box, see
README constraints). Devtools shows only the last path segment; mapping given below.

| # | Req (devtools name) | Status | Verified root cause |
|---|-----|--------|-------------|
| 1 | GET /chat (document) | 200 | chat_handler → backend `/` → HTML body-rewritten. OK today. |
| 2 | bundle.BBxhLRGv.js | 200 | served via `/chat/_app/{*path}`; `"/_app/` in index.html rewritten to `"./chat/_app/`. Hash-named bundle — upstream moved from `/bundle.js` to hashed `_app/immutable/` assets since this workaround was written. |
| 3 | bundle.DmJS8kUp.css | 200 | same mechanism |
| 4 | apple-touch-icon-180x180.png | 200 | `./apple-`→`./chat/apple/apple-` rewrite + route |
| 5 | favicon.svg | 200 | sloppy global `favicon.svg`→`chat/favicon.svg` rewrite (matches anywhere, incl. JS strings) |
| 6 | GET /tools → 404 | ✗ | **PROVEN**: served bundle contains `HI={EXECUTE:"/tools",LIST:"/tools"}` — W3 pattern expects `={LIST:"/tools",EXECUTE:"/tools"}`; **minifier flipped the key order** → no replacement → request to gateway root `/tools`, which has no open route → axum 404 (never proxied). |
| 7 | GET build.json | 200 | handled |
| 8 | GET props (= `/chat/props`) | 200 | `"./props"`→`"/chat/props"` replacement **does** match served bundle (`Tbe("/chat/props",r,...)`). Browser additionally had to send the Bearer key (route is secured, adapter does not inject backend key). |
| 9 | POST /tools → 404 | ✗ | same as #6; router also has no POST `/chat/tools` at all. |
| 10 | GET props?autoload=false | 200 | fine |
| 11 | GET mcp + OPTIONS 127.0.0.1:8000/mcp | ✗ | UI probes local MCP server on :8000; gateway-independent (needs `--ui-mcp-proxy` + actual MCP server). Not a gw bug. |
| 12 | POST lookup (= `/v1/streams/lookup`) → 404 | ✗ | **PROVEN**: bundle has `Aye={BASE:"./v1/stream",LOOKUP:"./v1/streams/lookup"}` — new upstream resumable-stream feature (PR #23226). Relative to document `/chat` (no trailing slash) resolves to root `/v1/streams/lookup`; gateway has no route → 404. Gateway also predates and ignores `X-Conversation-Id` semantics. |
| 13 | GET version.json (= `/chat/_app/version.json`) | 200 | fine (`${rl}/chat/_app/version.json` — rewrite applied) |
| 14 | GET props?model=… | 200 | per-model props works |
| 15 | GET models (= `/v1/models`) → 404 | ✗ | **PROVEN**: bundle has `...DOWNLOAD:"/models",LIST:"/v1/models",LOAD:"/models/load",SSE:"/models/sse"...`; W3 pattern requires literal `={LIST:"/v1/models",` — the `={` prefix is gone (other keys precede) → no replacement → browser requests gateway `/v1/models` → 404. Confirmed by direct probe: `/v1/models -> 404`, while `/api/v1/models` and `/api/1/v1/models` return 200. Also `/models`, `/models/load`, `/models/sse` (router-mode model manager) are unmapped → UI model-manager half-broken. |

Additional latent breakage proven by probe:
- `Ete="/cors-proxy"` — constant name minified away, so the W3 pattern
  `CORS_PROXY_ENDPOINT="/cors-proxy"` can never match again → MCP proxy path broken (latent; only
  visible when `--ui-mcp-proxy` is used).
- `/sw.js`, `/manifest.webmanifest`: backend returns **415 "gzip not supported"** if the client does
  not send `Accept-Encoding: gzip` (gateway forwards the client's identity encoding verbatim). Browsers
  always send gzip, so invisible in practice — but curl-style checks mislead.
- Gateway returns the **upstream ETag unchanged while serving a mutated body** (verified on `/chat`:
  upstream `etag: "eeca…"` kept after rewriting) → browser cache stores mutated content under the
  pristine ETag; any other consumer of the same etag semantics gets surprises.
- `/chat/tools` via gateway returns **403** (backend answers because tools/MCP not enabled server-side)
  — so even with correct routing, features 403/404 depending on backend launch flags.

Conclusion: today's 200s are coincidental coverage of a stale rewrite table; three observed breakages
are each literally "the minifier changed its mind" (key order, constant folding, adjacent keys) and one
is "upstream added a feature". The design cannot be fixed by adding more patterns.
