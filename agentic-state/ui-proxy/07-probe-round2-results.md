# 07 — Probe round 2: #22474 verdict, prefixed walk WITH key, W8 confirmed (2026-10-16)

Live environment note: owner now runs a **dev instance** `gw-server --port 11434 --override-host-ip
127.0.0.1 --no-https --api-key none --log-request-info --vulkan` (backends :11440/:11441, vulkan
builds). Production :8443 down at probe time. Probes against :11434 were read-only GET/POST to
stateless passthrough endpoints + admin GET — no model switches triggered.

## 1) Issue #22474 verdict: closed **not_planned (stale)** — never fixed upstream

Fetched the issue: author's proposed fix (prefix the allowlist entries) was never submitted as PR;
bot-closed 2026-06-12. Local `git log tools/server/server-http.cpp` confirms no prefix-aware
allowlist commit; newest auth-related change went the *other* way (ee0ea03ad #26347: models
endpoints made **private** when auth enabled — aligns with our probe: `/chat/v1/models` 401s w/o key).
→ **Do not wait for upstream. R1c (cookie bootstrap + server-side key injection) is the way.**
Also means: don't expect upstream to fix this; if we ever submit our local patch, keep it as a
carried patch? No — gateway-side injection avoids patching llama.cpp entirely. Good.

## 2) Full prefixed walk WITH injected key (router-mode probe, :11990, Accept-Encoding: gzip)

| request | result | note |
|---|---|---|
| `GET /chat/` +key | 200 html, ETag, `no-cache`, COOP=same-origin, COEP=require-corp | document fine |
| `GET /chat/` +key +If-None-Match | **304** ✓ | revalidation works through prefix |
| `HEAD /chat/` +key | 200 ✓ | |
| `GET /chat/_app/immutable/bundle.X.js` **no key** | **401** | hashed assets NOT public when prefix+key set (confirms #22474 impact on every asset) |
| same +key | 200, 9.08 MB | gateway injection covers it |
| `/chat/manifest.webmanifest`, `/sw.js`, `/build.json`, icons +key | 200, correct ct/cc/etag-immutable | |
| `GET /chat` (no trailing slash) +key | **404, NO redirect** | gateway catch-all MUST 308 `/chat` → `/chat/` (relative URLs + service-worker scope depend on trailing slash) |
| `GET /chat/<unknown>` +key | 404 JSON (llama.cpp format) | unknown future endpoints: forward = correct default |
| `POST /chat/v1/streams/lookup` +key | **200** | resumable-streams probe works under prefix |

**Conclusion: R1 + R1c fully viable.** Every UI request succeeds when the gateway injects the
backend key; only auth-on-the-edge needs the cookie design. Design detail firmed up: gateway
catch-all = (1) 308 `/chat`→`/chat/`, (2) intercept `POST /chat/v1/chat/completions[/control]` into
existing pipeline, (3) else forward verbatim + inject backend Authorization.

## 3) W8 CONFIRMED — and it's TWO bugs (live-verified on :11434 dev instance)

Current `api_fallback` feature ("forward any /api/v1/* to backend") is **completely broken**:

a) `POST /api/v1/tokenize` (valid backend endpoint) → reaches llama-server as
   `/v1/api/v1/tokenize` → backend 404 (llama.cpp-shaped 404 body proves it was forwarded).
   Root cause: handler keeps the incoming `/api/v1/tokenize` path intact and `forward_request`
   *adds* another `v1/` prefix — gateway never strips its own `/api/v1` mount point.
   Same for `GET /api/v1/health`, `/api/v1/props`, etc. → all 404.

b) **Any** request matching `/api/{n}/v1/{*path}` fallback → **HTTP 500** from axum:
   `"Wrong number of path arguments for Path. Expected 1 but got 2"` —
   `api_fallback_with_parallel_param` declares **two `Path` extractors**
   (`Path(n_parallel)` + `Path(api_path)`); axum allows only one. Always rejects → 500.
   (Dedicated routes models/completions/embeddings are unaffected — single Path each.)
   This is an *unhandled-panic-adjacent* landmine: any client probing e.g. `/api/1/v1/health`
   gets 500, and the parallel-param variant of the fallback has NEVER worked.

Fix direction (fits R1 naturally): ONE fallback handler per mount, extract `Path<String>` once
(`/api/v1/{*path}`), map `/{path}` → backend `{api_prefix}/v1/{path}` (strip own mount, add
backend prefix — no double-prefix), share the single-Path-with-parallel-param pattern
`Path<(u8, String)>` tuple for the `/api/{n}/v1/...` variant. Also: decide whether the fallback
stays at all (it currently forwards nothing that works — candidates to just 404 loudly and let
R1's `/chat` catch-all serve UI needs).

## 4) Heartbeat-relevant observation for R1b

Upstream `40f3aafc4` added `X-Accel-Buffering: no` on streaming endpoints — llama-server already
hints proxies not to buffer. Our gateway keepalive remains necessary (www-proxy idle-kill), but
we should *preserve* that header when forwarding (verbatim forwarding does; current
content-rewriting path drops nothing today but re-check after refactor).

## 5) OPERATIONAL RULES (owner-stated, hard) — add to README constraints

- **Never POST chat completions** to the dev gateway (:11434) or the running backends (:11440/
  :11441): this very conversation is served from their KV-cache; any request can evict it (model
  switch / run-config mismatch ⇒ process restart ⇒ session lost) or contend for the single slot.
  Owner-supplied dev-gateway apikey: `0123456789` (dev :11434 runs `--api-key none`, so gateway
  auth is open locally; backend keys are NOT this — read admin GET only, they're enough for state
  inspection).
- W6 (`stream:false` SSE-mangling) — **static proof, no runtime test allowed**:
  `post_chat_completions` (localllamacppclient/mod.rs) is request-type-agnostic: it always builds
  `sanitized_stream` and responds `Content-Type: text/event-stream` + forced `data: [DONE]`
  epilog. Upstream's JSON body for `stream:false` would be line-framed by `LinesCodec` and emitted
  as `data: ` chunks → mangled. Confidence: high (single unconditional code path).
  Additionally: `post_chat_completions_impl` forces `ensure_requested_languagemodel_is_served`, so
  a request naming a non-running alias restarts the backend — that is the KV-loss vector in code.
- Safe runtime testing exists for the REFACTOR: router-mode fixture trick (probe llama-server on a
  spare port, `--models-dir <empty>`, `--api-key <test>`, zero RAM, fully isolated from :11440) +
  gateway dev-instance on another port pointed at the fixture. Use that, never the live pair.

## Cleanup

- Probe server :11990 (pid 261067) — killed, verified (see below).
- :11434 dev instance + :11440/:11441 backends belong to owner — left untouched.

## Next actions (updated, replaces 06's list)

1. ✅ #22474 verdict → not planned; ✅ prefixed walk with key → all green; ✅ W8 → confirmed ×2.
2. Implement R1a′ + R1b + R1c behind a new `--chat` feature flag on the dev instance (:11434 is a
   perfect safe testbed — localhost, api-key none).
3. Compat-test harness R2: reuse router-mode probe trick (`llama-server --api-prefix /chat
   --api-key X --models-dir <empty>` on a spare port, zero RAM) as fixture for gateway assertions.
4. Decide fate of `/api/v1/{*path}` fallback (fix mapping vs. drop it — see 3b).
5. W6 still open: `stream:false` → SSE-mangled? Test on dev instance next (cheap: POST with
   `stream:false`, observe frame).
