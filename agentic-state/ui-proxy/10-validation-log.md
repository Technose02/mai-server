# 10 — Validation log: testability + mock tests + fixture smoke test (2026-10-18)

Session 3 on branch `chat-ui-passthrough`. Status: **implemented, unit-tested, integration-tested,
fixture-smoke-tested green. Production not yet switched over.** All checks below ran with zero model
loads; production (:11434 dev gw, :11440/:11441 backends) untouched and verified alive afterwards.

## 1) Gateway is now testable (task 1 — done)

`main.rs`:
- backend ports: `--llamacpp-llm-port <p>` / `--llamacpp-embeddings-port <p>` CLI flags, env
  `LLAMACPP_LLM_PORT` / `LLAMACPP_EMBEDDINGS_PORT`; default unchanged (11440/11441)
- exec dirs: `--llamacpp-execdir <dir>` flag + new env `LLAMACPP_EXECDIR_EMBEDDINGS`
  (`LLAMACPP_EXECDIR_LLM` already existed); default unchanged
- default-embedding-model startup at boot: **panic → logged warning** (degraded, not dead), plus
  `--no-default-embedding-startup` to skip entirely
- NEW safety valve `--no-chatui-model-ensure`: the `/chat/` document handler skips
  `ensure_any_languagemodel_is_served` (via new default-false `ApplicationConfig::
  skip_chatui_model_ensure()` hook) — a fixture instance can serve the UI without ever spawning a
  real llama-server with real model configs. Default behaviour unchanged.

## 2) Mock integration tests (task 2 — done): `gw-server/tests/chat_ui_integration.rs`

15 tests, zero RAM, no processes (mock `OpenAiClientOutPort`/`ModelsServiceInPort`/configs +
`tower::ServiceExt::oneshot` against the REAL router; three of them use a raw-socket mock upstream
on an OS-assigned **ephemeral** port through the real adapter):

- `/chat` → 308 `/chat/` (location asserted, backend untouched)
- `/chat/` document forwarded to backend root; model-ensure called by default, skipped with the flag
- writes (`POST /chat/props`) 401 without/with-wrong bearer, forwarded verbatim-with-bearer when OK;
  open when no apikey configured; reads always public, never key-injected
- `POST /chat/v1/chat/completions` + `/chat/v1/embeddings` re-enter the pipeline (model-ensure with
  requested alias observed; `/model x` sniffing extracts + strips the command)
- root aliases `/tools`, `/models{,/load}`, `/cors-proxy` secured + forwarded; `/health`,
  `/v1/health` open forwards; `/v1/models`, `/api/{,n/}v1/models` answered by gateway (unchanged)
- real-socket: `/chat/_app/x` → backend `/_app/x`; `/api/v1/tokenize` → `/v1/tokenize` and
  `/api/2/v1/tokenize` → `/v1/tokenize` with key injection (W8a+W8b over the wire)
- real-socket SSE: idle stream ⇒ first real chunk passes through, then `: keepalive\n\n` repeats;
  `content-length` dropped on the wrapped body

Plus 6 new unit tests in the adapter (`path_mapping_tests`, `keepalive_tests`): pure path-mapping
cases incl. `/chat?x=1` (now maps to `/?x=1` — was `?x=1`, an empty request-target; **bug fixed
while writing tests**) and keepalive timing/reset behavior. Lib suite: 7 passed.

## 3) Fixture smoke test on spare ports (task 3 — done, all green)

Fixture: `llama-server --port 11991 --api-key testkey --models-dir /tmp/chatui-smoke/empty-models`
(router mode, zero models → zero RAM; build b11718/547208dca — NOTE: newer than the b11672-3-g2851b3bcb
of note 04; behavior below verified against *this* binary) + gateway `--port 11435 -k testkey
--llamacpp-llm-port 11991 --llamacpp-embeddings-port 11992 --no-default-embedding-startup
--no-chatui-model-ensure`. Both stopped afterwards (cmdline-verified kill), production verified alive.

| check | result |
|---|---|
| `GET /chat` → 308 `/chat/` | ✅ |
| document byte-identical backend `/` vs gateway `/chat/` (raw gzip bytes AND decompressed; 1290 B) | ✅ zero content rewriting |
| response headers identical (case-insensitively — hyper lowercases field names, legal per RFC 9110 §5.1) incl. ETag, COOP/COEP, `X-Accel-Buffering: no` | ✅ |
| this build refuses non-gzip on `/` with 415 "gzip is not supported" — gateway reproduces the upstream error **byte-for-byte** (old code would have 500'd per W4) | ✅ transparency incl. errors |
| hashed asset `/_app/immutable/bundle.*.js` (9.08 MB) loads WITHOUT key, bytes identical backend vs gateway | ✅ |
| ETag revalidation: `If-None-Match` → 304 through gateway | ✅ |
| `/chat/props`: 401 without key / 200 with (backend decides, gateway passes through) | ✅ |
| `/tools` 401 w/o key; with key 403 = backend's "feature disabled" — identical to direct backend | ✅ |
| `/models`, `/models/load`, `/models/sse`, `/cors-proxy` aliases: gateway status == backend status exactly (200/404/200-stream/403) | ✅ |
| bundle absolute-URL literal scan (`/v1/models /models /models/{load,unload,sse} /tools /cors-proxy`) — every literal routable | ✅ (R2 compat test effectively executed; automatable next session) |
| `GET /api/v1/health` + key == backend `/v1/health`; `GET /api/3/v1/health` (parallel-param, old guaranteed-500) identical too — W8a+W8b live-verified | ✅ |
| unknown API endpoint → backend's own 404 body verbatim | ✅ forward-by-default |
| **keepalive on a real idle SSE stream through the full stack** (`/chat/models/sse`, router-mode event bus idle): `: keepalive` observed after ~10 s, headers otherwise identical | ✅ load-bearing heartbeat works |
| router-mode guard: completions against fixture answer `model 'nonexistent' not found` — no model can load in the harness | ✅ safe harness |

## Findings / notes for next session

- **Newer llama.cpp (b11718) serves `/props` at root only** (`/props`, not `/v1/props`) and has no
  `/v1/tokenize` — earlier probes that used those paths under the API mount "failed" correctly:
  gateway maps `/api/v1/<x>` → backend `/v1/<x>`, and the backend really has no such route. The W8a
  proof therefore uses `/v1/health` (exists in router mode). Not a gateway defect.
- **OPTIONS preflight deviation**: upstream answers `OPTIONS` *before* auth; our catch-all treats all
  non-GET/HEAD as writes → `OPTIONS /chat/props` without bearer = 401 where the backend would answer
  the preflight. Same-origin UI never sends preflights, so this only matters if something starts
  calling from another origin. If it bites: mirror upstream by short-circuiting OPTIONS in
  `chat_ui_dispatch` before `check_bearer`. Recorded, not changed (behaviour parity with old gateway).
- `/models/load|unload` answer **404 on router-mode even with key** on b11718 (router needs
  `model` names resolvable from `--models-dir`; empty dir → 404). Gateway passes the backend's own
  404 through — transparency verified.
- Kill hygiene: fixture processes launched via `setsid nohup` survive exec sandboxes; kill by
  cmdline-matched PID only (`*--port 11991*`, `*--port 11435*`).

## 4) Checks that still need a REAL model (run on the morning dev instance)

1. **`stream:false` returning real JSON** through `/api/v1/chat/completions` and
   `/chat/v1/chat/completions` (W6 fix): `curl -N -s -d '{"model":"<alias>","messages":[{"role":"user","content":"hi"}],"stream":false}'`
   — expect a single JSON body, NOT SSE frames / `[DONE]`.
2. **keepalive during long TTFT** end-to-end through the work www-proxy (the fixture proved the
   mechanism; confirm with a real thinking model + real proxy chain, incl. that 10 s is enough).
3. **`/model <alias>` and `/stop` typed into the chat UI** (UI → `POST /chat/v1/chat/completions` →
   pipeline): switch models mid-session, stop generation. Fixture covered the sniffing logic only.
4. **UI renders & chats in a real browser**: document + 9 MB bundle over TLS on the dev instance,
   model manager screen (router endpoints), PWA/service-worker scope under `/chat/` — one hard
   refresh expected (stale SW from the old rewritten bundle; see 08 "Open risks").
5. Resumable streams (`X-Conversation-Id`, `POST ./v1/streams/lookup`, reattach via `/v1/stream`):
   lookup now at least *reaches* the backend (was 404 before); actual resume across a gateway-side
   disconnect still needs the typed-request round-trip to carry the header — unchanged limitation.

## Definition-of-done status

- [x] workspace builds clean (no warnings), lib tests 7/7, integration tests 15/15
- [x] fixture smoke test green (table above), fixtures stopped, production verified untouched
- [x] written status of what remains (§4)
- [ ] owner decision: merge vs. keep on branch — see recommendation below

## Recommendation: keep on branch until §4 items 1–3 pass on the dev instance

The transparent forwarder is proven at HTTP level (byte-identity, auth parity, W8 fixes, keepalive).
What remains (§4) needs a real model and the work www-proxy — cheap to run in the morning, and item 3
(`/model` + `/stop` through the actual UI) is the only feature whose *plumbing* changed without full
coverage. If those pass, merge is low-risk: the OpenAI API surface is untouched code paths, and every
deviation we could provoke against the fixture matched the backend exactly.
