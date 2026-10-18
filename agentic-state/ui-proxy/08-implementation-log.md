# 08 — Implementation log: chat-ui passthrough (branch `chat-ui-passthrough`)

Session 2026-10-16. Status: **compiles, unit test green, NOT yet validated end-to-end.**

## Design actually implemented (differs from 05 in one important way)

**No `--api-prefix` on the backend.** The gateway strips its own `/chat` mount and forwards to a
backend that keeps serving the UI at its root. Reason: `--api-prefix` + `--api-key` is unusable on the
pinned build (probe 06/07: everything 401s, issue #22474 closed *not planned*). Stripping at the
gateway needs **no** backend launch change and no carried llama.cpp patch.

Auth model preserved exactly:
- reads (`GET`/`HEAD`) on `/chat/*` → public, forwarded; the **backend** decides what is public
  (its asset allowlist), so `/chat/props` without a bearer still 401s — same outcome as before
- writes and non-asset reads → `check_bearer` (same apikey), client bearer forwarded as-is
  (gateway key == backend key by construction; owner restarts the stack daily with `-k`)

## Changes per file

| file | change |
|---|---|
| `infrastructure/adapter/localllamacppclient/mod.rs` | rewritten: transparent forwarder. `retarget` (path + Host + optional key injection), `transparent_response` (status/headers/bytes pass through), `keepalive_stream` (10 s idle → `: keepalive\n\n`, **only** for `text/event-stream` without content-encoding, drops `content-length`). `forward_api_request` strips the `/api/v1` / `/api/{n}/v1` mount (fixes W8a). `post_chat_completions`/`post_embedding` are plain JSON POSTs — no SSE re-framing, no `:null→:""`, no forced `[DONE]`, no gzip round-trip |
| `infrastructure/adapter/localllamacppclient/responsepayload.rs` | **deleted** (all string surgery gone) |
| `application/openairouter.rs` | rewritten routing. `/api/*` routes unchanged in path+openness. New: `/chat` → 308 `/chat/`, `/chat/` → document (model-ensure), `/chat/{*path}` → dispatch (reads forward, writes `check_bearer` + forward, `POST v1/chat/completions` + `v1/embeddings` routed into the existing pipeline so `/model` & `/stop` keep working). Root aliases for the absolute paths the UI uses: `/v1/models`, `/health`, `/v1/health` (open) and `/models{,/*}`, `/tools`, `/cors-proxy` (secured). W8b fixed: `Path<(u8, String)>` tuple instead of two `Path` extractors |
| `application/middleware/securitymw.rs` | `check_bearer` extracted; `check_auth` is now a thin wrapper. Behaviour change: with **no** apikey configured requests are accepted without an Authorization header (previously a header was still required — a bug) |
| `application/model/mod.rs` | removed the `sent_from_ui` (Referer-based) model **override**. The UI now lists gateway aliases via `/v1/models` and its selection is honoured (`/model` text command still works). Missing-model requests still get the running/default alias inserted |
| `domain/ports/mod.rs`, `domain/service/openaiclientrequestforwardservice.rs` | dropped `get_chat`/`request_chat` (only used by the removed whitelist handlers) |
| `main.rs` | removed dead `--chatui` flag |
| `Cargo.toml` | + `hyper` (response type), + `tower` dev-dep (route test) |

## Verified

- `cargo build --workspace` — clean, no warnings
- `cargo test -p gw-server --lib` — 1 passed
  (`chat_ui_patterns_are_valid_and_match_document`): asserts the route patterns don't conflict and
  that `/chat/`, `/chat/x`, `/chat/props?…`, `/chat/v1/models`, `/v1/models`, `/models/load`,
  `/tools`, `/cors-proxy` all match.
- **Findings from writing that test:** `/chat/{*path}` does **not** match `/chat/` (the document
  needed an explicit route — would have been a white page); and `/chat/` + `/chat/{*path}` coexist
  without a matchit conflict.

## NOT done / next session

1. **End-to-end run** — blocked by hardcoded backend ports (`LLAMACPP_LLM_PORT` 11440 /
   `LLAMACPP_EMBEDDINGS_PORT` 11441 in `main.rs`); a test instance would collide with the live
   backends. First move next session: make those ports (and the exec dirs) CLI/env-configurable, then
   run a gateway on a spare port against a router-mode fixture
   (`llama-server --host 127.0.0.1 --port <spare> --api-key <t> --models-dir <empty>` — zero RAM).
2. Behavioural checks owed against that fixture: `/chat` 308, document render (rewritten? no — verify
   it is byte-identical to backend `/`), asset fetch without key, `/chat/props` with/without bearer,
   keepalive visible on a long generation, `POST /chat/v1/chat/completions` round-trip incl. `/model`
   and `/stop`, and `stream:false` returning real JSON (W6 fix).
3. Decide whether the root aliases (`/models*`, `/tools`, `/cors-proxy`) should also be reachable
   without the `/api` prefix by API users (currently gateway-only side effects).
4. `agentic-state/` should probably be committed as book-keeping.

## Open risks

- Unknown upstream JSON fields in chat-completions payloads are still dropped by the typed
  `CreateChatCompletionRequest` round-trip (pre-existing behaviour, unchanged).
- `X-Conversation-Id` (upstream resumable streams) does not survive the typed round-trip; a
  resumption attempt will just start a fresh stream. Not a regression, but the UI's new resume probe
  will still not do anything useful.
- Service workers cached the *old rewritten* bundle under old URLs; existing browsers may need a
  hard refresh once.