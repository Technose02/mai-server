# 09 — Copy-paste prompt for the next session

Use this verbatim to start a fresh session with full context.

---

Continue the gw-server chat-ui refactor.

Repo `/data0/dev/rust/mai-server-vibe`, branch **`chat-ui-passthrough`** (2 commits ahead of `main`,
clean tree). Work only on that branch.

**Read first, and treat as authoritative:** `agentic-state/README.md` and
`agentic-state/ui-proxy/08-implementation-log.md`. The repo's own `README.md` and old comments are
outdated — the notes were verified live against llama.cpp checkout `b11672-3-g2851b3bcb`. Use web
search for anything about current llama.cpp behaviour instead of assuming.

**Hard constraints — violating these breaks my live service:**
- This machine is production. `gw-server` serves live internet traffic, including the model you are
  talking to. Never restart it, never kill it, never bind its ports (`8443`, `11434`, `11440`, `11441`).
- **Never POST** to `/chat/completions` or `/v1/chat/completions` on `11434`/`11440`/`11441` — that
  evicts the live session's KV-cache and ends the conversation. Read-only `GET` to `/admin/*` is fine.
- **<5 GB RAM free — never load a model.** Use llama.cpp *router mode*
  (`llama-server --models-dir <empty dir>` = zero RAM) or mocks as fixtures.
- The OpenAI-compatible API (`/api/v1/*`, `/api/{n}/v1/*`) is the **primary product**: paths,
  semantics and auth must not change. The chat UI is a bonus. Single-apikey security stays; I rotate
  it daily by restarting the whole stack with `-k <newkey>` (gateway and backends share the key).
- No full reverse-proxy dependency — thin purpose-built forwarding inside gw-server only.
- SSE keepalives are load-bearing: my work proxy kills idle connections and TTFT can reach 30 min.

**State:** the rewrite is implemented, compiles clean (`cargo build --workspace`), one route-pattern
unit test passes — but it has **never served a request**. The three layers of compiled-in upstream
knowledge are gone: no route whitelist, no path-rewrite table, no string surgery on minified bundles
(`responsepayload.rs` deleted). Reads on `/chat/*` are forwarded transparently, writes require the
apikey, `POST /chat/v1/chat/completions|embeddings` re-enter the existing pipeline, `/chat` → 308
`/chat/`, generic 10 s SSE keepalive replaces the SSE re-framing, and two confirmed fallback bugs
(double `v1` prefix, guaranteed axum 500) are fixed. Details in note 08.

**Task, in order:**
1. **Make the gateway testable** (`gw-server/src/main.rs`): backend ports and exec dirs from CLI/env
   (default to today's 11440/11441), and turn the mandatory default-embedding-model startup at boot
   from `panic!` into a logged warning (or add `--no-default-embedding-startup`). Without this, any
   test instance collides with my live backends and tries to load a model.
2. **Integration tests with mocks** — zero RAM, no processes: stub `OpenAiClientOutPort` +
   `ApplicationConfig`, drive the real router via `tower::ServiceExt::oneshot`. Assert: `/chat` → 308;
   `/chat/` document; `/chat/props` 401 without bearer → 200 with; `POST /chat/v1/chat/completions`
   re-enters the pipeline; `/tools` and `/models` aliases; path mapping `/chat/_app/x` → `/_app/x`
   and `/api/v1/tokenize` → `/v1/tokenize`; keepalive bytes actually appear on an idle
   `text/event-stream` body.
3. **Fixture smoke test on spare ports:**
   ```
   mkdir -p /tmp/empty-models
   /data0/inference/llama.cpp/build-rocm/bin/llama-server --host 127.0.0.1 \
       --port 11991 --api-key testkey --models-dir /tmp/empty-models
   cargo run -p gw-server -- --port 11435 -H 127.0.0.1 --no-https -k testkey \
       --llamacpp-llm-port 11991 --llamacpp-embeddings-port 11992
   ```
   Verify `/chat` → 308 and that `diff <(curl -s <backend>/) <(curl -s <gateway>/chat/)` is
   **byte-identical** (proves zero content rewriting), assets load without a key, `/chat/props` is 401
   without a key / 200 with. Stop both fixtures afterwards.
4. **Report** which checks still need a real model (`stream:false` returning real JSON, keepalive on a
   long TTFT, `/model` + `/stop` through the UI) so I can run those on my morning dev instance.

**Definition of done:** workspace builds clean, new tests pass, fixture smoke test green, a short
written status of what remains, and an explicit recommendation on merge vs. keep-on-branch. Commit to
the branch (never `main`), and keep book-keeping notes in `agentic-state/`.