# 06 — Probe results: `--api-prefix` + `--api-key` on pinned build (2026-10-16)

Probe setup (zero RAM cost — router mode loads no model):
```
llama-server --host 127.0.0.1 --port 11990 --api-prefix /chat --api-key testkey123 \
             --models-dir /tmp/probe-models-empty        # build b11675-2851b3bcb, router mode
```
Script: `/tmp/probe_prefix.js` (recreate if needed). Router mode starts fine with 0 models and
serves the full HTTP surface — ideal harness for future compat tests.

## ★ CRITICAL FINDING — issue #22474 is REAL on this build

**With `--api-key` + `--api-prefix`, NOTHING is public, including `/chat/` (the UI document) and all
assets.** Public-allowlist entries are matched *unprefixed* (`/health`, asset names) while requests
arrive prefixed (`/chat/health`, `/chat/_app/...`) → everything 401s:

| request | no key | with key |
|---|---|---|
| `GET /chat/` (UI doc) | **401** | (expected 200 — not reached, doc is allowlist entry) |
| `GET /chat/health`, `/chat/build.json`, `/chat/favicon.ico`, `/chat/manifest.webmanifest` | **401** | — |
| `GET /chat/props`, `/chat/v1/models`, `/chat/models` | 401 | **200** ✓ |
| `GET /` , `/health` (unprefixed) | 404 ✓ (prefix strictly enforced, good) | |
| `GET /chat` (no trailing slash) | 401 — **no redirect observed** to `/chat/`; matters for relative-URL resolution in the SPA |

Consequence for R1: fallback **(a) "assets stay public at backend" is DEAD on this build**.
The plan must use one of:

- **(b) gateway-injected key + cookie bootstrap (R1c) becomes MANDATORY, not optional.**
  Order matters (already recorded in 05): open `/chat/bootstrap?key=<apikey>` → sets HttpOnly
  cookie → redirect `/chat/` ; catch-all requires cookie AND injects backend `Authorization` on
  every forwarded request. Security note: this is *not* an injection oracle because the bootstrap
  link itself is the auth gate (one-time, HTTPS-only; consider short TTL/nonce).
- **(a′) upgrade llama.cpp** — web verification of #22474 fix status FAILED (search provider down);
  TODO next session: check `git log -- tools/server/server-http.cpp` in local checkout for a commit
  making the allowlist prefix-aware, or grep GitHub issues via other means. If fixed upstream ≥ our
  build +2 weeks, upgrading may restore the simpler design (assets public at backend).
- **(c) last resort** (still viable): drop `--api-key` on backend entirely; security =
  `--host localhost` bind + gateway auth on everything. Review what else can reach :11440 locally.

## Other verified facts from this probe

- Router-mode `/props` works and reports `"role":"router"`, `build_info b11675` — UI renders against
  an empty model list (useful as a no-model test harness for the R2 compat-test).
- Unprefixed routes are strictly gone when prefix set (404/401, never served) → no namespace leaks.
- `OPTIONS /chat/props` without key: CORS preflight short-circuits BEFORE auth middleware in
  upstream code (`server-http.cpp` pre_routing_handler checks `req.method == "OPTIONS"` first) —
  gateway should mirror this (answer OPTIONS itself or forward; either way don't 401 preflights).

## Open items for next session (in order)

1. Check upstream for prefix-aware public-allowlist fix (#22474) — local `git log` on
   `tools/server/server-http.cpp` in `/data0/inference/llama.cpp`, or retry web search (provider was
   down; try again / different provider).
2. Decide fallback branch: (b) cookie bootstrap as designed, vs (a′) upgrade-then-(original R1),
   vs (c). Recommendation leans **(b)** regardless — it also fixes the "paste key into UI" UX and
   makes whole `/chat` gated (strictly stronger than today).
3. Trailing-slash: gateway catch-all should 308 `/chat` → `/chat/` so SPA relative URLs resolve
   inside the prefix (backend does NOT redirect; it 401s — verify again with key header sent:
   `GET /chat` WITH key was not tested! Do that in next probe round.)
4. Build R2 compat-test harness against router-mode probe server (no model needed) + live bundle.
5. Still-open verifications from before: W6 (`stream:false` mangled), W8 (`/api/v1/*` fallback
   double-prefixes `v1/`), `/chat` WITH-key full asset walk on the *real* gateway (:8443, read-only).

## Cleanup

- Probe server pid 134857 on :11990 — killed and verified stopped 2026-10-16; production (:8443,
  :11440, :11441) untouched throughout.
- `/tmp/probe-models-empty`, probe scripts in /tmp are disposable.
