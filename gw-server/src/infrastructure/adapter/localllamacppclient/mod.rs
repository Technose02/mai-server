use crate::{domain::ports::OpenAiClientOutPort, model::SecurityConfig};
use async_trait::async_trait;
use axum::{
    body::{Body, Bytes},
    extract::Request,
    http::{
        HeaderMap, HeaderName, HeaderValue, StatusCode, Uri, Version,
        header::{
            ACCEPT_ENCODING, AUTHORIZATION, CONNECTION, CONTENT_ENCODING, CONTENT_LENGTH,
            CONTENT_TYPE, HOST as HOST_HEADER, TE, TRAILER, TRANSFER_ENCODING, UPGRADE,
        },
    },
    response::Response,
};
use futures::StreamExt;
use hyper_util::{
    client::legacy::{Client as LegacyClient, connect::HttpConnector},
    rt::TokioExecutor,
};
use std::{sync::Arc, time::Duration};
use tracing::{error, trace, warn};

const LLAMACPP_HTTP_SCHEME: &str = "http";
const LLAMACPP_HOST: &str = "localhost";
const LLAMACPP_API_BASE_PATH: &str = "v1";

/// header carrying the upstream resumable-stream session id (llama.cpp
/// `server_stream_conv_id_from_headers`). It has to survive the gateway
/// untouched: with it the backend keeps the generation alive across client
/// disconnects and the ui can reattach at a byte offset; without it a dropped
/// socket loses the generation permanently.
const CONVERSATION_ID_HEADER: &str = "x-conversation-id";

/// idle time after which a keepalive comment is injected into an SSE response
const SSE_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(10);
const SSE_KEEPALIVE_COMMENT: &[u8] = b": keepalive\n\n";

/// headers that describe the client/upstream connection rather than the
/// request; they must never be copied onto a different connection.
const HOP_BY_HOP_HEADERS: [HeaderName; 5] = [
    CONNECTION,
    TRANSFER_ENCODING,
    TE,
    TRAILER,
    UPGRADE,
];

type Client = LegacyClient<HttpConnector, axum::body::Body>;

/// Transparent forwarding client for a locally running llama-server.
///
/// The whole point of this adapter is to NOT interpret upstream content: request
/// paths are mapped, everything else (status, headers, body bytes) is passed
/// through untouched. The only body-level operation is the generic SSE
/// keepalive, which is required because long TTFT generations would otherwise
/// be killed by intermediate proxies on idle connections.
pub struct LocalLlamaCppClientAdapter {
    client: Client,
    llamacpp_port: u16,
    security_config: Arc<dyn SecurityConfig>,
}

impl LocalLlamaCppClientAdapter {
    pub fn create_adapter(
        port: u16,
        security_config: Arc<dyn SecurityConfig>,
    ) -> Arc<dyn OpenAiClientOutPort> {
        let client = LegacyClient::builder(TokioExecutor::new()).build(HttpConnector::new());

        Arc::new(Self {
            client,
            llamacpp_port: port,
            security_config,
        })
    }

    fn backend_base_uri(&self) -> String {
        format!(
            "{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}",
            self.llamacpp_port
        )
    }

    /// retargets `request` onto the local llama-server.
    ///
    /// `target_path_and_query` is an absolute path (and optional query) as served
    /// by the backend, e.g. `/v1/chat/completions`.
    fn retarget(&self, request: &mut Request, target_path_and_query: &str, inject_api_key: bool) {
        let uri_string = format!("{}{target_path_and_query}", self.backend_base_uri());

        trace!("forwarding request to llama-server using uri {uri_string}");

        *request.uri_mut() = Uri::try_from(uri_string.clone())
            .unwrap_or_else(|_| panic!("{uri_string} expected to be a valid uri"));

        let headers = request.headers_mut();
        headers.insert(HOST_HEADER, HeaderValue::from_static(LLAMACPP_HOST));

        if inject_api_key
            && let Some(api_key) = self.security_config.get_apikey()
            && let Ok(auth_value) = HeaderValue::from_str(&format!("Bearer {api_key}"))
        {
            headers.insert(AUTHORIZATION, auth_value);
        }

        *request.version_mut() = Version::HTTP_11;
    }

    /// sends an already retargeted request and returns the transparently
    /// forwarded response.
    ///
    /// A transport failure becomes `503` (not `500`): the gateway is fine, the
    /// backend is not reachable/not answering. The body is an openai-shaped
    /// error so both api clients and the ui can render something sensible.
    async fn send(&self, request: Request, keepalive: bool) -> Result<Response, StatusCode> {
        let target = request
            .uri()
            .path_and_query()
            .map(|v| v.as_str().to_string())
            .unwrap_or_default();
        let method = request.method().clone();
        let started = std::time::Instant::now();

        match self.client.request(request).await {
            Ok(response) => {
                trace!(
                    "llama-server {method} {target} -> {} in {:?}",
                    response.status(),
                    started.elapsed()
                );
                Ok(transparent_response(response, keepalive))
            }
            Err(e) => {
                warn!(
                    "llama-server {method} {target} failed after {:?}: {e}",
                    started.elapsed()
                );
                Err(StatusCode::SERVICE_UNAVAILABLE)
            }
        }
    }

    /// body + headers for a gateway-initiated json POST to the backend.
    ///
    /// `headers` are the headers of the original client request: everything
    /// that describes the request itself (notably `X-Conversation-Id`) is
    /// carried over so upstream features keep working; connection-level
    /// headers, `host`, `content-length` and encoding negotiation are replaced
    /// because the body is rewritten by the gateway.
    fn build_json_post(
        &self,
        target_path: &str,
        client_headers: &HeaderMap,
        payload: String,
    ) -> Result<Request, StatusCode> {
        let mut request = Request::post(format!("{}{target_path}", self.backend_base_uri()))
            .body(Body::from(payload))
            .map_err(|e| {
                warn!("error building llama.cpp-request for {target_path}: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        let headers = request.headers_mut();
        for (name, value) in forwarded_headers(client_headers) {
            headers.insert(name, value);
        }
        // the payload is rewritten by the gateway, so length/type/encoding are ours
        headers.remove(CONTENT_LENGTH);
        // never let upstream compress an sse body we might have to interleave
        headers.insert(ACCEPT_ENCODING, HeaderValue::from_static("identity"));
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        self.retarget(&mut request, target_path, true);
        Ok(request)
    }

    async fn post_json(
        &self,
        target_path: &str,
        client_headers: &HeaderMap,
        payload: String,
    ) -> Result<Response, StatusCode> {
        let request = self.build_json_post(target_path, client_headers, payload)?;
        // keepalive only for streams that cannot be resumed by the client anyway
        let has_conversation_id = client_headers
            .get(CONVERSATION_ID_HEADER)
            .is_some_and(|value| !value.is_empty());
        let keepalive = !has_conversation_id;
        self.send(request, keepalive).await
    }
}

/// headers of the client request that are safe and useful to forward verbatim.
fn forwarded_headers(client_headers: &HeaderMap) -> impl Iterator<Item = (HeaderName, HeaderValue)> {
    client_headers
        .iter()
        .filter(|(name, _)| {
            !HOP_BY_HOP_HEADERS.contains(name)
                && *name != HOST_HEADER
                && *name != CONTENT_LENGTH
                && *name != ACCEPT_ENCODING
                && *name != CONTENT_TYPE
        })
        .map(|(name, value)| (name.clone(), value.clone()))
}

/// true when the request is part of an upstream resumable stream session.
///
/// Two shapes exist: the original `POST /v1/chat/completions` opts in with the
/// `X-Conversation-Id` header, and a reattach is a `GET /v1/stream?conv_id=...&from=N`.
/// Both must be forwarded byte-for-byte: the client resumes at a *wire byte
/// offset* that has to line up with the backend's replay buffer, so no keepalive
/// comments may be interleaved.
fn is_resumable_request(request: &Request) -> bool {
    let has_header = request
        .headers()
        .get(CONVERSATION_ID_HEADER)
        .is_some_and(|value| !value.is_empty());

    let has_conv_id_query = request.uri().query().is_some_and(|query| {
        query
            .split('&')
            .any(|param| param == "conv_id" || param.starts_with("conv_id="))
    });

    has_header || has_conv_id_query
}

/// transparently forwards an upstream response.
///
/// Status, headers and body are passed through unchanged. For SSE responses (and
/// only for those) an idle keepalive comment is interleaved, because upstream is
/// free to stay silent for many minutes while a model is thinking.
fn transparent_response(response: hyper::Response<hyper::body::Incoming>, keepalive: bool) -> Response {
    let (mut parts, body) = response.into_parts();

    if keepalive && is_keepalive_candidate(&parts.headers) {
        // the body grows by the keepalive comments -> upstream length is void
        parts.headers.remove(CONTENT_LENGTH);
        Response::from_parts(parts, Body::from_stream(keepalive_stream(Body::new(body))))
    } else {
        Response::from_parts(parts, Body::new(body))
    }
}

/// keepalive may only be interleaved into a plain (not content-encoded)
/// `text/event-stream` body - anything else is forwarded untouched.
fn is_keepalive_candidate(headers: &axum::http::HeaderMap) -> bool {
    let is_sse = headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));
    // never interleave bytes into an encoded stream
    let is_encoded = headers.contains_key(CONTENT_ENCODING);

    is_sse && !is_encoded
}

fn keepalive_stream(body: Body) -> impl futures::Stream<Item = Result<Bytes, std::io::Error>> {
    keepalive_stream_with(body, SSE_KEEPALIVE_INTERVAL)
}

fn keepalive_stream_with(
    body: Body,
    idle_interval: Duration,
) -> impl futures::Stream<Item = Result<Bytes, std::io::Error>> {
    let mut stream = body.into_data_stream();
    let mut interval = tokio::time::interval(idle_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    async_stream::stream! {
        // an interval's first tick is immediate; consume it so we only emit
        // keepalives after real idle time has passed
        interval.tick().await;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    yield Ok(Bytes::from_static(SSE_KEEPALIVE_COMMENT));
                }
                chunk = stream.next() => {
                    match chunk {
                        Some(Ok(bytes)) => {
                            interval.reset();
                            yield Ok(bytes);
                        }
                        Some(Err(e)) => {
                            error!("error streaming from llama-server: {e}");
                            yield Err(std::io::Error::other(e));
                            break;
                        }
                        None => break,
                    }
                }
            }
        }
    }
}

/// maps a gateway API path onto the path served by the backend.
///
/// The gateway mounts its openai-compatible API at `/api/v1` (and optionally
/// `/api/{n_parallel}/v1`), while the backend serves the very same API at `/v1`.
/// The mount prefix has to be stripped, otherwise the backend is asked for
/// `/v1/api/v1/...` and answers 404.
fn gateway_api_to_backend_path(uri: &Uri) -> Result<String, StatusCode> {
    let path = uri.path();

    let rest = path
        .strip_prefix("/api/v1")
        .or_else(|| {
            path.strip_prefix("/api/").and_then(|after_api| {
                let (n_parallel, after_n) = after_api.split_once('/')?;
                n_parallel.parse::<u8>().ok()?;
                after_n.strip_prefix("v1")
            })
        })
        .ok_or(StatusCode::NOT_FOUND)?;

    let query = uri.query().map(|q| format!("?{q}")).unwrap_or_default();

    Ok(format!("/{LLAMACPP_API_BASE_PATH}{rest}{query}"))
}

/// maps a gateway ui path onto the path served by the backend.
///
/// The gateway re-presents the upstream ui under `/chat`, the backend still
/// serves it at its root - so the mount prefix is stripped here. Root-absolute
/// endpoints the upstream ui uses (`/tools`, `/models/...`, `/cors-proxy`, ...)
/// are passed through unchanged.
fn ui_target_path_and_query(uri: &Uri) -> String {
    let path_and_query = uri
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or("/");

    match path_and_query.strip_prefix("/chat") {
        Some("") => "/".to_string(),
        // "/chat?query" -> the backend still needs an absolute path in front of
        // the query, otherwise the forwarded request-target would be empty
        Some(rest) if rest.starts_with('?') => format!("/{rest}"),
        Some(rest) if rest.starts_with('/') => rest.to_string(),
        _ => path_and_query.to_string(),
    }
}

#[async_trait]
impl OpenAiClientOutPort for LocalLlamaCppClientAdapter {
    async fn forward_api_request(&self, mut request: Request) -> Result<Response, StatusCode> {
        let target = gateway_api_to_backend_path(request.uri())?;
        let keepalive = !is_resumable_request(&request);
        self.retarget(&mut request, &target, true);
        self.send(request, keepalive).await
    }

    async fn forward_ui_request(&self, mut request: Request) -> Result<Response, StatusCode> {
        let target = ui_target_path_and_query(request.uri());
        let keepalive = !is_resumable_request(&request);
        // no api-key injection: the backend's own auth middleware decides what is
        // public, and a client-supplied bearer (the ui-user's apikey) is forwarded
        // as-is because gateway- and backend-key are the same by construction
        self.retarget(&mut request, &target, false);
        self.send(request, keepalive).await
    }

    async fn post_chat_completions_json(
        &self,
        client_headers: HeaderMap,
        payload: String,
    ) -> Result<Response, StatusCode> {
        trace!("entered post_chat_completions_json");
        self.post_json(
            &format!("/{LLAMACPP_API_BASE_PATH}/chat/completions"),
            &client_headers,
            payload,
        )
        .await
    }

    async fn post_embedding_json(
        &self,
        client_headers: HeaderMap,
        payload: String,
    ) -> Result<Response, StatusCode> {
        self.post_json(
            &format!("/{LLAMACPP_API_BASE_PATH}/embeddings"),
            &client_headers,
            payload,
        )
        .await
    }
}
#[cfg(test)]
mod path_mapping_tests {
    use super::*;

    fn uri(s: &str) -> Uri {
        Uri::try_from(s).expect("test uri must be valid")
    }

    #[test]
    fn gateway_api_path_is_stripped_to_backend_v1() {
        // /api/v1 mount -> backend /v1 (W8a: no double v1 prefix)
        assert_eq!(
            gateway_api_to_backend_path(&uri("/api/v1/tokenize")).unwrap(),
            "/v1/tokenize"
        );
        assert_eq!(
            gateway_api_to_backend_path(&uri("/api/v1/chat/completions")).unwrap(),
            "/v1/chat/completions"
        );
        // query must survive
        assert_eq!(
            gateway_api_to_backend_path(&uri("/api/v1/props?autoload=false")).unwrap(),
            "/v1/props?autoload=false"
        );
        // /api/{n}/v1 mount -> backend /v1 (W8b: parallel param variant)
        assert_eq!(
            gateway_api_to_backend_path(&uri("/api/1/v1/models")).unwrap(),
            "/v1/models"
        );
        assert_eq!(
            gateway_api_to_backend_path(&uri("/api/4/v1/tokenize?q=1")).unwrap(),
            "/v1/tokenize?q=1"
        );
    }

    #[test]
    fn non_api_paths_are_not_mapped() {
        assert!(gateway_api_to_backend_path(&uri("/chat/props")).is_err());
        assert!(gateway_api_to_backend_path(&uri("/v1/models")).is_err());
        // a non-numeric segment between /api/ and /v1 is not the parallel mount
        assert!(gateway_api_to_backend_path(&uri("/api/xx/v1/models")).is_err());
    }

    #[test]
    fn ui_mount_prefix_is_stripped_but_root_absolute_paths_pass_through() {
        // /chat document -> backend root
        assert_eq!(ui_target_path_and_query(&uri("/chat/")), "/");
        assert_eq!(ui_target_path_and_query(&uri("/chat")), "/");
        assert_eq!(ui_target_path_and_query(&uri("/chat?x=1")), "/?x=1");
        // sub-resources under the mount -> stripped
        assert_eq!(
            ui_target_path_and_query(&uri("/chat/_app/immutable/bundle.x.js")),
            "/_app/immutable/bundle.x.js"
        );
        assert_eq!(
            ui_target_path_and_query(&uri("/chat/props?a=1")),
            "/props?a=1"
        );
        // root-absolute ui endpoints (no /chat prefix) pass through unchanged
        assert_eq!(ui_target_path_and_query(&uri("/tools")), "/tools");
        assert_eq!(
            ui_target_path_and_query(&uri("/models/load")),
            "/models/load"
        );
        assert_eq!(ui_target_path_and_query(&uri("/cors-proxy")), "/cors-proxy");
    }
}

#[cfg(test)]
mod keepalive_tests {
    use super::*;
    use axum::body::Body;
    use futures::StreamExt;

    fn headers(content_type: &str, encoded: bool) -> axum::http::HeaderMap {
        let mut builder = axum::http::HeaderMap::new();
        builder.insert(CONTENT_TYPE, content_type.parse().unwrap());
        if encoded {
            builder.insert(CONTENT_ENCODING, "gzip".parse().unwrap());
        }
        builder
    }

    #[test]
    fn only_plain_sse_bodies_are_keepalive_candidates() {
        assert!(is_keepalive_candidate(&headers("text/event-stream", false)));
        assert!(is_keepalive_candidate(&headers(
            "text/event-stream; charset=utf-8",
            false
        )));
        // never interleave into encoded streams or non-SSE bodies
        assert!(!is_keepalive_candidate(&headers("text/event-stream", true)));
        assert!(!is_keepalive_candidate(&headers("application/json", false)));
        assert!(!is_keepalive_candidate(&headers("text/html", false)));
    }

    #[tokio::test]
    async fn keepalive_bytes_are_emitted_on_an_idle_sse_body() {
        // a body that never yields data but stays open (like a thinking model)
        let idle = Body::from_stream(futures::stream::pending::<Result<Bytes, std::io::Error>>());
        let mut stream = std::pin::pin!(keepalive_stream_with(idle, Duration::from_millis(20)));

        // first item should be a keepalive comment (after one interval), not data
        let first = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("expected a keepalive within 2s")
            .expect("stream must yield an item")
            .expect("keepalive must be Ok");
        assert_eq!(first.as_ref(), SSE_KEEPALIVE_COMMENT);

        // and it keeps ticking while idle
        let second = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("expected a second keepalive")
            .unwrap()
            .unwrap();
        assert_eq!(second.as_ref(), SSE_KEEPALIVE_COMMENT);
    }

    #[tokio::test]
    async fn real_data_is_forwarded_and_resets_the_idle_timer() {
        // one immediate chunk, then idle -> we should see the chunk first, then a keepalive
        let body = Body::from_stream(
            futures::stream::once(async {
                Ok::<_, std::io::Error>(Bytes::from_static(b"data: hi\n\n"))
            })
            .chain(futures::stream::pending::<Result<Bytes, std::io::Error>>()),
        );

        let mut stream = std::pin::pin!(keepalive_stream_with(body, Duration::from_millis(30)));

        let first = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(first.as_ref(), b"data: hi\n\n");

        let second = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("expected a keepalive after idle")
            .unwrap()
            .unwrap();
        assert_eq!(second.as_ref(), SSE_KEEPALIVE_COMMENT);
    }
}

#[cfg(test)]
mod header_forwarding_tests {
    use super::*;

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                HeaderName::from_static(name),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        map
    }

    #[test]
    fn hop_by_hop_and_rewritten_headers_are_dropped() {
        let client = headers(&[
            ("host", "gateway.example"),
            ("connection", "keep-alive"),
            ("transfer-encoding", "chunked"),
            ("te", "trailers"),
            ("trailer", "x-checksum"),
            ("upgrade", "h2c"),
            ("content-length", "123"),
            ("content-type", "application/json"),
            ("accept-encoding", "gzip, br"),
            // these must survive
            ("x-conversation-id", "conv-1::model"),
            ("authorization", "Bearer user-key"),
            ("accept", "text/event-stream"),
            ("x-custom", "keep-me"),
        ]);

        let forwarded: Vec<String> = forwarded_headers(&client)
            .map(|(name, _)| name.as_str().to_string())
            .collect();

        for dropped in [
            "host",
            "connection",
            "transfer-encoding",
            "te",
            "trailer",
            "upgrade",
            "content-length",
            "content-type",
            "accept-encoding",
        ] {
            assert!(
                !forwarded.contains(&dropped.to_string()),
                "'{dropped}' must not be forwarded: {forwarded:?}"
            );
        }
        for kept in ["x-conversation-id", "authorization", "accept", "x-custom"] {
            assert!(
                forwarded.contains(&kept.to_string()),
                "'{kept}' must be forwarded: {forwarded:?}"
            );
        }
    }

    #[test]
    fn resumable_requests_are_recognised_by_header_and_by_conv_id_query() {
        let by_header = Request::post("/v1/chat/completions")
            .header("x-conversation-id", "conv-1::model")
            .body(Body::empty())
            .unwrap();
        assert!(is_resumable_request(&by_header));

        let by_query = Request::get("/v1/stream?conv_id=conv-1%3A%3Amodel&from=42")
            .body(Body::empty())
            .unwrap();
        assert!(is_resumable_request(&by_query));

        let plain_post = Request::post("/v1/chat/completions")
            .body(Body::empty())
            .unwrap();
        assert!(!is_resumable_request(&plain_post));

        let empty_header = Request::post("/v1/chat/completions")
            .header("x-conversation-id", "")
            .body(Body::empty())
            .unwrap();
        assert!(!is_resumable_request(&empty_header));

        // an unrelated query parameter must not disable keepalives
        let other_query = Request::get("/props?autoload=false")
            .body(Body::empty())
            .unwrap();
        assert!(!is_resumable_request(&other_query));
    }

    #[test]
    fn empty_conversation_id_header_does_not_disable_keepalives() {
        let headers = headers(&[("x-conversation-id", "")]);
        assert!(!headers
            .get("x-conversation-id")
            .is_some_and(|value| !value.is_empty()));
    }
}
