use crate::{domain::ports::OpenAiClientOutPort, model::SecurityConfig};
use async_openai::types::{chat::CreateChatCompletionRequest, embeddings::CreateEmbeddingRequest};
use async_trait::async_trait;
use axum::{
    body::{Body, Bytes},
    extract::Request,
    http::{
        HeaderValue, StatusCode, Uri, Version,
        header::{
            ACCEPT_ENCODING, AUTHORIZATION, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE,
            HOST as HOST_HEADER,
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
use tracing::{error, trace};

const LLAMACPP_HTTP_SCHEME: &str = "http";
const LLAMACPP_HOST: &str = "localhost";
const LLAMACPP_API_BASE_PATH: &str = "v1";

/// idle time after which a keepalive comment is injected into an SSE response
const SSE_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(10);
const SSE_KEEPALIVE_COMMENT: &[u8] = b": keepalive\n\n";

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
        format!("{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}", self.llamacpp_port)
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
    async fn send(&self, request: Request) -> Result<Response, StatusCode> {
        self.client
            .request(request)
            .await
            .map(transparent_response)
            .map_err(|e| {
                error!("error forwarding request to llama.cpp: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })
    }

    async fn post_json<T: serde::Serialize>(
        &self,
        target_path: &str,
        payload: &T,
    ) -> Result<Response, StatusCode> {
        let json_string = serde_json::to_string(payload).map_err(|e| {
            error!("error converting payload to json-string: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        let mut request = Request::post(format!("{}{target_path}", self.backend_base_uri()))
            .header(ACCEPT_ENCODING, HeaderValue::from_static("identity"))
            .body(Body::from(json_string))
            .map_err(|e| {
                error!("error building llama.cpp-request: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        self.retarget(&mut request, target_path, true);
        self.send(request).await
    }
}

/// transparently forwards an upstream response.
///
/// Status, headers and body are passed through unchanged. For SSE responses (and
/// only for those) an idle keepalive comment is interleaved, because upstream is
/// free to stay silent for many minutes while a model is thinking.
fn transparent_response(response: hyper::Response<hyper::body::Incoming>) -> Response {
    let (mut parts, body) = response.into_parts();

    if is_keepalive_candidate(&parts.headers) {
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
            path.strip_prefix("/api/")
                .and_then(|after_api| {
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
        self.retarget(&mut request, &target, true);
        self.send(request).await
    }

    async fn forward_ui_request(&self, mut request: Request) -> Result<Response, StatusCode> {
        let target = ui_target_path_and_query(request.uri());
        // no api-key injection: the backend's own auth middleware decides what is
        // public, and a client-supplied bearer (the ui-user's apikey) is forwarded
        // as-is because gateway- and backend-key are the same by construction
        self.retarget(&mut request, &target, false);
        self.send(request).await
    }

    async fn post_chat_completions(
        &self,
        payload: CreateChatCompletionRequest,
    ) -> Result<Response, StatusCode> {
        trace!("entered post_chat_completions");
        self.post_json(
            &format!("/{LLAMACPP_API_BASE_PATH}/chat/completions"),
            &payload,
        )
        .await
    }

    async fn post_embedding(
        &self,
        payload: CreateEmbeddingRequest,
    ) -> Result<Response, StatusCode> {
        self.post_json(&format!("/{LLAMACPP_API_BASE_PATH}/embeddings"), &payload)
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
        assert_eq!(ui_target_path_and_query(&uri("/chat/props?a=1")), "/props?a=1");
        // root-absolute ui endpoints (no /chat prefix) pass through unchanged
        assert_eq!(ui_target_path_and_query(&uri("/tools")), "/tools");
        assert_eq!(ui_target_path_and_query(&uri("/models/load")), "/models/load");
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
        let body = Body::from_stream(futures::stream::once(async {
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: hi\n\n"))
        })
        .chain(futures::stream::pending::<Result<Bytes, std::io::Error>>()));

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
