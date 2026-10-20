//! Integration tests for the chat-ui passthrough routing, driven entirely with
//! mocks (zero RAM, no llama-server processes) via `tower::ServiceExt::oneshot`.
//!
//! Some tests use a loopback mock-upstream on an OS-assigned *ephemeral* port
//! to prove path-mapping and SSE-keepalive behavior over real sockets. The
//! production ports (8443/11434/11440/11441) are never touched.
#![allow(dead_code)]

use async_openai::types::{chat::CreateChatCompletionRequest, embeddings::CreateEmbeddingRequest};
use async_trait::async_trait;
use axum::body::Body;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::response::Response;
use futures::StreamExt;
use gw_server::application;
use gw_server::domain::ports::{
    ModelManagerServiceInPort, ModelsServiceInPort, OpenAiClientOutPort,
    OpenAiRequestForwardPServiceInPort, StableDiffusionServiceInPort,
};
use gw_server::infrastructure::adapter::LocalLlamaCppClientAdapter;
use gw_server::model::{ApplicationConfig, SecurityConfig};
use http_body_util::BodyExt;
use inference_backends::{LlamaCppProcessState, LlamaCppRunConfig};
use serde_json::json;
use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use staticmodelconfig::ModelList;
use std::time::Duration;
use tower::ServiceExt;

const TEST_KEY: &str = "test-apikey-123";

// ---------------------------------------------------------------------------
// mocks
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Captured {
    method: String,
    path_and_query: String,
    authorization: Option<String>,
    body: String,
}

#[derive(Default)]
struct MockClientInner {
    captured: Vec<Captured>,
}

/// records every request the router sends to "the backend" and answers with a
/// canned 200 text/plain "mocked-backend"
#[derive(Default)]
struct MockClient {
    inner: Arc<Mutex<MockClientInner>>,
}

impl MockClient {
    fn captures(&self) -> Vec<Captured> {
        self.inner.lock().unwrap().captured.clone()
    }

    fn record(&self, captured: Captured) {
        self.inner.lock().unwrap().captured.push(captured);
    }

    fn canned() -> Response {
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/plain")
            .body(Body::from("mocked-backend"))
            .unwrap()
    }
}

async fn capture(request: Request, prefix: &str) -> Captured {
    let method = request.method().to_string();
    let path_and_query = request
        .uri()
        .path_and_query()
        .map(|v| v.as_str())
        .unwrap_or("/")
        .to_string();
    let authorization = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let body = request
        .into_body()
        .collect()
        .await
        .map(|b| String::from_utf8_lossy(&b.to_bytes()).to_string())
        .unwrap_or_default();
    Captured {
        method: format!("{prefix}{method}"),
        path_and_query,
        authorization,
        body,
    }
}

#[async_trait]
impl OpenAiClientOutPort for MockClient {
    async fn post_chat_completions(
        &self,
        payload: CreateChatCompletionRequest,
    ) -> Result<Response, StatusCode> {
        self.record(Captured {
            method: "PIPELINE".into(),
            path_and_query: "/v1/chat/completions".into(),
            authorization: None,
            body: serde_json::to_string(&payload).unwrap(),
        });
        Ok(Self::canned())
    }

    async fn post_embedding(
        &self,
        payload: CreateEmbeddingRequest,
    ) -> Result<Response, StatusCode> {
        self.record(Captured {
            method: "PIPELINE".into(),
            path_and_query: "/v1/embeddings".into(),
            authorization: None,
            body: serde_json::to_string(&payload).unwrap(),
        });
        Ok(Self::canned())
    }

    async fn forward_api_request(&self, request: Request) -> Result<Response, StatusCode> {
        let captured = capture(request, "API ").await;
        self.record(captured);
        Ok(Self::canned())
    }

    async fn forward_ui_request(&self, request: Request) -> Result<Response, StatusCode> {
        let captured = capture(request, "UI ").await;
        self.record(captured);
        Ok(Self::canned())
    }
}

/// records model-ensure calls; never spawns anything
#[derive(Default)]
struct MockModelsService {
    ensure_any_calls: Arc<Mutex<Vec<String>>>,
    ensure_llm_calls: Arc<Mutex<Vec<String>>>,
    ensure_embedding_calls: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl ModelsServiceInPort for MockModelsService {
    async fn ensure_any_languagemodel_is_served(
        &self,
        default_model_alias: &str,
        _timeout: Duration,
    ) -> Result<(), ()> {
        self.ensure_any_calls
            .lock()
            .unwrap()
            .push(default_model_alias.to_string());
        Ok(())
    }
    async fn ensure_requested_languagemodel_is_served(
        &self,
        requested_model_variant: &str,
        _timeout: Duration,
    ) -> Result<(), ()> {
        self.ensure_llm_calls
            .lock()
            .unwrap()
            .push(requested_model_variant.to_string());
        Ok(())
    }
    async fn ensure_requested_embeddingmodel_is_served(
        &self,
        requested_model_variant: &str,
        _timeout: Duration,
    ) -> Result<(), ()> {
        self.ensure_embedding_calls
            .lock()
            .unwrap()
            .push(requested_model_variant.to_string());
        Ok(())
    }
    fn get_models(&self) -> Arc<ModelList> {
        Arc::new(ModelList::default())
    }
    fn get_model_names(&self) -> String {
        "mock-alias".into()
    }
    fn get_default_languagemodel_alias(&self) -> String {
        "mock-default-llm".into()
    }
    fn get_default_embeddingmodel_alias(&self) -> String {
        "mock-default-embed".into()
    }
    async fn get_running_languagemodel_alias(&self) -> Option<String> {
        Some("mock-running-llm".into())
    }
    fn set_parallel_backend_requests(&self, _parallel_backend_requests: u8) {}
}

#[derive(Default)]
struct MockModelManagerService;

#[async_trait]
impl ModelManagerServiceInPort for MockModelManagerService {
    async fn get_llamacpp_state(&self) -> LlamaCppProcessState {
        managed_process::ProcessState::Stopped
    }
    async fn stop_llamacpp_process(&self) {}
    async fn start_llamacpp_process(&self, _config: LlamaCppRunConfig) -> LlamaCppProcessState {
        managed_process::ProcessState::Stopped
    }
}

#[derive(Default)]
struct MockStableDiffusionService;

#[async_trait]
impl StableDiffusionServiceInPort for MockStableDiffusionService {
    async fn process_prompt(
        &self,
        _sd_config: &str,
        _prompt_dto: gw_server::application::model::StableDiffusionPromptDto,
    ) -> Result<Response, StatusCode> {
        Ok(Response::default())
    }
    async fn list_apis(&self) -> Result<serde_json::Value, StatusCode> {
        Ok(json!([]))
    }
    async fn abort_all(&self) {}
}

struct MockSecurityConfig {
    apikey: Option<String>,
}

impl SecurityConfig for MockSecurityConfig {
    fn get_apikey(&self) -> Option<Cow<'_, str>> {
        self.apikey.as_deref().map(Cow::Borrowed)
    }
}

struct TestConfig {
    forward_service: Arc<dyn OpenAiRequestForwardPServiceInPort>,
    model_manager: Arc<dyn ModelManagerServiceInPort>,
    models: Arc<dyn ModelsServiceInPort>,
    stable_diffusion: Arc<dyn StableDiffusionServiceInPort>,
    skip_chatui_model_ensure: bool,
}

impl ApplicationConfig for TestConfig {
    fn openai_chat_completions_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort> {
        self.forward_service.clone()
    }
    fn openai_embeddings_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort> {
        self.forward_service.clone()
    }
    fn languagemodelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort> {
        self.model_manager.clone()
    }
    fn embeddingmodelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort> {
        self.model_manager.clone()
    }
    fn models_service(&self) -> Arc<dyn ModelsServiceInPort> {
        self.models.clone()
    }
    fn stable_diffusion_service(&self) -> Arc<dyn StableDiffusionServiceInPort> {
        self.stable_diffusion.clone()
    }
    fn skip_chatui_model_ensure(&self) -> bool {
        self.skip_chatui_model_ensure
    }
}

struct TestApp {
    router: axum::routing::Router,
    client_captures: Arc<Mutex<MockClientInner>>,
    ensure_any_calls: Arc<Mutex<Vec<String>>>,
    ensure_llm_calls: Arc<Mutex<Vec<String>>>,
    ensure_embedding_calls: Arc<Mutex<Vec<String>>>,
}

impl TestApp {
    /// mock-backend router: `skip_chatui_model_ensure` controls the new hook
    fn new(apikey: Option<&str>, skip_chatui_model_ensure: bool) -> Self {
        let client = Arc::new(MockClient::default());
        let captures = client.inner.clone();
        let models = Arc::new(MockModelsService::default());
        let ensure_any_calls = models.ensure_any_calls.clone();
        let ensure_llm_calls = models.ensure_llm_calls.clone();
        let ensure_embedding_calls = models.ensure_embedding_calls.clone();
        let forward_service: Arc<dyn OpenAiRequestForwardPServiceInPort> =
            gw_server::domain::service::OpenAiClientRequestForwardService::create_service(client);
        let config: Arc<dyn ApplicationConfig> = Arc::new(TestConfig {
            forward_service,
            model_manager: Arc::new(MockModelManagerService),
            models,
            stable_diffusion: Arc::new(MockStableDiffusionService),
            skip_chatui_model_ensure,
        });
        let security: Arc<dyn SecurityConfig> = Arc::new(MockSecurityConfig {
            apikey: apikey.map(str::to_string),
        });

        TestApp {
            // the real router under test
            router: application::open_ai_router(config, security),
            client_captures: captures,
            ensure_any_calls,
            ensure_llm_calls,
            ensure_embedding_calls,
        }
    }

    async fn request(&self, req: Request) -> Response {
        self.router.clone().oneshot(req).await.unwrap()
    }

    async fn get(&self, uri: &str) -> Response {
        self.request(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
    }

    fn captures(&self) -> Vec<Captured> {
        self.client_captures.lock().unwrap().captured.clone()
    }
}

async fn body_to_string(response: &mut Response) -> String {
    let bytes = response.body_mut().collect().await.unwrap().to_bytes();
    String::from_utf8_lossy(&bytes).to_string()
}

// ---------------------------------------------------------------------------
// chat-ui routing tests (pure mocks, oneshot — zero sockets)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_redirects_to_slash_with_308() {
    let app = TestApp::new(Some(TEST_KEY), true);
    let mut response = app.get("/chat").await;
    assert_eq!(response.status(), StatusCode::PERMANENT_REDIRECT);
    assert_eq!(
        response
            .headers()
            .get(axum::http::header::LOCATION)
            .and_then(|v| v.to_str().ok()),
        Some("/chat/")
    );
    // the redirect must not hit the backend at all
    assert!(app.captures().is_empty());
    let _ = &mut response;
}

#[tokio::test]
async fn chat_document_is_served_as_a_document() {
    let app = TestApp::new(Some(TEST_KEY), true);
    let mut response = app.get("/chat/").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_to_string(&mut response).await, "mocked-backend");

    let captures = app.captures();
    assert_eq!(captures.len(), 1);
    // NOTE: the mock client records the request BEFORE retargeting, so the
    // gateway-side path is visible here. Backend-path mapping (prefix
    // stripping) is asserted over a real socket in the e2e tests below.
    assert_eq!(captures[0].path_and_query, "/chat/");
    assert_eq!(captures[0].method, "UI GET");
    // skip-flag honored: no model-ensure happened
    assert!(app.ensure_any_calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn chat_document_ensures_a_language_model_by_default() {
    let app = TestApp::new(Some(TEST_KEY), false);
    let response = app.get("/chat/").await;
    assert_eq!(response.status(), StatusCode::OK);
    // default behaviour: the gateway made sure *some* model is served
    assert_eq!(
        app.ensure_any_calls.lock().unwrap().as_slice(),
        ["mock-default-llm".to_string()]
    );
}

#[tokio::test]
async fn chat_writes_require_bearer() {
    // with no apikey configured everything is open
    let open = TestApp::new(None, true);
    let response = open
        .request(
            Request::builder()
                .method("POST")
                .uri("/chat/props")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);

    let app = TestApp::new(Some(TEST_KEY), true);

    // POST (a write) without bearer -> 401, backend never called
    let response = app
        .request(
            Request::builder()
                .method("POST")
                .uri("/chat/props")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(app.captures().is_empty());

    // wrong bearer -> 401
    let response = app
        .request(
            Request::builder()
                .method("POST")
                .uri("/chat/props")
                .header("authorization", "Bearer wrong")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(app.captures().is_empty());

    // correct bearer -> forwarded, client bearer passed through untouched
    let mut response = app
        .request(
            Request::builder()
                .method("POST")
                .uri("/chat/props")
                .header("authorization", format!("Bearer {TEST_KEY}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_to_string(&mut response).await, "mocked-backend");
    let captures = app.captures();
    assert_eq!(captures[0].path_and_query, "/chat/props");
    // the client bearer is forwarded verbatim (gateway does not rewrite it)
    assert_eq!(
        captures[0].authorization.as_deref(),
        Some(format!("Bearer {TEST_KEY}").as_str())
    );
}

#[tokio::test]
async fn ui_reads_are_public_and_forwarded_without_key_injection() {
    let app = TestApp::new(Some(TEST_KEY), true);
    // GET without any authorization header stays public (backend decides)
    let mut response = app.get("/chat/_app/immutable/chunks/bundle.js").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_to_string(&mut response).await, "mocked-backend");
    let captures = app.captures();
    assert_eq!(captures[0].path_and_query, "/chat/_app/immutable/chunks/bundle.js");
    // no key injection on ui reads — the client bearer (here: none) is forwarded as-is
    assert_eq!(captures[0].authorization, None);
}

#[tokio::test]
async fn post_chat_v1_chat_completions_re_enters_the_pipeline() {
    let app = TestApp::new(Some(TEST_KEY), true);
    let body = json!({
        "model": "mock-alias",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let mut response = app
        .request(
            Request::builder()
                .method("POST")
                .uri("/chat/v1/chat/completions")
                .header("authorization", format!("Bearer {TEST_KEY}"))
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_to_string(&mut response).await, "mocked-backend");

    let captures = app.captures();
    assert_eq!(captures.len(), 1);
    // it went through the gateway pipeline (model-ensure + typed request),
    // NOT through a plain ui-forward
    assert_eq!(captures[0].method, "PIPELINE");
    assert!(captures[0].body.contains("mock-alias"));
    assert_eq!(
        app.ensure_llm_calls.lock().unwrap().as_slice(),
        ["mock-alias".to_string()]
    );

    // a write without bearer must be rejected before the pipeline runs
    let app2 = TestApp::new(Some(TEST_KEY), true);
    let response = app2
        .request(
            Request::builder()
                .method("POST")
                .uri("/chat/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(app2.captures().is_empty());
}

#[tokio::test]
async fn post_chat_v1_embeddings_re_enters_the_pipeline() {
    let app = TestApp::new(Some(TEST_KEY), true);
    let body = json!({"model": "mock-embed", "input": "hello"});
    let response = app
        .request(
            Request::builder()
                .method("POST")
                .uri("/chat/v1/embeddings")
                .header("authorization", format!("Bearer {TEST_KEY}"))
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let captures = app.captures();
    assert_eq!(captures[0].method, "PIPELINE");
    assert_eq!(captures[0].path_and_query, "/v1/embeddings");
    assert_eq!(
        app.ensure_embedding_calls.lock().unwrap().as_slice(),
        ["mock-embed".to_string()]
    );
}

#[tokio::test]
async fn model_command_sniffing_still_works_via_ui_path() {
    let app = TestApp::new(Some(TEST_KEY), true);
    let body = json!({
        "model": "mock-alias",
        "messages": [{"role": "user", "content": "/model other-alias switch please"}]
    });
    let response = app
        .request(
            Request::builder()
                .method("POST")
                .uri("/chat/v1/chat/completions")
                .header("authorization", format!("Bearer {TEST_KEY}"))
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    // the /model command must have been extracted and used for model-ensure
    assert_eq!(
        app.ensure_llm_calls.lock().unwrap().as_slice(),
        ["other-alias".to_string()]
    );
    let captures = app.captures();
    // ...and removed from the prompt that reaches the backend
    assert!(!captures[0].body.contains("/model"));
    assert!(captures[0].body.contains("switch please"));
}

#[tokio::test]
async fn root_aliases_are_secured_and_forwarded() {
    let app = TestApp::new(Some(TEST_KEY), true);

    for uri in ["/tools", "/models", "/models/load", "/cors-proxy"] {
        let response = app.get(uri).await;
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "{uri} must require the apikey"
        );

        let mut response = app
            .request(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("authorization", format!("Bearer {TEST_KEY}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
        assert_eq!(body_to_string(&mut response).await, "mocked-backend");
    }

    let captures = app.captures();
    // root-absolute ui endpoints are forwarded (one capture per request;
    // path mapping itself is proven over a real socket in the e2e tests)
    let mapped: Vec<String> = captures
        .iter()
        .map(|c| format!("{} {}", c.method, c.path_and_query))
        .collect();
    assert_eq!(
        mapped,
        vec![
            "UI POST /tools",
            "UI POST /models",
            "UI POST /models/load",
            "UI POST /cors-proxy",
        ]
    );
}

#[tokio::test]
async fn root_health_endpoints_are_open() {
    let app = TestApp::new(Some(TEST_KEY), true);
    for uri in ["/health", "/v1/health"] {
        let response = app.get(uri).await;
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
    }
    // health endpoints are forwarded to the backend WITHOUT a key (public reads)
    let captures = app.captures();
    assert_eq!(captures.len(), 2);
    assert!(captures.iter().all(|c| c.authorization.is_none()));

    // /v1/models is answered by the gateway itself (its own model list)
    let mut response = app.get("/v1/models").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_string(&mut response).await;
    assert!(body.contains("\"object\":\"list\""), "got: {body}");
    // still only the two health forwards were captured
    assert_eq!(app.captures().len(), 2);
}

#[tokio::test]
async fn api_models_routes_are_unchanged_and_open() {
    let app = TestApp::new(Some(TEST_KEY), true);
    for uri in ["/api/v1/models", "/api/2/v1/models"] {
        let response = app.get(uri).await;
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
    }
    // answered by the gateway itself — backend never called
    assert!(app.captures().is_empty());
}

#[tokio::test]
async fn api_completions_still_work_and_are_secured() {
    let app = TestApp::new(Some(TEST_KEY), true);
    let body = json!({
        "model": "mock-alias",
        "messages": [{"role": "user", "content": "hi"}]
    });

    // without bearer -> 401 (unchanged openness of the product API)
    let response = app
        .request(
            Request::builder()
                .method("POST")
                .uri("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    // with bearer -> pipeline
    let response = app
        .request(
            Request::builder()
                .method("POST")
                .uri("/api/v1/chat/completions")
                .header("authorization", format!("Bearer {TEST_KEY}"))
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(app.captures()[0].method, "PIPELINE");
}

// ---------------------------------------------------------------------------
// end-to-end forwarding against a loopback mock-upstream (real sockets,
// OS-assigned ephemeral ports only) — proves path mapping and keepalive
// through the REAL LocalLlamaCppClientAdapter + hyper client.
//
// The upstream is a minimal hand-rolled HTTP/1.1 responder on raw tokio TCP
// sockets: no extra dependencies, full control over streamed bodies.
// ---------------------------------------------------------------------------

mod upstream {
    use super::*;
    use std::net::SocketAddr;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    pub struct MockUpstream {
        pub addr: SocketAddr,
        requests: Arc<Mutex<Vec<SeenRequest>>>,
    }

    type SeenRequest = (String, String, Option<String>);

    /// what the mock answers: status, headers, and either fixed bytes or a
    /// never-ending SSE body (first chunk, then silence)
    pub enum Answer {
        Fixed(u16, Vec<(String, String)>, Vec<u8>),
        IdleSse {
            first_chunk: &'static [u8],
        },
    }

    impl MockUpstream {
        /// binds 127.0.0.1:0 (OS-assigned port — can never collide with the
        /// production ports) and answers each request via `answer_for`
        pub async fn start<F>(answer_for: F) -> Self
        where
            F: Fn(&str) -> Answer + Send + Sync + 'static,
        {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let requests: Arc<Mutex<Vec<SeenRequest>>> =
                Arc::new(Mutex::new(Vec::new()));
            let answer_for = Arc::new(answer_for);

            let listener_requests = requests.clone();
            tokio::spawn(async move {
                loop {
                    let Ok((stream, _)) = listener.accept().await else {
                        continue;
                    };
                    let requests = listener_requests.clone();
                    let answer_for = answer_for.clone();                    tokio::spawn(async move {
                        let (mut reader, mut writer) = stream.into_split();
                        // read request head (headers); bodies of these tests are tiny
                        let mut buf = vec![0u8; 8192];
                        let mut head = String::new();
                        loop {
                            let Ok(Ok(n)) = tokio::time::timeout(
                                Duration::from_secs(5),
                                reader.read(&mut buf),
                            )
                            .await
                            else {
                                return;
                            };
                            if n == 0 {
                                break;
                            }
                            head.push_str(&String::from_utf8_lossy(&buf[..n]));
                            if head.contains("\r\n\r\n") {
                                break;
                            }
                        }
                        let mut lines = head.lines();
                        let Some(request_line) = lines.next() else { return };
                        let mut parts = request_line.split_whitespace();
                        let method = parts.next().unwrap_or("?").to_string();
                        let path = parts.next().unwrap_or("/").to_string();
                        let auth = head
                            .lines()
                            .find(|l| l.to_ascii_lowercase().starts_with("authorization:"))
                            .map(|l| l.split_once(':').map(|x| x.1).unwrap_or("").trim().to_string());
                        requests.lock().unwrap().push((method, path.clone(), auth));

                        match answer_for(&path) {
                            Answer::Fixed(status, headers, body) => {
                                let mut head_out =
                                    format!("HTTP/1.1 {status} OK\r\ncontent-length: {}\r\n", body.len());
                                for (k, v) in headers {
                                    head_out.push_str(&format!("{k}: {v}\r\n"));
                                }
                                head_out.push_str("connection: close\r\n\r\n");
                                let _ = writer.write_all(head_out.as_bytes()).await;
                                let _ = writer.write_all(&body).await;
                            }
                            Answer::IdleSse { first_chunk } => {
                                // no content-length, no chunked: body ends at
                                // connection close — hyper handles that fine and
                                // we can keep the (idle) stream open indefinitely
                                let head_out = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\nconnection: keep-alive\r\n\r\n";
                                if writer.write_all(head_out.as_bytes()).await.is_err() {
                                    return;
                                }
                                let _ = writer.flush().await;
                                let _ = writer.write_all(first_chunk).await;
                                let _ = writer.flush().await;
                                // stay silent but connected — like a thinking model
                                tokio::time::sleep(Duration::from_secs(3600)).await;
                            }
                        }
                        let _ = writer.shutdown().await;
                    });
                }
            });

            MockUpstream { addr, requests }
        }

        pub fn requests(&self) -> Vec<SeenRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    /// a router wired to the REAL adapter pointed at `upstream`
    pub fn router_to(upstream_port: u16, apikey: Option<&str>) -> axum::routing::Router {
        let security: Arc<dyn SecurityConfig> = Arc::new(MockSecurityConfig {
            apikey: apikey.map(str::to_string),
        });
        let client = LocalLlamaCppClientAdapter::create_adapter(upstream_port, security.clone());
        let service: Arc<dyn OpenAiRequestForwardPServiceInPort> =
            gw_server::domain::service::OpenAiClientRequestForwardService::create_service(client);

        struct Cfg(Arc<dyn OpenAiRequestForwardPServiceInPort>);
        impl ApplicationConfig for Cfg {
            fn openai_chat_completions_service(
                &self,
            ) -> Arc<dyn OpenAiRequestForwardPServiceInPort> {
                self.0.clone()
            }
            fn openai_embeddings_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort> {
                self.0.clone()
            }
            fn languagemodelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort> {
                Arc::new(MockModelManagerService)
            }
            fn embeddingmodelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort> {
                Arc::new(MockModelManagerService)
            }
            fn models_service(&self) -> Arc<dyn ModelsServiceInPort> {
                // no-op mock: never spawns anything
                Arc::new(MockModelsService::default())
            }
            fn stable_diffusion_service(&self) -> Arc<dyn StableDiffusionServiceInPort> {
                Arc::new(MockStableDiffusionService)
            }
            // fixture: never spawn real processes
            fn skip_chatui_model_ensure(&self) -> bool {
                true
            }
        }

        application::open_ai_router(Arc::new(Cfg(service)), security)
    }
}

#[tokio::test]
async fn ui_asset_path_is_mapped_on_a_real_socket() {
    use upstream::Answer;
    let upstream = upstream::MockUpstream::start(|path| match path {
        "/_app/x" => Answer::Fixed(
            200,
            vec![("content-type".into(), "text/plain".into())],
            b"asset-bytes".to_vec(),
        ),
        _ => Answer::Fixed(404, vec![], b"not-found".to_vec()),
    })
    .await;

    let router = upstream::router_to(upstream.addr.port(), Some(TEST_KEY));
    let mut response = router
        .oneshot(
            Request::builder()
                .uri("/chat/_app/x")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_to_string(&mut response).await, "asset-bytes");
    let requests = upstream.requests();
    assert_eq!(requests.len(), 1);
    // the /chat mount prefix was stripped on the wire
    assert_eq!(requests[0].1, "/_app/x");
}

#[tokio::test]
async fn api_fallback_path_is_mapped_on_a_real_socket() {
    use upstream::Answer;
    let upstream = upstream::MockUpstream::start(|path| match path {
        "/v1/tokenize" => Answer::Fixed(
            200,
            vec![("content-type".into(), "application/json".into())],
            br#"{"ok":true}"#.to_vec(),
        ),
        _ => Answer::Fixed(404, vec![], format!("unexpected path {path}").into_bytes()),
    })
    .await;

    let router = upstream::router_to(upstream.addr.port(), Some(TEST_KEY));

    // /api/v1/tokenize -> backend /v1/tokenize (W8a: exactly one v1)
    let mut response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/tokenize")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {TEST_KEY}"))
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_to_string(&mut response).await, r#"{"ok":true}"#);

    let requests = upstream.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].1, "/v1/tokenize");
    // the gateway injects its key for api forwards
    assert_eq!(
        requests[0].2.as_deref(),
        Some(format!("Bearer {TEST_KEY}").as_str())
    );

    // parallel-param variant (W8b: previously a guaranteed axum 500)
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/2/v1/tokenize")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {TEST_KEY}"))
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let requests = upstream.requests();
    assert_eq!(requests[1].1, "/v1/tokenize");
}

#[tokio::test]
async fn keepalive_bytes_appear_on_an_idle_sse_body_end_to_end() {
    use upstream::Answer;
    // upstream: SSE body that emits one event and then stays silent forever,
    // like a model thinking for 30 minutes
    let upstream = upstream::MockUpstream::start(|_path| Answer::IdleSse {
        first_chunk: b"data: first\n\n",
    })
    .await;

    // drive the real adapter through a real client socket
    let security: Arc<dyn SecurityConfig> = Arc::new(MockSecurityConfig { apikey: None });
    let client = LocalLlamaCppClientAdapter::create_adapter(upstream.addr.port(), security);
    let mut response = client
        .forward_ui_request(
            Request::builder()
                .uri(format!("http://localhost:{}/props", upstream.addr.port()))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // content-length must be gone (the body grows by keepalive comments)
    assert!(
        response
            .headers()
            .get(axum::http::header::CONTENT_LENGTH)
            .is_none(),
        "content-length must be removed from keepalive-wrapped SSE bodies"
    );

    let mut body = std::pin::pin!(response.body_mut().into_data_stream());

    // first: the real upstream event
    let first = tokio::time::timeout(Duration::from_secs(15), body.next())
        .await
        .expect("first chunk")
        .unwrap()
        .unwrap();
    assert_eq!(&first[..], b"data: first\n\n");

    // then, while upstream stays silent, keepalive comments must appear.
    // SSE_KEEPALIVE_INTERVAL is 10 s — allow two intervals.
    let second = tokio::time::timeout(Duration::from_secs(25), body.next())
        .await
        .expect("a keepalive comment must be emitted on an idle SSE stream")
        .unwrap()
        .unwrap();
    assert_eq!(&second[..], b": keepalive\n\n");

    // and it keeps ticking
    let third = tokio::time::timeout(Duration::from_secs(25), body.next())
        .await
        .expect("keepalives must repeat")
        .unwrap()
        .unwrap();
    assert_eq!(&third[..], b": keepalive\n\n");
}
