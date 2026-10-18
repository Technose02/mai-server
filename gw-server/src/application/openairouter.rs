use crate::{
    application::{
        middleware::{check_auth, check_bearer},
        model::{
            try_map_request_body_to_create_chat_completion_request,
            try_map_request_body_to_create_embedding_request,
        },
    },
    model::ApplicationConfig,
    model::SecurityConfig,
};
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
};
use axum::{
    Json,
    body::Body,
    extract::{Path, Query, Request, State},
    http::{Method, Response, StatusCode},
    response::{IntoResponse, Redirect},
    routing::{Router, any, get, post},
};
use futures_util::stream;
use staticmodelconfig::ModelList;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{error, info, trace, warn};

/// state shared by the chat-ui routes
#[derive(Clone)]
struct ChatUiState {
    config: Arc<dyn ApplicationConfig>,
    security_config: Arc<dyn SecurityConfig>,
}

pub fn create_router(
    config: Arc<dyn ApplicationConfig>,
    security_config: Arc<dyn SecurityConfig>,
) -> Router {
    let chat_ui_state = ChatUiState {
        config: config.clone(),
        security_config: security_config.clone(),
    };

    // --- openai-compatible api -------------------------------------------------
    //   paths and openness of these routes must not change: the api is the
    //   primary product of this server
    let api_open = Router::new()
        .route(
            "/api/{n_parallel}/v1/models",
            get(get_models_with_parallel_param),
        )
        .route("/api/v1/models", get(get_models))
        .with_state(config.clone());

    let api_secured = Router::new()
        .route(
            "/api/{n_parallel}/v1/chat/completions",
            post(post_completions_with_parallel_param),
        )
        .route("/api/v1/chat/completions", post(post_completions))
        .route(
            "/api/{n_parallel}/v1/embeddings",
            post(post_embeddings_with_parallel_param),
        )
        .route("/api/v1/embeddings", post(post_embeddings))
        .route(
            "/api/{n_parallel}/v1/{*path}",
            any(api_fallback_with_parallel_param),
        )
        .route("/api/v1/{*path}", any(api_fallback))
        .route("/api/v2/{*path}", any(|| async { StatusCode::NOT_FOUND }))
        .layer(axum::middleware::from_fn_with_state(
            security_config.clone(),
            check_auth,
        ))
        .with_state(config.clone());

    // --- chat-ui ---------------------------------------------------------------
    //   the upstream ui is re-presented under /chat. There is intentionally no
    //   route- or content-whitelist any more: reads are forwarded transparently
    //   (the backend's own auth middleware decides what is public), writes are
    //   secured, and endpoints the ui requests root-absolute are aliased below.

    let chat_ui = Router::new()
        .route("/chat", get(redirect_chat))
        .route("/chat/", get(chat_ui_document_handler))
        .route("/chat/{*path}", any(chat_ui_dispatch))
        .with_state(chat_ui_state);

    // root-absolute read-endpoints used by the upstream ui
    let chat_ui_open_aliases = Router::new()
        .route("/v1/models", get(get_models))
        .route("/health", get(forward_ui))
        .route("/v1/health", get(forward_ui))
        .with_state(config.clone());

    let chat_ui_secured_aliases = Router::new()
        .route("/models", any(forward_ui))
        .route("/models/{*path}", any(forward_ui))
        .route("/tools", any(forward_ui))
        .route("/cors-proxy", any(forward_ui))
        .layer(axum::middleware::from_fn_with_state(
            security_config,
            check_auth,
        ))
        .with_state(config);

    Router::new()
        .merge(api_open)
        .merge(api_secured)
        .merge(chat_ui)
        .merge(chat_ui_open_aliases)
        .merge(chat_ui_secured_aliases)
}

/// `/chat` must be redirected to `/chat/`: the ui is a single-page app whose
/// relative urls are resolved against the document url, so the trailing slash is
/// what keeps every request inside the prefix.
async fn redirect_chat() -> Redirect {
    Redirect::permanent("/chat/")
}

async fn chat_ui_dispatch(
    State(state): State<ChatUiState>,
    Path(path): Path<String>,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    let is_read = matches!(*request.method(), Method::GET | Method::HEAD);

    // reads stay public (upstream serves the ui and its assets without a key);
    // everything else needs the apikey, exactly like the rest of the gateway
    if !is_read {
        check_bearer(&state.security_config, request.headers())?;
    }

    match (is_read, path.as_str()) {
        // the document: make sure *some* language model is being served
        (true, "") => chat_ui_document(state, request).await,
        // requests issued by the ui itself are routed through the same pipeline
        // as the public api, so model-switching and the /model & /stop commands
        // keep working
        (false, "v1/chat/completions") => {
            post_completions(State(state.config.clone()), request).await
        }
        (false, "v1/embeddings") => post_embeddings(State(state.config.clone()), request).await,
        // anything else: forwarded transparently - unknown upstream endpoints
        // work by construction
        _ => {
            state
                .config
                .openai_chat_completions_service()
                .forward_ui_request(request)
                .await
        }
    }
}

async fn chat_ui_document_handler(
    State(state): State<ChatUiState>,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    chat_ui_document(state, request).await
}

async fn chat_ui_document(
    state: ChatUiState,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    let default_model_alias = state.config.models_service().get_default_languagemodel_alias();
    state
        .config
        .models_service()
        .ensure_any_languagemodel_is_served(&default_model_alias, Duration::from_mins(3))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state
        .config
        .openai_chat_completions_service()
        .forward_ui_request(request)
        .await
}

async fn forward_ui(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    application_config
        .openai_chat_completions_service()
        .forward_ui_request(request)
        .await
}

// MODELS
async fn get_models_with_parallel_param(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Path(n_parallel): Path<u8>,
    Query(query_map): Query<HashMap<String, String>>,
) -> Result<Response<Body>, StatusCode> {
    trace!("info: models-endpoint called for parallel={n_parallel}");
    get_models_impl(application_config, query_map).await
}

async fn get_models(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Query(query_map): Query<HashMap<String, String>>,
) -> Result<Response<Body>, StatusCode> {
    get_models_impl(application_config, query_map).await
}

async fn get_models_impl(
    application_config: Arc<dyn ApplicationConfig>,
    query_map: HashMap<String, String>,
) -> Result<Response<Body>, StatusCode> {
    if let Some(val) = query_map.get("names-only")
        && val == "true"
    {
        Ok((
            StatusCode::OK,
            application_config.models_service().get_model_names(),
        )
            .into_response())
    } else {
        Ok((
            StatusCode::OK,
            Json::<&ModelList>::from(application_config.models_service().get_models().as_ref()),
        )
            .into_response())
    }
}

// CHAT COMPLETIONS
async fn post_completions_with_parallel_param(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Path(n_parallel): Path<u8>,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    post_chat_completions_impl(application_config, Some(n_parallel), request).await
}

async fn post_completions(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    post_chat_completions_impl(application_config, None, request).await
}

async fn post_chat_completions_impl(
    application_config: Arc<dyn ApplicationConfig>,
    optional_parallel_backend_requests_to_set: Option<u8>,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    let mut chat_completions_request = try_map_request_body_to_create_chat_completion_request(
        request,
        application_config
            .models_service()
            .get_running_languagemodel_alias()
            .await
            .unwrap_or(
                application_config
                    .models_service()
                    .get_default_languagemodel_alias(),
            ),
    )
    .await?;

    trace!("request: {:#?}", chat_completions_request);

    if check_user_prompt_for_stop_llamacpp(&mut chat_completions_request) {
        use async_openai::types::chat::{
            ChatChoiceStream, ChatCompletionStreamResponseDelta,
            CreateChatCompletionStreamResponse, FinishReason, Role,
        };
        use axum::response::sse::{Event, Sse};
        use std::convert::Infallible;
        application_config
            .languagemodelmanager_service()
            .stop_llamacpp_process()
            .await;
        let response_json = serde_json::to_string(&CreateChatCompletionStreamResponse {
            id: "none".into(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    role: Some(Role::Assistant),
                    content: Some("No llama.cpp-processes running".into()),
                    tool_calls: None,
                    #[allow(deprecated)]
                    function_call: None,
                    refusal: None,
                },
                finish_reason: Some(FinishReason::Stop), // Wichtig für das Client-Ende
                logprobs: None,
            }],
            service_tier: None,
            created: 0,
            model: "none".into(),
            #[allow(deprecated)]
            system_fingerprint: None,
            object: "chat.completion.chunk".into(), // Wichtig für Kompatibilität
            usage: None,
        })
        .unwrap();

        let stream = stream::iter(vec![
            Ok::<_, Infallible>(Event::default().data(response_json)),
            Ok::<_, Infallible>(Event::default().data("[DONE]")), // Beendet den Stream beim Client
        ]);

        return Ok(Sse::new(stream).into_response());
    }

    let requested_model = extract_model_override_from_last_user_text_in_chat_completions_request(
        &mut chat_completions_request,
    )
    .unwrap_or(chat_completions_request.model.clone());

    application_config
        .stable_diffusion_service()
        .abort_all()
        .await;

    if let Some(parallel_backend_requests_to_set) = optional_parallel_backend_requests_to_set {
        application_config
            .models_service()
            .set_parallel_backend_requests(parallel_backend_requests_to_set);
    }

    application_config
        .models_service()
        .ensure_requested_languagemodel_is_served(&requested_model, Duration::from_mins(3))
        .await
        .map_err(|_| {
            error!("error serving requested model");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    application_config
        .openai_chat_completions_service()
        .process_chat_completions_request(chat_completions_request)
        .await
}

// EMBEDDINGS
async fn post_embeddings_with_parallel_param(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Path(n_parallel): Path<u8>,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    post_embeddings_impl(application_config, Some(n_parallel), request).await
}

async fn post_embeddings(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    post_embeddings_impl(application_config, None, request).await
}

async fn post_embeddings_impl(
    application_config: Arc<dyn ApplicationConfig>,
    optional_parallel_backend_requests_to_set: Option<u8>,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    let embedding_request = try_map_request_body_to_create_embedding_request(request).await?;

    let requested_model = embedding_request.model.clone();

    if let Some(parallel_backend_requests_to_set) = optional_parallel_backend_requests_to_set {
        application_config
            .models_service()
            .set_parallel_backend_requests(parallel_backend_requests_to_set);
    }

    application_config
        .models_service()
        .ensure_requested_embeddingmodel_is_served(&requested_model, Duration::from_mins(3))
        .await
        .map_err(|_| {
            error!("error serving requested embedding model");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    application_config
        .openai_embeddings_service()
        .process_embedding_request(embedding_request)
        .await
}

// FALLBACK
async fn api_fallback_with_parallel_param(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Path((n_parallel, api_path)): Path<(u8, String)>,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    api_fallback_impl(application_config, Some(n_parallel), api_path, request).await
}

async fn api_fallback(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Path(api_path): Path<String>,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    api_fallback_impl(application_config, None, api_path, request).await
}

async fn api_fallback_impl(
    application_config: Arc<dyn ApplicationConfig>,
    optional_parallel_backend_requests_to_set: Option<u8>,
    api_path: String,
    request: Request,
) -> Result<Response<Body>, StatusCode> {
    warn!("unexpected {}-request to {api_path}", request.method());

    if let Some(parallel_backend_requests_to_set) = optional_parallel_backend_requests_to_set {
        application_config
            .models_service()
            .set_parallel_backend_requests(parallel_backend_requests_to_set);
    }

    application_config
        .openai_chat_completions_service()
        .forward_api_request(request)
        .await
}

fn process_last_user_prompt_in_chat_completions(
    chat_completions_request: &mut CreateChatCompletionRequest,
    proc: impl Fn(&mut String) -> Option<String>,
) -> Option<String> {
    chat_completions_request
        .messages
        .iter_mut()
        .filter_map(|m| {
            if let ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                name: _,
                content: ChatCompletionRequestUserMessageContent::Text(text),
            }) = m
            {
                Some(text)
            } else {
                None
            }
        })
        .next_back()
        .and_then(proc)
}

fn extract_model_override_from_last_user_text_in_chat_completions_request(
    chat_completions_request: &mut CreateChatCompletionRequest,
) -> Option<String> {
    process_last_user_prompt_in_chat_completions(chat_completions_request, |text| {
        const MODEL_COMMAND_PREFIX: &str = "/model ";
        let detected = text.strip_prefix(MODEL_COMMAND_PREFIX).and_then(|rest| {
            rest.split_whitespace().next().map(|s| {
                info!("detected /model command in user-text - will set requested model to '{s}'");
                String::from(s)
            })
        });
        if let Some(detected_model) = &detected {
            *text = text
                .strip_prefix(MODEL_COMMAND_PREFIX)
                .unwrap()
                .strip_prefix(detected_model)
                .unwrap()
                .trim_start()
                .to_owned();
            info!("removed extracted model-command from user-prompt.");
            info!("cleaned user-prompt: '{text}'");
        }
        detected
    })
}

fn check_user_prompt_for_stop_llamacpp(
    chat_completions_request: &mut CreateChatCompletionRequest,
) -> bool {
    const STOP_COMMAND: &str = "/stop";
    if let Some(text) =
        process_last_user_prompt_in_chat_completions(chat_completions_request, |text| {
            if text.trim() == STOP_COMMAND {
                Some(String::new())
            } else {
                text.strip_prefix(STOP_COMMAND).map(str::to_owned)
            }
        })
    {
        info!("detected {STOP_COMMAND} command in user-text - will set stop llama.cpp-processes");
        info!("removed extracted stop-command from user-prompt.");
        info!("cleaned user-prompt: '{text}'");
        true
    } else {
        false
    }
}

#[cfg(test)]
mod route_pattern_tests {
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode},
        routing::any,
    };
    use tower::ServiceExt;

    /// The chat-ui routing relies on specific pattern combinations. Axum panics
    /// at registration time on conflicting patterns, and the `/chat/` document
    /// must be matched by the catch-all wildcard - both are asserted here so a
    /// future refactor cannot silently break the ui routes.
    #[tokio::test]
    async fn chat_ui_patterns_are_valid_and_match_document() {
        let dummy = || async { StatusCode::OK };

        let router = Router::new()
            .route("/chat", any(dummy))
            .route("/chat/", any(dummy))
            .route("/chat/{*path}", any(dummy))
            .route("/v1/models", any(dummy))
            .route("/models", any(dummy))
            .route("/models/{*path}", any(dummy))
            .route("/tools", any(dummy))
            .route("/cors-proxy", any(dummy));

        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/chat/")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        for uri in [
            "/chat",
            "/chat/_app/immutable/bundle.x.js",
            "/chat/props?autoload=false",
            "/chat/v1/models",
            "/v1/models",
            "/models/load",
            "/tools",
            "/cors-proxy",
        ] {
            let response = router
                .clone()
                .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "unexpected status for {uri}");
        }
    }
}