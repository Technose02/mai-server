use crate::{ApplicationConfig, SecurityConfig, application::middleware::check_auth};
use async_openai::types::chat::CreateChatCompletionRequest;
use axum::{
    Json, debug_handler,
    extract::{Path, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{Router, any, get, post},
};
use staticmodelconfig::ModelList;
use std::{sync::Arc, time::Duration};

pub fn create_router(
    config: Arc<dyn ApplicationConfig>,
    security_config: Arc<dyn SecurityConfig>,
) -> Router {
    let open_routes = Router::new()
        .route("/api/v1/models", get(get_models))
        .route("/chat", get(chat_handler));

    let secured = Router::new()
        .route("/api/v1/chat/completions", post(post_completions))
        .route("/api/v1/{*path}", any(fallback))
        .route("/api/v2/{*path}", any(|| async { StatusCode::NOT_FOUND }))
        .layer(axum::middleware::from_fn_with_state(
            security_config.clone(),
            check_auth,
        ));

    Router::new()
        .merge(open_routes)
        .merge(secured)
        .with_state(config) // injects state in open_routes and secured_routes
}

async fn get_models(
    State(config): State<Arc<dyn ApplicationConfig>>,
) -> Result<Response, StatusCode> {
    Ok((
        StatusCode::OK,
        Json::<ModelList>::from(config.models_service().get_models().await),
    )
        .into_response())
}

#[debug_handler]
async fn post_completions(
    State(config): State<Arc<dyn ApplicationConfig>>,
    Json(chat_completions_request): Json<CreateChatCompletionRequest>, //request: Request,
) -> Result<Response, StatusCode> {
    config
        .models_service()
        .ensure_requested_model_is_served(&chat_completions_request.model, Duration::from_mins(3))
        .await
        .map_err(|_| {
            eprintln!("error serving requested model");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    config
        .openai_service()
        .process_chat_completions_request(chat_completions_request)
        .await
}

async fn fallback(
    State(service_state): State<Arc<dyn ApplicationConfig>>,
    Path(api_path): Path<String>,
    request: Request,
) -> Result<Response, StatusCode> {
    println!("unexpected {}-request to {api_path}", request.method());
    service_state
        .openai_service()
        .forward_openai_request(request)
        .await
}

async fn chat_handler(
    State(service_state): State<Arc<dyn ApplicationConfig>>,
) -> Result<Response, StatusCode> {
    let default_model_alias = service_state.models_service().get_default_model_alias();
    service_state
        .models_service()
        .ensure_any_model_is_served(&default_model_alias, Duration::from_mins(3))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    service_state.openai_service().get_chat().await
}
