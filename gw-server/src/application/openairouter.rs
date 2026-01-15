use crate::{
    ApplicationConfig, SecurityConfig,
    application::{middleware::check_auth, model::ModelList},
};
use async_openai::types::chat::CreateChatCompletionRequest;
use axum::{
    Json, debug_handler,
    extract::{Path, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{Router, any, get, post},
};
use std::{sync::Arc, time::Duration};

pub fn create_router(
    config: Arc<dyn ApplicationConfig>,
    security_config: Arc<dyn SecurityConfig>,
) -> Router {
    let open_routes = Router::new().route("/api/v1/models", get(get_models));

    let secured_routes = Router::new()
        .route("/api/v1/chat/completions", post(post_completions))
        .route("/api/v1/{*path}", any(fallback))
        .route("/api/v2/{*path}", any(|| async { StatusCode::NOT_FOUND }))
        .layer(axum::middleware::from_fn_with_state(
            security_config,
            check_auth,
        ));

    Router::new()
        .merge(open_routes)
        .merge(secured_routes)
        .with_state(config) // injects state in open_routes and secured_routes
}

async fn get_models(
    State(config): State<Arc<dyn ApplicationConfig>>,
) -> Result<Response, StatusCode> {
    let mut model_list = ModelList::new();
    for model_configuration in config.models_service().get_model_configuration_list().await {
        model_list.extend_from_domain_model_configuration(&model_configuration);
    }
    Ok((StatusCode::OK, Json::<ModelList>::from(model_list)).into_response())
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
        .openapi_service()
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
        .openapi_service()
        .forward_openai_request(request)
        .await
}
