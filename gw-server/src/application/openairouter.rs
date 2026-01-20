use crate::{
    ApplicationConfig, SecurityConfig,
    application::{
        middleware::check_auth, model::try_map_request_body_to_create_chat_completion_request,
    },
};
use axum::{
    Json,
    extract::{Path, Query, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{Router, any, get, post},
};
use staticmodelconfig::ModelList;
use std::{collections::HashMap, sync::Arc, time::Duration};

pub fn create_router(
    config: Arc<dyn ApplicationConfig>,
    security_config: Arc<dyn SecurityConfig>,
) -> Router {
    let open_routes = Router::new()
        .route(
            "/api/{n_parallel}/v1/models",
            get(get_models_with_parallel_param),
        )
        .route("/api/v1/models", get(get_models))
        .route("/chat", get(chat_handler));

    let secured = Router::new()
        .route("/chat/props", get(chat_handler_assets))
        .route(
            "/api/{n_parallel}/v1/chat/completions",
            post(post_completions_with_parallel_param),
        )
        .route("/api/v1/chat/completions", post(post_completions))
        .route(
            "/api/{n_parallel}/v1/{*path}",
            any(api_fallback_with_parallel_param),
        )
        .route("/api/v1/{*path}", any(api_fallback))
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

async fn get_models_with_parallel_param(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Path(n_parallel): Path<u8>,
    Query(query_map): Query<HashMap<String, String>>,
) -> Result<Response, StatusCode> {
    println!("info: models-endpoint called for parallel={n_parallel}");
    get_models_impl(application_config, query_map).await
}

async fn get_models(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Query(query_map): Query<HashMap<String, String>>,
) -> Result<Response, StatusCode> {
    get_models_impl(application_config, query_map).await
}

async fn get_models_impl(
    application_config: Arc<dyn ApplicationConfig>,
    query_map: HashMap<String, String>,
) -> Result<Response, StatusCode> {
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

async fn post_completions_with_parallel_param(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Path(n_parallel): Path<u8>,
    request: Request,
) -> Result<Response, StatusCode> {
    post_chat_completions_impl(application_config, Some(n_parallel), request).await
}

async fn post_completions(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    request: Request,
) -> Result<Response, StatusCode> {
    post_chat_completions_impl(application_config, None, request).await
}

async fn post_chat_completions_impl(
    application_config: Arc<dyn ApplicationConfig>,
    optional_parallel_backend_requests_to_set: Option<u8>,
    request: Request,
) -> Result<Response, StatusCode> {
    let chat_completions_request = try_map_request_body_to_create_chat_completion_request(
        request,
        application_config
            .models_service()
            .get_running_model_alias()
            .await
            .unwrap_or(
                application_config
                    .models_service()
                    .get_default_model_alias(),
            ),
    )
    .await?;

    if let Some(parallel_backend_requests_to_set) = optional_parallel_backend_requests_to_set {
        application_config
            .models_service()
            .set_parallel_backend_requests(parallel_backend_requests_to_set);
    }

    application_config
        .models_service()
        .ensure_requested_model_is_served(&chat_completions_request.model, Duration::from_mins(3))
        .await
        .map_err(|_| {
            eprintln!("error serving requested model");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    application_config
        .openai_service()
        .process_chat_completions_request(chat_completions_request)
        .await
}

async fn api_fallback_with_parallel_param(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Path(n_parallel): Path<u8>,
    Path(api_path): Path<String>,
    request: Request,
) -> Result<Response, StatusCode> {
    api_fallback_impl(application_config, Some(n_parallel), api_path, request).await
}

async fn api_fallback(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Path(api_path): Path<String>,
    request: Request,
) -> Result<Response, StatusCode> {
    api_fallback_impl(application_config, None, api_path, request).await
}

async fn api_fallback_impl(
    application_config: Arc<dyn ApplicationConfig>,
    optional_parallel_backend_requests_to_set: Option<u8>,
    api_path: String,
    request: Request,
) -> Result<Response, StatusCode> {
    println!("unexpected {}-request to {api_path}", request.method());

    if let Some(parallel_backend_requests_to_set) = optional_parallel_backend_requests_to_set {
        application_config
            .models_service()
            .set_parallel_backend_requests(parallel_backend_requests_to_set);
    }

    application_config
        .openai_service()
        .forward_api_request(request)
        .await
}

async fn chat_handler(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
) -> Result<Response, StatusCode> {
    let default_model_alias = application_config
        .models_service()
        .get_default_model_alias();
    application_config
        .models_service()
        .ensure_any_model_is_served(&default_model_alias, Duration::from_mins(3))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    application_config.openai_service().get_chat().await
}

async fn chat_handler_assets(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    request: Request,
) -> Result<Response, StatusCode> {
    application_config
        .openai_service()
        .forward_ui_request(request)
        .await
}
