use crate::{
    ApplicationConfig, SecurityConfig,
    application::{middleware::check_auth, model::Llmodels},
};
use axum::{
    Json,
    body::{Body, to_bytes as body_to_bytes},
    debug_handler,
    extract::{Path, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{Router, any, get, post},
};
use serde::Deserialize;
use std::{sync::Arc, time::Duration};

pub fn create_router(
    config: Arc<dyn ApplicationConfig>,
    security_config: Arc<dyn SecurityConfig>,
) -> Router {
    let open_routes = Router::new().route("/api/v1/models", get(get_models));

    let secured_routes = Router::new()
        .route("/api/v1/chat/completions", post(post_completions))
        .route("/api/v1/{*path}", any(fallback))
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
    println!("todo: return generated models response");
    Ok((
        StatusCode::OK,
        Json::<&Llmodels>::from(config.models_service().llmodels().as_ref()),
    )
        .into_response())
}

#[debug_handler]
async fn post_completions(
    State(config): State<Arc<dyn ApplicationConfig>>,
    request: Request,
) -> Result<Response, StatusCode> {
    let (request, provided_requested_model) = extract_model_from_request_payload(request)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    if let Some(requested_model) = provided_requested_model {
        config
            .models_service()
            .ensure_requested_model_is_served(&requested_model, Duration::from_mins(5))
            .await
            .map_err(|e| {
                eprintln!("error serving requested model: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        config
            .openapi_service()
            .process_openai_request(request, "chat/completions")
            .await
    } else {
        Err(StatusCode::BAD_REQUEST)
    }
}

async fn fallback(
    State(service_state): State<Arc<dyn ApplicationConfig>>,
    Path(api_path): Path<String>,
    request: Request,
) -> Result<Response, StatusCode> {
    service_state
        .openapi_service()
        .process_openai_request(request, &api_path)
        .await
}

async fn extract_model_from_request_payload(
    request: Request,
) -> Result<(Request, Option<String>), String> {
    #[derive(Deserialize)]
    struct ModelContainer {
        model: String,
    }

    let (parts, body) = request.into_parts();
    let body_data = body_to_bytes(body, usize::MAX)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
        .map_err(|e| format!("error serializing payload to Bytes: {e}"))?;

    if let Ok(Json(model_container)) =
        Json::<ModelContainer>::from_bytes(body_data.clone().trim_ascii())
    {
        Ok((
            Request::from_parts(parts, Body::from(body_data)),
            Some(model_container.model),
        ))
    } else {
        Ok((Request::from_parts(parts, Body::from(body_data)), None))
    }
}
