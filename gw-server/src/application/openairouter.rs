use crate::{
    ApplicationConfig, SecurityConfig,
    application::{middleware::check_auth, model::ModelList},
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
    let mut model_list = ModelList::new();
    for model_configuration in config.models_service().get_model_configuration_list().await {
        model_list.extend_from_domain_model_configuration(&model_configuration);
    }
    Ok((StatusCode::OK, Json::<ModelList>::from(model_list)).into_response())
}

#[debug_handler]
async fn post_completions(
    State(config): State<Arc<dyn ApplicationConfig>>,
    request: Request,
) -> Result<Response, StatusCode> {
    let (request, provided_requested_model) = extract_model_from_request_payload(request)
        .await
        .map_err(|_| {
            println!("error reading request-payload");
            StatusCode::BAD_REQUEST
        })?;

    if let Some(requested_model) = provided_requested_model {
        config
            .models_service()
            .ensure_requested_model_is_served(&requested_model, Duration::from_mins(5))
            .await
            .map_err(|_| {
                eprintln!("error serving requested model");
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
    } else {
        println!("warning: no model found in request-payload");
    }
    config
        .openapi_service()
        .process_openai_request(request, "chat/completions")
        .await
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
) -> Result<(Request, Option<String>), ()> {
    #[derive(Deserialize)]
    struct ModelContainer {
        model: String,
    }

    let (parts, body) = request.into_parts();

    match body_to_bytes(body, usize::MAX).await {
        Ok(body_data) => {
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

        Err(e) => {
            eprintln!("error serializing payload to Bytes {e}");
            Err(())
        }
    }
}
