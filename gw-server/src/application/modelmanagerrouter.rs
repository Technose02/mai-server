use crate::{
    ApplicationConfig, SecurityConfig,
    application::{
        middleware::check_auth,
        model::{LlamaCppProcessStateResponse, LlamaCppRunConfigDto},
    },
};
use axum::{
    extract::{Json as JsonExtract, State},
    http::StatusCode,
    response::Json as JsonBody,
    routing::{Router, get},
};
use std::sync::Arc;

#[derive(Clone)]
struct CombinedState {
    config: Arc<dyn ApplicationConfig>,
    security_config: Arc<dyn SecurityConfig>,
}

pub fn create_router(
    config: Arc<dyn ApplicationConfig>,
    security_config: Arc<dyn SecurityConfig>,
) -> Router {
    let combined_state = CombinedState {
        config,
        security_config: security_config.clone(),
    };

    Router::new()
        .route(
            "/admin/llamacpp",
            get(get_llama_cpp_state)
                .put(start_llama_cpp_process)
                .delete(stop_llamacpp),
        )
        .layer(axum::middleware::from_fn_with_state(
            security_config,
            check_auth,
        ))
        .with_state(combined_state)
}

async fn get_llama_cpp_state(
    State(combined_state): State<CombinedState>,
) -> Result<JsonBody<LlamaCppProcessStateResponse>, StatusCode> {
    let llamacpp_process_state: LlamaCppProcessStateResponse = combined_state
        .config
        .modelmanager_service()
        .get_llamacpp_state()
        .await
        .into();
    Ok(JsonBody::from(llamacpp_process_state))
}

async fn start_llama_cpp_process(
    State(combined_state): State<CombinedState>,
    JsonBody(llamacpp_run_config_dto): JsonExtract<LlamaCppRunConfigDto>,
) -> Result<JsonBody<LlamaCppProcessStateResponse>, StatusCode> {
    let llama_cpp_run_config =
        llamacpp_run_config_dto.map_into_domain(combined_state.security_config.get_apikey());
    let llamacpp_process_state: LlamaCppProcessStateResponse = combined_state
        .config
        .modelmanager_service()
        .start_llamacpp_process(llama_cpp_run_config)
        .await
        .into();
    Ok(JsonBody::from(llamacpp_process_state))
}

async fn stop_llamacpp(State(combined_state): State<CombinedState>) -> StatusCode {
    combined_state
        .config
        .modelmanager_service()
        .stop_llamacpp_process()
        .await;
    StatusCode::NO_CONTENT
}
