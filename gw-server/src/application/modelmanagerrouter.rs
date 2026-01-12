use crate::{
    ApplicationConfig, SecurityConfig,
    application::{
        middleware::check_auth,
        model::{LlamaCppConfig, LlamaCppProcessState},
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
) -> Result<JsonBody<LlamaCppProcessState>, StatusCode> {
    let llamacpp_process_state: LlamaCppProcessState = combined_state
        .config
        .modelmanager_service()
        .get_llamacpp_state()
        .await
        .into();
    Ok(JsonBody::from(llamacpp_process_state))
}

async fn start_llama_cpp_process(
    State(combined_state): State<CombinedState>,
    JsonBody(llamacpp_config): JsonExtract<LlamaCppConfig>,
) -> Result<JsonBody<LlamaCppProcessState>, StatusCode> {
    //println!(
    //    "[modelmanagerrouter::start_llama_cpp_process] received llamacpp_config:\n{llamacpp_config:#?}"
    //);
    let llamacpp_config = llamacpp_config.map(Some(combined_state.security_config.get_apikey()));
    let llamacpp_process_state: LlamaCppProcessState = combined_state
        .config
        .modelmanager_service()
        .start_llamacpp_process(llamacpp_config)
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
