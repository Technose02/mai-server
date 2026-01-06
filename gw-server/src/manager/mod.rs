use crate::{AppState, auth_middleware, manager::model::LlamaCppConfig};
use axum::{
    extract::{Json as JsonExtract, State},
    http::StatusCode,
    response::Json as JsonBody,
    routing::{Router, get},
};
use std::sync::Arc;

mod model;
use model::LlamaCppProcessState;

pub fn router(app_state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/admin/llamacpp",
            get(get_llama_cpp_state)
                .put(start_llama_cpp_process)
                .delete(stop_llamacpp),
        )
        .layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            auth_middleware,
        ))
        .with_state(app_state)
}

#[axum::debug_handler]
async fn get_llama_cpp_state(
    State(app_state): State<Arc<AppState>>,
) -> Result<JsonBody<LlamaCppProcessState>, StatusCode> {
    let state: LlamaCppProcessState = app_state.llamacpp_controller().read_state().await.into();

    Ok(JsonBody::from(state))
}

async fn start_llama_cpp_process(
    State(app_state): State<Arc<AppState>>,
    config: JsonExtract<LlamaCppConfig>,
) -> Result<JsonBody<LlamaCppProcessState>, StatusCode> {
    app_state
        .llamacpp_controller()
        .start(config.map(app_state.api_key()))
        .await;
    let state: LlamaCppProcessState = app_state.llamacpp_controller().read_state().await.into();

    Ok(JsonBody::from(state))
}

async fn stop_llamacpp(State(app_state): State<Arc<AppState>>) -> StatusCode {
    app_state.llamacpp_controller().stop().await;

    StatusCode::NO_CONTENT
}
