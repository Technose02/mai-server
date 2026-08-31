use crate::{
    application::{middleware::check_auth, model::StableDiffusionPromptDto},
    model::{ApplicationConfig, SecurityConfig},
};
use axum::{
    body::Body,
    extract::{Json, Path, State},
    http::Response,
    http::StatusCode,
    routing::{Router, post},
};
use managed_process::ProcessState;
use std::sync::Arc;

pub fn create_router(
    config: Arc<dyn ApplicationConfig>,
    security_config: Arc<dyn SecurityConfig>,
) -> Router {
    Router::new()
        .route("/api/sd/{sd_config}", post(post_stablediffusion_request))
        .layer(axum::middleware::from_fn_with_state(
            security_config.clone(),
            check_auth,
        ))
        .with_state(config)
}

async fn post_stablediffusion_request(
    State(application_config): State<Arc<dyn ApplicationConfig>>,
    Path(sd_config): Path<String>,
    Json(prompt_dto): Json<StableDiffusionPromptDto>,
) -> Result<Response<Body>, StatusCode> {
    let _previously_running_model = match application_config
        .languagemodelmanager_service()
        .get_llamacpp_state()
        .await
    {
        ProcessState::Running(cfg) => {
            application_config
                .languagemodelmanager_service()
                .stop_llamacpp_process()
                .await;
            Some(cfg)
        }
        ProcessState::Starting(cfg) => {
            application_config
                .languagemodelmanager_service()
                .stop_llamacpp_process()
                .await;
            Some(cfg)
        }
        ProcessState::Stopping(_, ocfg) => {
            application_config
                .languagemodelmanager_service()
                .stop_llamacpp_process()
                .await;
            ocfg
        }
        _ => None,
    };

    let res = application_config
        .stable_diffusion_service()
        .process_prompt(&sd_config, prompt_dto)
        .await;

    // WRONG: STABLE DIFFUSION PROCESSES ASYNC, YOU MAY NO RELEAD OLD MODEL HERE, YET
    /*if let Some(llm) = previously_running_model
        && application_config
            .models_service()
            .ensure_requested_languagemodel_is_served(
                llm.args_handle.alias.as_str(),
                Duration::from_hours(10),
            )
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
            .is_err()
    {
        error!("failed to restart model {}", llm.args_handle.alias);
    }*/

    res
}
