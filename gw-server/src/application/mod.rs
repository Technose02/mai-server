use crate::model::{ApplicationConfig, SecurityConfig};
use axum::routing::Router;
use std::sync::Arc;

pub mod middleware;
pub mod model;
mod modelmanagerrouter;
mod openairouter;

pub fn open_ai_router(
    config: Arc<dyn ApplicationConfig>,
    security_config: Arc<dyn SecurityConfig>,
) -> Router {
    openairouter::create_router(config, security_config)
}

pub fn model_manager_router(
    config: Arc<dyn ApplicationConfig>,
    security_config: Arc<dyn SecurityConfig>,
) -> Router {
    modelmanagerrouter::create_router(config, security_config)
}
