use crate::model::SecurityConfig;
use axum::{
    extract::{Request, State},
    http::{StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

pub async fn check_auth(
    State(security_config): State<Arc<dyn SecurityConfig>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let auth_header = req
        .headers()
        .get(AUTHORIZATION)
        .ok_or(StatusCode::UNAUTHORIZED)?
        .to_str()
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    if let Some(bearer_token) = auth_header.strip_prefix("Bearer ")
        && security_config.get_apikey() == bearer_token
    {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}
