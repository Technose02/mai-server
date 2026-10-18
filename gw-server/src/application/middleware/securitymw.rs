use crate::model::SecurityConfig;
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

/// validates the bearer-token in `headers` against the configured apikey.
///
/// Used both by the `check_auth` middleware and by handlers that have to decide
/// per-request (e.g. the chat-ui catch-all, which is public for reads and
/// secured for everything else).
pub fn check_bearer(
    security_config: &Arc<dyn SecurityConfig>,
    headers: &HeaderMap,
) -> Result<(), StatusCode> {
    // no apikey configured -> everything is open
    let Some(security_apikey) = security_config.get_apikey() else {
        return Ok(());
    };

    let auth_header = headers
        .get(AUTHORIZATION)
        .ok_or(StatusCode::UNAUTHORIZED)?
        .to_str()
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    match auth_header.strip_prefix("Bearer ") {
        Some(bearer_token) if bearer_token == security_apikey => Ok(()),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

pub async fn check_auth(
    State(security_config): State<Arc<dyn SecurityConfig>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    check_bearer(&security_config, req.headers())?;
    Ok(next.run(req).await)
}