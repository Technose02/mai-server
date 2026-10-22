use axum::{extract::Request, middleware::Next, response::Response};
use std::time::Instant;
use tracing::info;

pub async fn request_logger(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let path = req
        .uri()
        .path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());
    let started = Instant::now();

    let response = next.run(req).await;

    // status + duration per request: without it a "the ui says server error"
    // report cannot be matched to what the gateway actually answered
    info!(
        "{method} {path} -> {} in {:?}",
        response.status(),
        started.elapsed()
    );

    response
}