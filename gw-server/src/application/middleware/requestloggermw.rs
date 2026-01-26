use axum::{extract::Request, http::StatusCode, middleware::Next, response::Response};
use tracing::info;

pub async fn request_logger(req: Request, next: Next) -> Result<Response, StatusCode> {
    info!("{} {}", req.method(), req.uri().path());
    let res = next.run(req).await;
    info!("\t-> {}", res.status());
    Ok(res)
}
