use crate::ApplicationConfig;
use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

pub async fn limit_max_parallel_requests(
    State(config): State<Arc<dyn ApplicationConfig>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if config.increase_number_of_parallel_requests() {
        let res = next.run(req).await;
        config.decrease_increase_number_of_parallel_requests();
        Ok(res)
    } else {
        println!("reject request: too many parallel requests");
        Err(StatusCode::TOO_EARLY)
    }
}
