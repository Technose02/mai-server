use axum::{extract::Request, http::StatusCode, middleware::Next, response::Response};

pub async fn request_logger(req: Request, next: Next) -> Result<Response, StatusCode> {
    println!("{} {}", req.method(), req.uri().path());
    let res = next.run(req).await;
    println!("\t-> {}", res.status());
    Ok(res)
}
