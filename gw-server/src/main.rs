use std::{net::SocketAddr, path::PathBuf};
use axum::{Router, response::{IntoResponse, Response}, routing::get};
use axum_server::{tls_rustls::{RustlsConfig}};

fn create_app() -> Router {
    Router::new().route("/", get(hello))
}

#[tokio::main]
async fn main() {

    let app = create_app();

    // configure certificate and private key used by https
    let config = RustlsConfig::from_pem_file(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("self_signed_certs")
            .join("cert.pem"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("self_signed_certs")
            .join("key.pem"),
    )
    .await
    .unwrap();

    let addr = SocketAddr::from(([0,0,0,0], 5050));

    println!("server started");
    axum_server::bind_rustls(addr, config)
        .serve(app.into_make_service()).await.expect("error serving via tls on 0.0.0.0:5050");
    println!("server shutdown");
}

async fn hello() -> Response {
    return "Hello from Axum".into_response()
}