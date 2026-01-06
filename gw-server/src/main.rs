use axum::Router;
use axum_reverse_proxy::ReverseProxy;
use axum_server::tls_rustls::RustlsConfig;
use inference_backends::LlamaCppBackend;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};

mod manager;
mod model;
use manager::router as manager_router;

mod middleware;
pub(crate) use middleware::auth_middleware;

use crate::model::AppState;

async fn create_app(provided_api_key: Option<String>) -> Router {
    let llamacpp_backend = LlamaCppBackend {
        host: "0.0.0.0".to_owned(),
        port: 11440,
        llama_cpp_command: "./build/bin/llama-server".to_owned(),
        llama_cpp_execdir: "/data0/inference/llama.cpp/".to_owned(),
    };

    let mut builder = AppState::builder(llamacpp_backend);
    if let Some(provided_api_key) = provided_api_key {
        builder.with_api_key(provided_api_key);
    }
    let app_state = Arc::new(builder.build().await);

    println!("your current api-key is '{}'", app_state.api_key());

    Router::new()
        // LLAMA.CPP (implicitly protected via llama.cpp)
        .merge(ReverseProxy::new("/chat", "http://localhost:11440"))
        .merge(ReverseProxy::new("/v1", "http://localhost:11440/v1"))
        .merge(ReverseProxy::new("/props", "http://localhost:11440/props"))
        // Manager (explicitly proteced)
        .merge(manager_router(app_state))
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args();
    let (provided_port, provided_api_key) = {
        let mut port = None;
        let mut api_key = None;
        while let Some(a) = args.next() {
            if port.is_none() && (a == "--port" || a == "-p") {
                if let Some(port_value) = args.next() {
                    if let Ok(provided_port) = port_value.parse::<u16>() {
                        port = Some(provided_port);
                    } else {
                        panic!("invalid value for \"-p\" (\"--port\") provided: {port_value}")
                    }
                } else {
                    panic!("no value for \"-p\" (\"--port\") provided")
                }
            }

            if api_key.is_none() && a == "--api-key" || a == "-k" {
                if let Some(api_key_value) = args.next() {
                    api_key = Some(api_key_value);
                } else {
                    panic!("no value for \"-k\" (\"--api-key\") provided")
                }
            }
        }
        (port, api_key)
    };

    let app = create_app(provided_api_key).await;

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

    let addr = SocketAddr::from(([0, 0, 0, 0], provided_port.unwrap_or(5050)));

    println!("server started on {addr}");
    axum_server::bind_rustls(addr, config)
        .serve(app.into_make_service())
        .await
        .unwrap_or_else(|_| panic!("error serving via tls on {addr}"));
    println!("server shutdown");
}
