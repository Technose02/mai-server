use axum::routing::Router;
use axum_reverse_proxy::ReverseProxy;
use axum_server::tls_rustls::RustlsConfig;
use inference_backends::LlamaCppBackend;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};

mod manager;
mod model;
use manager::router as manager_router;

mod middleware;
pub(crate) use middleware::{auth_middleware, request_logger};

use crate::model::AppState;

async fn create_app(
    provided_api_key: Option<String>,
    log_request_info: bool,
    llama_cpp_chatui: bool,
) -> Router {
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

    fn llamacpp_backend_reverse_proxy(
        path: &'static str,
        target: &'static str,
        log_request_info: bool,
    ) -> Router {
        let mut llamacpp_backend_reverse_proxy: Router = ReverseProxy::new(path, target).into();
        if log_request_info {
            llamacpp_backend_reverse_proxy =
                llamacpp_backend_reverse_proxy.layer(axum::middleware::from_fn(request_logger));
        }
        llamacpp_backend_reverse_proxy
    }

    let mut router = Router::new()
        // Manager (explicitly proteced)
        .merge(manager_router(app_state))
        // LLAMA.CPP (implicitly protected via llama.cpp)
        .merge(llamacpp_backend_reverse_proxy(
            "/api/v1/",
            "http://localhost:11440/v1/",
            log_request_info,
        ));

    if llama_cpp_chatui {
        // LLAMA.CPP WITH UI (implicitly protected via llama.cpp) - replicated at context-root, necessary for frontend
        // !! THIS SHOULD BE IN THE LAST POSITION !!
        router = router.merge(llamacpp_backend_reverse_proxy(
            "/",
            "http://localhost:11440",
            log_request_info,
        ))
    }

    router
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args();
    let (provided_port, provided_api_key, provided_log_request_info, provided_llama_cpp_chatui) = {
        let mut port = None;
        let mut api_key = None;
        let mut log_request_info = false;
        let mut llama_cpp_chatui = false;
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

            if a == "--log-request-info" || a == "-l" {
                log_request_info = true;
            }

            if a == "--chatui" {
                llama_cpp_chatui = true;
            }
        }
        (port, api_key, log_request_info, llama_cpp_chatui)
    };

    let app = create_app(
        provided_api_key,
        provided_log_request_info,
        provided_llama_cpp_chatui,
    )
    .await;

    // configure certificate and private key used by https
    let config = RustlsConfig::from_pem_file(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("letsencrypt/mai-server.ipv64.net")
            .join("fullchain.pem"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("letsencrypt/mai-server.ipv64.net")
            .join("privkey.pem"),
    )
    .await
    .unwrap();

    let addr = SocketAddr::from(([0, 0, 0, 0], provided_port.unwrap_or(8443)));

    println!("server started on {addr}");
    axum_server::bind_rustls(addr, config)
        .serve(app.into_make_service())
        .await
        .unwrap_or_else(|_| panic!("error serving via tls on {addr}"));
    println!("server shutdown");
}
