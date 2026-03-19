use crate::{
    domain::{
        ports::{
            ModelManagerServiceInPort, ModelsServiceInPort, OpenAiRequestForwardPServiceInPort,
        },
        service::{
            DefaultModelsService, InferenceBackendModelManagerService,
            OpenAiClientRequestForwardService,
        },
    },
    infrastructure::adapter::{
        LlamaCppControllerAdapter, LocalLlamaCppClientAdapter, StaticModelLoader,
    },
};
use axum::routing::Router;
use axum_server::tls_rustls::RustlsConfig;
use rand::Rng;
use rustls::pki_types::{IpAddr, Ipv4Addr};
use std::{
    borrow::Cow, collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration,
};
use tracing::{Level, info};

mod application;
mod domain;
mod infrastructure;

mod model;
pub(crate) use model::{ApplicationConfig, SecurityConfig};

const RANDOM_APIKEY_LEN: u8 = 25;
const LLAMACPP_LLM_PORT: u16 = 11440;
const LLAMACPP_EMBEDDINGS_PORT: u16 = 11441;
const LLAMACPP_COMMAND: &str = "./build/bin/llama-server";
const LLAMACPP_EXECDIR: &str = "/data0/inference/llama.cpp/";

struct MyAppState {
    openai_chat_completions_service: Arc<dyn OpenAiRequestForwardPServiceInPort>,
    openai_embeddings_service: Arc<dyn OpenAiRequestForwardPServiceInPort>,
    languagemodelmanager_service: Arc<dyn ModelManagerServiceInPort>,
    embeddingmodelmanager_service: Arc<dyn ModelManagerServiceInPort>,
    models_service: Arc<dyn ModelsServiceInPort>,
}

impl ApplicationConfig for MyAppState {
    fn openai_chat_completions_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort> {
        self.openai_chat_completions_service.clone()
    }

    fn openai_embeddings_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort> {
        self.openai_embeddings_service.clone()
    }

    fn languagemodelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort> {
        self.languagemodelmanager_service.clone()
    }

    fn embeddingmodelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort> {
        self.embeddingmodelmanager_service.clone()
    }

    fn models_service(&self) -> Arc<dyn ModelsServiceInPort> {
        self.models_service.clone()
    }
}

struct MySecurityConfig {
    apikey: Option<String>,
}

impl SecurityConfig for MySecurityConfig {
    fn get_apikey(&self) -> Option<std::borrow::Cow<'_, str>> {
        if let Some(api_key) = &self.apikey {
            Some(Cow::Owned(api_key.clone()))
        } else {
            None
        }
    }
}

async fn create_app(provided_apikey: Option<String>, localhost: bool, log_request_info: bool) -> Router {
    let security_config = match provided_apikey {

        None if localhost => Arc::new(MySecurityConfig { apikey: None }),
        Some(apikey) => Arc::new(MySecurityConfig { apikey: Some(apikey) }),
        None => {
            const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            let mut rng = rand::rng();
        
            let apikey = (0..RANDOM_APIKEY_LEN)
            .map(|_| {
                let idx = rng.random_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect();
        
            info!("your current api-key is '{apikey}'");
            Arc::new(MySecurityConfig { apikey: Some(apikey) })
        }
    };
    //let apikey = provided_apikey.unwrap_or_else(|| {
    //    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    //    let mut rng = rand::rng();

    //    let apikey = (0..RANDOM_APIKEY_LEN)
    //        .map(|_| {
    //            let idx = rng.random_range(0..CHARSET.len());
    //            CHARSET[idx] as char
    //        })
    //        .collect();

    //    info!("your current api-key is '{apikey}'");

    //    apikey
    //});

    //let security_config = Arc::new(MySecurityConfig { apikey });

    // init adapters
    let llamacpp_llm_client =
        LocalLlamaCppClientAdapter::create_adapter(LLAMACPP_LLM_PORT, security_config.clone());
    let llamacpp_embeddings_client = LocalLlamaCppClientAdapter::create_adapter(
        LLAMACPP_EMBEDDINGS_PORT,
        security_config.clone(),
    );
    let llamacpp_llm_backend_controller = LlamaCppControllerAdapter::create_adapter(
        LLAMACPP_LLM_PORT,
        LLAMACPP_COMMAND,
        LLAMACPP_EXECDIR,
    )
    .await;
    let llamacpp_embeddings_backend_controller = LlamaCppControllerAdapter::create_adapter(
        LLAMACPP_EMBEDDINGS_PORT,
        LLAMACPP_COMMAND,
        LLAMACPP_EXECDIR,
    )
    .await;

    let model_loader = StaticModelLoader::create_adapter(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../staticmodelconfig/static_config_files"),
        security_config.clone(),
    )
    .unwrap();

    // init services
    let parallel_llamacpp_requests = 1_u8;
    let number_of_llamacpp_threads = 8_i8;
    let environment_args = {
        let mut environment_args = HashMap::new();
        environment_args.insert("GGML_CUDA_ENABLE_UNIFIED_MEMORY".into(), "1".into());
        environment_args.insert("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL".into(), "1".into());
        environment_args
    };

    let openai_chat_completions_service =
        OpenAiClientRequestForwardService::create_service(llamacpp_llm_client);
    let openai_embeddings_service =
        OpenAiClientRequestForwardService::create_service(llamacpp_embeddings_client);
    let models_service = DefaultModelsService::create_service(
        llamacpp_llm_backend_controller.clone(),
        llamacpp_embeddings_backend_controller.clone(),
        model_loader,
        parallel_llamacpp_requests,
        number_of_llamacpp_threads,
        environment_args,
    );

    {
        // start default embeddings-model
        let default_model = models_service.get_default_embeddingmodel_alias();
        models_service
            .ensure_requested_embeddingmodel_is_served(&default_model, Duration::from_millis(60000))
            .await
            .unwrap_or_else(|_| {
                panic!("error starting default embedding-model ('{default_model}')")
            });
    }

    let languagemodelmanager_service =
        InferenceBackendModelManagerService::create_service(llamacpp_llm_backend_controller);
    let embeddingmodelmanager_service =
        InferenceBackendModelManagerService::create_service(llamacpp_embeddings_backend_controller);

    // build configuration(s)
    let config = Arc::new(MyAppState {
        openai_chat_completions_service,
        openai_embeddings_service,
        languagemodelmanager_service,
        embeddingmodelmanager_service,
        models_service,
    });

    let router = Router::new()
        .merge(application::open_ai_router(
            config.clone(),
            security_config.clone(),
        ))
        .merge(application::model_manager_router(config, security_config));

    if log_request_info {
        router.layer(axum::middleware::from_fn(
            application::middleware::request_logger,
        ))
    } else {
        router
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let mut args = std::env::args();
    let (host, port, tls, provided_api_key, provided_log_request_info, _provided_llama_cpp_chatui) = {
        let mut port = None;
        let mut api_key = None;
        let mut log_request_info = false;
        let mut llama_cpp_chatui = false;
        let mut no_https = false;
        let mut override_host = None;
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
            } else if a == "--port" || a == "-p" {
                panic!("you must not provide \"-p\" (\"--port\") more than once")
            }

            if api_key.is_none() && a == "--api-key" || a == "-k" {
                if let Some(api_key_value) = args.next() {
                    api_key = Some(api_key_value);
                } else {
                    panic!("no value for \"-k\" (\"--api-key\") provided")
                }
            } else if a == "--api-key" || a == "-k" {
                panic!("you must not provide \"-k\" (\"--api-key\") more than once")
            }

            if override_host.is_none() && a == "--override-host-ip" || a == "-H" {
                if let Some(override_host_val) = args.next() {
                    override_host = match IpAddr::try_from(override_host_val.as_str()) {
                        Ok(addr) => Some(addr),
                        Err(_) => {
                            panic!("invalid value for \"-H\" (\"-override-host-ip\") provided")
                        }
                    };
                } else {
                    panic!("no value for \"-H\" (\"-override-host-ip\") provided")
                }
            } else if a == "--override-host-ip" || a == "-H" {
                panic!("you must not provide \"-H\" (\"-override-host-ip\") more than once")
            }

            if a == "--log-request-info" || a == "-l" {
                log_request_info = true;
            }

            if a == "--chatui" {
                llama_cpp_chatui = true;
            }

            if a == "--no-https" {
                no_https = true;
            }
        }
        let port = match port {
            Some(p) => p,
            None if no_https => 8080,
            None => 8443,
        };
        let host = override_host.unwrap_or(IpAddr::V4(Ipv4Addr::from([0, 0, 0, 0])));
        (
            host,
            port,
            !no_https,
            api_key,
            log_request_info,
            llama_cpp_chatui,
        )
    };

    let app = create_app(provided_api_key, host==IpAddr::V4(Ipv4Addr::from([127, 0, 0, 1])), provided_log_request_info).await;
    let addr = SocketAddr::from((host, port));
    let app_service = app.into_make_service();

    let tls_config = if tls {
        rustls::crypto::ring::default_provider()
            .install_default()
            .expect("Failed to install rustls crypto provider");

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
        Some(config)
    } else {
        None
    };

    info!("server started on {addr}");

    if let Some(tls_config) = tls_config {
        axum_server::bind_rustls(addr, tls_config)
            .serve(app_service)
            .await
            .unwrap_or_else(|_| panic!("error serving via tls on {addr}"));
    } else {
        axum_server::bind(addr)
            .serve(app_service)
            .await
            .unwrap_or_else(|_| panic!("error serving without tls on {addr}"));
    }

    info!("server shutdown");
}
