use crate::infrastructure::adapter::{LlamaCppControllerAdapter, LocalLlamaCppClientAdapter};
use crate::{
    application::model::{Data, DataMeta, Llmodels, Model, ModelDetails},
    domain::{
        ports::{
            ModelManagerServiceInPort, ModelsServiceInPort, OpenAiRequestForwardPServiceInPort,
        },
        service::{
            DefaultModelsService, InferenceBackendModelManagerService,
            OpenAiClientRequestForwardService,
        },
    },
};
use axum::routing::Router;
use axum_server::tls_rustls::RustlsConfig;
use rand::Rng;
use std::{borrow::Cow, net::SocketAddr, path::PathBuf, sync::Arc};

mod application;
mod domain;
mod infrastructure;

mod model;
pub(crate) use model::{ApplicationConfig, SecurityConfig};

const RANDOM_APIKEY_LEN: u8 = 25;
const LLAMACPP_PORT: u16 = 11440;
const LLAMACPP_COMMAND: &str = "./build/bin/llama-server";
const LLAMACPP_EXECDIR: &str = "/data0/inference/llama.cpp/";

struct MyAppState {
    apikey: String,
    openai_service: Arc<dyn OpenAiRequestForwardPServiceInPort>,
    modelmanager_service: Arc<dyn ModelManagerServiceInPort>,
    models_service: Arc<dyn ModelsServiceInPort>,
}

impl ApplicationConfig for MyAppState {
    fn openapi_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort> {
        self.openai_service.clone()
    }

    fn modelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort> {
        self.modelmanager_service.clone()
    }

    fn models_service(&self) -> Arc<dyn ModelsServiceInPort> {
        self.models_service.clone()
    }
}

impl SecurityConfig for MyAppState {
    fn get_apikey(&self) -> std::borrow::Cow<'_, str> {
        Cow::Borrowed(&self.apikey)
    }
}

async fn create_app(provided_apikey: Option<String>, log_request_info: bool) -> Router {
    let apikey = provided_apikey.unwrap_or_else(|| {
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

        let mut rng = rand::rng();

        let apikey = (0..RANDOM_APIKEY_LEN)
            .map(|_| {
                let idx = rng.random_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect();

        println!("your current api-key is '{apikey}'");

        apikey
    });

    // init adapters
    let llamacpp_client = LocalLlamaCppClientAdapter::create_adapter(LLAMACPP_PORT);
    let llamacpp_backend_controller = LlamaCppControllerAdapter::create_adapter(
        LLAMACPP_PORT,
        LLAMACPP_COMMAND,
        LLAMACPP_EXECDIR,
    )
    .await;

    // init services
    let openai_service = OpenAiClientRequestForwardService::create_service(llamacpp_client);
    let models_service = DefaultModelsService::create_service(
        llamacpp_backend_controller.clone(),
        create_llmodels_list(),
    );
    let modelmanager_service =
        InferenceBackendModelManagerService::create_service(llamacpp_backend_controller);

    // build configuration(s)
    let config = Arc::new(MyAppState {
        apikey,
        openai_service,
        modelmanager_service,
        models_service,
    });
    let security_config = config.clone();

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

fn create_llmodels_list() -> Llmodels {
    let nemotron = String::from("nemotron-3-nano-30b-a3b");

    let m = Model {
        name: nemotron.clone(),
        model: nemotron.clone(),
        modified_at: String::new(),
        size: String::new(),
        digest: String::new(),
        description: String::new(),
        tags: Vec::new(),
        capabilities: vec!["Completion".into()],
        parameters: String::new(),
        details: ModelDetails {
            parent_model: String::new(),
            format: "gguf".into(),
            family: String::new(),
            families: Vec::new(),
            parameter_size: String::new(),
            quantization_level: String::new(),
        },
    };
    let d = Data {
        id: nemotron,
        created: 1768231590,
        meta: DataMeta {
            vocab_type: 2,
            n_vocab: 131072,
            n_ctx_train: 1048576,
            n_embd: 2688,
            n_params: 31577940288,
            size: 40440063744,
        },
    };

    let mut llmodels = Llmodels::new();
    llmodels.add(m, d);
    llmodels
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args();
    let (provided_port, provided_api_key, provided_log_request_info, _provided_llama_cpp_chatui) = {
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

    let app = create_app(provided_api_key, provided_log_request_info).await;

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
