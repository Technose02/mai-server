use axum::routing::Router;
use axum_server::tls_rustls::RustlsConfig;
use gw_server::{
    application,
    domain::{
        ports::{
            ModelManagerServiceInPort, ModelsServiceInPort, OpenAiRequestForwardPServiceInPort,
            StableDiffusionServiceInPort,
        },
        service::{
            DefaultModelsService, DefaultStableDiffusionService,
            InferenceBackendModelManagerService, OpenAiClientRequestForwardService,
        },
    },
    infrastructure::adapter::{
        LlamaCppControllerAdapter, LocalLlamaCppClientAdapter, StableDiffusionConfigRunnerAdapter,
        StaticModelLoader,
    },
    model::{ApplicationConfig, SecurityConfig},
};
use rand::Rng;
use rustls::pki_types::{IpAddr, Ipv4Addr};
use std::{
    borrow::Cow, collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration,
};
use tracing::{info, warn};

//const SERVER_DOMAIN: &str = "mai-server.ipv64.net";
const SERVER_DOMAIN: &str = "w0zteduifhh6rjyc.myfritz.net";

const MAISERVER_LOG_KEY: &str = "MAISERVER_LOG";
const RANDOM_APIKEY_LEN: u8 = 25;
pub const LLAMACPP_LLM_PORT: u16 = 11440;
pub const LLAMACPP_LLM_PORT_ENV: &str = "LLAMACPP_LLM_PORT";
pub const LLAMACPP_EMBEDDINGS_PORT: u16 = 11441;
pub const LLAMACPP_EMBEDDINGS_PORT_ENV: &str = "LLAMACPP_EMBEDDINGS_PORT";
const LLAMACPP_LLM_TIMEOUT_SECS: u16 = 60000;
const STABLEDIFFUSIONCPP_EXEDIR: &str = "/data0/inference/stable-diffusion.cpp";

/// where the gateway expects (and starts) its two llama-server backends.
/// every value is overridable via CLI-flag or environment-variable so a test
/// instance can be pointed at spare ports / fixture directories without ever
/// touching the live pair on the default ports.
#[derive(Clone)]
pub struct LlamaCppBackendConfig {
    pub llm_port: u16,
    pub embeddings_port: u16,
    pub llm_execdir: String,
    pub embeddings_execdir: String,
}

impl Default for LlamaCppBackendConfig {
    fn default() -> Self {
        let env_or = |key: &str, fallback: &str| {
            std::env::var(key).unwrap_or_else(|_| {
                info!("{key} not set in env, defaulting to '{fallback}'");
                fallback.to_string()
            })
        };
        let env_port = |key: &str, fallback: u16| match std::env::var(key) {
            Ok(value) => value.parse::<u16>().unwrap_or_else(|_| {
                panic!("invalid value for {key} in env: '{value}' (expected a port number)")
            }),
            Err(_) => {
                info!("{key} not set in env, defaulting to '{fallback}'");
                fallback
            }
        };
        Self {
            llm_port: env_port(LLAMACPP_LLM_PORT_ENV, LLAMACPP_LLM_PORT),
            embeddings_port: env_port(LLAMACPP_EMBEDDINGS_PORT_ENV, LLAMACPP_EMBEDDINGS_PORT),
            llm_execdir: env_or(
                gw_server::LLAMACPP_EXECDIR_LLM_ENV,
                gw_server::LLAMACPP_EXECDIR_LLM,
            ),
            embeddings_execdir: env_or(
                gw_server::LLAMACPP_EXECDIR_EMBEDDINGS_ENV,
                gw_server::LLAMACPP_EXECDIR_EMBEDDINGS,
            ),
        }
    }
}

/// runtime switches that only matter for testing / development instances
#[derive(Clone, Default)]
pub struct RunOptions {
    /// skip starting the default embedding-model at boot (a test instance must
    /// not load models on machines without spare RAM)
    pub skip_default_embedding_startup: bool,
    /// skip the "ensure a language model is served" step when the chat-ui
    /// document is requested (prevents a fixture-test from spawning real
    /// llama-server processes with real model configs)
    pub skip_chatui_model_ensure: bool,
}

#[derive(Default)]
enum AcceleratorBackend {
    Vulkan,
    #[default]
    RocM,
}

impl AcceleratorBackend {
    pub fn llamacpp_command(&self) -> &'static str {
        match self {
            AcceleratorBackend::RocM => "./build-rocm/bin/llama-server",
            AcceleratorBackend::Vulkan => "./build-vulkan/bin/llama-server",
        }
    }
    pub fn stablediffusioncpp_command(&self) -> &'static str {
        match self {
            AcceleratorBackend::RocM => "build-rocm/bin/sd-cli",
            AcceleratorBackend::Vulkan => "build-vulkan/bin/sd-cli",
        }
    }
    pub fn environment_args(&self) -> HashMap<String, String> {
        let mut environment_args = HashMap::new();

        match self {
            AcceleratorBackend::RocM => {
                // Zwingend nötig: Erzwingt den korrekten RDNA 3.5 Pfad (gfx1151)
                environment_args.insert("HSA_OVERRIDE_GFX_VERSION".into(), "11.5.1".into());

                // Erlaubt llama.cpp, den gemeinsamen Systemspeicher der APU nativ anzusprechen
                environment_args.insert("GGML_HIP_ENABLE_UNIFIED_MEMORY".into(), "1".into());

                // Verhindert Stream-Asynchronitäten beim schnellen Single-Token-Decoding
                environment_args.insert("HIP_LAUNCH_BLOCKING".into(), "1".into());

                // Der VMM-Kompromiss:
                // Testen Sie es im ersten Durchlauf OHNE "GGML_HIP_NO_VMM".
                // Durch das korrekte "ROCWMMA_FATTN" und den symmetrischen Q8_0-Cache ist die Speicherstruktur
                // jetzt mathematisch sauber ausgerichtet. Sollte das Modell beim Start dennoch hängenbleiben,
                // fügen Sie die Zeile wieder ein – mit dem neuen Compiler-Build bremst sie das Decoding nicht mehr so stark aus.
            }
            AcceleratorBackend::Vulkan => {
                // Schaltet die interne Shader-Optimierung frei – extrem wichtig für RDNA3/3.5-APUs,
                // da es das zeitaufwendige Zwischenspeichern (Spilling) im Grafikspeicher blockiert.
                environment_args.insert("RADV_PERFTEST".into(), "nogttspill".into());

                // Zwingt llama.cpp, die integrierte iGPU (Radeon 8060S) direkt als primäres
                // Rechengerät anzusprechen (hilfreich, falls die CPU fälschlicherweise als Device 0 gelistet wird).
                environment_args.insert("GGML_VK_VISIBLE_DEVICES".into(), "0".into());

                // Optimiert die Befehlswarteschlange (Queue Execution) in neueren llama.cpp-Versionen,
                // um Latenzen bei sequenziellen Vektoroperationen (Decoding) zu minimieren.
                environment_args.insert("GGML_VK_NODES_PER_SUBMIT".into(), "2".into());
            }
        }
        environment_args
    }
}

struct MyAppState {
    openai_chat_completions_service: Arc<dyn OpenAiRequestForwardPServiceInPort>,
    openai_embeddings_service: Arc<dyn OpenAiRequestForwardPServiceInPort>,
    languagemodelmanager_service: Arc<dyn ModelManagerServiceInPort>,
    embeddingmodelmanager_service: Arc<dyn ModelManagerServiceInPort>,
    models_service: Arc<dyn ModelsServiceInPort>,
    stable_diffusion_service: Arc<dyn StableDiffusionServiceInPort>,
    run_options: RunOptions,
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

    fn stable_diffusion_service(&self) -> Arc<dyn StableDiffusionServiceInPort> {
        self.stable_diffusion_service.clone()
    }

    fn skip_chatui_model_ensure(&self) -> bool {
        self.run_options.skip_chatui_model_ensure
    }
}

struct MySecurityConfig {
    apikey: Option<String>,
}

impl SecurityConfig for MySecurityConfig {
    fn get_apikey(&self) -> Option<std::borrow::Cow<'_, str>> {
        self.apikey
            .as_ref()
            .map(|apikey| Cow::Owned(apikey.clone()))
    }
}

async fn create_app(
    provided_apikey: Option<String>,
    localhost: bool,
    log_request_info: bool,
    accelerator_backend: AcceleratorBackend,
    backend_config: LlamaCppBackendConfig,
    run_options: RunOptions,
) -> Router {
    let security_config = match provided_apikey {
        None if localhost => Arc::new(MySecurityConfig { apikey: None }),
        Some(apikey) => Arc::new(MySecurityConfig {
            apikey: Some(apikey),
        }),
        None => {
            const CHARSET: &[u8] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            let mut rng = rand::rng();

            let apikey = (0..RANDOM_APIKEY_LEN)
                .map(|_| {
                    let idx = rng.random_range(0..CHARSET.len());
                    CHARSET[idx] as char
                })
                .collect();

            info!("your current api-key is '{apikey}'");
            Arc::new(MySecurityConfig {
                apikey: Some(apikey),
            })
        }
    };

    // init adapters
    let llamacpp_llm_client = LocalLlamaCppClientAdapter::create_adapter(
        backend_config.llm_port,
        security_config.clone(),
    );
    let llamacpp_embeddings_client = LocalLlamaCppClientAdapter::create_adapter(
        backend_config.embeddings_port,
        security_config.clone(),
    );
    let llamacpp_llm_backend_controller = LlamaCppControllerAdapter::create_adapter(
        backend_config.llm_port,
        Some(LLAMACPP_LLM_TIMEOUT_SECS),
        accelerator_backend.llamacpp_command(),
        backend_config.llm_execdir.clone(),
    )
    .await;
    let llamacpp_embeddings_backend_controller = LlamaCppControllerAdapter::create_adapter(
        backend_config.embeddings_port,
        None,
        accelerator_backend.llamacpp_command(),
        backend_config.embeddings_execdir.clone(),
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
    let number_of_llamacpp_batch_threads = 32_i8;

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
        number_of_llamacpp_batch_threads,
        accelerator_backend.environment_args(),
    );

    if run_options.skip_default_embedding_startup {
        info!("skipping default embedding-model startup (--no-default-embedding-startup)");
    } else {
        // start default embeddings-model
        let default_model = models_service.get_default_embeddingmodel_alias();
        // a failure here must not kill the gateway: on machines without spare
        // ram (or without the model file) this is a degraded, not a dead server
        if models_service
            .ensure_requested_embeddingmodel_is_served(&default_model, Duration::from_millis(60000))
            .await
            .is_err()
        {
            warn!("error starting default embedding-model ('{default_model}') - continuing without it; embeddings requests will retry on demand");
        }
    }

    let languagemodelmanager_service =
        InferenceBackendModelManagerService::create_service(llamacpp_llm_backend_controller);
    let embeddingmodelmanager_service =
        InferenceBackendModelManagerService::create_service(llamacpp_embeddings_backend_controller);

    let path_to_stablediffusioncpp = PathBuf::from(STABLEDIFFUSIONCPP_EXEDIR)
        .join(accelerator_backend.stablediffusioncpp_command());
    let stable_diffusion_config_runner =
        StableDiffusionConfigRunnerAdapter::create_adapter(path_to_stablediffusioncpp);
    let stable_diffusion_service =
        DefaultStableDiffusionService::create_service(stable_diffusion_config_runner);

    // build configuration(s)
    let config = Arc::new(MyAppState {
        openai_chat_completions_service,
        openai_embeddings_service,
        languagemodelmanager_service,
        embeddingmodelmanager_service,
        models_service,
        stable_diffusion_service,
        run_options: run_options.clone(),
    });

    let router = Router::new()
        .merge(application::open_ai_router(
            config.clone(),
            security_config.clone(),
        ))
        .merge(application::stable_diffusion_router(
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

fn init_logging() {
    use std::fs::File;
    use tracing_subscriber::{Registry, filter::LevelFilter, fmt, prelude::*};

    // 1. Log-Level als `LevelFilter` ermitteln
    let max_level = if let Ok(max_log_level) = std::env::var(MAISERVER_LOG_KEY) {
        match max_log_level.to_lowercase().as_str() {
            "info" => LevelFilter::INFO,
            "trace" => LevelFilter::TRACE,
            "debug" => LevelFilter::DEBUG,
            "error" => LevelFilter::ERROR,
            "warn" => LevelFilter::WARN,
            _ => LevelFilter::ERROR,
        }
    } else {
        LevelFilter::INFO
    };

    println!("max-level: {max_level}");

    // 2. Datei-Writer erstellen
    let file = File::options()
        .append(true)
        .create(true)
        .open("./log.txt")
        .expect("Konnte Logdatei nicht erstellen oder öffnen");

    // 3. Die Layer definieren
    let stdout_layer = fmt::layer().with_writer(std::io::stdout);
    let file_layer = fmt::layer().with_writer(file).with_ansi(false);

    // 4. Layer mit dem globalen LevelFilter kombinieren und initialisieren
    Registry::default()
        .with(stdout_layer)
        .with(file_layer)
        .with(max_level) // LevelFilter implementiert das `Layer`-Trait korrekt
        .init();
}

#[tokio::main]
async fn main() {
    /*
    if let Ok(max_log_level) = std::env::var(MAISERVER_LOG_KEY) {
        let max_level = match max_log_level.to_lowercase().as_str() {
            "info" => Level::INFO,
            "trace" => Level::TRACE,
            "debug" => Level::DEBUG,
            "error" => Level::ERROR,
            "warn" => Level::WARN,
            _ => Level::ERROR,
        };
        tracing_subscriber::fmt().with_max_level(max_level).init();
    } else {
        tracing_subscriber::fmt().init();
    }
    */
    init_logging();

    let mut args = std::env::args();
    let (
        host,
        port,
        tls,
        provided_api_key,
        provided_log_request_info,
        llamacpp_backend,
        backend_config,
        run_options,
    ) = {
        let mut port = None;
        let mut api_key = None;
        let mut log_request_info = false;
        let mut no_https = false;
        let mut override_host = None;
        let mut llamacpp_backend = AcceleratorBackend::default();
        // env (or built-in defaults) provide the baseline; CLI flags below win
        let mut backend_config = LlamaCppBackendConfig::default();
        let mut run_options = RunOptions::default();

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

            if a == "--llamacpp-llm-port" {
                let value = args.next().expect("no value for \"--llamacpp-llm-port\" provided");
                backend_config.llm_port = value
                    .parse::<u16>()
                    .unwrap_or_else(|_| panic!("invalid value for \"--llamacpp-llm-port\": {value}"));
            }

            if a == "--llamacpp-embeddings-port" {
                let value =
                    args.next().expect("no value for \"--llamacpp-embeddings-port\" provided");
                backend_config.embeddings_port = value.parse::<u16>().unwrap_or_else(|_| {
                    panic!("invalid value for \"--llamacpp-embeddings-port\": {value}")
                });
            }

            if a == "--llamacpp-execdir" {
                let value = args.next().expect("no value for \"--llamacpp-execdir\" provided");
                backend_config.llm_execdir = value.clone();
                backend_config.embeddings_execdir = value;
            }

            if a == "--no-default-embedding-startup" {
                run_options.skip_default_embedding_startup = true;
            }

            if a == "--no-chatui-model-ensure" {
                run_options.skip_chatui_model_ensure = true;
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

            if a == "--no-https" {
                no_https = true;
            }

            if a == "--rocm" {
                llamacpp_backend = AcceleratorBackend::RocM;
            }

            if a == "--vulkan" {
                llamacpp_backend = AcceleratorBackend::Vulkan;
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
            llamacpp_backend,
            backend_config,
            run_options,
        )
    };

    let app = create_app(
        provided_api_key,
        host == IpAddr::V4(Ipv4Addr::from([127, 0, 0, 1])),
        provided_log_request_info,
        llamacpp_backend,
        backend_config,
        run_options,
    )
    .await;
    let addr = SocketAddr::from((host, port));
    let app_service = app.into_make_service();

    let tls_config = if tls {
        rustls::crypto::ring::default_provider()
            .install_default()
            .expect("Failed to install rustls crypto provider");

        // configure certificate and private key used by https
        let config = RustlsConfig::from_pem_file(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(format!("letsencrypt/{SERVER_DOMAIN}"))
                .join("fullchain.pem"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(format!("letsencrypt/{SERVER_DOMAIN}"))
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
