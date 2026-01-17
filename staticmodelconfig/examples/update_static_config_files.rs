use inference_backends::{
    LlamaCppBackend, LlamaCppBackendController, LlamaCppConfig, LlamaCppConfigArgs,
    LlamaCppProcessState,
};
use reqwest::get;
use serde_json::from_slice;
use staticmodelconfig::{ModelConfiguration, ModelList};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

const LLAMA_SERVER_HOST: &str = "localhost";
const LLAMA_SERVER_PORT: u16 = 11440;

#[tokio::main]
async fn main() {
    let llamacpp_controller = LlamaCppBackendController::init_backend(LlamaCppBackend {
        host: LLAMA_SERVER_HOST.to_owned(),
        port: LLAMA_SERVER_PORT,
        llama_cpp_command: "./build/bin/llama-server".into(),
        llama_cpp_execdir: "/data0/inference/llama.cpp/".into(),
    })
    .await;

    let env_handle = Arc::new({
        let mut env = HashMap::<String, String>::new();
        env.insert("GGML_CUDA_ENABLE_UNIFIED_MEMORY".into(), "1".into());
        env
    });

    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("static_config_files");

    let mut read_dir = dir
        .read_dir()
        .unwrap_or_else(|e| panic!("error reading dir '{:#?}': {e}", dir));
    while let Some(Ok(entry)) = read_dir.next() {
        if entry.path().is_file()
            && entry
                .file_name()
                .to_string_lossy()
                .to_ascii_lowercase()
                .ends_with(".json")
        {
            process_configuration(
                entry.path().as_path(),
                &llamacpp_controller,
                env_handle.clone(),
            )
            .await;
        }
    }
}

async fn process_configuration(
    json_file: &Path,
    llamacpp_backend_controller: &LlamaCppBackendController,
    env_handle: Arc<HashMap<String, String>>,
) {
    println!("processing {:#?}", json_file);

    let mut model_configuration = {
        // read file to Vec<u8>
        let data = std::fs::read(json_file).unwrap_or_else(|e| {
            panic!(
                "error parsing file '{}': {e}",
                json_file.as_os_str().to_string_lossy()
            )
        });

        // parse json
        from_slice::<ModelConfiguration>(&data).unwrap_or_else(|e| {
            panic!(
                "error parsing file '{}': {e}",
                json_file.as_os_str().to_string_lossy()
            )
        })
    };

    // create llamacpp_config from model_configuration
    let llamacpp_config = LlamaCppConfig {
        env_handle,
        args_handle: Arc::new(LlamaCppConfigArgs {
            alias: model_configuration.alias.clone(),
            api_key: None,
            model_path: model_configuration.model_path.clone(),
            mmproj_path: model_configuration.mmproj_path.clone(),
            prio: model_configuration.prio,
            threads: model_configuration.threads,
            n_gpu_layers: model_configuration.n_gpu_layers,
            jinja: model_configuration.jinja,
            ctx_size: Some(inference_backends::ContextSize::T8192),
            no_mmap: model_configuration.no_mmap,
            flash_attn: model_configuration.flash_attn.clone(),
            fit: model_configuration.fit.clone(),
            batch_size: model_configuration.batch_size,
            ubatch_size: model_configuration.ubatch_size,
            cache_type_v: model_configuration.cache_type_v.clone(),
            cache_type_k: model_configuration.cache_type_k.clone(),
            parallel: model_configuration.parallel, // FixMe: DAS SOLLTE KEIN TEIL DER MODELCONFIG SEIN
            no_context_shift: model_configuration.no_context_shift,
            no_cont_batching: model_configuration.no_cont_batching,
            min_p: model_configuration.min_p,
            temp: model_configuration.temp,
            repeat_penalty: model_configuration.repeat_penalty,
            presence_penalty: model_configuration.presence_penalty,
            seed: model_configuration.seed,
            top_k: model_configuration.top_k,
            top_p: model_configuration.top_p,
        }),
    };

    // start llama-server
    llamacpp_backend_controller.start(llamacpp_config).await;
    loop {
        match llamacpp_backend_controller.read_state().await {
            LlamaCppProcessState::Running(s)
                if &s.args_handle.alias == &model_configuration.alias =>
            {
                break;
            }
            LlamaCppProcessState::Running(_) | LlamaCppProcessState::Stopped => {
                panic!("failed to run model '{}'", &model_configuration.alias)
            }
            _ => continue,
        }
    }
    println!("llama-server now running '{}'", model_configuration.alias);

    // get models-endpoint as json
    let models: ModelList = get(format!(
        "http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/v1/models"
    ))
    .await
    .unwrap_or_else(|e| panic!("error retrieving models-json from v1/models endpoint: {e}"))
    .json()
    .await
    .unwrap_or_else(|e| panic!("error retrieving models-json from v1/models endpoint: {e}"));

    // retrieve model meta-data and capabilities as determined by llama-server

    let (model_meta_data, capabilities) = models.as_list_of_meta_data_and_capabilities().remove(0);

    // update provided model_configuration
    model_configuration.capabilities = capabilities;
    model_configuration.vocab_type = model_meta_data.vocab_type;
    model_configuration.n_vocab = model_meta_data.n_vocab;
    model_configuration.n_ctx_train = model_meta_data.n_ctx_train;
    model_configuration.n_embd = model_meta_data.n_embd;
    model_configuration.n_params = model_meta_data.n_params;
    model_configuration.size = model_meta_data.size;

    // save with updated model_configuration
    std::fs::write(
        json_file,
        serde_json::to_string_pretty(&model_configuration).unwrap_or_else(|e| {
            panic!("error serializing model_configuration to pretty-string: {e}")
        }),
    )
    .unwrap_or_else(|e| panic!("error writing serialized model_configuration to file: {e}"));

    // stop llama-server
    llamacpp_backend_controller.stop().await;
}
