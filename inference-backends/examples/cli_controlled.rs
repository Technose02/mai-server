use std::{collections::HashMap, io::stdin, sync::Arc};

use inference_backends::{
    AttnSetting, ComfyUiBackend, ComfyUiBackendController, ComfyUiConfig, ComfyUiConfigArgs,
    ContextSize, LlamaCppBackend, LlamaCppBackendController, LlamaCppConfig, LlamaCppConfigArgs,
    OnOffValue, VRamSetting,
};

#[tokio::main]
async fn main() {
    let mut shared_env = HashMap::new();
    shared_env.insert("GGML_CUDA_ENABLE_UNIFIED_MEMORY".into(), "1".into());
    let shared_env = Arc::new(shared_env);

    let llamacpp_backend = LlamaCppBackend {
        host: "0.0.0.0".to_owned(),
        port: 11440,
        llama_cpp_command: "./build/bin/llama-server".to_owned(),
        llama_cpp_execdir: "/data0/inference/llama.cpp/".to_owned(),
    };

    let llama_cpp_backend_controller =
        LlamaCppBackendController::init_backend(llamacpp_backend).await;

    let comfyui_backend = ComfyUiBackend {
        listen: "0.0.0.0".to_owned(),
        port: 3000,
        comfyui_main_py: "main.py".to_owned(),
        comfyui_setup_sh: "setup_for_normal_run.sh".to_owned(),
        comfyui_execdir: "/data0/inference/ComfyUI".to_owned(),
    };
    let comfyui_backend_controller = ComfyUiBackendController::init_backend(comfyui_backend).await;

    let config_1 = LlamaCppConfig {
        env_handle: shared_env.clone(),
        args_handle: LlamaCppConfigArgs {
            alias: "devstral-small-2-24B-instruct-2512".to_string(),
            api_key: Some("apikey1".to_string()),
            model_path: "/model_data/huggingface/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-UD-Q8_K_XL.gguf".to_string(),
            mmproj_path: Some("/model_data/huggingface/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/mmproj-F16.gguf".to_string()),
            prio: Some(3),
            min_p: Some(0.01),
            threads: Some(-1),
            n_gpu_layers: Some(99),
            jinja: true,
            ctx_size: Some(ContextSize::T262144),
            no_mmap: true,
            flash_attn: Some(OnOffValue::On),
            batch_size: None,
            ubatch_size: None,
            cache_type_v: None,
            cache_type_k: None,
            parallel: None,
            no_context_shift: false,
            no_cont_batching: false,
            temp: None,
            repeat_penalty: None,
            seed:Some(1),
            top_k:None,
            top_p: None,
            fit: None,
            presence_penalty: None,
        }
        .into()
    };

    let config_2 = LlamaCppConfig {
        env_handle: shared_env.clone(),
        args_handle: LlamaCppConfigArgs {
            alias: "gpt-oss-120b-Q8_0".to_string(),
            api_key: Some("apikey2".to_string()),
            model_path: "/model_data/huggingface/unsloth/gpt-oss-120b-GGUF/Q8_0/gpt-oss-120b-Q8_0-00001-of-00002.gguf"
                .to_string(),
            mmproj_path: None,
            prio: None,
            min_p: None,
            threads: None,
            n_gpu_layers: Some(99),
            jinja: true,
            ctx_size: Some(ContextSize::T32768),
            no_mmap: true,
            flash_attn: Some(OnOffValue::On),
            batch_size: Some(2048),
            ubatch_size: Some(2048),
            cache_type_v: Some("q8_0".to_string()),
            cache_type_k: Some("q8_0".to_string()),
            parallel: Some(1),
            no_context_shift: true,
            no_cont_batching: true,
            temp: None,
            repeat_penalty: None,
            seed: Some(1),
            top_k: None,
            top_p: None,
            fit: None,
            presence_penalty: None,
        }
        .into(),
    };

    let comfyui_config = ComfyUiConfig {
        env_handle: shared_env,
        args_handle: ComfyUiConfigArgs {
            allow_origin: Some(String::from("localhost:5050")),
            fp32_vae: true,
            use_flash_attention: true,
            vram_setting: Some(VRamSetting::HighVram),
            attn_setting: Some(AttnSetting::PytorchCrossAttention),
        }
        .into(),
    };

    loop {
        let mut cmd = String::new();
        stdin().read_line(&mut cmd).unwrap();
        match cmd.trim() {
            "1" => llama_cpp_backend_controller.start(config_1.clone()).await,
            "2" => llama_cpp_backend_controller.start(config_2.clone()).await,
            "s" => llama_cpp_backend_controller.stop().await,
            "r" => println!(
                "llamacpp-state: {:?}",
                llama_cpp_backend_controller.read_state().await
            ),
            "q" => {
                println!("bye");
                break;
            }
            "3" => {
                comfyui_backend_controller
                    .start(comfyui_config.clone())
                    .await
            }
            "d" => comfyui_backend_controller.stop().await,
            "t" => println!(
                "comfyui-state: {:?}",
                comfyui_backend_controller.read_state().await
            ),
            _ => {
                println!(
                    r#"type...
    '1' to start llamacpp-backend with configuration 1 (devstral),
    '2' to start llamacpp-backend with configuration 2 (gpt-oss-120b),
    '3' to start comfyui-backend
    'r' to read and print the current llamacpp-backend-state,
    't' to read and print the current comfyui-backend-state,
    's' to stop llamacpp-backend,
    'd' to stop comfyui-backend and
    'q' to quit"#
                )
            }
        }
    }
}
