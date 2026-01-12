use crate::{
    application::model::Llmodels,
    domain::ports::{LlamaCppControllerOutPort, ModelsServiceInPort},
};
use async_trait::async_trait;
use inference_backends::{ContextSize, LlamaCppConfig, LlamaCppConfigArgs, OnOffValue};
use std::{collections::HashMap, sync::Arc, time::Duration};

pub struct DefaultModelsService {
    llamacpp_controller: Arc<dyn LlamaCppControllerOutPort>,
    llmodels: Arc<Llmodels>,
}

impl DefaultModelsService {
    pub fn create_service(
        llamacpp_controller: Arc<dyn LlamaCppControllerOutPort>,
        llmodels: Llmodels,
    ) -> Arc<dyn ModelsServiceInPort> {
        Arc::new(Self {
            llamacpp_controller,
            llmodels: Arc::new(llmodels),
        })
    }

    fn get_llamacpp_config(&self, requested_model: &str) -> Option<LlamaCppConfig> {
        if requested_model == "nemotron-3-nano-30b-a3b" {
            let env_handle = {
                let mut map = HashMap::<String, String>::new();
                map.insert("GGML_CUDA_ENABLE_UNIFIED_MEMORY".into(), "1".into());
                Arc::new(map)
            };

            let args_handle = Arc::new(LlamaCppConfigArgs {
                alias: String::from("nemotron-3-nano-30b-a3b"),
                api_key: None,
                model_path: String::from(
                    "/model_data/huggingface/unsloth/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf",
                ),
                n_gpu_layers: Some(99),
                jinja: true,
                mmproj_path: None,
                threads: Some(-1),
                ctx_size: Some(ContextSize::T32768),
                temp: Some(1.0),
                top_p: Some(1.0),
                fit: Some(OnOffValue::On),
                flash_attn: Some(OnOffValue::On),
                prio: None,
                no_mmap: false,
                batch_size: None,
                ubatch_size: None,
                cache_type_k: None,
                cache_type_v: None,
                parallel: None,
                no_cont_batching: false,
                no_context_shift: false,
                min_p: None,
                repeat_penalty: None,
                presence_penalty: None,
                seed: None,
                top_k: None,
            });

            Some(LlamaCppConfig {
                env_handle,
                args_handle,
            })
        } else {
            None
        }
    }
}

#[async_trait]
impl ModelsServiceInPort for DefaultModelsService {
    async fn ensure_requested_model_is_served(
        &self,
        requested_model: &str,
        timeout: Duration,
    ) -> Result<(), String> {
        let start_time = std::time::Instant::now();

        loop {
            if start_time.elapsed() >= timeout {
                return Err(format!(
                    "starting model '{requested_model}' ran into timeout"
                ));
            }
            if let inference_backends::LlamaCppProcessState::Running(s) =
                self.llamacpp_controller.get_llamacpp_state().await
                && s.args_handle.alias == requested_model
            {
                return Ok(());
            }
            if let Some(llamacpp_config) = self.get_llamacpp_config(requested_model) {
                self.llamacpp_controller
                    .start_llamacpp_process(llamacpp_config)
                    .await;
                println!(
                    "waiting for backend to serve '{requested_model}' ({}s)",
                    start_time.elapsed().as_secs()
                );
                tokio::time::sleep(Duration::from_millis(500)).await
            } else {
                return Err(format!(
                    "could not retrieve a configuration for model '{requested_model}'"
                ));
            }
        }
    }

    fn llmodels(&self) -> Arc<Llmodels> {
        self.llmodels.clone()
    }
}
