use crate::domain::ports::{LlamaCppControllerOutPort, ModelLoaderOutPort, ModelsServiceInPort};
use async_trait::async_trait;
use inference_backends::{ContextSize, LlamaCppConfigArgs, LlamaCppRunConfig};
use staticmodelconfig::{ContextSizeAwareAlias, ModelList};
use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, RwLock},
    time::Duration,
};

pub struct DefaultModelsService {
    llamacpp_controller: Arc<dyn LlamaCppControllerOutPort>,
    model_loader: Arc<dyn ModelLoaderOutPort>,
    llamacpp_parallel_processings: RwLock<u8>,
    environment_args: Arc<HashMap<String, String>>,
    cached_model_list: OnceLock<Arc<ModelList>>,
}

impl DefaultModelsService {
    pub fn create_service(
        llamacpp_controller: Arc<dyn LlamaCppControllerOutPort>,
        model_loader: Arc<dyn ModelLoaderOutPort>,
        llamacpp_parallel_processings: u8,
        environment_args: HashMap<String, String>,
    ) -> Arc<dyn ModelsServiceInPort> {
        Arc::new(Self {
            llamacpp_controller,
            model_loader,
            llamacpp_parallel_processings: RwLock::new(llamacpp_parallel_processings),
            environment_args: Arc::new(environment_args),
            cached_model_list: OnceLock::new(),
        })
    }

    fn create_run_config_from_args_and_current_state(
        &self,
        llamacpp_config_args: Arc<LlamaCppConfigArgs>,
    ) -> LlamaCppRunConfig {
        let llamacpp_parallel_processings = *self.llamacpp_parallel_processings.read().unwrap();
        LlamaCppRunConfig {
            args_handle: llamacpp_config_args.clone(),
            env_handle: self.environment_args.clone(),
            parallel: llamacpp_parallel_processings,
        }
    }
}

#[async_trait]
impl ModelsServiceInPort for DefaultModelsService {
    async fn ensure_any_model_is_served(
        &self,
        default_model_alias: &str,
        timeout: Duration,
    ) -> Result<(), ()> {
        let current_state = self.llamacpp_controller.get_llamacpp_state().await;
        if matches!(
            current_state,
            inference_backends::LlamaCppProcessState::Running(_)
        ) {
            Ok(())
        } else {
            self.ensure_requested_model_is_served(default_model_alias, timeout)
                .await
        }
    }

    fn set_parallel_backend_requests(&self, parallel_backend_requests: u8) {
        let old = {
            let _guard = self.llamacpp_parallel_processings.read().unwrap();
            *_guard
        };
        if parallel_backend_requests != old {
            println!("switching parallel_backend_requests to {parallel_backend_requests}");
            let mut _guard = self.llamacpp_parallel_processings.write().unwrap();
            *_guard = parallel_backend_requests;
        }
    }

    async fn ensure_requested_model_is_served(
        &self,
        requested_model: &str,
        timeout: Duration,
    ) -> Result<(), ()> {
        let start_time = std::time::Instant::now();

        let mut waiting_notified = false;
        loop {
            if start_time.elapsed() >= timeout {
                println!("starting model variant '{requested_model}' ran into timeout");
                return Err(());
            }
            if let inference_backends::LlamaCppProcessState::Running(running_config) =
                self.llamacpp_controller.get_llamacpp_state().await
            {
                let running_config_args_handle = running_config.args_handle.clone();
                if requested_model == running_config_args_handle.alias {
                    let runconfig_as_requested = self
                        .create_run_config_from_args_and_current_state(running_config_args_handle);
                    if runconfig_as_requested == running_config {
                        return Ok(());
                    } else {
                        println!(
                            "problem: requested-model is '{requested_model}' is running but the process was started with different run-params"
                        );
                    }
                } else {
                    println!(
                        "problem: requested-model is '{requested_model}' but a model '{}' is still running",
                        running_config.args_handle.alias
                    )
                }
            }

            match self
                .model_loader
                .get_model_configuration(requested_model)
                .await
            {
                Ok(llamacpp_config_args) => {
                    let llamacpp_run_config =
                        self.create_run_config_from_args_and_current_state(llamacpp_config_args);

                    self.llamacpp_controller
                        .start_llamacpp_process(llamacpp_run_config)
                        .await;
                    if !waiting_notified {
                        println!("waiting for backend to serve '{requested_model}'...)");
                    }
                    waiting_notified = true;
                    tokio::time::sleep(Duration::from_millis(500)).await
                }
                Err(()) => {
                    eprintln!("could not retrieve a configuration for model '{requested_model}'");
                    return Err(());
                }
            }
        }
    }

    fn get_models(&self) -> Arc<ModelList> {
        let handle = self.cached_model_list.get_or_init(|| {
            let static_model_configurations = self.model_loader.get_static_model_configurations();
            let mut model_list = ModelList::with_capacity(static_model_configurations.len());

            for base_configuration in static_model_configurations.iter().cloned() {
                'inner: for ctx_size in [
                    ContextSize::T8192,
                    ContextSize::T16384,
                    ContextSize::T32768,
                    ContextSize::T65536,
                    ContextSize::T131072,
                    ContextSize::T262144,
                ] {
                    if ctx_size > base_configuration.max_ctx_size {
                        break 'inner;
                    }
                    let mut base_configuration = base_configuration.clone();
                    let caa = ContextSizeAwareAlias::from((base_configuration.alias, ctx_size));
                    base_configuration.alias = caa.alias();
                    println!("adding model '{}' to list", base_configuration.alias);
                    model_list.add_model_configuration(&base_configuration);
                }
            }

            Arc::new(model_list)
        });
        handle.clone()
    }

    fn get_model_names(&self) -> String {
        self.get_models().names()
    }

    fn get_default_model_alias(&self) -> String {
        //"gpt-oss-120b-Q8_0-small".to_string()
        "gemma-3-12b-it-qat-Q8_0-moderate".to_string()
    }
}
