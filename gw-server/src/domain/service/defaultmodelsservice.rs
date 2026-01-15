use crate::domain::{
    model::ModelConfiguration,
    ports::{LlamaCppControllerOutPort, ModelLoaderOutPort, ModelsServiceInPort},
};
use async_trait::async_trait;
use std::{sync::Arc, time::Duration};

pub struct DefaultModelsService {
    llamacpp_controller: Arc<dyn LlamaCppControllerOutPort>,
    model_loader: Arc<dyn ModelLoaderOutPort>,
}

impl DefaultModelsService {
    pub fn create_service(
        llamacpp_controller: Arc<dyn LlamaCppControllerOutPort>,
        model_loader: Arc<dyn ModelLoaderOutPort>,
    ) -> Arc<dyn ModelsServiceInPort> {
        Arc::new(Self {
            llamacpp_controller,
            model_loader,
        })
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

    async fn ensure_requested_model_is_served(
        &self,
        requested_model: &str,
        timeout: Duration,
    ) -> Result<(), ()> {
        let start_time = std::time::Instant::now();

        loop {
            if start_time.elapsed() >= timeout {
                println!("starting model variant '{requested_model}' ran into timeout");
                return Err(());
            }
            if let inference_backends::LlamaCppProcessState::Running(s) =
                self.llamacpp_controller.get_llamacpp_state().await
            {
                if s.args_handle.alias == requested_model {
                    return Ok(());
                } else {
                    println!(
                        "problem: requested-model is '{requested_model}' but a model '{}' is still running",
                        s.args_handle.alias
                    )
                }
            }

            match self
                .model_loader
                .get_model_configuration(requested_model)
                .await
            {
                Ok(llamacpp_config) => {
                    self.llamacpp_controller
                        .start_llamacpp_process(llamacpp_config.as_ref())
                        .await;
                    println!(
                        "waiting for backend to serve '{requested_model}' ({}s)",
                        start_time.elapsed().as_secs()
                    );
                    tokio::time::sleep(Duration::from_millis(500)).await
                }
                Err(()) => {
                    eprintln!("could not retrieve a configuration for model '{requested_model}'");
                    return Err(());
                }
            }
        }
    }

    async fn get_model_configuration_list(&self) -> Vec<ModelConfiguration> {
        self.model_loader.get_model_configurations().await
    }

    fn get_default_model_alias(&self) -> String {
        //"gpt-oss-120b-Q8_0-small".to_string()
        "gemma-3-12b-it-qat-Q8_0-moderate".to_string()
    }
}
