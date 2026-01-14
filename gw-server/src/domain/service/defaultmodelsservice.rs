use crate::{
    application::model::ContextSizeAwareAlias,
    domain::{
        model::ModelConfiguration,
        ports::{LlamaCppControllerOutPort, ModelLoaderOutPort, ModelsServiceInPort},
    },
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
    async fn ensure_requested_model_is_served(
        &self,
        requested_model: &str,
        timeout: Duration,
    ) -> Result<(), String> {
        let start_time = std::time::Instant::now();

        let requested_model_alias = if let Ok(context_size_aware_alias) =
            ContextSizeAwareAlias::try_from(requested_model.to_owned())
        {
            context_size_aware_alias.model()
        } else {
            requested_model.to_owned()
        };

        loop {
            if start_time.elapsed() >= timeout {
                return Err(format!(
                    "starting model '{requested_model}' ran into timeout"
                ));
            }
            if let inference_backends::LlamaCppProcessState::Running(s) =
                self.llamacpp_controller.get_llamacpp_state().await
                && s.args_handle.alias == requested_model_alias
            {
                return Ok(());
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
                Err(e) => {
                    return Err(format!(
                        "could not retrieve a configuration for model '{requested_model}': {e}"
                    ));
                }
            }
        }
    }

    async fn get_model_configuration_list(&self) -> Vec<ModelConfiguration> {
        self.model_loader.get_model_configurations().await
    }
}
