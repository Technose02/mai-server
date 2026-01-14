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
        requested_model_variant: &str,
        timeout: Duration,
    ) -> Result<(), ()> {
        let start_time = std::time::Instant::now();

        let (model, optional_context_size) = if let Ok(context_size_aware_alias) =
            ContextSizeAwareAlias::try_from(requested_model_variant.to_owned())
        {
            println!(
                "requested model-variant is '{requested_model_variant}' (of model '{}' and ctx-size {})",
                context_size_aware_alias.model(),
                context_size_aware_alias.context_size().as_ref()
            );
            (
                context_size_aware_alias.model(),
                Some(context_size_aware_alias.context_size()),
            )
        } else {
            println!("requested model '{requested_model_variant}'");
            (requested_model_variant.to_owned(), None)
        };

        loop {
            if start_time.elapsed() >= timeout {
                println!("starting model variant '{requested_model_variant}' ran into timeout");
                return Err(());
            }
            if let inference_backends::LlamaCppProcessState::Running(s) =
                self.llamacpp_controller.get_llamacpp_state().await
            {
                if s.args_handle.alias == model {
                    return Ok(());
                } else {
                    println!(
                        "problem: requested-model is '{model}' but a model '{}' is still running",
                        s.args_handle.alias
                    )
                }
            }

            match self
                .model_loader
                .get_model_configuration(&model, optional_context_size)
                .await
            {
                Ok(llamacpp_config) => {
                    self.llamacpp_controller
                        .start_llamacpp_process(llamacpp_config.as_ref())
                        .await;
                    println!(
                        "waiting for backend to serve '{model}' with requested Context-Size '{}' ({}s)",
                        optional_context_size
                            .map(|c| Into::<u64>::into(&c))
                            .unwrap_or(0),
                        start_time.elapsed().as_secs()
                    );
                    tokio::time::sleep(Duration::from_millis(500)).await
                }
                Err(()) => {
                    eprintln!(
                        "could not retrieve a configuration for model '{requested_model_variant}'"
                    );
                    return Err(());
                }
            }
        }
    }

    async fn get_model_configuration_list(&self) -> Vec<ModelConfiguration> {
        self.model_loader.get_model_configurations().await
    }
}
