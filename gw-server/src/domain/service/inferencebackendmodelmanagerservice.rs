use crate::domain::ports::{LlamaCppControllerOutPort, ModelManagerServiceInPort};
use async_trait::async_trait;
use inference_backends::{LlamaCppConfig, LlamaCppProcessState};
use std::sync::Arc;

pub struct InferenceBackendModelManagerService {
    llamacpp_controller: Arc<dyn LlamaCppControllerOutPort>,
}

impl InferenceBackendModelManagerService {
    pub fn create_service(
        llamacpp_controller: Arc<dyn LlamaCppControllerOutPort>,
    ) -> Arc<dyn ModelManagerServiceInPort> {
        Arc::new(Self {
            llamacpp_controller,
        })
    }
}

#[async_trait]
impl ModelManagerServiceInPort for InferenceBackendModelManagerService {
    async fn get_llamacpp_state(&self) -> LlamaCppProcessState {
        self.llamacpp_controller.get_llamacpp_state().await
    }

    async fn stop_llamacpp_process(&self) {
        self.llamacpp_controller.stop_llamacpp_process().await
    }

    async fn start_llamacpp_process(
        &self,
        llamacpp_config: &LlamaCppConfig,
        parallel: u8,
    ) -> LlamaCppProcessState {
        self.llamacpp_controller
            .start_llamacpp_process(llamacpp_config, parallel)
            .await
    }
}
