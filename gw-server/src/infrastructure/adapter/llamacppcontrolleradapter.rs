use crate::domain::ports::LlamaCppControllerOutPort;
use async_trait::async_trait;
use inference_backends::{
    LlamaCppBackend, LlamaCppBackendController, LlamaCppConfig, LlamaCppProcessState,
};
use std::sync::Arc;

pub struct LlamaCppControllerAdapter {
    llamacpp_controller: LlamaCppBackendController,
}

impl LlamaCppControllerAdapter {
    pub async fn create_adapter(
        port: u16,
        llama_cpp_command: impl Into<String>,
        llama_cpp_execdir: impl Into<String>,
    ) -> Arc<dyn LlamaCppControllerOutPort> {
        let llamacpp_controller = LlamaCppBackendController::init_backend(LlamaCppBackend {
            host: "localhost".to_owned(),
            port,
            llama_cpp_command: llama_cpp_command.into(),
            llama_cpp_execdir: llama_cpp_execdir.into(),
        })
        .await;

        Arc::new(Self {
            llamacpp_controller,
        })
    }
}

#[async_trait]
impl LlamaCppControllerOutPort for LlamaCppControllerAdapter {
    async fn get_llamacpp_state(&self) -> LlamaCppProcessState {
        self.llamacpp_controller.read_state().await
    }

    async fn start_llamacpp_process(
        &self,
        llamacpp_config: &LlamaCppConfig,
    ) -> LlamaCppProcessState {
        self.llamacpp_controller
            .start(llamacpp_config.clone())
            .await;
        self.llamacpp_controller.read_state().await
    }

    async fn stop_llamacpp_process(&self) {
        self.llamacpp_controller.stop().await;
    }
}
