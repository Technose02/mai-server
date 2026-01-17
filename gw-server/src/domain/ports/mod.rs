use async_openai::types::chat::CreateChatCompletionRequest;
use async_trait::async_trait;
use axum::{extract::Request, http::StatusCode, response::Response};
use inference_backends::{LlamaCppConfig, LlamaCppProcessState};
use staticmodelconfig::ModelConfiguration;
use staticmodelconfig::ModelList;
use std::{sync::Arc, time::Duration};

/// IN-PORTS

#[async_trait]
pub trait OpenAiRequestForwardPServiceInPort: Send + Sync + 'static {
    async fn process_chat_completions_request(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<Response, StatusCode>;

    async fn forward_openai_request(&self, request: Request) -> Result<Response, StatusCode>;

    async fn get_chat(&self) -> Result<Response, StatusCode>;
}

#[async_trait]
pub trait ModelManagerServiceInPort: Send + Sync + 'static {
    async fn get_llamacpp_state(&self) -> LlamaCppProcessState;
    async fn stop_llamacpp_process(&self);
    async fn start_llamacpp_process(
        &self,
        llamacpp_config: &LlamaCppConfig,
    ) -> LlamaCppProcessState;
}

#[async_trait]
pub trait ModelsServiceInPort: Send + Sync + 'static {
    async fn ensure_any_model_is_served(
        &self,
        default_model_alias: &str,
        timeout: Duration,
    ) -> Result<(), ()>;
    async fn ensure_requested_model_is_served(
        &self,
        requested_model_variant: &str,
        timeout: Duration,
    ) -> Result<(), ()>;

    /// returns the ModelList to return on the models-endpoint
    async fn get_models(&self) -> ModelList;
    //async fn get_model_configuration_list(&self) -> Vec<ModelConfiguration>;

    fn get_default_model_alias(&self) -> String;
}

/// OUT-PORTS

#[async_trait]
pub trait OpenAiClientOutPort: Send + Sync + 'static {
    async fn post_chat_completions(
        &self,
        payload: CreateChatCompletionRequest,
    ) -> Result<Response, StatusCode>;
    async fn forward_request(&self, request: Request) -> Result<Response, StatusCode>;
    async fn request_chat(&self) -> Result<Response, StatusCode>;
}

#[async_trait]
pub trait LlamaCppControllerOutPort: Send + Sync + 'static {
    async fn get_llamacpp_state(&self) -> LlamaCppProcessState;
    async fn start_llamacpp_process(
        &self,
        llamacpp_config: &LlamaCppConfig,
    ) -> LlamaCppProcessState;
    async fn stop_llamacpp_process(&self);
}

#[async_trait]
pub trait ModelLoaderOutPort: Send + Sync + 'static {
    async fn get_static_model_configurations(&self) -> Vec<ModelConfiguration>;
    async fn get_model_configuration(&self, alias: &str) -> Result<Arc<LlamaCppConfig>, ()>;
}
