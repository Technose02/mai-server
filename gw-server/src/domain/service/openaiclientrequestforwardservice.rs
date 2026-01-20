use crate::domain::ports::{OpenAiClientOutPort, OpenAiRequestForwardPServiceInPort};
use async_openai::types::chat::CreateChatCompletionRequest;
use async_trait::async_trait;
use axum::{extract::Request, http::StatusCode, response::Response};
use std::sync::Arc;

pub struct OpenAiClientRequestForwardService {
    llamacpp_client: Arc<dyn OpenAiClientOutPort>,
}

impl OpenAiClientRequestForwardService {
    pub fn create_service(
        llamacpp_client: Arc<dyn OpenAiClientOutPort>,
    ) -> Arc<dyn OpenAiRequestForwardPServiceInPort> {
        Arc::new(Self { llamacpp_client })
    }
}

#[async_trait]
impl OpenAiRequestForwardPServiceInPort for OpenAiClientRequestForwardService {
    async fn process_chat_completions_request(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<Response, StatusCode> {
        self.llamacpp_client.post_chat_completions(request).await
    }

    async fn forward_api_request(&self, request: Request) -> Result<Response, StatusCode> {
        self.llamacpp_client.forward_api_request(request).await
    }

    async fn forward_ui_request(&self, request: Request) -> Result<Response, StatusCode> {
        self.llamacpp_client.forward_ui_request(request).await
    }

    async fn get_chat(&self) -> Result<Response, StatusCode> {
        self.llamacpp_client.request_chat().await
    }
}
