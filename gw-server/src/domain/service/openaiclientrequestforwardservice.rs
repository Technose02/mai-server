use crate::domain::ports::{OpenAiClientOutPort, OpenAiRequestForwardPServiceInPort};
use async_trait::async_trait;
use axum::{extract::Request, http::StatusCode, response::Response};
use openai_api_rust::chat::ChatBody as ChatCompletionsRequest;
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
        request: ChatCompletionsRequest,
    ) -> Result<Response, StatusCode> {
        self.llamacpp_client.post_chat_completions(request).await
    }

    async fn forward_openai_request(&self, request: Request) -> Result<Response, StatusCode> {
        self.llamacpp_client.forward_request(request).await
    }
}
