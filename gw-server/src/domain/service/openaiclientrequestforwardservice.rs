use crate::domain::ports::{OpenAiClientOutPort, OpenAiRequestForwardPServiceInPort};
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
    async fn process_openai_request(
        &self,
        request: Request,
        api_path: &str,
    ) -> Result<Response, StatusCode> {
        self.llamacpp_client
            .send_request(request, Some(api_path))
            .await
    }
}
