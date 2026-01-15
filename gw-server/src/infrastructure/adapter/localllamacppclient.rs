use crate::{SecurityConfig, domain::ports::OpenAiClientOutPort};
use async_openai::types::chat::CreateChatCompletionRequest;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{
        StatusCode, Uri, Version,
        header::{AUTHORIZATION, HOST as HOST_HEADER, HeaderValue},
    },
    response::{IntoResponse, Response},
};
use std::sync::Arc;

const LLAMACPP_HTTP_SCHEME: &str = "http";
const LLAMACPP_HOST: &str = "localhost";
const LLAMACPP_BASE_PATH: &str = "v1";

use hyper_util::{
    client::legacy::{Client as LegacyClient, connect::HttpConnector},
    rt::TokioExecutor,
};

type Client = LegacyClient<HttpConnector, axum::body::Body>;

pub struct LocalLlamaCppClientAdapter {
    client: Client,
    llamacpp_port: u16,
    security_config: Arc<dyn SecurityConfig>,
}

impl LocalLlamaCppClientAdapter {
    pub fn create_adapter(
        port: u16,
        security_config: Arc<dyn SecurityConfig>,
    ) -> Arc<dyn OpenAiClientOutPort> {
        let client = LegacyClient::builder(TokioExecutor::new()).build(HttpConnector::new());

        Arc::new(Self {
            client,
            llamacpp_port: port,
            security_config,
        })
    }
}

#[async_trait]
impl OpenAiClientOutPort for LocalLlamaCppClientAdapter {
    async fn forward_request(&self, mut request: Request) -> Result<Response, StatusCode> {
        let path_and_query = {
            let path = request.uri().path();
            request
                .uri()
                .path_and_query()
                .map(|v| v.as_str())
                .unwrap_or(path)
        };

        let uri_string = format!(
            "{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}/{LLAMACPP_BASE_PATH}/{}",
            self.llamacpp_port, path_and_query
        );

        *request.uri_mut() = Uri::try_from(uri_string.clone())
            .unwrap_or_else(|_| panic!("{uri_string} expected to be a valid uri"));

        request.headers_mut().insert(
            HOST_HEADER,
            HeaderValue::from_str(LLAMACPP_HOST).expect("hostname expected as valid headervalue"),
        );

        *request.version_mut() = Version::HTTP_11;

        Ok(self
            .client
            .request(request)
            .await
            .map_err(|e| {
                eprintln!("error forwarding request to llama.cpp: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .into_response())
    }

    async fn post_chat_completions(
        &self,
        payload: CreateChatCompletionRequest,
    ) -> Result<Response, StatusCode> {
        let json_string = serde_json::to_string(&payload).map_err(|e| {
            eprintln!("error converting payload to json-string: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        let request = Request::post(format!(
            "{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}/{LLAMACPP_BASE_PATH}/chat/completions",
            self.llamacpp_port
        ))
        .header(
            AUTHORIZATION,
            format!("Bearer {}", self.security_config.get_apikey()),
        )
        .header(HOST_HEADER, LLAMACPP_HOST)
        .version(Version::HTTP_11)
        .body(Body::from(json_string))
        .map_err(|e| {
            eprintln!("error building llama.cpp-request: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        Ok(self
            .client
            .request(request)
            .await
            .map_err(|e| {
                eprintln!("error posting chat completions to llama.cpp: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .into_response())
    }
}
