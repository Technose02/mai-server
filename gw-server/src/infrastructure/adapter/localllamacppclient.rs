use crate::domain::ports::OpenAiClientOutPort;
use async_trait::async_trait;
use axum::{
    extract::Request,
    http::{
        StatusCode, Uri, Version,
        header::{HOST as HOST_HEADER, HeaderValue},
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
}

impl LocalLlamaCppClientAdapter {
    pub fn create_adapter(port: u16) -> Arc<dyn OpenAiClientOutPort> {
        let client = LegacyClient::builder(TokioExecutor::new()).build(HttpConnector::new());

        Arc::new(Self {
            client,
            llamacpp_port: port,
        })
    }
}

#[async_trait]
impl OpenAiClientOutPort for LocalLlamaCppClientAdapter {
    async fn send_request(
        &self,
        mut request: Request,
        path_and_query: Option<&str>,
    ) -> Result<Response, StatusCode> {
        let path_and_query = {
            if let Some(path_and_query) = path_and_query {
                path_and_query
            } else {
                let path = request.uri().path();
                request
                    .uri()
                    .path_and_query()
                    .map(|v| v.as_str())
                    .unwrap_or(path)
            }
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
                eprintln!("{e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .into_response())
    }
}
