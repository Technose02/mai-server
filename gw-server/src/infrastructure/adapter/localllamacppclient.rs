use crate::domain::ports::OpenAiClientOutPort;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{
        StatusCode, Uri, Version,
        header::{HOST as HOST_HEADER, HeaderValue},
    },
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::BodyExt;
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

        println!("forwarding request {:#?} to llama-server", request,);
        match self.client.request(request).await {
            Ok(res) => {
                let status = res.status();
                let headers = res.headers().clone();
                let upstream_body = res.into_body();

                type BoxError = Box<dyn std::error::Error + Send + Sync>;

                let transformed_stream = upstream_body.into_data_stream().map(|result| {
                    result
                        .map(|data| {
                            let text = String::from_utf8_lossy(&data).clone().to_string();
                            // process text if necessary
                            println!("received some data: {text}");
                            Bytes::from(text)
                        })
                        .map_err(|e| Box::new(e) as BoxError)
                });

                let body = Body::from_stream(transformed_stream);

                Ok((status, headers, body).into_response())
            }
            Err(e) => {
                println!("received error: {e}");
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
}
