use crate::{SecurityConfig, domain::ports::OpenAiClientOutPort};
use async_openai::types::chat::CreateChatCompletionRequest;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{
        StatusCode, Uri, Version,
        header::{
            ACCEPT_ENCODING, AUTHORIZATION, CONTENT_LENGTH, HOST as HOST_HEADER, HeaderValue,
        },
    },
    response::{IntoResponse, Response},
};
use flate2::{Compression, read::GzDecoder};
use http_body_util::BodyExt;
use std::{
    io::{Read, Write},
    sync::Arc,
};

const LLAMACPP_HTTP_SCHEME: &str = "http";
const LLAMACPP_HOST: &str = "localhost";
const LLAMACPP_API_BASE_PATH: &str = "v1";

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

    async fn forward_request(
        &self,
        mut request: Request,
        strip_source_path_prefix: Option<&str>,
        prefix_target_path: Option<&str>,
    ) -> Result<Response, StatusCode> {
        let path_and_query = {
            let path = request.uri().path();
            let path_and_query = request
                .uri()
                .path_and_query()
                .map(|v| v.as_str())
                .unwrap_or(path);

            if let Some(source_path_prefix_to_strip) = strip_source_path_prefix {
                println!(
                    "orginal path_and_query: {path_and_query}, source_path_prefix_to_strip: {source_path_prefix_to_strip}"
                );
                if let Some(path_and_query_stripped_prefix) =
                    path_and_query.strip_prefix(source_path_prefix_to_strip)
                {
                    path_and_query_stripped_prefix
                } else {
                    path_and_query
                }
            } else {
                path_and_query
            }
        };

        let uri_string = if let Some(path_prefix) = prefix_target_path {
            format!(
                "{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}/{path_prefix}/{}",
                self.llamacpp_port, path_and_query
            )
        } else {
            format!(
                "{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}/{}",
                self.llamacpp_port, path_and_query
            )
        };

        println!("forwarding request to llama-server using uri {uri_string}");

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
}

#[async_trait]
impl OpenAiClientOutPort for LocalLlamaCppClientAdapter {
    async fn forward_api_request(&self, request: Request) -> Result<Response, StatusCode> {
        self.forward_request(request, None, Some(LLAMACPP_API_BASE_PATH))
            .await
    }

    async fn forward_ui_request(&self, request: Request) -> Result<Response, StatusCode> {
        self.forward_request(request, Some("/chat/"), None).await
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
            "{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}/{LLAMACPP_API_BASE_PATH}/chat/completions",
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

    async fn request_chat(&self) -> Result<Response, StatusCode> {
        let request = Request::get(format!(
            "{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}",
            self.llamacpp_port
        ))
        .header(HOST_HEADER, LLAMACPP_HOST)
        .header(ACCEPT_ENCODING, "gzip")
        //        .header(ACCEPT_ENCODING, "identity, gzip;q=0, deflate;q=0, br;q=0")
        .version(Version::HTTP_11)
        .body(Body::empty())
        .map_err(|e| {
            eprintln!("error building llama.cpp-request: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        let response = self.client.request(request).await.map_err(|e| {
            eprintln!("error loading llama-server's chat: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        let (mut res_parts, res_body) = response.into_parts();

        // load whole body and unzip
        let bytes = res_body
            .collect()
            .await
            .map_err(|e| {
                eprintln!("Fehler beim Sammeln des Body: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .to_bytes();

        println!("received {} bytes", bytes.len());
        let mut decoder = GzDecoder::new(&bytes[..]);
        let mut decoded_text = String::new();
        decoder.read_to_string(&mut decoded_text).map_err(|e| {
            eprintln!("Gzip Dekomprimierung fehlgeschlagen: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        //println!("uncompressed body");

        // process decoded text
        decoded_text = decoded_text.replace("}/props", "}/chat/props");
        decoded_text = decoded_text.replace("./props", "./chat/props");
        decoded_text = decoded_text.replace("}/v1", "}/api/1/v1");
        decoded_text = decoded_text.replace("./v1", "./api/1/v1");

        //println!("processed body");

        // repack body
        let data = decoded_text.into_bytes();

        let buf = Vec::with_capacity(data.len());
        let mut enc = flate2::write::GzEncoder::new(buf, Compression::best());
        enc.write_all(&data).map_err(|e| {
            eprintln!("error zipping payload: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        let repacked = enc.finish().map_err(|e| {
            eprintln!("error zipping payload: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        //println!("recompressed body");

        res_parts.headers.insert(
            CONTENT_LENGTH,
            HeaderValue::from_str(&format!("{}", repacked.len())).map_err(|e| {
                eprintln!(
                    "error creating HeaderValue from payload-length {}: {e}",
                    repacked.len()
                );
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
        );

        Ok(Response::from_parts(res_parts, repacked.into()))
    }
}
