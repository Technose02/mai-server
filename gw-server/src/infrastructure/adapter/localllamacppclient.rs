use crate::{SecurityConfig, domain::ports::OpenAiClientOutPort};
use async_openai::types::{chat::CreateChatCompletionRequest, embeddings::CreateEmbeddingRequest};
use async_trait::async_trait;
use axum::{
    body::{Body, Bytes},
    extract::Request,
    http::{
        StatusCode, Uri, Version,
        header::{
            ACCEPT_ENCODING, AUTHORIZATION, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE,
            HOST as HOST_HEADER, HeaderValue,
        },
    },
    response::{IntoResponse, Response},
};
use flate2::{Compression, read::GzDecoder};
use futures::StreamExt;
use http_body_util::BodyExt;
use std::{
    io::{Error as IoError, Read, Write},
    sync::Arc,
};
use tokio_util::{
    codec::{FramedRead, LinesCodec},
    io::StreamReader,
};
use tracing::{debug, error, info, trace, warn};

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
        prefix_target_path: Option<&str>,
    ) -> Result<Response, StatusCode> {
        let path_and_query = {
            let path = request.uri().path();
            request
                .uri()
                .path_and_query()
                .map(|v| v.as_str())
                .unwrap_or(path)
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

        trace!("forwarding request to llama-server using uri {uri_string}");

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
                error!("error forwarding request to llama.cpp: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .into_response())
    }

    async fn forward_uichat_request_and_process(
        &self,
        mut request: Request,
    ) -> Result<Response, StatusCode> {
        let path_and_query = {
            let path = request.uri().path();
            let path_and_query = request
                .uri()
                .path_and_query()
                .map(|v| v.as_str())
                .unwrap_or(path);

            trace!("mapping path-and-query from '{}'", path_and_query);

            match path_and_query {
                "/" => "/",
                "/chat/bundle.css" => "/bundle.css",
                "/chat/bundle.js" => "/bundle.js",
                s if s.starts_with("/chat/props") => s.strip_prefix("/chat").unwrap(),
                other => {
                    warn!("no mapping-rule for path-and-query of '{other} ; forwarding directly '");
                    other
                }
            }
        };

        let uri_string = format!(
            "{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}{}",
            self.llamacpp_port, path_and_query
        );

        trace!("forwarding request to llama-server using uri {uri_string}");

        *request.uri_mut() = Uri::try_from(uri_string.clone())
            .unwrap_or_else(|_| panic!("{uri_string} expected to be a valid uri"));

        *request.version_mut() = Version::HTTP_11;

        let response = self.client.request(request).await.map_err(|e| {
            error!("error forwarding to uichat: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        let (mut res_parts, res_body) = response.into_parts();

        let content_encoding = res_parts.headers.get(CONTENT_ENCODING);

        // load whole body and unzip
        let bytes = res_body
            .collect()
            .await
            .map_err(|e| {
                error!("Fehler beim Sammeln des Body: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .to_bytes();

        debug!("received {} bytes", bytes.len());

        let mut plain_content = match content_encoding.and_then(|v| v.to_str().ok()) {
            Some("gzip") => {
                let mut decoder = GzDecoder::new(&bytes[..]);
                let mut decoded_text = String::new();
                decoder.read_to_string(&mut decoded_text).map_err(|e| {
                    error!("Gzip Dekomprimierung fehlgeschlagen: {e}");
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;
                Ok(decoded_text)
            }
            Some(encoding) => {
                error!("unexpected content-encoding of llama-cpp-response: {encoding}");
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
            None => Ok(String::from_utf8_lossy(&bytes).to_string()),
        }?;

        // process decoded text
        trace!(
            "performing replacement in payload of content-type {}",
            res_parts
                .headers
                .get(CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown")
        );
        plain_content = plain_content.replace(r#"./bundle."#, r#"/chat/bundle."#);
        plain_content = plain_content.replace("}/props", "}/chat/props");
        plain_content = plain_content.replace("./props", "/chat/props");
        plain_content = plain_content.replace("}/v1", "}/api/1/v1");
        plain_content = plain_content.replace("./v1", "./api/1/v1");
        plain_content = plain_content.replace("/v1/models", "/api/1/v1/models");

        debug!("processed body");

        trace!("received content: {plain_content}");

        // repack body
        let data = plain_content.into_bytes();

        let buf = Vec::with_capacity(data.len());
        let mut enc = flate2::write::GzEncoder::new(buf, Compression::best());
        enc.write_all(&data).map_err(|e| {
            error!("error zipping payload: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        let repacked = enc.finish().map_err(|e| {
            error!("error zipping payload: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        debug!("recompressed body");

        res_parts.headers.insert(
            CONTENT_LENGTH,
            HeaderValue::from_str(&format!("{}", repacked.len())).map_err(|e| {
                error!(
                    "error creating HeaderValue from payload-length {}: {e}",
                    repacked.len()
                );
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
        );

        res_parts.headers.insert(
            CONTENT_ENCODING,
            HeaderValue::from_str("gzip").map_err(|e| {
                error!(
                    "error creating HeaderValue for content-encoding 'gzip' {}: {e}",
                    repacked.len()
                );
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
        );

        Ok(Response::from_parts(res_parts, repacked.into()))
    }
}

#[async_trait]
impl OpenAiClientOutPort for LocalLlamaCppClientAdapter {
    async fn forward_api_request(&self, request: Request) -> Result<Response, StatusCode> {
        self.forward_request(request, Some(LLAMACPP_API_BASE_PATH))
            .await
    }

    async fn forward_ui_request(&self, request: Request) -> Result<Response, StatusCode> {
        self.forward_uichat_request_and_process(request).await
    }

    async fn post_chat_completions(
        &self,
        payload: CreateChatCompletionRequest,
    ) -> Result<Response, StatusCode> {
        info!("entered post_chat_completions");
        let json_string = serde_json::to_string(&payload).map_err(|e| {
            error!("error converting payload to json-string: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        let mut request_builder = Request::post(format!(
            "{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}/{LLAMACPP_API_BASE_PATH}/chat/completions",
            self.llamacpp_port
        ));

        if let Some(security_apikey) = self.security_config.get_apikey() {
            request_builder =
                request_builder.header(AUTHORIZATION, format!("Bearer {}", security_apikey));
        }

        let mut request = request_builder
            .header(HOST_HEADER, LLAMACPP_HOST)
            .version(Version::HTTP_11)
            .body(Body::from(json_string))
            .map_err(|e| {
                error!("error building llama.cpp-request: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        request
            .headers_mut()
            .insert(ACCEPT_ENCODING, HeaderValue::from_static("identity"));

        info!("request built: {request:#?} -> forwarding to llama-server");

        let response = self
            .client
            .request(request)
            .await
            .map_err(|e| {
                error!("error posting chat completions to llama.cpp: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .into_response();

        info!("now streaming response");

        let (mut parts, body) = response.into_parts();

        info!("received response with parts: {parts:#?}");

        let stream_reader = StreamReader::new(
            body.into_data_stream()
                .map(|res| res.map_err(IoError::other)),
        );

        let mut lines = FramedRead::new(stream_reader, LinesCodec::new());

        info!("receiving lines from stream");
        let sanitized_stream = async_stream::stream! {
            let next = lines.next().await;
            info!("read '{next:#?}'");
            while let Some(Ok(line)) = lines.next().await {
                if line.is_empty() { continue; }

                // FIX 1: "content":null -> "content":""
                if line.find("\"content\":null").is_some() {
                    warn!(r#"santizer found '"content":null' ; replacing with '"content":""'"#);
                }
                let mut sanitized = line.replace("\"content\":null", "\"content\":\"\"");

                // FIX 2: Sicherstellen, dass nach 'data:' ein Leerzeichen kommt (Rig/OpenAI Standard)
                if sanitized.starts_with("data:") && !sanitized.starts_with("data: ") {
                    warn!(r#"santizer found '"data:"' ; replacing with '"data: "'"#);
                    sanitized = sanitized.replacen("data:", "data: ", 1);
                }

                // FIX 3: SSE-Formatierung wahren (\n\n am Ende jedes Events)
                let sanitized = format!("{}\n\n", sanitized);
                info!(r#"now yielding '{sanitized}'"#);
                yield Ok::<Bytes, std::io::Error>(Bytes::from(sanitized));
            }

            // FIX 4: Sicherstellen, dass der Stream sauber terminiert
            let epilogue = "data: [DONE]\n\n";
            info!(r#"finalizing stream with '{epilogue}'"#);
            yield Ok::<Bytes, std::io::Error>(Bytes::from(epilogue));
        };

        // 3. Neue Response mit dem transformierten Stream zurückgeben
        let new_body = Body::from_stream(sanitized_stream);

        // Entferne diese Header zwingend, da dein Sanitizer-Stream nun "reiner" Text ist
        parts.headers.remove(CONTENT_ENCODING);
        parts.headers.remove(CONTENT_LENGTH);

        // Optional: Erzwinge den richtigen Content-Type für SSE
        parts
            .headers
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        Ok(Response::from_parts(parts, new_body))
    }

    async fn post_embedding(
        &self,
        payload: CreateEmbeddingRequest,
    ) -> Result<Response, StatusCode> {
        let json_string = serde_json::to_string(&payload).map_err(|e| {
            error!("error converting payload to json-string: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        let mut request_builder = Request::post(format!(
            "{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}/{LLAMACPP_API_BASE_PATH}/embeddings",
            self.llamacpp_port
        ));

        if let Some(security_apikey) = self.security_config.get_apikey() {
            request_builder =
                request_builder.header(AUTHORIZATION, format!("Bearer {}", security_apikey));
        }

        let request = request_builder
            .header(HOST_HEADER, LLAMACPP_HOST)
            .version(Version::HTTP_11)
            .body(Body::from(json_string))
            .map_err(|e| {
                error!("error building llama.cpp-request: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        Ok(self
            .client
            .request(request)
            .await
            .map_err(|e| {
                error!("error posting embeddings to llama.cpp: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .into_response())
    }

    async fn request_chat(&self) -> Result<Response, StatusCode> {
        let url = format!(
            "{LLAMACPP_HTTP_SCHEME}://{LLAMACPP_HOST}:{}",
            self.llamacpp_port
        );
        trace!("requesting '{url}'");
        let request = Request::get(url)
            .header(HOST_HEADER, LLAMACPP_HOST)
            .header(ACCEPT_ENCODING, "gzip")
            .version(Version::HTTP_11)
            .body(Body::empty())
            .map_err(|e| {
                error!("error building llama.cpp-request: {e}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        self.forward_uichat_request_and_process(request).await
    }
}
