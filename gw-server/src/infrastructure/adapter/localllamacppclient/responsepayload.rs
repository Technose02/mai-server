use axum::{
    body::Bytes,
    http::{HeaderValue, StatusCode},
};
use flate2::read::GzDecoder;
use std::io::Read;
use tracing::{debug, error, info};

pub enum InnerPayload {
    Zipped(Bytes),
    UnEncoded(Bytes),
}

impl TryFrom<InnerPayload> for Vec<u8> {
    type Error = StatusCode;

    fn try_from(value: InnerPayload) -> Result<Self, Self::Error> {
        match value {
            InnerPayload::UnEncoded(data) => Ok(data.into()),
            InnerPayload::Zipped(data) => {
                let mut decoder = GzDecoder::new(&data[..]);
                let mut decoded = Vec::new();
                decoder.read_to_end(&mut decoded).map_err(|e| {
                    error!("Gzip Dekomprimierung fehlgeschlagen: {e}");
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;
                Ok(decoded)
            }
        }
    }
}

impl TryFrom<InnerPayload> for String {
    type Error = StatusCode;

    fn try_from(value: InnerPayload) -> Result<Self, Self::Error> {
        match value {
            InnerPayload::UnEncoded(data) => Ok(String::from_utf8_lossy(&data).to_string()),
            InnerPayload::Zipped(data) => {
                let mut decoder = GzDecoder::new(&data[..]);
                let mut decoded_text = String::new();
                decoder.read_to_string(&mut decoded_text).map_err(|e| {
                    error!("Gzip Dekomprimierung fehlgeschlagen: {e}");
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;
                Ok(decoded_text)
            }
        }
    }
}

pub enum ResponsePayload {
    Html(InnerPayload),
    Js(InnerPayload),
    Json(InnerPayload),
    Css(InnerPayload),
    Raw(InnerPayload),
}

impl ResponsePayload {
    pub fn try_from_body(
        content_type: &str,
        content_encoding: Option<&HeaderValue>,
        bytes: Bytes,
    ) -> Result<Self, StatusCode> {
        let inner_payload = match content_encoding.and_then(|v| v.to_str().ok()) {
            Some("gzip") => Ok(InnerPayload::Zipped(bytes)),
            Some(encoding) => {
                error!("unexpected content-encoding: {encoding}");
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
            None => Ok(InnerPayload::UnEncoded(bytes)),
        }?;

        match content_type.to_lowercase() {
            css if css.starts_with("text/css") => Ok(Self::Css(inner_payload)),
            json if json.starts_with("application/json") => Ok(Self::Json(inner_payload)),
            js if js.starts_with("application/javascript") => Ok(Self::Js(inner_payload)),
            html if html.starts_with("text/html") => Ok(Self::Html(inner_payload)),
            _ => Ok(Self::Raw(inner_payload)),
        }
    }

    pub fn process(self) -> Result<Vec<u8>, StatusCode> {
        match self {
            Self::Css(inner) => {
                info!("performing replacement in css-payload");
                let processed = Ok(Self::process_css(inner.try_into()?).into());
                debug!("processed body");
                processed
            }
            Self::Js(inner) => {
                info!("performing replacement in javascript-payload");
                let processed = Ok(Self::process_js(inner.try_into()?).into());
                debug!("processed body");
                processed
            }
            Self::Json(inner) => {
                info!("performing replacement in JSON-payload");
                let processed = Ok(Self::process_json(inner.try_into()?).into());
                debug!("processed body");
                processed
            }
            Self::Html(inner) => {
                info!("performing replacement in html-payload");
                let processed = Ok(Self::process_html(inner.try_into()?).into());
                debug!("processed body");
                processed
            }
            Self::Raw(inner) => {
                info!("not performing replacement in raw payload");
                inner.try_into()
            }
        }
    }
}

impl ResponsePayload {
    fn universal_replacements(mut content: String) -> String {
        // replacing in html
        content = content.replace(r#"./bundle."#, r#"/chat/bundle."#);

        // replacing in js
        content = content.replace("}/props", "}/chat/props");
        content = content.replace("./props", "/chat/props");
        content = content.replace(
            r#"CORS_PROXY_ENDPOINT="/cors-proxy""#,
            r#"CORS_PROXY_ENDPOINT="/chat/cors-proxy""#,
        );

        content = content.replace("./v1/chat/completions", "./api/1/v1/chat/completions");

        content = content.replace(
            r#"={LIST:"/v1/models",LOAD:"/models/load",UNLOAD:"/models/unload"}"#,
            r#"={LIST:"/api/1/v1/models",LOAD:"/models/load",UNLOAD:"/models/unload"}"#,
        );
        content = content.replace(
            r#"={LIST:"/tools",EXECUTE:"/tools"}"#,
            r#"={LIST:"/chat/tools",EXECUTE:"/tools"}"#,
        );
        content = content.replace(r#"/_app/"#, r#"/chat/_app/"#);
        content = content.replace(r#"favicon.ico"#, r#"chat/favicon.ico"#);
        content = content.replace(r#"favicon.svg"#, r#"chat/favicon.svg"#);
        content = content.replace(r#"./apple-"#, r#"./chat/apple/apple-"#);
        content = content.replace(r#"href="apple-"#, r#"href="chat/apple/apple-"#);
        content = content.replace(r#"/build.json"#, r#"/chat/build.json"#);

        content
    }

    fn process_json(content: String) -> String {
        Self::universal_replacements(content)
    }
    fn process_css(content: String) -> String {
        Self::universal_replacements(content)
    }
    fn process_js(content: String) -> String {
        Self::universal_replacements(content)
    }
    fn process_html(content: String) -> String {
        Self::universal_replacements(content)
    }
}

/*
                || t.starts_with("application/javascript")
                || t.starts_with("text/css") =>
            {
                info!("not performing replacement in payload of content-type {content_type}");
                true
            }
            _ => {
                info!("not performing replacement in payload of content-type {content_type}");
                false
            }






        let payload = match content_type.to_lowercase() {
            json if json.starts_with("application/json") => Payload::Json(bytes),
            js if js.starts_with("application/javascript") => Payload::Js(bytes),
            css if css.starts_with("text/css") => Payload::Css(bytes),
            _ => Payload::Raw(bytes),
        };

        let process_as_text = match content_type.to_lowercase() {
            t if t.starts_with("application/json")
                || t.starts_with("application/javascript")
                || t.starts_with("text/css") =>
            {
                info!("not performing replacement in payload of content-type {content_type}");
                true
            }
            _ => {
                info!("not performing replacement in payload of content-type {content_type}");
                false
            }
        };

        let data = if process_as_text {
            // decode if zipped
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

            // replacing in html
            plain_content = plain_content.replace(r#"./bundle."#, r#"/chat/bundle."#);

            // replacing in js
            plain_content = plain_content.replace("}/props", "}/chat/props");
            plain_content = plain_content.replace("./props", "/chat/props");
            plain_content = plain_content.replace(
                r#"CORS_PROXY_ENDPOINT="/cors-proxy""#,
                r#"CORS_PROXY_ENDPOINT="/chat/cors-proxy""#,
            );

            plain_content =
                plain_content.replace("./v1/chat/completions", "./api/1/v1/chat/completions");

            plain_content = plain_content.replace(
                r#"={LIST:"/v1/models",LOAD:"/models/load",UNLOAD:"/models/unload"}"#,
                r#"={LIST:"/api/1/v1/models",LOAD:"/models/load",UNLOAD:"/models/unload"}"#,
            );
            plain_content = plain_content.replace(
                r#"={LIST:"/tools",EXECUTE:"/tools"}"#,
                r#"={LIST:"/chat/tools",EXECUTE:"/tools"}"#,
            );
            plain_content = plain_content.replace(r#"/_app/"#, r#"/chat/_app/"#);
            plain_content = plain_content.replace(r#"favicon.ico"#, r#"chat/favicon.ico"#);
            plain_content = plain_content.replace(r#"favicon.svg"#, r#"chat/favicon.svg"#);
            plain_content = plain_content.replace(r#"./apple-"#, r#"./chat/apple/apple-"#);
            plain_content = plain_content.replace(r#"href="apple-"#, r#"href="chat/apple/apple-"#);
            plain_content = plain_content.replace(r#"/build.json"#, r#"/chat/build.json"#);

            debug!("processed body");

            trace!("received content: {plain_content}");

            // repack body
            plain_content.into_bytes()
        } else {
            bytes.into()
        };
*/
