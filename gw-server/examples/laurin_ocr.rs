use std::path::PathBuf;

use base64::{Engine, prelude::BASE64_STANDARD};
use dotenv;
use rig::{
    client::CompletionClient,
    completion::Prompt,
    http_client::ReqwestClient,
    message::{DocumentSourceKind, Image, ImageDetail, ImageMediaType},
};
use serde_json::json;
use tracing::info;

const MAI_SERVER_APIKEY: &str = "MAI_SERVER_APIKEY";
const SCANNED_PAGES_DIR: &str = "/data0/dev/python/laurins-geheimnis-md/data/images";
const OUT_DIR: &str = "/home/technose02/Documents/laurin_ocr_out/take2";
const SEED: u64 = 2;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();
    dotenv::dotenv().expect("failed to load env from .env");

    let apikey = std::env::var(MAI_SERVER_APIKEY)
        .unwrap_or_else(|_| panic!("env-var {MAI_SERVER_APIKEY} not set"));

    let client = rig::providers::openai::client::CompletionsClient::<ReqwestClient>::builder()
        .base_url("https://mai-server.ipv64.net:8080/api/v1")
        .api_key(apikey)
        .http_client(
            ReqwestClient::builder()
                .timeout(std::time::Duration::from_millis(300000))
                .build()
                .expect("expected building a reqwest-client to succeed"),
        )
        .build()
        .expect("failed to build openai-client");

    let agent = client
        .agent("qwen3.5-35b-a3b-non-thinking-general-min")
        .preamble(
            r#"Du bist ein Experte für die Erstellung von E-Books als Markdown-Formatierte Textseiten aus
eingescannten Buchseiten im PNG-Format.
Du erhältst von dem User gleich den Scan einer Buchseite des deutschen Romans "Laurins Geheimnis".
Gib den Inhalt dieser Seite als Text aus. Erhalte die Formatierung und Struktur.
Verwende Markdown für die Formatierung und nutze die in Markdown verfügbaren Möglichkeiten:
- nutze Überschriften für neue Kapitel
- nutze kursive und fett gedruckte Schrift innerhalb des Textflusses, wenn das original entsprechende Hervorhebungen nutzt
- Grenze normale Zeilenumbrüche von Absätze-Wechseln ab
- halte die einzelnen Text-Absätzen strikt ein, ignoriere aber die einfachen Zeilenumbrüche
- achte insbesondere bei Gesprächen auf die Einteilung der einzelnen Gesprächsanteile auf Absätze,
  so dass es sich der als Markdown gerenderte Text optimal und gewohnt liest

Antworte ausschließlich mit dem generierten Text."#,
        )
        .additional_params(json!({
            "seed": SEED
        }))
        .build();

    let mut idx = 1_u16;
    let laurin_ocr_out = std::path::PathBuf::from(OUT_DIR);
    if let Err(e) = std::fs::create_dir_all(&laurin_ocr_out) {
        panic!("could not create dir(s) '{:#?}': {e}", &laurin_ocr_out)
    }
    loop {
        if let Some((image, path)) = load_image(idx as usize) {
            info!("analyzing image {path:#?}...");
            match agent.prompt(image).await {
                Ok(res) => {
                info!("\twriting output to {path:#?}...");
                tokio::fs::write(laurin_ocr_out.join(&format!("page_{:03}.md", idx)), res)
                    .await
                    .unwrap_or_else(|e| panic!("error writing output as md-file: {e}"));
                idx += 1;
                info!("\tdone");
                continue;
                },
                Err(e) => {
                    panic!("{e}")
                }
            }

        }
        break;
    }

    info!("all done");
}

fn load_image(idx: usize) -> Option<(Image, PathBuf)> {
    let src = std::path::PathBuf::from(SCANNED_PAGES_DIR).join(format!("laurin_{idx:03}.png"));
    if src.exists() && src.is_file() {
        let image_bytes = std::fs::read(&src).unwrap();
        let base64_image = BASE64_STANDARD.encode(image_bytes);

        Some((
            Image {
                data: DocumentSourceKind::Base64(base64_image),
                media_type: Some(ImageMediaType::PNG),
                detail: Some(ImageDetail::Auto),
                additional_params: None,
            },
            src,
        ))
    } else {
        None
    }
}
