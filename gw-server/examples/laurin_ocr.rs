use std::{
    fs::{create_dir_all, read_to_string},
    io::Write,
    path::PathBuf,
};

use base64::{Engine, prelude::BASE64_STANDARD};
use dotenv;
use futures::StreamExt;
use image::GenericImageView;
use rig_core::{
    OneOrMany,
    agent::MultiTurnStreamItem,
    client::CompletionClient,
    http_client::ReqwestClient,
    message::{DocumentSourceKind, Image, ImageDetail, ImageMediaType, Message, UserContent},
    providers::openai::CompletionsClient,
    streaming::StreamingPrompt,
};
use serde_json::json;
use tracing::info;

const MAI_SERVER_APIKEY: &str = "MAI_SERVER_APIKEY";
const SCANNED_PAGES_DIR: &str = "/data0/dev/python/laurins-geheimnis-md/data/images";
const OUT_DIR: &str = "/home/technose02/Documents/laurin_ocr_out/take3";
const SEED: u64 = 2;
const CONTINUE: bool = true;

//const MODEL: &str = "qwen3.6-27b-agentic-coding-no-reasoning-60000";
const MODEL: &str = "qwen3.6-27b-mtp-ud-q8-k-xl-non-thinking-reasoning-200000";

struct PathUtil(u16);
impl PathUtil {
    fn path_to_src_image(&self) -> PathBuf {
        PathBuf::from(SCANNED_PAGES_DIR).join(format!("laurin_{:03}.png", self.0))
    }

    fn path_to_dest_doc(&self) -> PathBuf {
        PathBuf::from(OUT_DIR).join(&format!("page_{:03}.md", self.0))
    }

    fn create_outdir() {
        create_dir_all(Self::outdir()).unwrap_or_else(|e| {
            panic!(
                "could not create dir(s) '{}': {e}",
                Self::outdir().to_string_lossy()
            )
        });
    }

    fn outdir() -> PathBuf {
        PathBuf::from(OUT_DIR)
    }

    fn path_to_all_pages_md() -> PathBuf {
        Self::outdir().join("all_pages.md")
    }
}

struct AllMd(String);
impl AllMd {
    fn new() -> Self {
        Self(String::new())
    }

    fn append(&mut self, content: impl AsRef<str>) -> &mut Self {
        if !self.0.is_empty() {
            self.0.push_str("\n\n---\n\n");
        }
        self.0.push_str(content.as_ref());
        self
    }

    async fn persist(&self) -> &Self {
        tokio::fs::write(PathUtil::path_to_all_pages_md(), &self.0)
            .await
            .unwrap_or_else(|e| panic!("error updating 'all_pages.md': {e}"));
        self
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();
    dotenv::dotenv().expect("failed to load env from .env");

    let apikey = std::env::var(MAI_SERVER_APIKEY)
        .unwrap_or_else(|_| panic!("env-var {MAI_SERVER_APIKEY} not set"));

    let client = CompletionsClient::<ReqwestClient>::builder()
        .base_url("https://mai-server.ipv64.net:8080/api/v1")
        .api_key(apikey)
        .http_client(
            ReqwestClient::builder()
                .timeout(std::time::Duration::from_millis(u64::MAX))
                .build()
                .expect("expected building a reqwest-client to succeed"),
        )
        .build()
        .expect("failed to build openai-client");

    let agent = client
        .agent(MODEL)
        .preamble(
            r#"Du bist ein Experte für die Erstellung von E-Books als Markdown-Formatierte Textseiten aus
eingescannten Buchseiten im PNG-Format.
Du erhältst von dem User gleich den Scan einer Buchseite des deutschen Romans "Laurins Geheimnis".
Gib den Inhalt dieser Seite als Text aus. Erhalte die Formatierung und Struktur.
Als Hilfe für die Einordnung des ermittelten Textes erhältst du mit dem Bild der Buchseite zusätzlich den
Text der vorherigen Seite, an den der neue Text anschließt.

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

    // Verzeichnisstruktur für die konvertiereten Seiten erstellen
    PathUtil::create_outdir();

    // Initialisierung
    let (mut idx, mut content_previous_page, mut all_md) = {
        if CONTINUE {
            let mut all_md = AllMd::new();
            {
                let mut idx_of_last_existing_doc = 0_u16;
                let mut content_previous_page = Option::None;
                while PathUtil(idx_of_last_existing_doc + 1)
                    .path_to_dest_doc()
                    .is_file()
                {
                    let path_to_last_existing_doc =
                        PathUtil(idx_of_last_existing_doc + 1).path_to_dest_doc();
                    let content_previous_page_inner = read_to_string(&path_to_last_existing_doc)
                        .unwrap_or_else(|e| {
                            format!(
                                "Error reading '{}': '{e}'",
                                path_to_last_existing_doc.to_string_lossy()
                            )
                        });
                    all_md.append(&content_previous_page_inner);
                    content_previous_page = Some(content_previous_page_inner);
                    idx_of_last_existing_doc += 1;
                }
                all_md.persist().await;
                (idx_of_last_existing_doc + 1, content_previous_page, all_md)
            }
        } else {
            (1_u16, Option::None, AllMd::new())
        }
    };

    // Forlaufende Konvertierung der gescannten Seiten
    loop {
        if let Some((image, path)) = load_image(idx) {
            info!("analyzing image {path:#?}...");

            let mut usercontent = if let Some(content_previous_page) = content_previous_page {
                OneOrMany::one(UserContent::text(format!(
                    r#"Dies ist der Text der vorherigen Seite:
<TEXT_VORHERIGE_SEITE>
{content_previous_page}
</TEXT_VORHERIGE_SEITE>
Die nächste Seite ist als Bild angefügt. Generiere nun den Text der nächsten Seite."#
                )))
            } else {
                OneOrMany::one(UserContent::text(
                    r#"Die erste Seite des Buches ist als Bild angefügt. Es gibt keine vorherige Seite. Generiere nun den Text der nächsten Seite."#,
                ))
            };

            usercontent.push(UserContent::Image(image));

            let mut final_response = None;
            let mut stream = agent
                .stream_prompt(Message::User {
                    content: usercontent,
                })
                .await;
            while let Some(n) = stream.next().await {
                match n {
                    Ok(MultiTurnStreamItem::FinalResponse(fr)) => final_response = Some(fr),
                    Ok(MultiTurnStreamItem::StreamAssistantItem(_)) => {
                        print!(".");
                        std::io::stdout().flush().unwrap();
                    }
                    Ok(_) => {}
                    Err(e) => panic!("{e}"),
                }
            }

            if let Some(final_response) = final_response {
                all_md.append(final_response.response());

                content_previous_page = Some(final_response.response().to_string());

                info!("\twriting output to {path:#?}...");

                all_md.persist().await;

                tokio::fs::write(PathUtil(idx).path_to_dest_doc(), final_response.response())
                    .await
                    .unwrap_or_else(|e| panic!("error writing output as md-file: {e}"));
                idx += 1;
                info!("\tdone");
                continue;
            } else {
                panic!("no 'final response' received!");
            }
        }
        break;
    }

    info!("all done");
}

fn load_image(idx: u16) -> Option<(Image, PathBuf)> {
    let src = PathUtil(idx).path_to_src_image();
    if src.exists() && src.is_file() {
        let mut img = image::ImageReader::open(&src)
            .unwrap()
            .decode()
            .unwrap()
            .grayscale();
        let (mut w, mut h) = img.dimensions();
        let scale = 1024.0 / ((w as f32).sqrt() * (h as f32).sqrt());
        w = (scale * w as f32).round() as u32;
        h = (scale * h as f32).round() as u32;
        let fix_w = w % 16;
        if fix_w >= 8 {
            w += 16 - fix_w;
        } else {
            w -= fix_w;
        }
        let fix_h = h % 16;
        if fix_h >= 8 {
            h += 16 - fix_h;
        } else {
            h -= fix_h;
        }
        img = img.resize(w, h, image::imageops::FilterType::Lanczos3);
        let mut img = img.to_luma8();
        let mut px_min = u8::MAX;
        let mut px_max = 0;
        for px in img.pixels_mut() {
            let val = px.0[0];
            if val > px_max {
                px_max = val;
            }
            if val < px_min {
                px_min = val;
            }
        }
        let t = (px_max - px_min) / 2;
        for px in img.pixels_mut() {
            //println!("thresholding - cur is {}, t is {t}, px_min is {px_min}, px_max is {px_max}", px.0[0]);
            px.0[0] = if px.0[0] >= t { u8::MAX } else { 0_u8 };
        }
        let mut img_buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut img_buf, image::ImageFormat::Png).unwrap();
        let image_bytes = img_buf.into_inner();
        let fname = src.file_name().unwrap().to_string_lossy().to_string();
        let fname = fname.to_lowercase().replace(".png", "converted.png");
        let processed_dst = src.parent().unwrap().to_owned().join(fname);
        std::fs::write(processed_dst, image_bytes).unwrap();

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
