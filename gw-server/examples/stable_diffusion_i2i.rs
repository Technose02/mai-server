use axum::http::Method;
use base64::prelude::{BASE64_STANDARD, Engine};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use gw_server::application::model::StableDiffusionSse;
use rig_core::http_client::ReqwestClient;

const BASE_URL: &str = "https://mai-server.ipv64.net:8080";

const I2IMODEL: &str = "flux2klein9b";

const PROMPT: &str = r#"replace the man including arms, hair, hands, and clothing with a cute, cuddly, photorealistic, naked groundhog preserving the pose and proportions"#;

fn read_png_to_b64_string(path: impl AsRef<std::path::Path>) -> String {
    let bytes = std::fs::read(path.as_ref())
        .unwrap_or_else(|_| panic!("file not found: '{}'", path.as_ref().to_string_lossy()));
    BASE64_STANDARD.encode(bytes)
}

#[tokio::main]
async fn main() {
    let apikey = {
        dotenv::dotenv().ok();
        std::env::var("MAI_SERVER_APIKEY").unwrap()
    };

    let dto = gw_server::application::model::StableDiffusionPromptDto {
        prompt: String::from(PROMPT),
        width: 1024,
        height: 1024,
        ref_png_1: Some(read_png_to_b64_string("./out.png")),
        ..Default::default()
    };

    let client = ReqwestClient::new();
    let r = client
        .request(Method::POST, format!("{BASE_URL}/api/sd/{I2IMODEL}"))
        .header("Authorization", format!("Bearer {apikey}"))
        .json(&dto)
        .build()
        .unwrap();

    let res = client.execute(r).await.unwrap();
    let mut stream = res.bytes_stream().eventsource();
    while let Some(e) = stream.next().await {
        match e {
            Ok(event) => {
                if let Ok(sse) = serde_json::de::from_str::<StableDiffusionSse>(&event.data) {
                    if let StableDiffusionSse::GenerationFinished { b64_encoded_image } = sse {
                        let data = BASE64_STANDARD.decode(b64_encoded_image).unwrap();
                        std::fs::write("edited.png", data).unwrap();
                    } else {
                        println!("{sse:#?}");
                    }
                } else {
                    println!("failed to deserialize event: {}", event.event);
                }
            }
            Err(e) => {
                eprintln!("received error: {}", e);
            }
        }
    }
}
