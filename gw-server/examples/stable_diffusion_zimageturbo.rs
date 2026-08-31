use axum::http::Method;
use base64::Engine;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use gw_server::application::model::StableDiffusionSse;
use rig_core::http_client::ReqwestClient;

const BASE_URL: &str = "https://mai-server.ipv64.net:8080";

#[tokio::main]
async fn main() {
    let apikey = {
        dotenv::dotenv().ok();
        std::env::var("MAI_SERVER_APIKEY").unwrap()
    };

    let dto = gw_server::application::model::StableDiffusionPromptDto {
        prompt: String::from(
            "an exhausted rust developer who just fell asleep while hacking another crate",
        ),
        width: 1024,
        height: 1024,
        ..Default::default()
    };

    let client = ReqwestClient::new();
    let r = client
        .request(Method::POST, format!("{BASE_URL}/api/sd/zimageturbo"))
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
                        let data = base64::prelude::BASE64_STANDARD
                            .decode(b64_encoded_image)
                            .unwrap();
                        std::fs::write("out.png", data).unwrap();
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
