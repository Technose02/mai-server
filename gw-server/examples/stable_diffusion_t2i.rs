use axum::http::Method;
use base64::prelude::{BASE64_STANDARD, Engine};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use gw_server::application::model::StableDiffusionSse;
use rig_core::http_client::ReqwestClient;

const BASE_URL: &str = "https://mai-server.ipv64.net:8080";

//const T2IMODEL: &str = "animaturbo";
const T2IMODEL: &str = "booguimageturbo";
//const T2IMODEL: &str = "flux2klein9b";
//const T2IMODEL: &str = "fluxdev";
//const T2IMODEL: &str = "fluxschnell";
//const T2IMODEL: &str = "krea2turbo";
//const T2IMODEL: &str = "mageflowturbo";
//const T2IMODEL: &str = "zimageturbo";

const PROMPT: &str = r#"an exhausted german software developer who just fell asleep while hacking together a new crate (lib) for the rust programming language"#;

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
        ..Default::default()
    };

    let client = ReqwestClient::new();
    let r = client
        .request(Method::POST, format!("{BASE_URL}/api/sd/{T2IMODEL}"))
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
