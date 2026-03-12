use dotenv;
use rig::{client::EmbeddingsClient, embeddings::EmbeddingModel};
use tracing::info;

const MAI_SERVER_APIKEY: &str = "MAI_SERVER_APIKEY";

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();
    dotenv::dotenv().expect("failed to load env from .env");

    let apikey = std::env::var(MAI_SERVER_APIKEY)
        .unwrap_or_else(|_| panic!("env-var {MAI_SERVER_APIKEY} not set"));

    let client = rig::providers::openai::Client::builder()
        .base_url("https://mai-server.ipv64.net:8080/api/v1")
        .api_key(apikey)
        .build()
        .expect("failed to create openai-client");

    //let embedding_model = client.embedding_model("text-embedding-3-small");
    //let embedding_model = client.embedding_model("all-MiniLM-L6-v2");
    //let embedding_model = client.embedding_model("jina-embeddings-v5-text-small-retrieval");
    let embedding_model = client.embedding_model("qwen3-embedding-0.6B");
    //let embedding_model = client.embedding_model("qwen3-embedding-4B");
    //let embedding_model = client.embedding_model("qwen3-embedding-8B");

    let text_to_embed = std::env::args()
        .nth(1)
        .unwrap_or("this is an embedding test".into());

    let embeddings = embedding_model
        .embed_text(&text_to_embed)
        .await
        .expect("failed to embed text");

    info!("embeddings: {:#?}", embeddings);
    info!("length of embeddings-vec: {}", embeddings.vec.len())
}
