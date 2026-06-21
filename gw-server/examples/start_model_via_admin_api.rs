use axum::http::Method;
use rig_core::http_client::ReqwestClient;
use serde_json::json;

const BASE_URL: &str = "https://mai-server.ipv64.net:8080";

#[tokio::main]
async fn main() {
    let apikey = {
        dotenv::dotenv().ok();
        std::env::var("MAI_SERVER_APIKEY").unwrap()
    };

    let client = ReqwestClient::new();
    let r = client.request(Method::PUT, format!("{BASE_URL}/admin/llamacpp/llm"))
    .header("Authorization", format!("Bearer {apikey}"))
    .json(&json!({
        "env": {
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY" : "1"
        },
        "ctx-size" : 131072,
        "alias": "qwen3.6-27b-agentic-coding-no-reasoning",
        "model-path": "/model_data/huggingface/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q8_K_XL.gguf",
        "mmproj-path": "/model_data/huggingface/unsloth/Qwen3.6-27B-MTP-GGUF/mmproj-BF16.gguf",
        "n-gpu-layers": -1,
        "flash-attn": "on",
        "batch-size": 512,
        "ubatch-size": 2048,
        "cache-type-v": "q8_0",
        "cache-ram": -1,
        "cache-reuse": 64,
        "spec-type": "draft-mtp",
        "spec-draft-n-max": 2,
        "cache-type-k": "q8_0",
        "temp": 0.3,
        "repeat-penalty": 1.0,
        "presence-penalty": 0.0,
        "min-p": 0.0,
        "top-k": 20,
        "top-p": 0.95,
        "jinja": true,
        "mlock": true,
        "chat-template-kwargs": "{\"preserve_thinking\":\"true\"}",
        "reasoning": "off"
    })).build().unwrap();
    client.execute(r).await.unwrap();
}
