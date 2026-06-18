use axum::http::Method;
use rig::http_client::ReqwestClient;
use serde_json::json;

#[tokio::main]
async fn main() {
    let apikey = {
        dotenv::dotenv().ok();
        std::env::var("MAI_SERVER_APIKEY").unwrap()
    };


    let client = ReqwestClient::new();
    let r = client.request(Method::PUT, "https://mai-server.ipv64.net:8080/admin/llamacpp/llm")
    .header("Authorization", format!("Bearer {apikey}"))
    .json(&json!({
        "env": {
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY" : "1"
        },
        "ctx-size" : 131072,
        "alias": "qwen3.6-35b-a3b-mtp-thinking-precise-coding",
        "model-path": "/model_data/huggingface/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf",
        "mmproj-path": "/model_data/huggingface/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/mmproj-BF16.gguf",
        "n-gpu-layers": 99,
        "flash-attn": "on",
        "batch-size": 1024,
        "spec-type": "draft-mtp",
        "spec-draft-n-max": 2,
        "temp": 0.6,
        "repeat-penalty": 1.0,
        "presence-penalty": 0.0,
        "min-p": 0.0,
        "top-k": 20,
        "top-p": 0.95,
        "jinja": true,
        "mlock": true
    })).build().unwrap();
    client.execute(r).await.unwrap();
}
