pub mod application;
pub mod domain;
pub mod infrastructure;

pub mod model;

// OVERRIDES FOR ENV-VARS
const DEFAULT_CHAT_LLM_ENV: &str = "DEFAULT_CHAT_LLM";
const DEFAULT_CHAT_LLM: &str = "qwen3.8-27b-thinking-swift15-gsq-rco-iq3"; // "qwen3.8-27b-thinking-gsq-rco-iq3-cq8-max"

pub const LLAMACPP_EXECDIR_LLM_ENV: &str = "LLAMACPP_EXECDIR_LLM";
pub const LLAMACPP_EXECDIR_LLM: &str = "/data0/inference/llama.cpp"; // "/data0/inference/llama-cpp-kolibri-1/llama.cpp"
