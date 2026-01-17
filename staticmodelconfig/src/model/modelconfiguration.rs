use inference_backends::{ContextSize, OnOffValue};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case")]
pub struct ModelConfiguration {
    pub alias: String,

    pub model_path: String,

    pub max_ctx_size: ContextSize,

    pub vocab_type: u8,
    pub n_vocab: u64,
    pub n_ctx_train: u64,
    pub n_embd: u64,
    pub n_params: u64,
    pub size: u64,
    pub capabilities: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub mmproj_path: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub prio: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub threads: Option<i8>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub n_gpu_layers: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub flash_attn: Option<OnOffValue>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub fit: Option<OnOffValue>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub batch_size: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub ubatch_size: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub cache_type_v: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub cache_type_k: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub parallel: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub temp: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub top_k: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub top_p: Option<f32>,

    #[serde(
        skip_serializing_if = "std::ops::Not::not",
        default = "default_to_false"
    )]
    pub jinja: bool,
    #[serde(
        skip_serializing_if = "std::ops::Not::not",
        default = "default_to_false"
    )]
    pub no_mmap: bool,
    #[serde(
        skip_serializing_if = "std::ops::Not::not",
        default = "default_to_false"
    )]
    pub no_context_shift: bool,
    #[serde(
        skip_serializing_if = "std::ops::Not::not",
        default = "default_to_false"
    )]
    pub no_cont_batching: bool,
}

fn default_to_false() -> bool {
    false
}
