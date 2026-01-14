use inference_backends::{ContextSize, OnOffValue};

#[derive(Clone, Debug)]
pub struct ModelConfiguration {
    pub alias: String,

    pub vocab_type: u8,
    pub n_vocab: u64,
    pub n_ctx_train: u64,
    pub n_embd: u64,
    pub n_params: u64,
    pub size: u64,
    pub capabilities: Vec<String>,

    pub max_ctx_size: ContextSize,
    pub model_path: String,
    pub mmproj_path: Option<String>,

    pub prio: Option<u8>,
    pub threads: Option<i8>,
    pub n_gpu_layers: Option<u8>,
    pub jinja: bool,
    pub no_mmap: bool,

    pub flash_attn: Option<OnOffValue>,
    pub fit: Option<OnOffValue>,
    pub batch_size: Option<u16>,
    pub ubatch_size: Option<u16>,
    pub cache_type_v: Option<String>,
    pub cache_type_k: Option<String>,
    pub parallel: Option<u8>,
    pub no_context_shift: bool,
    pub no_cont_batching: bool,
    pub min_p: Option<f32>,
    pub temp: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub top_k: Option<u16>,
    pub top_p: Option<f32>,
}
