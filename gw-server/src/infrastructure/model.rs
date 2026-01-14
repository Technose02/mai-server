use crate::domain::model::ModelConfiguration as DomainModelConfiguration;
use inference_backends::{ContextSize, OnOffValue};
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
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

impl ModelConfiguration {
    pub fn into_domain(self) -> DomainModelConfiguration {
        DomainModelConfiguration {
            alias: self.alias,
            model_path: self.model_path,
            mmproj_path: self.mmproj_path,
            prio: self.prio,
            threads: self.threads,
            n_gpu_layers: self.n_gpu_layers,
            jinja: self.jinja,
            no_mmap: self.no_mmap,
            flash_attn: self.flash_attn,
            fit: self.fit,
            batch_size: self.batch_size,
            ubatch_size: self.ubatch_size,
            cache_type_k: self.cache_type_k,
            cache_type_v: self.cache_type_v,
            parallel: self.parallel,
            no_context_shift: self.no_context_shift,
            no_cont_batching: self.no_cont_batching,
            min_p: self.min_p,
            temp: self.temp,
            repeat_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            top_k: self.top_k,
            top_p: self.top_p,
            max_ctx_size: self.max_ctx_size,

            vocab_type: self.vocab_type,
            n_vocab: self.n_vocab,
            n_ctx_train: self.n_ctx_train,
            n_embd: self.n_embd,
            n_params: self.n_params,
            size: self.size,
            capabilities: self.capabilities,
        }
    }
}

fn default_to_false() -> bool {
    false
}
