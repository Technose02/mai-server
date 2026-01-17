use std::{collections::HashMap, sync::Arc};

use inference_backends::{ContextSize, LlamaCppConfigArgs, OnOffValue};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum LlamaCppProcessState {
    Stopped,
    Running(LlamaCppConfig),
    Starting(LlamaCppConfig),
    Stopping,
}

impl From<inference_backends::LlamaCppProcessState> for LlamaCppProcessState {
    fn from(value: inference_backends::LlamaCppProcessState) -> Self {
        match value {
            managed_process::ProcessState::Running(config) => {
                LlamaCppProcessState::Running(config.into())
            }
            managed_process::ProcessState::Starting(config) => {
                LlamaCppProcessState::Starting(config.into())
            }
            managed_process::ProcessState::Stopped => LlamaCppProcessState::Stopped,
            managed_process::ProcessState::Stopping(_, None) => LlamaCppProcessState::Stopping,
            managed_process::ProcessState::Stopping(_, Some(config)) => {
                LlamaCppProcessState::Starting(config.into())
            }
        }
    }
}

fn default_to_false() -> bool {
    false
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case")]
pub struct LlamaCppConfig {
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,

    pub alias: String,

    pub model_path: String,

    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub mmproj_path: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub prio: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub threads: Option<i8>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub n_gpu_layers: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none", default = "Option::default")]
    pub ctx_size: Option<ContextSize>,
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

impl From<inference_backends::LlamaCppConfig> for LlamaCppConfig {
    fn from(value: inference_backends::LlamaCppConfig) -> Self {
        let env = (*value.env_handle).clone();
        LlamaCppConfig {
            env,
            alias: value.args_handle.alias.clone(),
            batch_size: value.args_handle.batch_size,
            model_path: value.args_handle.model_path.clone(),
            mmproj_path: value.args_handle.mmproj_path.clone(),
            prio: value.args_handle.prio,
            min_p: value.args_handle.min_p,
            threads: value.args_handle.threads,
            n_gpu_layers: value.args_handle.n_gpu_layers,
            jinja: value.args_handle.jinja,
            ctx_size: value.args_handle.ctx_size,
            cache_type_k: value.args_handle.cache_type_k.clone(),
            cache_type_v: value.args_handle.cache_type_v.clone(),
            no_mmap: value.args_handle.no_mmap,
            flash_attn: value.args_handle.flash_attn.clone(),
            fit: value.args_handle.fit.clone(),
            ubatch_size: value.args_handle.ubatch_size,
            no_context_shift: value.args_handle.no_context_shift,
            no_cont_batching: value.args_handle.no_cont_batching,
            parallel: value.args_handle.parallel,
            temp: value.args_handle.temp,
            repeat_penalty: value.args_handle.repeat_penalty,
            presence_penalty: value.args_handle.presence_penalty,
            seed: value.args_handle.seed,
            top_k: value.args_handle.top_k,
            top_p: value.args_handle.top_p,
        }
    }
}

impl LlamaCppConfig {
    pub fn map(&self, apikey: Option<impl Into<String>>) -> inference_backends::LlamaCppConfig {
        let env_handle = Arc::new(self.env.clone());
        let args_handle = Arc::new(LlamaCppConfigArgs {
            alias: self.alias.clone(),
            api_key: apikey.map(Into::<String>::into),
            batch_size: self.batch_size,
            cache_type_k: self.cache_type_k.clone(),
            cache_type_v: self.cache_type_v.clone(),
            ctx_size: self.ctx_size,
            model_path: self.model_path.clone(),
            mmproj_path: self.mmproj_path.clone(),
            prio: self.prio,
            min_p: self.min_p,
            threads: self.threads,
            n_gpu_layers: self.n_gpu_layers,
            jinja: self.jinja,
            no_mmap: self.no_mmap,
            flash_attn: self.flash_attn.clone(),
            fit: self.fit.clone(),
            ubatch_size: self.ubatch_size,
            parallel: self.parallel,
            no_cont_batching: self.no_cont_batching,
            no_context_shift: self.no_context_shift,
            temp: self.temp,
            repeat_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            top_k: self.top_k,
            top_p: self.top_p,
        });

        inference_backends::LlamaCppConfig {
            env_handle,
            args_handle,
        }
    }
}
