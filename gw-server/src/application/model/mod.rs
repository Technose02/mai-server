use std::{collections::HashMap, sync::Arc};

use async_openai::types::chat::CreateChatCompletionRequest;
use axum::{
    extract::Request,
    http::{StatusCode, header},
};
use http_body_util::BodyExt;
use inference_backends::{ContextSize, LlamaCppConfigArgs, LlamaCppRunConfig, OnOffValue};
use serde::{Deserialize, Serialize};
use tracing::{error, trace};

const DEFAULT_PARALLEL: u8 = 1;

// Structs here are wrappers for the models from inference backends but enriched with Serialization/Deserialization capabilities

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum LlamaCppProcessStateResponse {
    Stopped,
    Running(LlamaCppRunConfigDto),
    Starting(LlamaCppRunConfigDto),
    Stopping,
}

impl From<inference_backends::LlamaCppProcessState> for LlamaCppProcessStateResponse {
    fn from(value: inference_backends::LlamaCppProcessState) -> Self {
        match value {
            managed_process::ProcessState::Running(run_config) => {
                LlamaCppProcessStateResponse::Running(run_config.into())
            }
            managed_process::ProcessState::Starting(run_config) => {
                LlamaCppProcessStateResponse::Starting(run_config.into())
            }
            managed_process::ProcessState::Stopped => LlamaCppProcessStateResponse::Stopped,
            managed_process::ProcessState::Stopping(_, None) => {
                LlamaCppProcessStateResponse::Stopping
            }
            managed_process::ProcessState::Stopping(_, Some(run_config)) => {
                LlamaCppProcessStateResponse::Starting(run_config.into())
            }
        }
    }
}

fn default_to_false() -> bool {
    false
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case")]
pub struct LlamaCppRunConfigDto {
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
    pub parallel: Option<u8>,
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

impl LlamaCppRunConfigDto {
    pub fn map_into_domain(self, api_key: impl Into<String>) -> LlamaCppRunConfig {
        LlamaCppRunConfig {
            env_handle: Arc::new(self.env),
            parallel: self.parallel.unwrap_or(DEFAULT_PARALLEL),
            threads: self.threads.unwrap_or(-1),
            args_handle: Arc::new(LlamaCppConfigArgs {
                alias: self.alias.clone(),
                api_key: Some(api_key.into()),
                batch_size: self.batch_size,
                cache_type_k: self.cache_type_k.clone(),
                cache_type_v: self.cache_type_v.clone(),
                ctx_size: self.ctx_size,
                model_path: self.model_path.clone(),
                mmproj_path: self.mmproj_path.clone(),
                prio: self.prio,
                min_p: self.min_p,
                n_gpu_layers: self.n_gpu_layers,
                jinja: self.jinja,
                no_mmap: self.no_mmap,
                flash_attn: self.flash_attn.clone(),
                fit: self.fit.clone(),
                ubatch_size: self.ubatch_size,
                no_cont_batching: self.no_cont_batching,
                no_context_shift: self.no_context_shift,
                temp: self.temp,
                repeat_penalty: self.repeat_penalty,
                presence_penalty: self.presence_penalty,
                seed: self.seed,
                top_k: self.top_k,
                top_p: self.top_p,
            }),
        }
    }
}

impl From<LlamaCppRunConfig> for LlamaCppRunConfigDto {
    fn from(value: LlamaCppRunConfig) -> Self {
        LlamaCppRunConfigDto {
            env: (*value.env_handle).clone(),
            parallel: Some(value.parallel),
            alias: value.args_handle.alias.clone(),
            batch_size: value.args_handle.batch_size,
            model_path: value.args_handle.model_path.clone(),
            mmproj_path: value.args_handle.mmproj_path.clone(),
            prio: value.args_handle.prio,
            min_p: value.args_handle.min_p,
            threads: Some(value.threads),
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
            temp: value.args_handle.temp,
            repeat_penalty: value.args_handle.repeat_penalty,
            presence_penalty: value.args_handle.presence_penalty,
            seed: value.args_handle.seed,
            top_k: value.args_handle.top_k,
            top_p: value.args_handle.top_p,
        }
    }
}

pub async fn try_map_request_body_to_create_chat_completion_request(
    request: Request,
    model_alias: impl AsRef<str>,
) -> Result<CreateChatCompletionRequest, StatusCode> {
    let sent_from_ui = {
        if let Some(referer) = request.headers().get(header::REFERER) {
            if let Ok(referer) = referer.to_str()
                && referer.ends_with("/chat")
            {
                trace!("request is assumed to be sent from the chat-ui");
                true
            } else {
                false
            }
        } else {
            false
        }
    };

    let request_body = request
        .into_body()
        .into_data_stream()
        .collect()
        .await
        .map_err(|e| {
            error!("error reading request-body as bytes: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .to_bytes();
    let mut request_body = String::from_utf8_lossy(request_body.trim_ascii()).to_string();

    request_body = request_body.replace(",\"max_tokens\":-1", "");
    request_body = request_body.replace("\"max_tokens\":-1,", "");

    if !request_body.contains("\"model\":") {
        let repl = format!("\"model\":\"{}\",\"messages\":[", model_alias.as_ref());
        request_body = request_body.replace("\"messages\":[", &repl);
    }

    serde_json::from_str::<CreateChatCompletionRequest>(&request_body)
        .map_err(|e| {
            error!("error deserializing payload (expected as CreateChatCompletionRequest): {e}");
            StatusCode::UNPROCESSABLE_ENTITY
        })
        .map(|mut create_chat_completions_request| {
            if sent_from_ui {
                create_chat_completions_request.model = model_alias.as_ref().to_string();
                create_chat_completions_request
            } else {
                create_chat_completions_request
            }
        })
}
