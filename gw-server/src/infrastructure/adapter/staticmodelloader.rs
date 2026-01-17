use crate::{
    SecurityConfig,
    domain::{model::ModelConfiguration as DomainModelConfiguration, ports::ModelLoaderOutPort},
    infrastructure::model::ModelConfiguration,
};
use async_trait::async_trait;
use inference_backends::{ContextSize, LlamaCppConfig, LlamaCppConfigArgs};
use std::{collections::HashMap, sync::Arc};

pub struct StaticModelLoader {
    static_model_configuration_list: Vec<DomainModelConfiguration>,
    env_handle: Arc<HashMap<String, String>>,
    security_config: Arc<dyn SecurityConfig>,
}

impl StaticModelLoader {
    pub fn create_adapter(security_config: Arc<dyn SecurityConfig>) -> Arc<dyn ModelLoaderOutPort> {
        let mut env = HashMap::new();
        env.insert("GGML_CUDA_ENABLE_UNIFIED_MEMORY".into(), "1".into());
        Arc::new(Self {
            static_model_configuration_list: default_model_configuration_list(),
            env_handle: Arc::new(env),
            security_config,
        })
    }
}

#[async_trait]
impl ModelLoaderOutPort for StaticModelLoader {
    async fn get_model_configurations(&self) -> Vec<DomainModelConfiguration> {
        let mut models = Vec::with_capacity(self.static_model_configuration_list.len() * 6);
        for base in self.static_model_configuration_list.iter().cloned() {
            'inner: for ctx_size in [
                ContextSize::T8192,
                ContextSize::T16384,
                ContextSize::T32768,
                ContextSize::T65536,
                ContextSize::T131072,
                ContextSize::T262144,
            ] {
                if ctx_size > base.max_ctx_size {
                    break 'inner;
                }
                let mut base = base.clone();
                let alias = ContextSizeAwareAlias::from((base.alias, ctx_size));
                base.alias = alias.alias();
                //println!("adding model '{}' to list",base.alias);
                models.push(base);
            }
        }
        models
    }
    async fn get_model_configuration(&self, alias: &str) -> Result<Arc<LlamaCppConfig>, ()> {
        let alias = alias.to_owned();
        let (model_key, optional_context_size) =
            match ContextSizeAwareAlias::try_from(alias.clone()) {
                Ok(caa) => (caa.model(), Some(caa.context_size())),
                Err(_) => (alias.clone(), None),
            };

        if let Some(model_configuration) = self
            .static_model_configuration_list
            .iter()
            .find(|&config| config.alias == model_key)
        {
            Ok(Arc::new(LlamaCppConfig {
                env_handle: self.env_handle.clone(),
                args_handle: Arc::new(LlamaCppConfigArgs {
                    alias,
                    api_key: Some(self.security_config.get_apikey().to_string()),
                    model_path: model_configuration.model_path.clone(),
                    mmproj_path: model_configuration.mmproj_path.clone(),
                    prio: model_configuration.prio,
                    threads: model_configuration.threads,
                    n_gpu_layers: model_configuration.n_gpu_layers,
                    jinja: model_configuration.jinja,
                    ctx_size: optional_context_size,
                    no_mmap: model_configuration.no_mmap,
                    flash_attn: model_configuration.flash_attn.clone(),
                    fit: model_configuration.fit.clone(),
                    batch_size: model_configuration.batch_size,
                    ubatch_size: model_configuration.ubatch_size,
                    cache_type_k: model_configuration.cache_type_k.clone(),
                    cache_type_v: model_configuration.cache_type_v.clone(),
                    parallel: model_configuration.parallel,
                    no_context_shift: model_configuration.no_context_shift,
                    no_cont_batching: model_configuration.no_cont_batching,
                    min_p: model_configuration.min_p,
                    temp: model_configuration.temp,
                    repeat_penalty: model_configuration.repeat_penalty,
                    presence_penalty: model_configuration.presence_penalty,
                    seed: model_configuration.seed,
                    top_k: model_configuration.top_k,
                    top_p: model_configuration.top_p,
                }),
            }))
        } else {
            eprintln!("no model-configuration found for alias '{model_key}'");
            Err(())
        }
    }
}

fn default_model_configuration_list() -> Vec<DomainModelConfiguration> {
    let list: Vec<ModelConfiguration> = serde_json::from_str(r#"
[{
    "alias" : "devstral-small-2-24B-instruct-2512",
    "model-path" : "/model_data/huggingface/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-UD-Q8_K_XL.gguf",
    "mmproj-path" : "/model_data/huggingface/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/mmproj-F16.gguf",
    "prio" : 3,
    "min-p" : 0.01,
    "threads" : 8,
    "n-gpu-layers" : 99,
    "jinja" : true,
    "max-ctx-size" : 262144,
    "no-mmap" : true,
    "flash-attn" : "on",
    "vocab-type" : 2,
    "n-vocab": 131072,
    "n-ctx-train" : 393216,
    "n-embd" : 5120,
    "n-params" : 23572403200,
    "size" : 28983971840,
    "capabilities": [
        "completion",
        "multimodal"
    ]
},
{
    "alias" : "gemma-3-12b-it-qat-Q8_0",
    "model-path" : "/model_data/huggingface/unsloth/gemma-3-12b-it-qat-GGUF/gemma-3-12b-it-qat-Q8_0.gguf",
    "mmproj-path" : "/model_data/huggingface/unsloth/gemma-3-12b-it-qat-GGUF/mmproj-BF16.gguf",
    "n-gpu-layers" : 99,
    "jinja" : true,
    "max-ctx-size" : 131072,
    "no-mmap" : true,
    "flash-attn" : "on",
    "prio": 2,
    "threads": 8,
    "temp": 1.0,
    "repeat-penalty": 1.0,
    "seed": 3407,
    "min-p": 0.01,
    "top-k": 64,
    "top-p": 0.95,
    "vocab-type" : 1,
    "n-vocab": 262208,
    "n-ctx-train" : 1048576,
    "n-embd" : 3840,
    "n-params" : 11766034176,
    "size" : 12503660544,
    "capabilities": [
        "completion",
        "multimodal"
    ]
},
{
    "alias" : "gemma-3-27b-it-qat-Q8_0",
    "model-path" : "/model_data/huggingface/unsloth/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q8_0.gguf",
    "mmproj-path" : "/model_data/huggingface/unsloth/gemma-3-27b-it-GGUF/mmproj-BF16.gguf",
    "n-gpu-layers" : 99,
    "jinja" : true,
    "max-ctx-size" : 131072,
    "no-mmap" : true,
    "flash-attn" : "on",
    "prio": 2,
    "threads": 8,
    "temp": 1.0,
    "repeat-penalty": 1.0,
    "seed": 3407,
    "min-p": 0.01,
    "top-k": 64,
    "top-p": 0.95,
    "vocab-type" : 1,
    "n-vocab": 262208,
    "n-ctx-train" : 1048576,
    "n-embd" : 5376,
    "n-params" : 27009346304,
    "size" : 28701409280,
    "capabilities": [
        "completion",
        "multimodal"
    ]
},
{
    "alias" : "glm-4.6v-flash",
    "model-path" : "/model_data/huggingface/unsloth/GLM-4.6V-Flash-GGUF/GLM-4.6V-Flash-BF16.gguf",
    "mmproj-path" : "/model_data/huggingface/unsloth/GLM-4.6V-Flash-GGUF/mmproj-F16.gguf",
    "n-gpu-layers" : 99,
    "jinja" : true,
    "max-ctx-size" : 16384,
    "flash-attn" : "on",
    "temp": 0.8,
    "top-p": 0.6,
    "top-k": 2,
    "repeat-penalty": 1.1,
    "prio": 3,
    "vocab-type" : 2,
    "n-vocab": 151552,
    "n-ctx-train" : 131072,
    "n-embd" : 4096,
    "n-params" : 9400279040,
    "size" : 18802245632,
    "capabilities": [
        "completion",
        "multimodal"
    ]
},
{
    "alias" : "gpt-oss-120b-Q8_0",
    "model-path" : "/model_data/huggingface/unsloth/gpt-oss-120b-GGUF/Q8_0/gpt-oss-120b-Q8_0-00001-of-00002.gguf",
    "n-gpu-layers" : 99,
    "jinja" : true,
    "max-ctx-size" : 32768,
    "no-mmap" : true,
    "flash-attn" : "on",
    "batch-size" : 2048,
    "ubatch-size" : 2048,
    "cache-type_v" : "q8_0",
    "cache-type_k" : "q8_0",
    "parallel" : 1,
    "no-context-shift" : true,
    "no-cont-batching" : true,
    "vocab-type" : 2,
    "n-vocab": 201088,
    "n-ctx-train" : 131072,
    "n-embd" : 2880,
    "n-params" : 116829156672,
    "size" : 63374323968,
    "capabilities": [
        "completion"
    ]
},
{
    "alias" : "granite-4.0-h-small",
    "model-path" : "/model_data/huggingface/unsloth/granite-4.0-h-small-GGUF/granite-4.0-h-small-Q8_0.gguf",
    "n-gpu-layers" : 99,
    "jinja" : true,
    "max-ctx-size" : 131072,
    "no-mmap" : true,
    "flash-attn" : "on",
    "threads": 8,
    "temp": 0.0,
    "top-k": 0,
    "top-p": 1.0,
    "vocab-type" : 2,
    "n-vocab": 100352,
    "n-ctx-train" : 1048576,
    "n-embd" : 4096,
    "n-params" : 32207337984,
    "size" : 34261297152,
    "capabilities": [
        "completion"
    ]
},
{
    "alias" : "nemotron-3-nano-30b-a3b",
    "model-path" : "//model_data/huggingface/unsloth/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf",
    "n-gpu-layers" : 99,
    "jinja" : true,
    "threads": 8,
    "max-ctx-size": 32768,
    "temp": 1.0,
    "top-p": 1.0,
    "fit":  "on",
    "flash-attn" : "on",
    "vocab-type" : 2,
    "n-vocab": 131072,
    "n-ctx-train" : 1048576,
    "n-embd" : 2688,
    "n-params" : 31577940288,
    "size" : 40440063744,
    "capabilities": [
        "completion"
    ]
},
{
    "alias" : "phi-4-reasoning-plus",
    "model-path" : "/model_data/huggingface/unsloth/Phi-4-reasoning-plus-GGUF/Phi-4-reasoning-plus-Q8_0.gguf",
    "jinja" : true,
    "prio": 3,
    "threads": 8,
    "max-ctx-size" : 32768,
    "n-gpu-layers" : 99,
    "temp": 0.8,
    "top-p": 0.95,
    "min-p": 0.00,
    "vocab-type" : 2,
    "n-vocab": 100352,
    "n-ctx-train" : 32768,
    "n-embd" : 5120,
    "n-params" : 14659507200,
    "size" : 15576944640,
    "capabilities": [
        "completion"
    ]
},
{
    "alias" : "qwen3-vl-30b-a3b-instruct",
    "model-path" : "/model_data/huggingface/unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF/BF16/Qwen3-VL-30B-A3B-Instruct-BF16-00001-of-00002.gguf",
    "mmproj-path" : "/model_data/huggingface/unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-BF16.gguf",
    "n-gpu-layers" : 99,
    "jinja" : true,
    "top-p": 0.8,
    "top-k": 20,
    "temp": 0.7,
    "min-p": 0.0,
    "flash-attn" : "on",
    "max-ctx-size" : 131072,
    "presence-penalty": 1.5,
    "vocab-type" : 2,
    "n-vocab": 151936,
    "n-ctx-train" : 262144,
    "n-embd" : 2048,
    "n-params" : 30532122624,
    "size" : 61089832960,
    "capabilities": [
        "completion",
        "multimodal"
    ]
},
{
    "alias" : "qwen3-vl-30b-a3b-thinking",
    "model-path" : "/model_data/huggingface/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF/BF16/Qwen3-VL-30B-A3B-Thinking-BF16-00001-of-00002.gguf",
    "mmproj-path" : "/model_data/huggingface/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF/mmproj-BF16.gguf",
    "n-gpu-layers" : 99,
    "jinja" : true,
    "top-p": 0.95,
    "top-k": 20,
    "temp": 1.0,
    "min-p": 0.0,
    "flash-attn": "on",
    "max-ctx-size" : 131072,
    "presence-penalty": 0.0,
    "vocab-type" : 2,
    "n-vocab": 151936,
    "n-ctx-train" : 262144,
    "n-embd" : 2048,
    "n-params" : 30532122624,
    "size" : 61089832960,
    "capabilities": [
        "completion",
        "multimodal"
    ]
}]
"#).unwrap();
    list.into_iter()
        .map(ModelConfiguration::into_domain)
        .collect()
}

const CTX_SIZE_HINT_T262144: &str = "max";
const CTX_SIZE_HINT_T131072: &str = "large";
const CTX_SIZE_HINT_T65536: &str = "moderate";
const CTX_SIZE_HINT_T32768: &str = "small";
const CTX_SIZE_HINT_T16384: &str = "tiny";
const CTX_SIZE_HINT_T8192: &str = "min";

pub struct ContextSizeAwareAlias(String, ContextSize);

impl ContextSizeAwareAlias {
    pub fn model(&self) -> String {
        self.0.clone()
    }
    pub fn context_size(&self) -> ContextSize {
        self.1
    }
    pub fn alias(&self) -> String {
        let suffix = match self.context_size() {
            ContextSize::T262144 => CTX_SIZE_HINT_T262144,
            ContextSize::T131072 => CTX_SIZE_HINT_T131072,
            ContextSize::T65536 => CTX_SIZE_HINT_T65536,
            ContextSize::T32768 => CTX_SIZE_HINT_T32768,
            ContextSize::T16384 => CTX_SIZE_HINT_T16384,
            ContextSize::T8192 => CTX_SIZE_HINT_T8192,
        };
        format!("{}-{suffix}", self.0)
    }
}

impl From<(String, ContextSize)> for ContextSizeAwareAlias {
    fn from(value: (String, ContextSize)) -> Self {
        Self(value.0, value.1)
    }
}

impl TryFrom<String> for ContextSizeAwareAlias {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if let Some((alias, context_size_hint)) = value.rsplit_once("-") {
            let context_size = match context_size_hint {
                CTX_SIZE_HINT_T262144 => ContextSize::T262144,
                CTX_SIZE_HINT_T131072 => ContextSize::T131072,
                CTX_SIZE_HINT_T65536 => ContextSize::T65536,
                CTX_SIZE_HINT_T32768 => ContextSize::T32768,
                CTX_SIZE_HINT_T16384 => ContextSize::T16384,
                CTX_SIZE_HINT_T8192 => ContextSize::T8192,
                _ => {
                    return Err(format!(
                        "value '{value}' contains invalid content-size-hint '{context_size_hint}'"
                    ));
                }
            };
            Ok(ContextSizeAwareAlias(alias.into(), context_size))
        } else {
            Err(format!(
                "value '{value}' does not contain context-size-hint"
            ))
        }
    }
}
