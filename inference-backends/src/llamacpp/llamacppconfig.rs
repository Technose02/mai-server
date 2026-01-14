use serde::{Deserialize, Serialize, de::Visitor};
use std::{collections::HashMap, sync::Arc};
use tokio::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaCppConfig {
    pub env_handle: Arc<HashMap<String, String>>,
    pub args_handle: Arc<LlamaCppConfigArgs>,
}

impl LlamaCppConfig {
    pub fn apply_args(&self, cmd: &mut Command) {
        self.args_handle.apply(cmd);
    }

    pub fn apply_env(&self, cmd: &mut Command) {
        for (key, val) in self.env_handle.iter() {
            cmd.env(key, val);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ContextSize {
    T8192,
    T16384,
    T32768,
    T65536,
    T131072,
    T262144,
}

impl From<&ContextSize> for u64 {
    fn from(value: &ContextSize) -> Self {
        match value {
            ContextSize::T8192 => 8192,
            ContextSize::T16384 => 16384,
            ContextSize::T32768 => 32768,
            ContextSize::T65536 => 65536,
            ContextSize::T131072 => 131072,
            ContextSize::T262144 => 262144,
        }
    }
}

impl From<u64> for ContextSize {
    fn from(value: u64) -> Self {
        match value {
            0..12288 => Self::T8192,
            12288..24576 => Self::T16384,
            24576..49152 => Self::T32768,
            49152..98304 => Self::T65536,
            98304..196608 => Self::T131072,
            _ => Self::T262144,
        }
    }
}

impl PartialEq for ContextSize {
    fn eq(&self, other: &Self) -> bool {
        Into::<u64>::into(self) == Into::<u64>::into(other)
    }
}

impl PartialOrd for ContextSize {
    fn ge(&self, other: &Self) -> bool {
        Into::<u64>::into(self) >= Into::<u64>::into(other)
    }
    fn gt(&self, other: &Self) -> bool {
        Into::<u64>::into(self) > Into::<u64>::into(other)
    }
    fn le(&self, other: &Self) -> bool {
        Into::<u64>::into(self) <= Into::<u64>::into(other)
    }
    fn lt(&self, other: &Self) -> bool {
        Into::<u64>::into(self) < Into::<u64>::into(other)
    }
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.gt(other) {
            Some(std::cmp::Ordering::Greater)
        } else if self.lt(other) {
            Some(std::cmp::Ordering::Less)
        } else {
            Some(std::cmp::Ordering::Equal)
        }
    }
}

#[derive(Debug, Clone)]
pub enum OnOffValue {
    On,
    Off,
    Other(String),
}

impl<'a, 's> From<&'a OnOffValue> for &'s str
where
    'a: 's,
{
    fn from(value: &'a OnOffValue) -> Self {
        match value {
            OnOffValue::Off => "off",
            OnOffValue::On => "on",
            OnOffValue::Other(s) => s,
        }
    }
}

impl From<OnOffValue> for String {
    fn from(value: OnOffValue) -> Self {
        match value {
            OnOffValue::Off => "off".into(),
            OnOffValue::On => "on".into(),
            OnOffValue::Other(s) => s,
        }
    }
}

impl From<String> for OnOffValue {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "on" => OnOffValue::On,
            "off" => OnOffValue::Off,
            _ => OnOffValue::Other(value),
        }
    }
}

impl PartialEq for OnOffValue {
    fn eq(&self, other: &Self) -> bool {
        match self {
            OnOffValue::On => matches!(other, OnOffValue::On),
            OnOffValue::Off => matches!(other, OnOffValue::Off),
            OnOffValue::Other(s) => {
                if let OnOffValue::Other(other_s) = other
                    && other_s == s
                {
                    true
                } else {
                    false
                }
            }
        }
    }
}

struct OnOffValueVisitor();

impl<'de> Visitor<'de> for OnOffValueVisitor {
    type Value = OnOffValue;

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Into::<OnOffValue>::into(v))
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Into::<OnOffValue>::into(String::from(v)))
    }

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("expected any String")
    }
}

impl<'de> Deserialize<'de> for OnOffValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_string(OnOffValueVisitor())
    }
}

impl Serialize for OnOffValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.into())
    }
}

struct ContextSizeVisitor();
impl<'de> Visitor<'de> for ContextSizeVisitor {
    type Value = ContextSize;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("expected a u64")
    }
    fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Into::<ContextSize>::into(v))
    }
}

impl<'de> Deserialize<'de> for ContextSize {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_u64(ContextSizeVisitor())
    }
}

impl Serialize for ContextSize {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(self.into())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaCppConfigArgs {
    pub alias: String,
    pub api_key: Option<String>,

    //pub model_path: PathBuf,
    pub model_path: String,

    //pub mmproj_path: Option<PathBuf>,
    pub mmproj_path: Option<String>,

    pub prio: Option<u8>,
    pub threads: Option<i8>,
    pub n_gpu_layers: Option<u8>,
    pub jinja: bool,
    pub ctx_size: Option<ContextSize>,
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

impl LlamaCppConfigArgs {
    fn apply(&self, cmd: &mut Command) {
        cmd.arg("--alias");
        cmd.arg(&self.alias);

        cmd.arg("--model");
        //cmd.arg(self.model_path.to_string_lossy().as_ref());
        cmd.arg(self.model_path.as_str());

        if let Some(api_key) = &self.api_key {
            cmd.arg("--api-key");
            cmd.arg(api_key);
        }

        if let Some(mmproj_path) = &self.mmproj_path {
            cmd.arg("--mmproj");
            //cmd.arg(mmproj_path.to_string_lossy().as_ref());
            cmd.arg(mmproj_path.as_str());
        }

        if let Some(prio) = self.prio {
            cmd.arg("--prio");
            cmd.arg(prio.to_string());
        }

        if let Some(threads) = self.threads {
            cmd.arg("--threads");
            cmd.arg(threads.to_string());
        }

        if let Some(n_gpu_layers) = self.n_gpu_layers {
            cmd.arg("--n-gpu-layers");
            cmd.arg(n_gpu_layers.to_string());
        }

        if self.jinja {
            cmd.arg("--jinja");
        }

        if self.no_mmap {
            cmd.arg("--no-mmap");
        }

        if let Some(ctx_size) = self.ctx_size {
            cmd.arg("--ctx-size");
            cmd.arg(Into::<u64>::into(&ctx_size).to_string());
        }

        if let Some(flash_attn) = &self.flash_attn {
            cmd.arg("--flash-attn");
            cmd.arg(Into::<String>::into(flash_attn.clone()));
        }

        if let Some(fit) = &self.fit {
            cmd.arg("--fit");
            cmd.arg(Into::<String>::into(fit.clone()));
        }

        if let Some(batch_size) = self.batch_size {
            cmd.arg("--batch-size");
            cmd.arg(batch_size.to_string());
        }

        if let Some(ubatch_size) = self.ubatch_size {
            cmd.arg("--ubatch-size");
            cmd.arg(ubatch_size.to_string());
        }

        if let Some(cache_type_v) = &self.cache_type_v {
            cmd.arg("--cache-type-v");
            cmd.arg(cache_type_v);
        }

        if let Some(cache_type_k) = &self.cache_type_k {
            cmd.arg("--cache-type-k");
            cmd.arg(cache_type_k);
        }

        if let Some(parallel) = self.parallel {
            cmd.arg("--parallel");
            cmd.arg(parallel.to_string());
        }

        if self.no_context_shift {
            cmd.arg("--no-context-shift");
        }

        if self.no_cont_batching {
            cmd.arg("--no-cont-batching");
        }

        if let Some(min_p) = self.min_p {
            cmd.arg("--min-p");
            cmd.arg(format!("{min_p:.2}"));
        }

        if let Some(temp) = self.temp {
            cmd.arg("--temp");
            cmd.arg(format!("{temp:.2}"));
        }

        if let Some(repeat_penalty) = self.repeat_penalty {
            cmd.arg("--repeat-penalty");
            cmd.arg(format!("{repeat_penalty:.2}"));
        }

        if let Some(presence_penalty) = self.presence_penalty {
            cmd.arg("--presence-penalty");
            cmd.arg(format!("{presence_penalty:.2}"));
        }

        if let Some(seed) = self.seed {
            cmd.arg("--seed");
            cmd.arg(seed.to_string());
        }

        if let Some(top_k) = self.top_k {
            cmd.arg("--top-k");
            cmd.arg(top_k.to_string());
        }

        if let Some(top_p) = self.top_p {
            cmd.arg("--top-p");
            cmd.arg(format!("{top_p:.2}"));
        }
    }
}
