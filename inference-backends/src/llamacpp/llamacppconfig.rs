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

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaCppConfigArgs {
    pub alias: String,
    pub api_key: String,

    //pub model_path: PathBuf,
    pub model_path: String,

    //pub mmproj_path: Option<PathBuf>,
    pub mmproj_path: Option<String>,

    pub prio: Option<u8>,
    pub threads: Option<i8>,
    pub n_gpu_layers: Option<u8>,
    pub jinja: bool,
    pub ctx_size: Option<u64>,
    pub no_mmap: bool,

    pub flash_attn: Option<String>,
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

        cmd.arg("--api-key");
        cmd.arg(&self.api_key);

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
            cmd.arg(ctx_size.to_string());
        }

        if let Some(flash_attn) = &self.flash_attn {
            cmd.arg("--flash-attn");
            cmd.arg(flash_attn);
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
